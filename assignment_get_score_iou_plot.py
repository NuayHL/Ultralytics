from ultralytics import YOLO
from ultralytics.utils.ops import Profile, xywh2xyxy
from ultralytics.utils import LOGGER, TQDM, callbacks, colorstr, emojis
from ultralytics.nn.autobackend import AutoBackend
from ultralytics.cfg import get_cfg, get_save_dir
from ultralytics.data.utils import check_cls_dataset, check_det_dataset
from ultralytics.utils.checks import check_imgsz
from ultralytics.utils.torch_utils import attempt_compile, select_device, smart_inference_mode, unwrap_model
from ultralytics.utils.metrics import box_iou

import json
import torch
import os
import pickle
import matplotlib.pyplot as plt
from pathlib import Path

import numpy as np

from ultralytics.engine.validator import BaseValidator
from ultralytics.models.yolo.detect.val import DetectionValidator


def _draw_score_iou_plot(before_nms, after_nms, conf_threshold, output_path):
    """绘制score与iou散点图（子图为正方形范围0-1）"""
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 6))

    # 绘制postprocess之前的散点图
    if before_nms:
        scores_before, ious_before = zip(*before_nms)
        ax1.scatter(scores_before, ious_before, alpha=0.5, s=1)
        ax1.set_title(f'Score vs IoU (Before NMS, conf>{conf_threshold})\nTotal: {len(before_nms)} points')
    else:
        ax1.text(0.5, 0.5, 'No data', ha='center', va='center', transform=ax1.transAxes)
        ax1.set_title('Score vs IoU (Before NMS) - No Data')

    # 绘制postprocess之后的散点图
    if after_nms:
        scores_after, ious_after = zip(*after_nms)
        ax2.scatter(scores_after, ious_after, alpha=0.5, s=1)
        ax2.set_title(f'Score vs IoU (After NMS)\nTotal: {len(after_nms)} points')
    else:
        ax2.text(0.5, 0.5, 'No data', ha='center', va='center', transform=ax2.transAxes)
        ax2.set_title('Score vs IoU (After NMS) - No Data')

    for ax, xlabel in zip((ax1, ax2), ('Score (Confidence)', 'Score (Confidence)')):
        ax.set_xlabel(xlabel)
    ax1.set_ylabel('IoU with Matched GT')
    ax2.set_ylabel('IoU with Matched GT')

    for ax in (ax1, ax2):
        ax.grid(True, alpha=0.3)
        ax.set_xlim(0, 1)
        ax.set_ylim(0, 1)
        ax.set_aspect('equal', adjustable='box')  # 子图保持正方形

    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close(fig)
    LOGGER.info(f"Score-IoU plot saved to {output_path}")


def plot_score_iou_from_pkl(pkl_path, conf_threshold=0.001, output_path=None):
    """
    从pkl文件中读取score & IoU数据并绘制散点图

    Args:
        pkl_path (str | Path): score_iou_data.pkl 的路径
        conf_threshold (float): 绘制标题中使用的阈值说明
        output_path (str | Path | None): 输出图像路径，默认与pkl同目录的score_iou_plot.png
    """
    pkl_path = Path(pkl_path)
    if not pkl_path.is_file():
        raise FileNotFoundError(f"{pkl_path} not found")

    with open(pkl_path, 'rb') as f:
        data = pickle.load(f)

    before_nms = data.get('before_nms', [])
    after_nms = data.get('after_nms', [])

    if output_path is None:
        output_path = pkl_path.with_name("score_iou_plot.png")

    _draw_score_iou_plot(before_nms, after_nms, conf_threshold, output_path)


def val_with_record(model_path, dataset_path, dir_name, project_path='runs_assignment', conf_threshold=0.001):
    """
    验证模型并绘制score与iou的散点图
    
    Args:
        model_path: 模型路径
        dataset_path: 数据集路径
        dir_name: 保存目录名称
        project_path: 项目路径
        conf_threshold: 置信度阈值，用于过滤postprocess之前的预测
    """
    class DetectionValidatorWithSaveScoreIou(DetectionValidator):
        def __init__(self, dataloader=None, save_dir=None, args=None, _callbacks=None):
            super().__init__(dataloader, save_dir, args, _callbacks)
            # 存储散点图数据
            self.score_iou_before_nms = []  # (score, iou) 对，postprocess之前
            self.score_iou_after_nms = []   # (score, iou) 对，postprocess之后
            self.conf_threshold = conf_threshold
            
        @smart_inference_mode()
        def __call__(self, trainer=None, model=None):
            """重写__call__方法，在postprocess之前收集数据"""
            self.training = trainer is not None
            augment = self.args.augment and (not self.training)
            if self.training:
                self.device = trainer.device
                self.data = trainer.data
                self.args.half = self.device.type != "cpu" and trainer.amp
                model = trainer.ema.ema or trainer.model
                if trainer.args.compile and hasattr(model, "_orig_mod"):
                    model = model._orig_mod
                model = model.half() if self.args.half else model.float()
                self.loss = torch.zeros_like(trainer.loss_items, device=trainer.device)
                self.args.plots &= trainer.stopper.possible_stop or (trainer.epoch == trainer.epochs - 1)
                model.eval()
            else:
                if str(self.args.model).endswith(".yaml") and model is None:
                    LOGGER.warning("validating an untrained model YAML will result in 0 mAP.")
                callbacks.add_integration_callbacks(self)
                model = AutoBackend(
                    model=model or self.args.model,
                    device=select_device(self.args.device, self.args.batch),
                    dnn=self.args.dnn,
                    data=self.args.data,
                    fp16=self.args.half,
                )
                self.device = model.device
                self.args.half = model.fp16
                stride, pt, jit = model.stride, model.pt, model.jit
                imgsz = check_imgsz(self.args.imgsz, stride=stride)
                if not (pt or jit or getattr(model, "dynamic", False)):
                    self.args.batch = model.metadata.get("batch", 1)
                    LOGGER.info(f"Setting batch={self.args.batch} input of shape ({self.args.batch}, 3, {imgsz}, {imgsz})")

                if str(self.args.data).rsplit(".", 1)[-1] in {"yaml", "yml"}:
                    self.data = check_det_dataset(self.args.data)
                elif self.args.task == "classify":
                    self.data = check_cls_dataset(self.args.data, split=self.args.split)
                else:
                    raise FileNotFoundError(emojis(f"Dataset '{self.args.data}' for task={self.args.task} not found ❌"))

                if self.device.type in {"cpu", "mps"}:
                    self.args.workers = 0
                if not (pt or (getattr(model, "dynamic", False) and not model.imx)):
                    self.args.rect = False
                self.stride = model.stride
                self.dataloader = self.dataloader or self.get_dataloader(self.data.get(self.args.split), self.args.batch)

                model.eval()
                if self.args.compile:
                    model = attempt_compile(model, device=self.device)
                model.warmup(imgsz=(1 if pt else self.args.batch, self.data["channels"], imgsz, imgsz))

            self.run_callbacks("on_val_start")
            dt = (
                Profile(device=self.device),
                Profile(device=self.device),
                Profile(device=self.device),
                Profile(device=self.device),
            )
            bar = TQDM(self.dataloader, desc=self.get_desc(), total=len(self.dataloader))
            self.init_metrics(unwrap_model(model))
            self.jdict = []
            # 重置数据存储
            self.score_iou_before_nms = []
            self.score_iou_after_nms = []
            
            for batch_i, batch in enumerate(bar):
                self.run_callbacks("on_val_batch_start")
                self.batch_i = batch_i
                # Preprocess
                with dt[0]:
                    batch = self.preprocess(batch)

                # Inference
                with dt[1]:
                    preds_raw = model(batch["img"], augment=augment)

                # Loss
                with dt[2]:
                    if self.training:
                        self.loss += model.loss(batch, preds_raw)[1]

                # 在postprocess之前收集数据
                self._collect_before_nms(preds_raw, batch)

                # Postprocess
                with dt[3]:
                    preds = self.postprocess(preds_raw)

                self.update_metrics(preds, batch)
                if self.args.plots and batch_i < 3:
                    self.plot_val_samples(batch, batch_i)
                    self.plot_predictions(batch, preds, batch_i)

                self.run_callbacks("on_val_batch_end")
            
            stats = self.get_stats()
            self.speed = dict(zip(self.speed.keys(), (x.t / len(self.dataloader.dataset) * 1e3 for x in dt)))
            self.finalize_metrics()
            self.print_results()
            self.run_callbacks("on_val_end")
            
            # 绘制散点图
            self._plot_score_iou()
            
            if self.training:
                model.float()
                results = {**stats, **trainer.label_loss_items(self.loss.cpu() / len(self.dataloader), prefix="val")}
                return {k: round(float(v), 5) for k, v in results.items()}
            else:
                LOGGER.info(
                    "Speed: {:.1f}ms preprocess, {:.1f}ms inference, {:.1f}ms loss, {:.1f}ms postprocess per image".format(
                        *tuple(self.speed.values())
                    )
                )
                if self.args.save_json and self.jdict:
                    with open(str(self.save_dir / "predictions.json"), "w", encoding="utf-8") as f:
                        LOGGER.info(f"Saving {f.name}...")
                        json.dump(self.jdict, f)
                    stats = self.eval_json(stats)
                if self.args.plots or self.args.save_json:
                    LOGGER.info(f"Results saved to {colorstr('bold', self.save_dir)}")
                return stats
        
        def _collect_before_nms(self, preds_raw, batch):
            """在postprocess之前收集score和iou数据"""
            if isinstance(preds_raw, (list, tuple)):
                preds_raw = preds_raw[0]  # 只取推理输出
            
            # 处理每个batch中的每张图片
            for si in range(len(batch["im_file"])):
                # 准备ground truth
                pbatch = self._prepare_batch(si, batch)
                if len(pbatch["cls"]) == 0:
                    continue  # 没有ground truth，跳过
                
                # 处理原始预测，与NMS函数逻辑保持一致
                if preds_raw.shape[-1] == 6:  # end-to-end模型，格式(batch, num_detections, 6)
                    pred_img = preds_raw[si]  # (num_detections, 6)
                    boxes = pred_img[:, :4]  # xyxy格式
                    conf = pred_img[:, 4]
                    cls = pred_img[:, 5].long()
                else:
                    # 标准YOLO格式: (batch, num_classes+4, num_anchors)
                    pred_img = preds_raw[si]  # (num_classes+4, num_anchors)
                    nc = pred_img.shape[0] - 4  # number of classes
                    pred_img = pred_img.transpose(-1, -2)  # (num_anchors, num_classes+4)
                    
                    # 转换为xyxy格式（与NMS一致）
                    pred_img = pred_img.clone()
                    pred_img[:, :4] = xywh2xyxy(pred_img[:, :4])
                    
                    # 提取box和class scores
                    boxes = pred_img[:, :4]  # xyxy格式
                    cls_scores = pred_img[:, 4:4+nc]  # (num_anchors, nc)
                    
                    # 获取每个anchor的最大class score和对应的class index（与NMS的best class only模式一致）
                    conf, cls_idx = cls_scores.max(1)
                    cls = cls_idx.long()
                
                # 过滤置信度大于threshold的预测
                mask = conf > self.conf_threshold
                if not mask.any():
                    continue
                
                boxes = boxes[mask]
                conf = conf[mask]
                cls = cls[mask]
                
                # 计算与ground truth的iou
                gt_boxes = pbatch["bboxes"]
                gt_cls = pbatch["cls"]
                
                if len(boxes) == 0 or len(gt_boxes) == 0:
                    continue
                
                iou_matrix = box_iou(boxes, gt_boxes)  # (N_pred, M_gt)
                
                # 匹配预测和ground truth（考虑类别匹配）
                # 确保gt_cls是tensor格式以便比较
                if not isinstance(gt_cls, torch.Tensor):
                    gt_cls = torch.tensor(gt_cls, device=boxes.device)
                
                for i, (pred_conf, pred_cls) in enumerate(zip(conf, cls)):
                    # 只考虑类别匹配的ground truth
                    cls_match = (gt_cls == pred_cls)
                    if not cls_match.any():
                        continue  # 没有匹配的类别
                    
                    # 获取与匹配类别的最大iou
                    matched_ious = iou_matrix[i][cls_match]
                    if len(matched_ious) > 0:
                        max_iou = matched_ious.max().item()
                        self.score_iou_before_nms.append((pred_conf.item(), max_iou))
        
        def update_metrics(self, preds, batch):
            """重写update_metrics，在postprocess之后收集数据"""
            for si, pred in enumerate(preds):
                self.seen += 1
                pbatch = self._prepare_batch(si, batch)
                predn = self._prepare_pred(pred)
                
                # 收集postprocess后的数据
                if len(predn["cls"]) > 0 and len(pbatch["cls"]) > 0:
                    # 计算iou
                    iou_matrix = box_iou(predn["bboxes"], pbatch["bboxes"])
                    # 确保类型一致
                    if isinstance(predn["cls"], torch.Tensor):
                        pred_cls = predn["cls"].cpu().numpy()
                    else:
                        pred_cls = np.array(predn["cls"])
                    
                    if isinstance(pbatch["cls"], torch.Tensor):
                        gt_cls = pbatch["cls"].cpu().numpy()
                    else:
                        gt_cls = np.array(pbatch["cls"])
                    
                    if isinstance(predn["conf"], torch.Tensor):
                        conf = predn["conf"].cpu().numpy()
                    else:
                        conf = np.array(predn["conf"])
                    
                    # 匹配预测和ground truth
                    for i, (pred_c, pred_conf) in enumerate(zip(pred_cls, conf)):
                        cls_match = (gt_cls == pred_c)
                        if cls_match.any():
                            matched_ious = iou_matrix[i][cls_match]
                            if len(matched_ious) > 0:
                                max_iou = matched_ious.max().item()
                                self.score_iou_after_nms.append((float(pred_conf), max_iou))
                
                # 调用父类方法更新指标
                cls = pbatch["cls"].cpu().numpy()
                no_pred = len(predn["cls"]) == 0
                self.metrics.update_stats(
                    {
                        **self._process_batch(predn, pbatch),
                        "target_cls": cls,
                        "target_img": np.unique(cls),
                        "conf": np.zeros(0) if no_pred else predn["conf"].cpu().numpy(),
                        "pred_cls": np.zeros(0) if no_pred else predn["cls"].cpu().numpy(),
                    }
                )
                # Evaluate
                if self.args.plots:
                    self.confusion_matrix.process_batch(predn, pbatch, conf=self.args.conf)
                    if self.args.visualize:
                        self.confusion_matrix.plot_matches(batch["img"][si], pbatch["im_file"], self.save_dir)

                if no_pred:
                    continue

                # Save
                if self.args.save_json or self.args.save_txt:
                    predn_scaled = self.scale_preds(predn, pbatch)
                if self.args.save_json:
                    self.pred_to_json(predn_scaled, pbatch)
                if self.args.save_txt:
                    self.save_one_txt(
                        predn_scaled,
                        self.args.save_conf,
                        pbatch["ori_shape"],
                        self.save_dir / "labels" / f"{Path(pbatch['im_file']).stem}.txt",
                    )
        
        def _plot_score_iou(self):
            """保存散点数据并绘制图像"""
            save_dir = Path(self.save_dir)

            data_path = save_dir / "score_iou_data.pkl"
            with open(data_path, 'wb') as f:
                pickle.dump(
                    {
                        'before_nms': self.score_iou_before_nms,
                        'after_nms': self.score_iou_after_nms,
                    },
                    f,
                )
            LOGGER.info(f"Score-IoU data saved to {data_path}")

            plot_score_iou_from_pkl(
                data_path,
                conf_threshold=self.conf_threshold,
                output_path=save_dir / "score_iou_plot.png",
            )
    
    model = YOLO(model_path)
    model.val(data=dataset_path, name=dir_name, project=project_path, validator=DetectionValidatorWithSaveScoreIou)


if __name__ == "__main__":

    model_path = "runs/detect/visdrone/v12s_assign_scale/weights/best.pt"
    dataset_path = "VisDrone.yaml"
    dir_name = "yolo12s_assign_scale_visdrone"
    project_path = "runs_assignment"

    val_with_record(model_path=model_path,
                    dataset_path=dataset_path,
                    dir_name=dir_name,
                    project_path=project_path)

    # plot_score_iou_from_pkl(pkl_path=f"{project_path}/{dir_name}/score_iou_data.pkl",
    #                         conf_threshold=0.001,
    #                         output_path=f"{project_path}/{dir_name}/score_iou_plot.png")


