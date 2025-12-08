from ultralytics import YOLO
from ultralytics.utils.ops import Profile
from ultralytics.utils import LOGGER, TQDM, callbacks, colorstr, emojis
from ultralytics.nn.autobackend import AutoBackend
from ultralytics.cfg import get_cfg, get_save_dir
from ultralytics.data.utils import check_cls_dataset, check_det_dataset
from ultralytics.utils.checks import check_imgsz
from ultralytics.utils.torch_utils import attempt_compile, select_device, smart_inference_mode, unwrap_model

import json
import torch
import os
import pickle

import numpy as np

from ultralytics.utils.tal import TaskAlignedAssigner
from ultralytics.engine.validator import BaseValidator
from ultralytics.models.yolo.detect.val import DetectionValidator
from ultralytics.utils.mla_dab import TaskAlignedAssigner_VaryingIoU_Sep

from assignment_utils import load_hyper_pair


_STEP = 0
_SAVE_DIR = "assign_detail"
os.makedirs(_SAVE_DIR, exist_ok=True)


def compress_mask_pos(mask_pos):
    """
    压缩mask_pos为稀疏格式，只存储True的anchor索引
    
    Args:
        mask_pos: numpy array of shape (b, max_num_obj, h*w) or torch.Tensor
                布尔数组，表示每个gt分配给哪些anchor
    
    Returns:
        dict: 包含压缩后的数据和元信息
            {
                'format': 'sparse',
                'shape': (b, max_num_obj, num_anchors),
                'dtype': 'uint16' or 'uint32',  # 根据anchor数量选择，避免溢出
                'data': list of list of arrays,  # [batch][gt_idx] = array of anchor indices
            }
    """
    # 转换为numpy
    if isinstance(mask_pos, torch.Tensor):
        mask_pos_np = mask_pos.detach().cpu().numpy()
    else:
        mask_pos_np = mask_pos
    
    b, max_num_obj, num_anchors = mask_pos_np.shape
    index_dtype = np.uint16 if num_anchors <= np.iinfo(np.uint16).max else np.uint32
    
    # 稀疏存储：对于每个(batch, gt_idx)，只存储True的anchor索引
    sparse_data = []
    for b_idx in range(b):
        batch_sparse = []
        for gt_idx in range(max_num_obj):
            # 获取该gt的anchor索引（True的位置）
            anchor_indices = np.where(mask_pos_np[b_idx, gt_idx, :])[0]
            # 存储为紧凑整数类型以节省空间
            batch_sparse.append(anchor_indices.astype(index_dtype))
        sparse_data.append(batch_sparse)
    
    # print(f"sparse_data_len: {[len(batch) for batch in sparse_data]}")

    return {
        'format': 'sparse',
        'shape': (b, max_num_obj, num_anchors),
        'dtype': index_dtype.__name__,
        'data': sparse_data,
    }


def decompress_mask_pos(compressed_mask_pos):
    """
    解压缩mask_pos，从稀疏格式恢复为dense格式
    
    Args:
        compressed_mask_pos: dict from compress_mask_pos, or numpy array (向后兼容)
    
    Returns:
        numpy.ndarray: shape (b, max_num_obj, num_anchors), dtype=bool
    """
    # 向后兼容：如果已经是numpy数组，直接返回
    if isinstance(compressed_mask_pos, np.ndarray):
        return compressed_mask_pos
    
    # 如果是dict格式（压缩格式）
    if isinstance(compressed_mask_pos, dict):
        if compressed_mask_pos.get('format') == 'sparse':
            b, max_num_obj, num_anchors = compressed_mask_pos['shape']
            sparse_data = compressed_mask_pos['data']
            index_dtype = compressed_mask_pos.get('dtype', 'uint16')
            
            # 恢复为dense格式
            dense_mask = np.zeros((b, max_num_obj, num_anchors), dtype=np.bool_)
            for b_idx, batch_sparse in enumerate(sparse_data):
                for gt_idx, anchor_indices in enumerate(batch_sparse):
                    if len(anchor_indices) > 0:
                        # 将索引转换为int64，然后设置True
                        dense_mask[b_idx, gt_idx, anchor_indices.astype(np.int64)] = True
            
            return dense_mask
        else:
            raise ValueError(f"Unknown mask_pos format: {compressed_mask_pos.get('format')}")
    else:
        raise TypeError(f"Unsupported mask_pos type: {type(compressed_mask_pos)}")


def save_assign_info(dir_name, **kwargs):
    """
    保存assignment信息，只保存计算稳定性需要的关键信息
    自动压缩mask_pos以节省存储空间
    anc_points单独保存一次（因为所有batch都一样）
    """
    save_path = os.path.join(dir_name, f"{_STEP}.pkl")
    save_dict = {}
    
    # anc_points单独保存到anc_points.pkl（只在第一次保存）
    anc_points_path = os.path.join(dir_name, "anc_points.pkl")
    if 'anc_points' in kwargs and not os.path.exists(anc_points_path):
        anc_points = kwargs['anc_points']
        if isinstance(anc_points, torch.Tensor):
            anc_points_np = anc_points.detach().cpu().numpy()
        else:
            anc_points_np = anc_points
        with open(anc_points_path, 'wb') as f:
            pickle.dump(anc_points_np, f)
    
    # 保存其他数据（不包括anc_points）
    for key, value in kwargs.items():
        if key == 'anc_points':
            continue  # anc_points已经单独保存，跳过
        
        if isinstance(value, torch.Tensor):
            value_np = value.detach().cpu().numpy()
            
            # 特殊处理mask_pos：使用稀疏格式压缩
            if key == 'mask_pos':
                compressed = compress_mask_pos(value_np)
                # 可选校验：确保压缩/解压无损（仅在环境变量打开时启用）
                if os.environ.get("ASSIGN_VERIFY_COMPRESS", "0") == "1":
                    restored = decompress_mask_pos(compressed)
                    if restored.shape != value_np.shape or not np.array_equal(restored, value_np):
                        raise ValueError("mask_pos compress/decompress mismatch detected")
                save_dict[key] = compressed
            else:
                save_dict[key] = value_np
        else:
            save_dict[key] = value
    
    with open(save_path, 'wb') as f:
        pickle.dump(save_dict, f)

class TaskAlignedAssigner_Record(TaskAlignedAssigner):
    """用于记录标准TaskAlignedAssigner的assignment信息"""
    def __init__(self, topk: int = 13, num_classes: int = 80, alpha: float = 1.0, beta: float = 6.0, eps: float = 1e-9,
                 dir_name: str = 'test'):
        super().__init__()
        self.topk = topk
        self.num_classes = num_classes
        self.alpha = alpha
        self.beta = beta
        self.eps = eps
        self.dir_name = os.path.join(_SAVE_DIR, dir_name)
        os.makedirs(self.dir_name, exist_ok=True)

    @torch.no_grad()
    def forward(self, pd_scores, pd_bboxes, anc_points, gt_labels, gt_bboxes, mask_gt, stride=None, **kwargs):
        """
        Compute the task-aligned assignment.

        Args:
            pd_scores (torch.Tensor): Predicted classification scores with shape (bs, num_total_anchors, num_classes).
            pd_bboxes (torch.Tensor): Predicted bounding boxes with shape (bs, num_total_anchors, 4).
            anc_points (torch.Tensor): Anchor points with shape (num_total_anchors, 2).
            gt_labels (torch.Tensor): Ground truth labels with shape (bs, n_max_boxes, 1).
            gt_bboxes (torch.Tensor): Ground truth boxes with shape (bs, n_max_boxes, 4).
            mask_gt (torch.Tensor): Mask for valid ground truth boxes with shape (bs, n_max_boxes, 1).

        Returns:
            target_labels (torch.Tensor): Target labels with shape (bs, num_total_anchors).
            target_bboxes (torch.Tensor): Target bounding boxes with shape (bs, num_total_anchors, 4).
            target_scores (torch.Tensor): Target scores with shape (bs, num_total_anchors, num_classes).
            fg_mask (torch.Tensor): Foreground mask with shape (bs, num_total_anchors).
            target_gt_idx (torch.Tensor): Target ground truth indices with shape (bs, num_total_anchors).

        References:
            https://github.com/Nioolek/PPYOLOE_pytorch/blob/master/ppyoloe/assigner/tal_assigner.py
        """
        self.bs = pd_scores.shape[0]
        self.n_max_boxes = gt_bboxes.shape[1]
        device = gt_bboxes.device
        global _STEP
        _STEP += 1

        if self.n_max_boxes == 0:
            return (
                torch.full_like(pd_scores[..., 0], self.num_classes),
                torch.zeros_like(pd_bboxes),
                torch.zeros_like(pd_scores),
                torch.zeros_like(pd_scores[..., 0]),
                torch.zeros_like(pd_scores[..., 0]),
            )

        try:
            return self._forward(pd_scores, pd_bboxes, anc_points, gt_labels, gt_bboxes, mask_gt)
        except torch.cuda.OutOfMemoryError:
            # Move tensors to CPU, compute, then move back to original device
            LOGGER.warning("CUDA OutOfMemoryError in TaskAlignedAssigner, using CPU")
            cpu_tensors = [t.cpu() for t in (pd_scores, pd_bboxes, anc_points, gt_labels, gt_bboxes, mask_gt)]
            result = self._forward(*cpu_tensors)
            return tuple(t.to(device) for t in result)

    def _forward(self, pd_scores, pd_bboxes, anc_points, gt_labels, gt_bboxes, mask_gt):
        """
        Compute the task-aligned assignment.

        Args:
            pd_scores (torch.Tensor): Predicted classification scores with shape (bs, num_total_anchors, num_classes).
            pd_bboxes (torch.Tensor): Predicted bounding boxes with shape (bs, num_total_anchors, 4).
            anc_points (torch.Tensor): Anchor points with shape (num_total_anchors, 2).
            gt_labels (torch.Tensor): Ground truth labels with shape (bs, n_max_boxes, 1).
            gt_bboxes (torch.Tensor): Ground truth boxes with shape (bs, n_max_boxes, 4).
            mask_gt (torch.Tensor): Mask for valid ground truth boxes with shape (bs, n_max_boxes, 1).

        Returns:
            target_labels (torch.Tensor): Target labels with shape (bs, num_total_anchors).
            target_bboxes (torch.Tensor): Target bounding boxes with shape (bs, num_total_anchors, 4).
            target_scores (torch.Tensor): Target scores with shape (bs, num_total_anchors, num_classes).
            fg_mask (torch.Tensor): Foreground mask with shape (bs, num_total_anchors).
            target_gt_idx (torch.Tensor): Target ground truth indices with shape (bs, num_total_anchors).
        """
        """
        mask_pos: (b, max_num_obj, h*w) boolean tensor indicating the positive (foreground) anchor points.
        align_metric: (b, max_num_obj, h*w) alignment metric for positive anchor points.
        overlaps: (b, max_num_obj, h*w) IoU_based overlaps between predicted and ground truth boxes for positive anchor points.
        """
        mask_pos, align_metric, overlaps = self.get_pos_mask(
            pd_scores, pd_bboxes, gt_labels, gt_bboxes, anc_points, mask_gt
        )

        # dealing with more than one assigned anchor,
        """
        target_gt_idx : (b, h*w) indicator of the assigned ground truth object for positive anchor points, with shape (b, h*w)
        fg_mask : (b, h*w) boolean tensor indicating the positive (foreground) anchor points.
        mask_pos : (b, max_num_obj, h*w) boolean tensor indicating the positive (foreground) anchor points.
        """
        target_gt_idx, fg_mask, mask_pos = self.select_highest_overlaps(mask_pos, overlaps, self.n_max_boxes)

        # Assigned target
        """
        target_labels : (b, h*w) target labels for positive anchor points, with shape (b, h*w).
        target_bboxes : (b, h*w, 4) target bounding boxes for positive anchor points, with shape (b, h*w, 4).
        target_scores : (b, h*w, num_classes) target scores for positive anchor points, with shape (b, h*w, num_classes).
        """
        target_labels, target_bboxes, target_scores = self.get_targets(gt_labels, gt_bboxes, target_gt_idx, fg_mask)

        # Normalize: max score assigned with max overlap score, rest using propotions with the max score.
        align_metric_ = align_metric * mask_pos
        pos_align_metrics = align_metric_.amax(dim=-1, keepdim=True)  # b, max_num_obj
        pos_overlaps = (overlaps * mask_pos).amax(dim=-1, keepdim=True)  # b, max_num_obj
        norm_align_metric = (align_metric_ * pos_overlaps / (pos_align_metrics + self.eps)).amax(-2).unsqueeze(-1)
        target_scores = target_scores * norm_align_metric

        # 只保存计算稳定性需要的关键信息
        # mask_pos: (b, max_num_obj, h*w) 就是assign给每个gt的mask
        save_assign_info(
            dir_name=self.dir_name,
            mask_pos=mask_pos.detach().cpu(),  # 直接保存mask_pos，这是assign给每个gt的mask
            gt_labels=gt_labels.detach().cpu(),
            gt_bboxes=gt_bboxes.detach().cpu(),
            mask_gt=mask_gt.detach().cpu(),
            anc_points=anc_points.detach().cpu(),
        )

        return target_labels, target_bboxes, target_scores, fg_mask.bool(), target_gt_idx


class TaskAlignedAssigner_VaryingIoU_Sep_Record(TaskAlignedAssigner_VaryingIoU_Sep):
    """用于记录TaskAlignedAssigner_VaryingIoU_Sep的assignment信息"""
    def __init__(self, topk: int = 13, num_classes: int = 80, alpha=None, beta=None, eps: float = 1e-9,
                 dir_name: str = 'test', **kwargs):
        super().__init__(topk=topk, num_classes=num_classes, alpha=alpha, beta=beta, eps=eps, **kwargs)
        self.dir_name = os.path.join(_SAVE_DIR, dir_name)
        os.makedirs(self.dir_name, exist_ok=True)

    @torch.no_grad()
    def forward(self, pd_scores, pd_bboxes, anc_points, gt_labels, gt_bboxes, mask_gt, stride=None, **kwargs):
        """重写forward方法以记录assignment信息"""
        self.bs = pd_scores.shape[0]
        self.n_max_boxes = gt_bboxes.shape[1]
        device = gt_bboxes.device
        global _STEP
        _STEP += 1

        if self.n_max_boxes == 0:
            return (
                torch.full_like(pd_scores[..., 0], self.num_classes),
                torch.zeros_like(pd_bboxes),
                torch.zeros_like(pd_scores),
                torch.zeros_like(pd_scores[..., 0]),
                torch.zeros_like(pd_scores[..., 0]),
            )

        try:
            return self._forward(pd_scores, pd_bboxes, anc_points, gt_labels, gt_bboxes, mask_gt, stride=stride)
        except torch.cuda.OutOfMemoryError:
            LOGGER.warning("CUDA OutOfMemoryError in TaskAlignedAssigner_VaryingIoU_Sep, using CPU")
            cpu_tensors = [t.cpu() for t in (pd_scores, pd_bboxes, anc_points, gt_labels, gt_bboxes, mask_gt)]
            result = self._forward(*cpu_tensors, stride=stride)
            return tuple(t.to(device) for t in result)

    def _forward(self, pd_scores, pd_bboxes, anc_points, gt_labels, gt_bboxes, mask_gt, stride=None):
        """计算assignment并记录"""
        mask_pos, align_metric, overlaps = self.get_pos_mask(
            pd_scores, pd_bboxes, gt_labels, gt_bboxes, anc_points, mask_gt
        )

        target_gt_idx, fg_mask, mask_pos = self.select_highest_overlaps(mask_pos, overlaps, self.n_max_boxes)
        target_labels, target_bboxes, target_scores = self.get_targets(gt_labels, gt_bboxes, target_gt_idx, fg_mask)

        # Normalize: max score assigned with max overlap score, rest using propotions with the max score.
        align_metric *= mask_pos
        pos_align_metrics = align_metric.amax(dim=-1, keepdim=True)  # b, max_num_obj
        pos_overlaps = (overlaps * mask_pos).amax(dim=-1, keepdim=True)  # b, max_num_obj
        norm_align_metric = (align_metric * pos_overlaps / (pos_align_metrics + self.eps)).amax(-2).unsqueeze(-1)
        target_scores = target_scores * norm_align_metric

        # 只保存计算稳定性需要的关键信息
        # mask_pos: (b, max_num_obj, h*w) 就是assign给每个gt的mask
        save_assign_info(
            dir_name=self.dir_name,
            mask_pos=mask_pos.detach().cpu(),  # 直接保存mask_pos，这是assign给每个gt的mask
            gt_labels=gt_labels.detach().cpu(),
            gt_bboxes=gt_bboxes.detach().cpu(),
            mask_gt=mask_gt.detach().cpu(),
            anc_points=anc_points.detach().cpu(),
        )

        return target_labels, target_bboxes, target_scores, fg_mask.bool(), target_gt_idx


def val_with_record(model_path, dataset_path, dir_name, assign_kwargs=dict(), project_path='runs_assignment', 
                    assigner_type='TaskAlignedAssigner'):
    class BaseValidatorWithLoss(BaseValidator):
        @smart_inference_mode()
        def __call__(self, trainer=None, model=None):
            """
            Execute validation process, running inference on dataloader and computing performance metrics.

            Args:
                trainer (object, optional): Trainer object that contains the model to validate.
                model (nn.Module, optional): Model to validate if not using a trainer.

            Returns:
                (dict): Dictionary containing validation statistics.
            """
            self.training = trainer is not None
            # Initialize loss - will be properly set in training or non-training branch
            self.loss = None
            augment = self.args.augment and (not self.training)
            if self.training:
                self.device = trainer.device
                self.data = trainer.data
                # Force FP16 val during training
                self.args.half = self.device.type != "cpu" and trainer.amp
                model = trainer.ema.ema or trainer.model
                if trainer.args.compile and hasattr(model, "_orig_mod"):
                    model = model._orig_mod  # validate non-compiled original model to avoid issues
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
                self.device = model.device  # update device
                self.args.half = model.fp16  # update half
                stride, pt, jit = model.stride, model.pt, model.jit
                imgsz = check_imgsz(self.args.imgsz, stride=stride)
                if not (pt or jit or getattr(model, "dynamic", False)):
                    self.args.batch = model.metadata.get("batch", 1)  # export.py models default to batch-size 1
                    LOGGER.info(f"Setting batch={self.args.batch} input of shape ({self.args.batch}, 3, {imgsz}, {imgsz})")

                if str(self.args.data).rsplit(".", 1)[-1] in {"yaml", "yml"}:
                    self.data = check_det_dataset(self.args.data)
                elif self.args.task == "classify":
                    self.data = check_cls_dataset(self.args.data, split=self.args.split)
                else:
                    raise FileNotFoundError(emojis(f"Dataset '{self.args.data}' for task={self.args.task} not found ❌"))

                if self.device.type in {"cpu", "mps"}:
                    self.args.workers = 0  # faster CPU val as time dominated by inference, not dataloading
                if not (pt or (getattr(model, "dynamic", False) and not model.imx)):
                    self.args.rect = False
                self.stride = model.stride  # used in get_dataloader() for padding
                self.dataloader = self.dataloader or self.get_dataloader(self.data.get(self.args.split), self.args.batch)

                model.eval()
                if self.args.compile:
                    model = attempt_compile(model, device=self.device)
                model.warmup(imgsz=(1 if pt else self.args.batch, self.data["channels"], imgsz, imgsz))  # warmup
                # Initialize loss for non-training mode (only for PyTorch models)
                if pt and hasattr(model, 'model') and hasattr(model.model, 'loss'):
                    self.loss = torch.zeros(3, device=self.device)  # box, cls, dfl losses
                else:
                    self.loss = torch.zeros(3, device=self.device)  # Initialize even if loss not available

            self.run_callbacks("on_val_start")
            dt = (
                Profile(device=self.device),
                Profile(device=self.device),
                Profile(device=self.device),
                Profile(device=self.device),
            )
            bar = TQDM(self.dataloader, desc=self.get_desc(), total=len(self.dataloader))
            self.init_metrics(unwrap_model(model))
            self.jdict = []  # empty before each val
            from types import SimpleNamespace
            model.model.args = SimpleNamespace(**model.model.args)
            model.model.args.box = 7.5
            model.model.args.cls = 0.5
            model.model.args.dfl = 1.5

            #------------------------------------ adding record assignment ---------------------------------------------
            model.model.criterion = model.model.init_criterion()
            topk = assign_kwargs.get('topk', 10)
            num_classes = assign_kwargs.get('num_classes', 10)
            alpha = assign_kwargs.get('alpha', 0.5)
            beta = assign_kwargs.get('beta', 6)
            
            if assigner_type == 'TaskAlignedAssigner_VaryingIoU_Sep':
                # 从原始assigner获取参数
                original_assigner = model.model.criterion.assigner
                assigner_kwargs = {
                    'topk': topk,
                    'num_classes': num_classes,
                    'alpha': alpha,
                    'beta': beta,
                    'dir_name': dir_name,
                }
                # 传递VaryingIoU_Sep特有的参数
                if hasattr(original_assigner, 'align_iou_type'):
                    assigner_kwargs['align_iou_type'] = original_assigner.align_iou_type
                    assigner_kwargs['align_iou_kwargs'] = getattr(original_assigner, 'align_iou_kwargs', {})
                    assigner_kwargs['score_iou_type'] = original_assigner.score_iou_type
                    assigner_kwargs['score_iou_kwargs'] = getattr(original_assigner, 'score_iou_kwargs', {})
                model.model.criterion.assigner = TaskAlignedAssigner_VaryingIoU_Sep_Record(**assigner_kwargs)
            else:
                model.model.criterion.assigner = TaskAlignedAssigner_Record(topk=topk, num_classes=num_classes,
                                                                            alpha=alpha, beta=beta, dir_name=dir_name)
            #------------------------------------ adding record assignment ---------------------------------------------

            for batch_i, batch in enumerate(bar):
                self.run_callbacks("on_val_batch_start")
                self.batch_i = batch_i
                # Preprocess
                with dt[0]:
                    batch = self.preprocess(batch)

                # Inference
                with dt[1]:
                    preds = model(batch["img"], augment=augment)
                # Loss
                with dt[2]:
                    model.model.loss(batch)

                # Postprocess
                with dt[3]:
                    preds = self.postprocess(preds)

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
            if self.training:
                model.float()
                results = {**stats, **trainer.label_loss_items(self.loss.cpu() / len(self.dataloader), prefix="val")}
                return {k: round(float(v), 5) for k, v in results.items()}  # return results as 5 decimal place floats
            else:
                LOGGER.info(
                    "Speed: {:.1f}ms preprocess, {:.1f}ms inference, {:.1f}ms loss, {:.1f}ms postprocess per image".format(
                        *tuple(self.speed.values())
                    )
                )
                if self.args.save_json and self.jdict:
                    with open(str(self.save_dir / "predictions.json"), "w", encoding="utf-8") as f:
                        LOGGER.info(f"Saving {f.name}...")
                        json.dump(self.jdict, f)  # flatten and save
                    stats = self.eval_json(stats)  # update stats
                if self.args.plots or self.args.save_json:
                    LOGGER.info(f"Results saved to {colorstr('bold', self.save_dir)}")
                return stats

    DetectionValidator.__bases__ = (BaseValidatorWithLoss,)

    global _STEP
    _STEP = 0

    model = YOLO(model_path)
    model.val(data=dataset_path, project=project_path, validator=DetectionValidator)


def compute_assignment_stability(base_dir, epoch_dirs=None, area_thresholds=None,
                                 save_path=None, save_epoch_changes_path=None, iou_threshold=None,):
    """
    计算assignment的稳定性
    
    Args:
        base_dir: 包含所有epoch目录的基础目录，或者单个包含pkl文件的目录
        epoch_dirs: epoch目录列表，例如['epoch_0', 'epoch_1', ...]。如果为None，则从base_dir查找所有epoch_*目录
        area_thresholds: 面积阈值列表，用于按大小分组统计，例如[32*32, 96*96]
        save_path: 保存结果的路径（CSV格式）
        save_epoch_changes_path: 逐epoch（相邻epoch对）平均变化率及面积分桶的CSV
        iou_threshold: 预留参数（当前逻辑未使用），兼容旧接口
    Returns:
        dict: 包含稳定性统计结果的字典
    """
    import glob
    
    if area_thresholds is None:
        area_thresholds = [32.0 * 32.0, 96.0 * 96.0]
    
    # 确定epoch目录
    if epoch_dirs is None:
        # 尝试查找epoch_*目录
        epoch_pattern = os.path.join(base_dir, "epoch_*")
        epoch_dirs_found = sorted(glob.glob(epoch_pattern), 
                                 key=lambda x: int(os.path.basename(x).split('_')[1]))
        if epoch_dirs_found:
            epoch_dirs = [os.path.basename(d) for d in epoch_dirs_found]
        else:
            LOGGER.error(f"在 {base_dir} 中未找到epoch_*目录，请确保每个epoch的数据在独立的目录中")
            return None
    
    base_dir_actual = base_dir
    
    # 多目录模式：每个epoch有独立目录
    if epoch_dirs:
        LOGGER.info(f"找到{len(epoch_dirs)}个epoch目录")
        all_epochs_data = []
        # 添加进度条，显示epoch目录处理进度
        for epoch_dir_name in TQDM(epoch_dirs, desc="Loading epochs"):
            epoch_dir = os.path.join(base_dir_actual, epoch_dir_name)
            # 只匹配数字开头的pkl文件（如0.pkl, 1.pkl），排除anc_points.pkl等
            all_pkl_files = glob.glob(os.path.join(epoch_dir, "*.pkl"))
            # 过滤：只保留文件名是数字开头的pkl文件
            pkl_files = []
            for pkl_file in all_pkl_files:
                basename = os.path.basename(pkl_file)
                # 检查文件名是否以数字开头（排除anc_points.pkl等）
                if basename.split('.')[0].isdigit():
                    pkl_files.append(pkl_file)
            # 按数字排序
            pkl_files = sorted(pkl_files, key=lambda x: int(os.path.basename(x).split('.')[0]))
            if len(pkl_files) == 0:
                LOGGER.warning(f"epoch目录 {epoch_dir} 中没有找到pkl文件，跳过")
                continue
            
            # 从单独文件加载anc_points（所有batch共享）
            anc_points_path = os.path.join(epoch_dir, "anc_points.pkl")
            anc_points = None
            if os.path.exists(anc_points_path):
                with open(anc_points_path, 'rb') as f:
                    anc_points_np = pickle.load(f)
                    anc_points = torch.from_numpy(anc_points_np) if isinstance(anc_points_np, np.ndarray) else anc_points_np
            else:
                LOGGER.warning(f"epoch目录 {epoch_dir} 中未找到anc_points.pkl，尝试从batch数据中读取")
            
            # 合并该epoch的所有batch数据
            epoch_batches = []
            for pkl_file in pkl_files:
                # pkl_files已经过滤了非数字开头的文件，所以不需要再检查anc_points.pkl
                with open(pkl_file, 'rb') as f:
                    data = pickle.load(f)
                    batch_data = {}
                    for key in ['mask_pos', 'gt_labels', 'gt_bboxes', 'mask_gt']:
                        if key in data:
                            if key == 'mask_pos':
                                # 解压缩mask_pos
                                dense_mask = decompress_mask_pos(data[key])
                                batch_data[key] = torch.from_numpy(dense_mask)
                            else:
                                value = data[key]
                                if isinstance(value, np.ndarray):
                                    batch_data[key] = torch.from_numpy(value)
                                else:
                                    batch_data[key] = value
                    
                    # 如果anc_points不在单独文件中，尝试从batch数据读取（向后兼容）
                    if anc_points is None and 'anc_points' in data:
                        anc_points_value = data['anc_points']
                        if isinstance(anc_points_value, np.ndarray):
                            anc_points = torch.from_numpy(anc_points_value)
                        else:
                            anc_points = anc_points_value
                    
                    epoch_batches.append(batch_data)
            
            # 合并所有batch：根据mask_gt筛选valid gt，然后合并
            if epoch_batches:
                merged_epoch = {}
                
                # 收集所有batch的valid gt数据
                valid_data_list = {key: [] for key in ['mask_pos', 'gt_labels', 'gt_bboxes', 'mask_gt']}
                
                for batch_data in epoch_batches:
                    # 获取该batch的mask_gt，shape: (b, max_num_obj, 1)
                    batch_mask_gt = batch_data['mask_gt']  # (b, max_num_obj, 1)
                    b = batch_mask_gt.shape[0]
                    
                    # 对每个batch中的每个样本，筛选valid gt
                    for b_idx in range(b):
                        # 获取该样本的valid gt mask，shape: (max_num_obj,)
                        valid_mask = batch_mask_gt[b_idx].squeeze(-1).bool()  # (max_num_obj,)
                        valid_indices = torch.where(valid_mask)[0]  # valid gt的索引
                        
                        if len(valid_indices) == 0:
                            continue  # 跳过没有valid gt的样本
                        
                        # 对每个key，筛选出valid的gt
                        for key in ['mask_pos', 'gt_labels', 'gt_bboxes']:
                            if key in batch_data:
                                batch_tensor = batch_data[key]  # shape: (b, max_num_obj, ...)
                                # 获取该样本的数据并筛选valid gt
                                sample_data = batch_tensor[b_idx]  # (max_num_obj, ...)
                                valid_sample_data = sample_data[valid_indices]  # (valid_gt_num, ...)
                                valid_data_list[key].append(valid_sample_data)
                
                # 合并所有valid gt数据
                for key in ['mask_pos', 'gt_labels', 'gt_bboxes']:
                    if key in valid_data_list and len(valid_data_list[key]) > 0:
                        # cat所有batch的valid gt，最终shape: (total_valid_gt_num, ...)
                        merged_epoch[key] = torch.cat(valid_data_list[key], dim=0)
                
                # anc_points不需要合并，直接使用（所有batch共享）
                if anc_points is not None:
                    merged_epoch['anc_points'] = anc_points
                else:
                    LOGGER.warning(f"epoch目录 {epoch_dir} 中未找到anc_points")
                
                all_epochs_data.append(merged_epoch)
        
        if len(all_epochs_data) < 2:
            LOGGER.warning(f"需要至少2个epoch的有效数据，当前只有{len(all_epochs_data)}个")
            return None
    
    # 统计结果
    stability_stats = {
        'jaccard_similarity': [],  # 每个gt在不同epoch间的Jaccard相似度
        'anchor_change_ratio': [],  # anchor变化比例
        'gt_area': [],  # gt的面积
        'gt_label': [],  # gt的类别
        'epoch_pairs': [],  # epoch对
    }
    # 逐epoch分桶均值
    epoch_change_rows = []
    
    def compute_box_area(boxes):
        """计算box面积 (xyxy格式)"""
        if boxes.numel() == 0:
            return torch.tensor([])
        wh = (boxes[..., 2:4] - boxes[..., 0:2]).clamp(min=0)
        return wh[..., 0] * wh[..., 1]
    
    def bbox_iou_simple(box1, box2):
        """计算两个box的IoU (xyxy格式)"""
        # box1: (4,), box2: (4,)
        x1 = torch.max(box1[0], box2[0])
        y1 = torch.max(box1[1], box2[1])
        x2 = torch.min(box1[2], box2[2])
        y2 = torch.min(box1[3], box2[3])
        
        inter = torch.clamp(x2 - x1, min=0) * torch.clamp(y2 - y1, min=0)
        area1 = (box1[2] - box1[0]) * (box1[3] - box1[1])
        area2 = (box2[2] - box2[0]) * (box2[3] - box2[1])
        union = area1 + area2 - inter
        
        return inter / (union + 1e-9)
    
    # 定义面积区间（用于整体与逐epoch分桶）
    area_bins = [0] + area_thresholds + [float('inf')]
    bin_names = []
    for i in range(len(area_bins) - 1):
        if i == 0:
            bin_names.append(f"< {area_bins[1]:.0f}")
        elif i == len(area_bins) - 2:
            bin_names.append(f">= {area_bins[i]:.0f}")
        else:
            bin_names.append(f"{area_bins[i]:.0f} - {area_bins[i+1]:.0f}")
    
    # 新逻辑：对比相邻两个epoch的mask_pos，不做GT匹配，假设同一gt的顺序一致
    # 对每个gt计算：
    #   上一轮anchor数 = mask_pos[epoch_i, gt_idx].sum()
    #   未变动anchor数 = (mask_pos[epoch_i, gt_idx] & mask_pos[epoch_i+1, gt_idx]).sum()
    #   稳定率 = jaccard相似度
    num_epochs = len(all_epochs_data)
    from ultralytics.utils import TQDM as TQDM_LOCAL
    for epoch_idx in TQDM_LOCAL(range(num_epochs - 1), desc="Stability per-epoch"):
        epoch_i = all_epochs_data[epoch_idx]
        epoch_i1 = all_epochs_data[epoch_idx + 1]

        mask_pos_i = epoch_i['mask_pos'].bool()   # (N_i, A)
        mask_pos_i1 = epoch_i1['mask_pos'].bool() # (N_j, A)
        gt_bboxes_i = epoch_i['gt_bboxes']        # (N_i, 4)
        gt_labels_i = epoch_i['gt_labels']        # (N_i, 1) or (N_i,)

        if gt_labels_i.dim() > 1:
            gt_labels_i = gt_labels_i.squeeze(-1)

        assert mask_pos_i.shape[0] == mask_pos_i1.shape[0], f"mask_pos_i.shape: {mask_pos_i.shape}, mask_pos_i1.shape: {mask_pos_i1.shape}"

        gt_num = mask_pos_i.shape[0]              # N_i


        # 计算jaccard相似度
        jaccard_union = (mask_pos_i | mask_pos_i1).sum(dim=1)
        jaccard_similarity = (mask_pos_i & mask_pos_i1).sum(dim=1) / jaccard_union.clamp(min=1.0)      # (N_i,)
        zero_mask = jaccard_union == 0
        jaccard_similarity = torch.where(zero_mask, torch.ones_like(jaccard_similarity), jaccard_similarity)

        # 记录统计（逐GT）
        epoch_areas = []
        epoch_jaccards = []
        epoch_change_ratios = []
        for idx in range(gt_num):
            if zero_mask[idx]:
                continue
            area = compute_box_area(gt_bboxes_i[idx].unsqueeze(0)).item()
            label = gt_labels_i[idx].item()
            ratio = jaccard_similarity[idx].item()
            stability_stats['jaccard_similarity'].append(ratio)          # 稳定率
            stability_stats['anchor_change_ratio'].append(1.0 - ratio)   # 变化比例
            stability_stats['gt_area'].append(area)
            stability_stats['gt_label'].append(label)
            stability_stats['epoch_pairs'].append((epoch_idx, epoch_idx + 1))
            epoch_areas.append(area)
            epoch_jaccards.append(ratio)
            epoch_change_ratios.append(1.0 - ratio)

        # 当前epoch对的均值（含面积分桶）
        if epoch_jaccards:
            epoch_areas_np = np.array(epoch_areas)
            epoch_jacc_np = np.array(epoch_jaccards)
            epoch_change_np = np.array(epoch_change_ratios)

            # overall
            epoch_change_rows.append({
                'epoch_i': int(epoch_idx),
                'epoch_j': int(epoch_idx + 1),
                'area_bin': 'Overall',
                'count': int(len(epoch_jacc_np)),
                'mean_jaccard': float(epoch_jacc_np.mean()),
                'std_jaccard': float(epoch_jacc_np.std()),
                'mean_change_ratio': float(epoch_change_np.mean()),
                'std_change_ratio': float(epoch_change_np.std()),
            })

            # area buckets
            for bin_idx in range(len(area_bins) - 1):
                if bin_idx == 0:
                    mask = epoch_areas_np < area_bins[1]
                elif bin_idx == len(area_bins) - 2:
                    mask = epoch_areas_np >= area_bins[bin_idx]
                else:
                    mask = (epoch_areas_np >= area_bins[bin_idx]) & (epoch_areas_np < area_bins[bin_idx + 1])

                if mask.sum() > 0:
                    epoch_change_rows.append({
                        'epoch_i': int(epoch_idx),
                        'epoch_j': int(epoch_idx + 1),
                        'area_bin': bin_names[bin_idx],
                        'count': int(mask.sum()),
                        'mean_jaccard': float(epoch_jacc_np[mask].mean()),
                        'std_jaccard': float(epoch_jacc_np[mask].std()),
                        'mean_change_ratio': float(epoch_change_np[mask].mean()),
                        'std_change_ratio': float(epoch_change_np[mask].std()),
                    })
    
    # 按面积分组统计
    results = []
    areas = np.array(stability_stats['gt_area'])
    jaccards = np.array(stability_stats['jaccard_similarity'])
    change_ratios = np.array(stability_stats['anchor_change_ratio'])
    
    for bin_idx in range(len(area_bins) - 1):
        if bin_idx == 0:
            mask = areas < area_bins[1]
        elif bin_idx == len(area_bins) - 2:
            mask = areas >= area_bins[bin_idx]
        else:
            mask = (areas >= area_bins[bin_idx]) & (areas < area_bins[bin_idx + 1])
        
        if mask.sum() > 0:
            results.append({
                'area_bin': bin_names[bin_idx],
                'count': int(mask.sum()),
                'mean_jaccard': float(jaccards[mask].mean()),
                'std_jaccard': float(jaccards[mask].std()),
                'mean_change_ratio': float(change_ratios[mask].mean()),
                'std_change_ratio': float(change_ratios[mask].std()),
            })
    
    # 整体统计
    if len(jaccards) > 0:
        results.append({
            'area_bin': 'Overall',
            'count': len(jaccards),
            'mean_jaccard': float(jaccards.mean()),
            'std_jaccard': float(jaccards.std()),
            'mean_change_ratio': float(change_ratios.mean()),
            'std_change_ratio': float(change_ratios.std()),
        })
    
    # 保存结果
    if save_path:
        import csv
        os.makedirs(os.path.dirname(save_path) if os.path.dirname(save_path) else '.', exist_ok=True)
        with open(save_path, 'w', newline='') as f:
            writer = csv.DictWriter(f, fieldnames=['area_bin', 'count', 'mean_jaccard', 'std_jaccard', 
                                                   'mean_change_ratio', 'std_change_ratio'])
            writer.writeheader()
            writer.writerows(results)
        LOGGER.info(f"稳定性统计结果已保存到: {save_path}")

    # 保存逐epoch平均变化率（含面积分桶）
    if save_epoch_changes_path and epoch_change_rows:
        import csv
        os.makedirs(os.path.dirname(save_epoch_changes_path) if os.path.dirname(save_epoch_changes_path) else '.', exist_ok=True)
        with open(save_epoch_changes_path, 'w', newline='') as f:
            writer = csv.DictWriter(
                f,
                fieldnames=[
                    'epoch_i', 'epoch_j', 'area_bin', 'count',
                    'mean_jaccard', 'std_jaccard',
                    'mean_change_ratio', 'std_change_ratio'
                ]
            )
            writer.writeheader()
            writer.writerows(epoch_change_rows)
        LOGGER.info(f"逐epoch变化率均值已保存到: {save_epoch_changes_path}")
    
    return {
        'results': results,
        'raw_stats': stability_stats,
        'epoch_change_rows': epoch_change_rows,
    }


def plot_epoch_change(csv_path, out_dir=None, metric="mean_jaccard",
                      compare_csv_path=None, bin_name=None):
    """
    根据逐epoch变化率CSV绘制折线图（按面积分桶和Overall）。

    Args:
        csv_path: compute_assignment_stability生成的epoch_change_stats.csv
        out_dir: 输出目录；默认与csv同目录
        metric: 绘制的指标，支持'mean_jaccard'或'mean_change_ratio'
        compare_csv_path: 可选，第二个CSV路径，用于对比绘制
        bin_name: 可选，只绘制指定的area_bin；None则绘制全部bin

    Returns:
        保存的图片路径
    """
    import os
    import pandas as pd
    import matplotlib.pyplot as plt

    if not os.path.isfile(csv_path):
        raise FileNotFoundError(f"未找到CSV: {csv_path}")

    df = pd.read_csv(csv_path)
    df['source'] = 'A'

    if compare_csv_path:
        if not os.path.isfile(compare_csv_path):
            raise FileNotFoundError(f"未找到对比CSV: {compare_csv_path}")
        df_b = pd.read_csv(compare_csv_path)
        df_b['source'] = 'B'
        df = pd.concat([df, df_b], ignore_index=True)
    if metric not in {"mean_jaccard", "mean_change_ratio"}:
        raise ValueError("metric 仅支持 'mean_jaccard' 或 'mean_change_ratio'")

    if out_dir is None:
        out_dir = os.path.dirname(csv_path) or "."
    os.makedirs(out_dir, exist_ok=True)

    # 选择bin
    if bin_name is not None:
        df = df[df["area_bin"] == bin_name]
        if df.empty:
            raise ValueError(f"指定bin_name='{bin_name}'无数据")
        bins = [bin_name]
    else:
        bins = df["area_bin"].unique().tolist()

    # 使用epoch_j作为横轴（表示比较后的那一轮），按area_bin分组绘制
    plt.figure(figsize=(8, 5))
    for bin_name in sorted(bins):
        sub = df[df["area_bin"] == bin_name].sort_values("epoch_j")
        if sub.empty:
            continue
        if compare_csv_path:
            for src in sub["source"].unique():
                ssub = sub[sub["source"] == src]
                x = ssub["epoch_j"].to_numpy()
                y = ssub[metric].to_numpy()
                plt.plot(x, y, marker="o", markersize=3, label=f"{bin_name}-{src}")
        else:
            x = sub["epoch_j"].to_numpy()
            y = sub[metric].to_numpy()
            plt.plot(x, y, marker="o", markersize=3, label=bin_name)

    plt.xlabel("epoch (j)")
    plt.ylabel(metric)
    plt.title(f"{metric} vs epoch")
    plt.grid(True, ls="--", alpha=0.4)
    plt.legend()
    plt.tight_layout()

    suffix = "" if not compare_csv_path else "_compare"
    bin_suffix = "" if bin_name is None else f"_{bin_name}"
    out_path = os.path.join(out_dir, f"{metric}_epoch_curve{bin_suffix}{suffix}.png")
    plt.savefig(out_path, dpi=200)
    plt.close()
    LOGGER.info(f"折线图已保存: {out_path}")
    return out_path


def process_all_epochs(model_dir, dataset_path, output_dir, assign_kwargs=dict(), 
                       assigner_type='TaskAlignedAssigner', project_path='runs_assignment',
                       compute_stability=True, iou_threshold=0.5, area_thresholds=None):
    """
    处理所有epoch的模型，生成assignment数据并计算稳定性
    
    Args:
        model_dir: 包含所有epoch模型的目录（例如 runs/detect/visdrone/v12s_record/weights/）
        dataset_path: 数据集路径
        output_dir: 输出目录
        assign_kwargs: assigner参数
        assigner_type: assigner类型
        project_path: 项目路径
        compute_stability: 是否计算稳定性
        iou_threshold: 用于匹配同一个gt的IoU阈值
        area_thresholds: 面积阈值列表，用于按大小分组统计
    """
    import glob
    
    # 获取所有epoch模型文件
    epoch_files = sorted(glob.glob(os.path.join(model_dir, "epoch*.pt")), 
                        key=lambda x: int(os.path.basename(x).split('.')[0].split('epoch')[1]))
    
    LOGGER.info(f"找到{len(epoch_files)}个epoch模型文件")
    
    # 为每个epoch生成assignment数据
    for epoch_file in epoch_files:
        epoch_num = int(os.path.basename(epoch_file).split('.')[0].split('epoch')[1])
        epoch_dir_name = os.path.join(output_dir, f"epoch_{epoch_num}")
        
        LOGGER.info(f"处理epoch {epoch_num}...")
        val_with_record(
            model_path=epoch_file,
            dataset_path=dataset_path,
            dir_name=epoch_dir_name,
            assign_kwargs=assign_kwargs,
            project_path=project_path,
            assigner_type=assigner_type
        )
    
    # 计算稳定性（使用所有epoch的数据）
    if compute_stability:
        stability_dir = os.path.join(output_dir, "stability")
        os.makedirs(stability_dir, exist_ok=True)
        
        LOGGER.info("计算assignment稳定性...")
        stability_result = compute_assignment_stability(
            base_dir=output_dir,
            epoch_dirs=None,  # 自动查找epoch_*目录
            iou_threshold=iou_threshold,
            area_thresholds=area_thresholds,
            save_path=os.path.join(stability_dir, "stability_stats.csv"),
            save_epoch_changes_path=os.path.join(stability_dir, "epoch_change_stats.csv"),
        )
        
        return stability_result
    
    return None


def test_compress_decompress():
    """
    测试压缩和解压缩函数
    用于验证压缩/解压缩的正确性和性能
    """
    import time
    
    # 创建测试数据：模拟真实的mask_pos
    b, max_num_obj, num_anchors = 4, 50, 8400  # 典型的batch size, max objects, anchors
    topk = 10
    
    # 随机生成mask_pos，每个gt最多topk+1个True
    mask_pos_dense = np.zeros((b, max_num_obj, num_anchors), dtype=np.bool_)
    for b_idx in range(b):
        for gt_idx in range(max_num_obj):
            # 每个gt随机选择topk+1个anchor
            num_assigned = np.random.randint(0, topk + 2)
            if num_assigned > 0:
                anchor_indices = np.random.choice(num_anchors, size=num_assigned, replace=False)
                mask_pos_dense[b_idx, gt_idx, anchor_indices] = True
    
    # 测试压缩
    print("测试压缩...")
    start_time = time.time()
    compressed = compress_mask_pos(mask_pos_dense)
    compress_time = time.time() - start_time
    
    # 计算压缩率
    original_size = mask_pos_dense.nbytes
    compressed_size = sum(
        sum(len(arr) * 2 for arr in batch)  # uint16 = 2 bytes
        for batch in compressed['data']
    ) + len(pickle.dumps(compressed))  # 加上dict的overhead
    compression_ratio = original_size / compressed_size if compressed_size > 0 else 0
    
    print(f"原始大小: {original_size / 1024 / 1024:.2f} MB")
    print(f"压缩后大小: {compressed_size / 1024 / 1024:.2f} MB")
    print(f"压缩率: {compression_ratio:.2f}x")
    print(f"压缩时间: {compress_time:.4f}秒")
    
    # 测试解压缩
    print("\n测试解压缩...")
    start_time = time.time()
    decompressed = decompress_mask_pos(compressed)
    decompress_time = time.time() - start_time
    
    print(f"解压缩时间: {decompress_time:.4f}秒")
    
    # 验证正确性
    assert decompressed.shape == mask_pos_dense.shape, "形状不匹配"
    assert decompressed.dtype == mask_pos_dense.dtype, "数据类型不匹配"
    assert np.array_equal(decompressed, mask_pos_dense), "数据不匹配！"
    
    print("\n✓ 压缩/解压缩测试通过！")
    
    # 测试向后兼容性（直接传入numpy数组）
    print("\n测试向后兼容性...")
    decompressed_direct = decompress_mask_pos(mask_pos_dense)
    assert np.array_equal(decompressed_direct, mask_pos_dense), "向后兼容性测试失败！"
    print("✓ 向后兼容性测试通过！")
    
    return {
        'compression_ratio': compression_ratio,
        'compress_time': compress_time,
        'decompress_time': decompress_time,
    }


if __name__ == "__main__":
    # 运行测试（可选）
    # test_compress_decompress()
    # 示例1: 处理单个模型
    # val_with_record(model_path="runs/detect/visdrone/v12s/weights/best.pt",
    #                 dataset_path="VisDrone.yaml",
    #                 dir_name="v12s_visdrone",
    #                 assign_kwargs={"topk": 10, "num_classes": 10, "alpha": 0.5, "beta": 6},
    #                 project_path="runs_assignment",
    #                 assigner_type='TaskAlignedAssigner')
    
    # 示例2: 处理所有epoch并计算稳定性
    # process_all_epochs(
    #     model_dir="runs/detect/visdrone/v12s_record/weights/",
    #     dataset_path="VisDrone.yaml",
    #     output_dir="v12s_record_stability",
    #     assign_kwargs={"topk": 10, "num_classes": 10, "alpha": 0.5, "beta": 6},
    #     assigner_type='TaskAlignedAssigner',
    #     project_path='runs_assignment',
    #     compute_stability=False
    # )
    
    # process_all_epochs(
    #     model_dir="runs/detect/visdrone/v12s_record_hausdorff/weights/",
    #     dataset_path="VisDrone.yaml",
    #     output_dir="v12s_record_hausdorff_stability",
    #     assign_kwargs={"topk": 10, "num_classes": 10, "alpha": 0.5, "beta": 6},
    #     assigner_type='TaskAlignedAssigner_VaryingIoU_Sep',
    #     project_path='runs_assignment',
    #     compute_stability=False
    # )

    
    # 示例3: 只计算稳定性（如果已经生成了pkl文件）
    # compute_assignment_stability(
    #     base_dir=os.path.join(_SAVE_DIR, "v12s_record_stability"),
    #     save_path=os.path.join(_SAVE_DIR, "v12s_record_stability_stats", "stability_stats.csv"),
    #     save_epoch_changes_path=os.path.join(_SAVE_DIR, "v12s_record_stability_stats", "epoch_change_stats.csv"),
    #     area_thresholds=[8*8, 16*16, 32.0 * 32.0, 64.0 * 64.0]
    # )

    # compute_assignment_stability(
    #     base_dir=os.path.join(_SAVE_DIR, "v12s_record_hausdorff_stability"),
    #     save_path=os.path.join(_SAVE_DIR, "v12s_record_hausdorff_stability_stats", "stability_stats.csv"),
    #     save_epoch_changes_path=os.path.join(_SAVE_DIR, "v12s_record_hausdorff_stability_stats", "epoch_change_stats.csv"),
    #     area_thresholds=[8*8, 16*16, 32.0 * 32.0, 64.0 * 64.0]
    # )
    
    # 示例4: 处理v12s_record_hausdorff模型（使用TaskAlignedAssigner_VaryingIoU_Sep）

    
    # plot_epoch_change(
    #     csv_path=os.path.join(_SAVE_DIR, "v12s_record_stability_stats", "epoch_change_stats.csv"),
    #     out_dir=os.path.join(_SAVE_DIR, "v12s_record_stability_stats"),
    #     metric="mean_jaccard"
    # )
    # plot_epoch_change(
    #     csv_path=os.path.join(_SAVE_DIR, "v12s_record_hausdorff_stability_stats", "epoch_change_stats.csv"),
    #     out_dir=os.path.join(_SAVE_DIR, "v12s_record_hausdorff_stability_stats"),
    #     metric="mean_jaccard"
    # )

    plot_epoch_change(
        csv_path=os.path.join(_SAVE_DIR, "v12s_record_stability_stats", "epoch_change_stats.csv"),
        out_dir=os.path.join(_SAVE_DIR, "."),
        metric="mean_change_ratio",
        compare_csv_path=os.path.join(_SAVE_DIR, "v12s_record_hausdorff_stability_stats", "epoch_change_stats.csv"),
        bin_name="< 64"
    )


