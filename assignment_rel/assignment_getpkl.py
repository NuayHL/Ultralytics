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

from assignment_utils import load_hyper_pair


_STEP = 0
_SAVE_DIR = "assign_detail"
os.makedirs(_SAVE_DIR, exist_ok=True)


def save_assign_info(dir_name, **kwargs):
    save_path = os.path.join(dir_name, f"{_STEP}.pkl")
    with open(save_path, 'wb') as f:
        pickle.dump(kwargs, f)

class TaskAlignedAssigner_Record(TaskAlignedAssigner):
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

        save_assign_info(
            dir_name=self.dir_name,
            pd_scores=pd_scores.detach().cpu(),
            pd_bboxes=pd_bboxes.detach().cpu(),
            anc_points=anc_points.detach().cpu(),
            gt_labels=gt_labels.detach().cpu(),
            gt_bboxes=gt_bboxes.detach().cpu(),
            mask_gt=mask_gt.detach().cpu(),
        )

        return target_labels, target_bboxes, target_scores, fg_mask.bool(), target_gt_idx


def val_with_record(model_path, dataset_path, dir_name, assign_kwargs=dict(), project_path='runs_assignment'):
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
    # model.val(data=dataset_path, project=project_path, )
    model.val(data=dataset_path, project=project_path, validator=DetectionValidator)


if __name__ == "__main__":
    # pkl_storage_path = "hyper_ab"

    # load hyper-parameter range and get pkl files
    # alpha_range, beta_range = load_hyper_pair("hyper_ab/config.yaml", "")
    # for alpha in alpha_range:
    #     for beta in beta_range:
    #         val_with_record(model_path=f"runs/detect/VisDrone_AB_Search/yolo12n_a{alpha}_b{beta}/weights/best.pt",
    #                         dataset_path="VisDrone.yaml",
    #                         dir_name=f"{pkl_storage_path}/yolo12n_a{alpha}_b{beta}",
    #                         assign_kwargs={"topk": 10, "num_classes": 10, "alpha": float(alpha), "beta": float(beta)},
    #                         project_path="runs_assignment")

    val_with_record(model_path="runs/detect/visdrone/v12s/weights/best.pt",
                    dataset_path="VisDrone.yaml",
                    dir_name="v12s_visdrone",
                    assign_kwargs={"topk": 10, "num_classes": 10, "alpha": 0.5, "beta": 6},
                    project_path="runs_assignment")


