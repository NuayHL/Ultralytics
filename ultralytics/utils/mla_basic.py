import torch
import torch.nn as nn
import torch.nn.functional as F
from .metrics import bbox_iou


class FCOSAssigner(nn.Module):
    """
    FCOS-style Static Assigner for Anchor-free Detectors.
    
    Instead of calculating IoU between priors and GTs (since there are no anchor boxes),
    this assigner uses spatial constraints:
    1. The anchor point must be within the Ground Truth box (or a center radius).
    2. Ambiguities (point inside multiple GTs) are resolved by choosing the GT with minimal area.
    """

    def __init__(self, num_classes: int = 80, center_radius: float = 1.5):
        """
        Args:
            num_classes (int): Number of object classes.
            center_radius (float): The radius (in stride units) to sample around the GT center.
                                   If -1, use the pure FCOS "in_box" logic (all points inside GT).
                                   Commonly set to 1.5 or 2.5 to focus on high-quality centers.
        """
        super().__init__()
        self.num_classes = num_classes
        self.center_radius = center_radius

    @torch.no_grad()
    def forward(self, pd_scores, pd_bboxes, anc_points, gt_labels, gt_bboxes, mask_gt, stride=None, **kwargs):
        """
        Compute the FCOS-style static assignment.

        Args:
            pd_scores, pd_bboxes: Ignored for assignment logic (static assignment).
            anc_points (Tensor): (n_anchors, 2) - Grid center points.
            gt_labels (Tensor): (bs, n_max_boxes, 1).
            gt_bboxes (Tensor): (bs, n_max_boxes, 4) - GT boxes in xyxy format.
            mask_gt (Tensor): (bs, n_max_boxes, 1).
            stride (Tensor): (bs, n_anchors, 1) - Stride for each anchor point.
        """
        self.bs = gt_bboxes.shape[0]
        self.n_max_boxes = gt_bboxes.shape[1]
        self.n_anchors = anc_points.shape[0]
        device = gt_bboxes.device

        if self.n_max_boxes == 0:
            return (
                torch.full((self.bs, self.n_anchors), self.num_classes, dtype=torch.long, device=device),
                torch.zeros((self.bs, self.n_anchors, 4), device=device),
                torch.zeros((self.bs, self.n_anchors, self.num_classes), device=device),
                torch.zeros((self.bs, self.n_anchors), dtype=torch.bool, device=device),
                torch.zeros((self.bs, self.n_anchors), dtype=torch.long, device=device),
            )

        # Initialize targets
        target_labels = torch.full((self.bs, self.n_anchors), self.num_classes, dtype=torch.long, device=device)
        target_bboxes = torch.zeros((self.bs, self.n_anchors, 4), device=device)
        target_scores = torch.zeros((self.bs, self.n_anchors, self.num_classes), device=device) # Hard labels usually
        fg_mask = torch.zeros((self.bs, self.n_anchors), dtype=torch.bool, device=device)
        target_gt_idx = torch.zeros((self.bs, self.n_anchors), dtype=torch.long, device=device)

        # 1. Compute Offsets from Anchor Points to GT Boundaries
        # We process batch-wise or broadcast. Let's do batch loop for safety with dynamic shapes/masks.
        
        # anc_points: (n_anchors, 2) -> (1, n_anchors, 2)
        anchors_expand = anc_points.unsqueeze(0)

        for bs in range(self.bs):
            # Get valid GTs
            valid_mask = mask_gt[bs].squeeze(-1).bool()
            if not valid_mask.any():
                continue
            
            # (n_gt, 4)
            curr_gt_bboxes = gt_bboxes[bs][valid_mask]
            curr_gt_labels = gt_labels[bs][valid_mask].long()
            n_gt = curr_gt_bboxes.shape[0]

            if stride is not None:
                curr_stride = stride[bs].squeeze(-1) # (n_anchors,)
            else:
                curr_stride = torch.ones(self.n_anchors, device=device)

            # --- Condition A: Point inside GT Box ---
            # lt: (n_gt, n_anchors, 2), rb: (n_gt, n_anchors, 2)
            # anchors: (1, n_anchors, 2)
            # gt: (n_gt, 1, 4)
            curr_gt_expand = curr_gt_bboxes.unsqueeze(1) # (n_gt, 1, 4)
            
            # offsets: x - xmin, y - ymin, xmax - x, ymax - y
            xs, ys = anchors_expand[0, :, 0], anchors_expand[0, :, 1]
            
            # Check l, t, r, b
            l = xs.unsqueeze(0) - curr_gt_expand[..., 0] # (n_gt, n_anchors)
            t = ys.unsqueeze(0) - curr_gt_expand[..., 1]
            r = curr_gt_expand[..., 2] - xs.unsqueeze(0)
            b = curr_gt_expand[..., 3] - ys.unsqueeze(0)
            
            # (n_gt, n_anchors, 4)
            bbox_deltas = torch.stack([l, t, r, b], dim=2)
            is_in_box = bbox_deltas.min(dim=2).values > 0 # (n_gt, n_anchors)

            # --- Condition B: Point inside Center Radius (Optional but standard in FCOS/ATSS) ---
            if self.center_radius > 0:
                gt_centers_x = (curr_gt_expand[..., 0] + curr_gt_expand[..., 2]) / 2.0
                gt_centers_y = (curr_gt_expand[..., 1] + curr_gt_expand[..., 3]) / 2.0
                
                # Dynamic radius based on stride
                radius = self.center_radius * curr_stride.unsqueeze(0) # (1, n_anchors)
                
                c_l = xs.unsqueeze(0) - (gt_centers_x - radius)
                c_t = ys.unsqueeze(0) - (gt_centers_y - radius)
                c_r = (gt_centers_x + radius) - xs.unsqueeze(0)
                c_b = (gt_centers_y + radius) - ys.unsqueeze(0)
                
                center_deltas = torch.stack([c_l, c_t, c_r, c_b], dim=2)
                is_in_center = center_deltas.min(dim=2).values > 0
                
                # Final spatial mask: inside box AND inside center region
                is_in_box = is_in_box & is_in_center

            # --- Condition C: Ambiguity Resolution (Min Area) ---
            # If an anchor is inside multiple GTs, assign to the one with min area.
            
            # Calculate GT areas
            gt_areas = (curr_gt_bboxes[:, 2] - curr_gt_bboxes[:, 0]) * \
                       (curr_gt_bboxes[:, 3] - curr_gt_bboxes[:, 1]) # (n_gt,)
            
            # Create a matrix of areas for valid matches, infinity for invalid
            # (n_gt, n_anchors)
            areas_matrix = gt_areas.unsqueeze(1).repeat(1, self.n_anchors)
            areas_matrix[~is_in_box] = float('inf')
            
            # Find best GT for each anchor
            # min_areas: (n_anchors,), min_gt_idxs: (n_anchors,)
            min_areas, min_gt_idxs = areas_matrix.min(dim=0)
            
            # Identify anchors that matched at least one GT
            pos_mask = min_areas != float('inf') # (n_anchors,)

            if pos_mask.any():
                # Fill targets
                fg_mask[bs] = pos_mask
                
                assigned_gt_idx = min_gt_idxs[pos_mask]
                target_gt_idx[bs, pos_mask] = assigned_gt_idx
                
                # Labels
                assigned_labels = curr_gt_labels[assigned_gt_idx].squeeze(-1)
                target_labels[bs, pos_mask] = assigned_labels
                
                # BBoxes
                assigned_bboxes = curr_gt_bboxes[assigned_gt_idx]
                target_bboxes[bs, pos_mask] = assigned_bboxes
                
                # Scores (Static assignment usually gives hard 1.0 target)
                # In FCOS, there is also a 'centerness' target, but for this interface,
                # we usually put 1.0 into the classification target.
                target_scores[bs, pos_mask, assigned_labels] = 1.0

        return target_labels, target_bboxes, target_scores, fg_mask, target_gt_idx

class SimOTAAssigner(nn.Module):
    """
    SimOTA Assigner strictly following the logic in Megvii YOLOX source code.
    Adapted to match the TaskAlignedAssigner interface.
    """

    def __init__(self, center_radius: float = 2.5, num_classes: int = 80, use_vfl: bool = False):
        """
        Args:
            center_radius: Radius for the geometry constraint (default 2.5 in standard YOLOX,
                           though snippet showed 1.5, configurable here).
            num_classes: Number of classes.
        """
        super().__init__()
        self.center_radius = center_radius
        self.num_classes = num_classes

    @torch.no_grad()
    def forward(self, pd_scores, pd_bboxes, anc_points, gt_labels, gt_bboxes, mask_gt, stride=None, **kwargs):
        """
        Compute the SimOTA assignment.

        Args:
            pd_scores (Tensor): (bs, n_anchors, num_classes) - Assumed to be final confidence (sigmoid applied if needed).
            pd_bboxes (Tensor): (bs, n_anchors, 4) - Predicted boxes in xyxy format.
            anc_points (Tensor): (n_anchors, 2) - Center (x, y) of anchors.
            gt_labels (Tensor): (bs, n_max_boxes, 1).
            gt_bboxes (Tensor): (bs, n_max_boxes, 4) - GT boxes in xyxy format.
            mask_gt (Tensor): (bs, n_max_boxes, 1) - Mask indicating valid GTs.
            stride (Tensor): (bs, n_anchors, 1) - Stride values for each anchor.
        """
        self.bs = pd_scores.shape[0]
        self.n_max_boxes = gt_bboxes.shape[1]
        self.n_anchors = pd_bboxes.shape[1]
        device = pd_bboxes.device

        if self.n_max_boxes == 0:
            return (
                torch.full_like(pd_scores[..., 0], self.num_classes).long(),
                torch.zeros_like(pd_bboxes),
                torch.zeros_like(pd_scores),
                torch.zeros_like(pd_scores[..., 0]),
                torch.zeros_like(pd_scores[..., 0]),
            )

        # Initialize outputs
        target_labels = torch.full((self.bs, self.n_anchors), self.num_classes, dtype=torch.long, device=device)
        target_bboxes = torch.zeros_like(pd_bboxes)
        target_scores = torch.zeros_like(pd_scores)
        fg_mask_all = torch.zeros((self.bs, self.n_anchors), dtype=torch.bool, device=device)
        target_gt_idx = torch.zeros((self.bs, self.n_anchors), dtype=torch.long, device=device)

        # SimOTA is processed image by image
        for b in range(self.bs):
            # 1. Get valid GTs for this image
            valid_mask = mask_gt[b].squeeze(-1).bool()
            if not valid_mask.any():
                continue

            # (n_gt, 4) xyxy
            gt_bboxes_per_image = gt_bboxes[b][valid_mask]
            # (n_gt,)
            gt_classes = gt_labels[b][valid_mask].squeeze(-1).long()

            n_gt = gt_bboxes_per_image.shape[0]

            # 2. Geometry Constraint (Pre-filtering)
            # Source logic uses center_radius based on stride.
            # We need to compute this constraint to filter candidates.
            if stride is not None:
                current_stride = stride[b]  # (n_anchors, 1)
            else:
                current_stride = torch.ones((self.n_anchors, 1), device=device)

            # anchor_filter: (n_anchors,) boolean - crude filter
            # geometry_relation: (n_gt, n_filtered_anchors) boolean - detailed filter
            anchor_filter, geometry_relation = self.get_geometry_constraint(
                gt_bboxes_per_image,
                anc_points,
                current_stride,
                self.center_radius
            )

            if not anchor_filter.any():
                continue

            # Filter tensors to save memory/computation
            # (n_filtered_anchors, 4)
            bboxes_preds_per_image = pd_bboxes[b][anchor_filter]
            # (n_filtered_anchors, num_classes)
            cls_preds_ = pd_scores[b][anchor_filter]

            # 3. Calculate IoU (Pairwise)
            # gt: (n_gt, 4), pred: (n_filtered, 4) -> (n_gt, n_filtered)
            # Using basic IoU as per source (source uses bboxes_iou, assumed standard IoU)
            pair_wise_ious = bbox_iou(gt_bboxes_per_image.unsqueeze(1), bboxes_preds_per_image.unsqueeze(0), xywh=False,
                                      CIoU=True).squeeze(-1).clamp(0)

            pair_wise_ious_loss = -torch.log(pair_wise_ious + 1e-8)

            # 4. Calculate Cls Cost
            # Source: F.binary_cross_entropy(cls.sqrt(), gt)
            # We assume pd_scores is already sigmoid(logit).
            # If your model outputs logits, apply sigmoid here. Assuming inputs are [0,1].

            # Prepare GT one-hot: (n_gt, num_classes)
            gt_cls_per_image = F.one_hot(gt_classes, self.num_classes).float()

            # Expand for broadcasting:
            # preds: (1, n_filtered, C)
            # gts:   (n_gt, 1, C)
            # cost:  (n_gt, n_filtered)

            # NOTE: strict source logic implementation
            # cls_preds_.float().sigmoid_().sqrt() -> source assumes logits input and decoupled head
            # Here we assume cls_preds_ matches the shape and value range of "cls * obj".
            # To mimic strict source behavior: sqrt the scores before BCE.
            cls_preds_sqrt = cls_preds_.sqrt()

            # Disable autocast for binary_cross_entropy to avoid AMP safety issues
            # Since pd_scores is already sigmoided (probabilities), we cannot use binary_cross_entropy_with_logits
            with torch.amp.autocast("cuda", enabled=False):
                cls_preds_sqrt_fp32 = cls_preds_sqrt.float()
                gt_cls_per_image_fp32 = gt_cls_per_image.float()
                pair_wise_cls_loss = F.binary_cross_entropy(
                    cls_preds_sqrt_fp32.unsqueeze(0).expand(n_gt, -1, -1),
                    gt_cls_per_image_fp32.unsqueeze(1).expand(-1, cls_preds_.shape[0], -1),
                    reduction="none"
                ).sum(-1)

            # 5. Total Cost
            # cost = cls + 3.0 * iou + 1e6 * (~geometry)
            cost = (
                    pair_wise_cls_loss
                    + 3.0 * pair_wise_ious_loss
                    + 1000000.0 * (~geometry_relation)
            )

            # 6. SimOTA Matching
            (
                num_fg,
                gt_matched_classes,
                pred_ious_this_matching,
                matched_gt_inds,
                fg_mask_in_filtered
            ) = self.simota_matching(cost, pair_wise_ious, gt_classes, n_gt)

            # 7. Map back to full size
            # fg_mask_in_filtered is boolean mask inside the 'anchor_filter' subset

            # Identify indices in the full list
            valid_anchor_ind = torch.nonzero(anchor_filter, as_tuple=False).squeeze(
                -1)  # indices of passed geometry filter
            matched_anchor_ind = valid_anchor_ind[fg_mask_in_filtered]  # indices of final positives

            if len(matched_anchor_ind) > 0:
                fg_mask_all[b, matched_anchor_ind] = True
                target_gt_idx[b, matched_anchor_ind] = matched_gt_inds
                target_labels[b, matched_anchor_ind] = gt_matched_classes
                target_bboxes[b, matched_anchor_ind] = gt_bboxes_per_image[matched_gt_inds]

                # Create One-hot target scores with IoU awareness (Soft label) or Hard 1.0
                # Source: cls_target = one_hot * pred_ious
                target_scores_b = torch.zeros((len(matched_anchor_ind), self.num_classes), device=device,
                                              dtype=pd_scores.dtype)
                target_scores_b.scatter_(1, gt_matched_classes.unsqueeze(1), 1.0)
                # Multiply by IoU as per source logic for soft targets?
                # Source: cls_target = ... * pred_ious_this_matching.unsqueeze(-1)
                # -------------------------------------------------------------
                # # Multiply by IoU as per source logic for soft targets
                target_scores_b = target_scores_b * pred_ious_this_matching.unsqueeze(-1)

                # ----------------------use TAL's normalization-------------------------------
                # # Normalize IoU values per GT: similar to TAL's normalization
                # # For each GT, normalize by its max IoU to ensure best anchor gets score = 1.0
                # pred_ious_matched = pred_ious_this_matching.clone()
                
                # unique_gts, inverse_indices = torch.unique(matched_gt_inds, return_inverse=True)
                # for i, gt_id in enumerate(unique_gts):
                #     gt_mask = (inverse_indices == i)
                #     if gt_mask.any():
                #         gt_ious = pred_ious_matched[gt_mask]
                #         max_iou = gt_ious.max()
                #         if max_iou > 1e-8:  # Avoid division by zero
                #             # Normalize: each anchor's IoU / max IoU for its GT
                #             pred_ious_matched[gt_mask] = gt_ious / max_iou
                
                # # Multiply by normalized IoU as per source logic for soft targets
                # target_scores_b = target_scores_b * pred_ious_matched.unsqueeze(-1)
                # ---------------------------------------------------------------------------

                target_scores[b, matched_anchor_ind] = target_scores_b.to(target_scores.dtype)

        return target_labels, target_bboxes, target_scores, fg_mask_all, target_gt_idx

    def get_geometry_constraint(self, gt_bboxes, anc_points, stride, center_radius):
        """
        Calculate whether the center of an object is located in a fixed range of
        an anchor.

        Args:
            gt_bboxes: (n_gt, 4) in xyxy format.
            anc_points: (n_anchors, 2)
            stride: (n_anchors, 1)
            center_radius: float
        """
        # Source uses cx, cy. Convert xyxy -> cx, cy
        gt_cx = (gt_bboxes[:, 0] + gt_bboxes[:, 2]) / 2.0
        gt_cy = (gt_bboxes[:, 1] + gt_bboxes[:, 3]) / 2.0

        # Expand for broadcasting: (1, n_anchors)
        x_centers = anc_points[:, 0].unsqueeze(0)
        y_centers = anc_points[:, 1].unsqueeze(0)

        # (n_anchors, 1) -> (1, n_anchors)
        s = stride.transpose(0, 1)

        # Center dist
        center_dist = s * center_radius  # (1, n_anchors)

        # Calculate deltas
        # (n_gt, 1) - (1, n_anchors) -> (n_gt, n_anchors)
        # Check if anchor center is within [gt_center - dist, gt_center + dist]
        # Equivalent to: |anchor - gt| < dist

        dist_l = x_centers - (gt_cx.unsqueeze(1) - center_dist)
        dist_r = (gt_cx.unsqueeze(1) + center_dist) - x_centers
        dist_t = y_centers - (gt_cy.unsqueeze(1) - center_dist)
        dist_b = (gt_cy.unsqueeze(1) + center_dist) - y_centers

        center_deltas = torch.stack([dist_l, dist_t, dist_r, dist_b], dim=2)  # (n_gt, n_anchors, 4)

        is_in_centers = center_deltas.min(dim=-1).values > 0.0  # (n_gt, n_anchors)

        # Union of all GT valid areas to create the coarse filter
        anchor_filter = is_in_centers.any(dim=0)  # (n_anchors,)

        # The relationship matrix for valid anchors only
        geometry_relation = is_in_centers[:, anchor_filter]  # (n_gt, n_filtered)

        return anchor_filter, geometry_relation

    def simota_matching(self, cost, pair_wise_ious, gt_classes, num_gt):
        """
        SimOTA matching logic strictly following source.
        """
        matching_matrix = torch.zeros_like(cost, dtype=torch.uint8)

        # 1. Dynamic K estimation
        n_candidate_k = min(10, pair_wise_ious.size(1))
        # topk_ious: (n_gt, 10)
        topk_ious, _ = torch.topk(pair_wise_ious, n_candidate_k, dim=1)
        # dynamic_ks: (n_gt,) - sum of topk IoUs, rounded up to ensure sufficient positive samples
        # Standard YOLOX uses sum and converts to int, but we ensure at least 1
        # dynamic_ks = torch.clamp((topk_ious.sum(1)).int(), min=1)
        dynamic_ks = torch.clamp((topk_ious.sum(1) + 0.5).int(), min=1)

        # 2. Select top-k lowest cost
        for gt_idx in range(num_gt):
            _, pos_idx = torch.topk(
                cost[gt_idx], k=dynamic_ks[gt_idx].item(), largest=False
            )
            matching_matrix[gt_idx][pos_idx] = 1

        # 3. Handle Conflicts (Multiple GTs matching one anchor)
        anchor_matching_gt = matching_matrix.sum(0)  # (n_filtered,)

        if anchor_matching_gt.max() > 1:
            multiple_match_mask = anchor_matching_gt > 1
            # Select the GT with min cost for these anchors
            _, cost_argmin = torch.min(cost[:, multiple_match_mask], dim=0)
            matching_matrix[:, multiple_match_mask] = 0
            matching_matrix[cost_argmin, multiple_match_mask] = 1

        # 4. Prepare return values
        fg_mask_in_filtered = matching_matrix.sum(0) > 0  # (n_filtered,)
        num_fg = fg_mask_in_filtered.sum().item()

        # Get matched GT index for each positive anchor
        matched_gt_inds = matching_matrix[:, fg_mask_in_filtered].argmax(0)
        gt_matched_classes = gt_classes[matched_gt_inds]

        pred_ious_this_matching = (matching_matrix * pair_wise_ious).sum(0)[fg_mask_in_filtered]

        return num_fg, gt_matched_classes, pred_ious_this_matching, matched_gt_inds, fg_mask_in_filtered
