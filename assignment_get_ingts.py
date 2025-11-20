from assignment_utils import (Assignment_Analyzer,
                              iter_pkl_files,
                              Assigner)

import os
import pickle
import torch

if __name__ == "__main__":
    pkl_dir = "assign_detail/visdrone_train"
    files = iter_pkl_files(pkl_dir)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    for file in files:
        data = Assignment_Analyzer.load_pickle(file)
        pd_scores = data['pd_scores'].to(device)
        pd_bboxes = data['pd_bboxes'].to(device)
        anc_points = data['anc_points'].to(device)
        gt_labels = data['gt_labels'].to(device)
        gt_bboxes = data['gt_bboxes'].to(device)
        mask_gt = data['mask_gt'].to(device)
        assigner = Assigner(topk=10, num_classes=10, alpha=0.5, beta=6)
        assigner.forward(pd_scores, pd_bboxes, anc_points, gt_labels, gt_bboxes, mask_gt)