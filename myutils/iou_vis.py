from ultralytics.utils.metrics import bbox_iou_ext
import numpy as np
import matplotlib.pyplot as plt
import torch

STEP=100
BBOX_SIZE = [60, 60]
GT_BBOX = torch.tensor([70, 70, BBOX_SIZE[0], BBOX_SIZE[1]]).expand(STEP+1, -1)
START_PRED_BBOX = torch.tensor([130, 150, BBOX_SIZE[0], BBOX_SIZE[1]]).expand(STEP+1, -1)
coe = torch.tensor([[float(i)/STEP] for i in range(STEP+1)])
INTERP_BBOX = GT_BBOX * coe + START_PRED_BBOX * (1 - coe)
X_RANGE = np.arange(STEP+1)

def plot_iou_curve(iou_list: list):
    iou_values = list()
    for _, iou_type, iou_kwargs in iou_list:
        iou_values.append(bbox_iou_ext(GT_BBOX, INTERP_BBOX,
                                          iou_type=iou_type, iou_kargs=iou_kwargs,
                                          xywh=True).numpy())

    plt.figure(figsize=(8, 6))
    ax = plt.gca()

    for (name, _, _), iou_value in zip(iou_list, iou_values):
        ax.plot(X_RANGE, iou_value, label=name)

    ax.set_xlabel('Step', fontsize=14)
    ax.set_ylabel('Loss Value', fontsize=14)
    ax.grid(True, linestyle='--', alpha=0.6)
    ax.set_xlim(0, 100)
    ax.set_ylim(0, 2)
    plt.legend()
    plt.show(dpi=1000)

if __name__ == "__main__":
    plot_iou_curve([["GIoU", "GIoU",{}],
                    ["DIoU", "DIoU",{}],
                    ["SIoU", "SIoU",{}],
                    ["CIoU", "CIoU",{}],
                    ["InterpIoU", "InterpIoU",{"interp_coe": 0.98}],
                    ["D_InterpIoU", "D_InterpIoU", {"lv":0.9, "hv":0.98}],
                    ["D_InterpIoU1", "D_InterpIoU", {"lv":0.0, "hv":0.98}],
                    ["IoU", "IoU" ,{}],
                    ["SimD1", "SimD",{"sim_x":6.13, "sim_y":4.59}],])
