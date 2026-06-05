import json
import os
from pathlib import Path
from ultralytics import YOLO
import torch
import numpy as np
import platform
import datetime
from decimal import Decimal, ROUND_HALF_UP
from pycocotools.coco import COCO
from pycocotools.cocoeval import COCOeval
import matplotlib.pyplot as plt

class COCOeval_Custom(COCOeval):
    def __init__(self, cocoGt=None, cocoDt=None, iouType='bbox', area_rng=None, area_lbl=None, max_dets=None):
        super().__init__(cocoGt=cocoGt, cocoDt=cocoDt, iouType=iouType)
        self.params.areaRng = area_rng if area_rng else [[0 ** 2, 1e5 ** 2],
                                                         [0 ** 2, 8 ** 2],
                                                         [8 ** 2, 16 ** 2],
                                                         [16 ** 2, 32 ** 2],
                                                         [32 ** 2, 1e5 ** 2]]
        self.params.areaRngLbl = area_lbl if area_lbl else ['all', 'verytiny', 'tiny', 'small', 'medium']
        if max_dets is not None:
            self.params.maxDets = max_dets
        self.custom_area_lbl = area_lbl

    def summarize(self):
        '''
        Compute and display summary metrics for evaluation results.
        Note this functin can *only* be applied on the default parameter setting
        '''
        def _summarize( ap=1, iouThr=None, areaRng='all', maxDets=100 ):
            p = self.params
            iStr = ' {:<18} {} @[ IoU={:<9} | area={:>6s} | maxDets={:>3d} ] = {:0.3f}'
            titleStr = 'Average Precision' if ap == 1 else 'Average Recall'
            typeStr = '(AP)' if ap==1 else '(AR)'
            iouStr = '{:0.2f}:{:0.2f}'.format(p.iouThrs[0], p.iouThrs[-1]) \
                if iouThr is None else '{:0.2f}'.format(iouThr)

            aind = [i for i, aRng in enumerate(p.areaRngLbl) if aRng == areaRng]
            mind = [i for i, mDet in enumerate(p.maxDets) if mDet == maxDets]
            if ap == 1:
                # dimension of precision: [TxRxKxAxM]
                s = self.eval['precision']
                # IoU
                if iouThr is not None:
                    t = np.where(iouThr == p.iouThrs)[0]
                    s = s[t]
                s = s[:,:,:,aind,mind]
            else:
                # dimension of recall: [TxKxAxM]
                s = self.eval['recall']
                if iouThr is not None:
                    t = np.where(iouThr == p.iouThrs)[0]
                    s = s[t]
                s = s[:,:,aind,mind]
            if len(s[s>-1])==0:
                mean_s = -1
            else:
                mean_s = np.mean(s[s>-1])
            print(iStr.format(titleStr, typeStr, iouStr, areaRng, maxDets, mean_s))
            return mean_s
        def _summarizeDets():
            extra_area_lbl = self.custom_area_lbl[1:]
            _n = len(extra_area_lbl)
            # Layout matches official AITOD cocoapi (without oLRP):
            #  0: AP@0.50:0.95 all     1: AP@0.25 all         2: AP@0.50 all
            #  3: AP@0.75 all          4..3+_n: AP per area   4+_n: AR@maxDets[0]
            #  5+_n: AR@maxDets[1]     6+_n: AR@maxDets[2]   7+_n..: AR per area
            stats = np.zeros((7 + 2 * _n,))
            stats[0] = _summarize(1, maxDets=self.params.maxDets[2])
            stats[1] = _summarize(1, iouThr=.25, maxDets=self.params.maxDets[2])  # AITOD-specific
            stats[2] = _summarize(1, iouThr=.5, maxDets=self.params.maxDets[2])
            stats[3] = _summarize(1, iouThr=.75, maxDets=self.params.maxDets[2])
            for i, lbl in enumerate(extra_area_lbl):
                stats[4 + i] = _summarize(1, areaRng=lbl, maxDets=self.params.maxDets[2])
            stats[4+_n] = _summarize(0, maxDets=self.params.maxDets[0])
            stats[5+_n] = _summarize(0, maxDets=self.params.maxDets[1])
            stats[6+_n] = _summarize(0, maxDets=self.params.maxDets[2])
            for i, lbl in enumerate(extra_area_lbl):
                stats[7+_n+i] = _summarize(0, areaRng=lbl, maxDets=self.params.maxDets[2])
            return stats
        if not self.eval:
            raise Exception('Please run accumulate() first')
        iouType = self.params.iouType
        if iouType == 'segm' or iouType == 'bbox':
            summarize = _summarizeDets
        elif iouType == 'keypoints':
            raise Exception('keypoints evaluation not supported for this custom implementation')
        self.stats = summarize()

def _normalize_img_ids(coco):
    """Ensure all image IDs are strings (fix mixed int/str imgIds in some datasets
    like AITODv2 where filenames can be non-numeric hex strings)."""
    # Normalize imgs dict keys
    coco.imgs = {str(k): v for k, v in coco.imgs.items()}
    # Normalize imgToAnns dict keys
    coco.imgToAnns = {str(k): v for k, v in coco.imgToAnns.items()}
    # Normalize anns image_id values
    for ann in coco.anns.values():
        ann['image_id'] = str(ann['image_id'])
    # Rebuild the imgIds list
    coco.imgIds = list(coco.imgs.keys())


def format_metrics(model_path, data_path, area_rng, area_lbl,
                   anno_json='aitodv2_coco.json', batch=8, imgsz=640, max_dets=None):
    name = '_tmp_val'
    tmp_dir = Path(f'runs/detect/{name}')
    pred_json = tmp_dir / 'predictions.json'
    model = YOLO(model_path)
    metrics = model.val(data=data_path, name=name, batch=batch, imgsz=imgsz, save_json=True, conf=0.001)

    # Normalize prediction image_ids to str in memory, then write to temp file for loadRes
    with open(pred_json) as f:
        pred_data = json.load(f)
    for item in pred_data:
        item['image_id'] = str(item['image_id'])
    temp_pred = tmp_dir / '_pred_norm.json'
    with open(temp_pred, 'w') as f:
        json.dump(pred_data, f)

    anno = COCO(anno_json)  # init annotations api
    _normalize_img_ids(anno)  # in-memory: ensure str IDs
    pred = anno.loadRes(str(temp_pred))  # init predictions api (temp_pred already has str IDs)
    _normalize_img_ids(pred)  # in-memory: ensure str IDs (loadRes may copy anno's mixed types)
    eval = COCOeval_Custom(anno, pred, 'bbox', area_rng=area_rng, area_lbl=area_lbl, max_dets=max_dets)

    eval.evaluate()
    eval.accumulate()
    eval.summarize()

    return_dicts = dict()

    return_dicts['mAP50'] = map50 = metrics.box.map50
    return_dicts['mAP75'] = map75 = metrics.box.map75
    return_dicts['mAP'] = map = metrics.box.map
    return_dicts['COCO_stats'] = eval.stats

    temp_pred.unlink()
    for file in os.listdir(tmp_dir):
        os.remove(tmp_dir / file)
    tmp_dir.rmdir()
    return return_dicts



def plot_ap_per_area(stats, custom_area_rng, save_path=None):
    """
    绘制每个面积区间对应的AP折线图（X轴为log尺度）
    Args:
        stats: eval.stats (来自 format_metrics)
        custom_area_rng: 自定义面积区间列表 [[min1, max1], [min2, max2], ...]
        save_path: 图片保存路径，为 None 则仅显示不保存
    """
    # 提取每个区间的面积中心（几何平均更适合 log 坐标）
    area_centers = [np.sqrt(rng[0] * rng[1]) if rng[0] > 0 else np.sqrt(1 * rng[1]) for rng in custom_area_rng[1:]]  # 跳过第一个总区间
    # 提取对应的 AP 值
    ap_values = stats[4 : 4 + len(area_centers)]  # AP per area starts at index 4 (after AP@0.25, AP50, AP75)

    # 保留空位但忽略绘制 -1 值
    valid_x = []
    valid_y = []
    for x, y in zip(area_centers, ap_values):
        if y != -1:
            valid_x.append(x)
            valid_y.append(y)

    # 绘制
    plt.figure(figsize=(8,5))
    plt.plot(valid_x, valid_y, marker='o', linestyle='-', color='b', label='AP per area')

    # 设定X轴为对数坐标
    plt.xscale('log')

    # 设置刻度与标签
    plt.xticks(area_centers, [f"[{int(r[0])}, {int(r[1])}]" for r in custom_area_rng[1:]], rotation=45, ha='right')
    plt.xlabel("Area Range (log scale)")
    plt.ylabel("Average Precision (AP)")
    plt.title("AP vs Object Area (log scale)")
    plt.grid(True, which='both', linestyle='--', alpha=0.5)
    plt.legend()
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f'Plot saved to {save_path}')
    plt.show()

if __name__ == "__main__":
    # AITODv2 area ranges (matching official AITOD cocoapi):
    #   verytiny: area < 8*8   = [0, 64]
    #   tiny:     8*8 to 16*16  = [64, 256]
    #   small:    16*16 to 32*32 = [256, 1024]
    #   medium:   32*32 to inf   = [1024, 1e10] (open-ended, per official)
    custom_area_rng = [
        [0 ** 2, 1e5 ** 2],     # 'all'
        [0 ** 2, 8 ** 2],       # 'verytiny': area < 8x8
        [8 ** 2, 16 ** 2],      # 'tiny': 8x8 ~ 16x16
        [16 ** 2, 32 ** 2],     # 'small': 16x16 ~ 32x32
        [32 ** 2, 1e5 ** 2],    # 'medium': > 32x32 (open-ended, matches official)
    ]
    custom_area_lbl = ['all', 'verytiny', 'tiny', 'small', 'medium']

    exps = ['yolo12m.yaml', 'yolo12m_usaa_raw_dyabcalra64_ra32_rtadd_s10.yaml', 'yolo12l_usaa_raw_dyabcalra64_ra32_rtadd_s10.yaml']

    for exp in exps:
        model_path = f'runs/detect/aitodv2/{exp}/weights/best.pt'

        stats = format_metrics(
            model_path=model_path,
            data_path='ultralytics/cfg/datasets/ai-todv2.yaml',
            area_rng=custom_area_rng,
            area_lbl=custom_area_lbl,
            anno_json='myutils/aitodv2_coco_val.json',
            batch=8,
            imgsz=640,
            max_dets=[1, 100, 1500],  # matches official AITOD cocoapi
        )['COCO_stats']
        save_path = f'ap_per_area_{exp}.png'
        plot_ap_per_area(stats, custom_area_rng, save_path=str(save_path))
