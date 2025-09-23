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


class COCOeval_Custom(COCOeval):
    def __init__(self, cocoGt=None, cocoDt=None, iouType='bbox', area_rng=None, area_lbl=None):
        super().__init__(cocoGt=cocoGt, cocoDt=cocoDt, iouType=iouType)
        self.params.areaRng = area_rng if area_rng else [[0 ** 2, 1e5 ** 2],
                                                         [0 ** 2, 32 ** 2],
                                                         [32 ** 2, 96 ** 2],
                                                         [96 ** 2, 1e5 ** 2]]
        self.params.areaRngLbl = area_lbl if area_lbl else ['all', 'small', 'medium', 'large']
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
            stats = np.zeros((6 + 2 * _n,))
            stats[0] = _summarize(1)
            stats[1] = _summarize(1, iouThr=.5, maxDets=self.params.maxDets[2])
            stats[2] = _summarize(1, iouThr=.75, maxDets=self.params.maxDets[2])
            for i, lbl in enumerate(extra_area_lbl):
                stats[3 + i] = _summarize(1, areaRng=lbl, maxDets=self.params.maxDets[2])
            stats[2+_n] = _summarize(0, maxDets=self.params.maxDets[0])
            stats[3+_n] = _summarize(0, maxDets=self.params.maxDets[1])
            stats[4+_n] = _summarize(0, maxDets=self.params.maxDets[2])
            for i, lbl in enumerate(extra_area_lbl):
                stats[5+_n+i] = _summarize(0, areaRng=lbl, maxDets=self.params.maxDets[2])
            return stats
        if not self.eval:
            raise Exception('Please run accumulate() first')
        iouType = self.params.iouType
        if iouType == 'segm' or iouType == 'bbox':
            summarize = _summarizeDets
        elif iouType == 'keypoints':
            raise Exception('keypoints evaluation not supported for this custom implementation')
        self.stats = summarize()

def quantize_to_2dp(value):
    if isinstance(value, (int, float)):
        value_decimal = Decimal(str(value))
    elif isinstance(value, Decimal):
        value_decimal = value
    else:
        raise TypeError(f"Unsupported type for quantization: {type(value)}")

    two_places = Decimal('0.01')
    return str(value_decimal.quantize(two_places, rounding=ROUND_HALF_UP))

def format_metrics(model_path, data_path, area_rng, area_lbl,
                   anno_json='visdrone_coco.json', batch=8, imgsz=640):
    name = '_tmp_val'
    tmp_dir = Path(f'../runs/detect/{name}')
    pred_json = tmp_dir / 'predictions.json'
    model = YOLO(model_path)
    metrics = model.val(data=data_path, name=name, batch=batch, imgsz=imgsz, save_json=True, conf=0.001, rect=True)

    anno = COCO(anno_json)  # init annotations api
    pred = anno.loadRes(str(pred_json))  # init predictions api
    eval = COCOeval_Custom(anno, pred, 'bbox', area_rng=area_rng, area_lbl=area_lbl)
    eval.params.areaRng = area_rng
    eval.params.areaRngLbl = area_lbl

    eval.evaluate()
    eval.accumulate()
    eval.summarize()

    return_dicts = dict()

    return_dicts['mAP50'] = map50 = metrics.box.map50
    return_dicts['mAP75'] = map75 = metrics.box.map75
    return_dicts['mAP'] = map = metrics.box.map
    return_dicts['COCO_stats'] = eval.stats

    for file in os.listdir(tmp_dir):
        os.remove(tmp_dir / file)
    tmp_dir.rmdir()
    return return_dicts

def print_info(metrics_dicts):
    pass

if __name__ == "__main__":
    # custom_area_rng = [
    #     [0 ** 2, 1e5 ** 2],  # 默认的 'all' 范围，保留
    #     [0 ** 2, 16 ** 2],  # 'tiny'
    #     [16 ** 2, 32 ** 2],  # 'small'
    #     [32 ** 2, 96 ** 2],  # 'medium'
    #     [96 ** 2, 1e5 ** 2],  # 'large'
    # ]
    # custom_area_lbl = ['all', 'tiny', 'small', 'medium', 'large']

    # 目标范围
    area_min, area_max = 1, 1e5**2  # [1, 1e10]
    log_min, log_max = np.log2(area_min), np.log2(area_max)

    # 划分 N 段
    N = 10
    log_bins = np.linspace(log_min, log_max, N+1)

    custom_area_rng = [[0, area_max]]  # all
    custom_area_lbl = ["all"]

    # 按 log2 均匀划分
    for i in range(N):
        lo, hi = 2**log_bins[i], 2**log_bins[i+1]
        custom_area_rng.append([lo, hi])
        custom_area_lbl.append(f"log[{log_bins[i]:.1f},{log_bins[i+1]:.1f}]")

    print_info(
    format_metrics(
        model_path='../runs/detect/visdrone/v12s_interpiou/weights/best.pt',
        data_path='../ultralytics/cfg/datasets/VisDrone.yaml',
        area_rng=custom_area_rng,
        area_lbl=custom_area_lbl,
        anno_json='visdrone_coco.json',
        batch=8,
        imgsz=640,
    ))
