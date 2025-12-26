import os
import cv2
import json

import numpy as np
from tqdm import tqdm
import argparse

def yolo2coco(image_path, label_path, save_path, classes, use_letterbox=False, input_imgsize=640, area_save=''):
    print("Loading data from ", image_path, label_path)
    area_list = list()

    assert os.path.exists(image_path)
    assert os.path.exists(label_path)

    originImagesDir = image_path
    originLabelsDir = label_path
    # images dir name
    indexes = os.listdir(originImagesDir)

    dataset = {'info':{},'categories': [], 'annotations': [], 'images': []}
    for i, cls in enumerate(classes, 0):
        dataset['categories'].append({'id': i, 'name': cls, 'supercategory': 'mark'})

    ann_id_cnt = 0
    for k, index in enumerate(tqdm(indexes)):
        # 支持 png jpg 格式的图片.
        txtFile = f'{index[:index.rfind(".")]}.txt'
        base_name = index[:index.rfind(".")]
        # 处理stem：纯数字字符串转换为整数(去除前导零)，否则保持字符串
        stem = int(base_name) if base_name.isdigit() else base_name
        # 读取图像的宽和高
        try:
            im = cv2.imread(os.path.join(originImagesDir, index))
            height, width, _ = im.shape
        except Exception as e:
            print(f'{os.path.join(originImagesDir, index)} read error.\nerror:{e}')
        # 添加图像的信息
        if not os.path.exists(os.path.join(originLabelsDir, txtFile)):
            # 如没标签，跳过，只保留图片信息.
            continue
        dataset['images'].append({'file_name': index,
                                  'id': stem,
                                  'width': width,
                                  'height': height})
        r = 1.0
        if use_letterbox:
            r = float(input_imgsize) / max(width, height)
        with open(os.path.join(originLabelsDir, txtFile), 'r') as fr:
            labelList = fr.readlines()
            for label in labelList:
                label = label.strip().split()
                x = float(label[1])
                y = float(label[2])
                w = float(label[3])
                h = float(label[4])

                # convert x,y,w,h to x1,y1,x2,y2
                H, W, _ = im.shape
                x1 = (x - w / 2) * W
                y1 = (y - h / 2) * H
                x2 = (x + w / 2) * W
                y2 = (y + h / 2) * H
                # 标签序号从0开始计算, coco2017数据集标号混乱，不管它了。
                cls_id = int(label[0])
                width = max(0.0, w * W)
                height = max(0.0, h * H)

                area_list.append(width * height * r * r)

                dataset['annotations'].append({
                    'area': width * height * r * r,
                    'bbox': [x1, y1, width, height],
                    'category_id': cls_id + 1,
                    'id': ann_id_cnt,
                    'image_id': stem,
                    'iscrowd': 0,
                    # mask, 矩形是从左上角点按顺时针的四个顶点
                    'segmentation': [[x1, y1, x2, y1, x2, y2, x1, y2]]
                })
                ann_id_cnt += 1

    # 保存结果
    with open(save_path, 'w') as f:
        json.dump(dataset, f)
        print('Save annotation to {}'.format(save_path))

    if area_save:
        np.save(area_save, np.array(area_list))

if __name__ == "__main__":
    # others at first
    # classes = ['others', 'pedestrian', 'people', 'bicycle', 'car', 'van', 'truck', 'tricycle', 'awning-tricycle', 'bus',
    #            'motor',
    #            ]

    # classes = [
    #     "others",
    #     "aeroplane",
    #     "bicycle",
    #     "bird",
    #     "boat",
    #     "bottle",
    #     "bus",
    #     "car",
    #     "cat",
    #     "chair",
    #     "cow",
    #     "diningtable",
    #     "dog",
    #     "horse",
    #     "motorbike",
    #     "person",
    #     "pottedplant",
    #     "sheep",
    #     "sofa",
    #     "train",
    #     "tvmonitor"
    # ]

    # image_path="../../datasets/VisDrone/VisDrone2019-DET-val/images"
    # label_path="../../datasets/VisDrone/VisDrone2019-DET-val/labels"

    # image_path="../../datasets/VOC/images/test2007"
    # label_path="../../datasets/VOC/labels/test2007"
    # save_path='visdrone_coco_val.json'

    # classes = ['others', 'Person', 'Car', 'Bicycle', 'OtherVehicle', 'DontCare']
    # image_path = "../../datasets/HIT-UAV/images/val"
    # label_path = "../../datasets/HIT-UAV/labels/val"
    # save_path = "hituav_coco_val.json"
    # area_save = "hituav_coco_val_area.npy"

    classes = ['others', 'airplane', 'bridge', 'storage-tank', 'ship', 'swimming-pool',
               'vehicle', 'person', 'wind-mill']
    image_path = "../../datasets/ai-todv2/images/val"
    label_path = "../../datasets/ai-todv2/labels/val"
    save_path = "aitodv2_coco_val.json"
    area_save = "aitodv2_coco_val_area_letterbox.npy"

    yolo2coco(image_path=image_path,
              label_path=label_path,
              save_path=save_path,
              classes=classes,
              use_letterbox=True,
              input_imgsize=640,
              area_save=area_save)
