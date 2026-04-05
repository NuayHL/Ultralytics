import os
import random
import shutil
from collections import Counter, defaultdict

# 配置路径
DATASET_ROOT = "../../datasets/UAVDT"
TEST_IMG_DIR = os.path.join(DATASET_ROOT, "test/images")
TEST_LABEL_DIR = os.path.join(DATASET_ROOT, "test/labels")

VAL_IMG_DIR = os.path.join(DATASET_ROOT, "val/images")
VAL_LABEL_DIR = os.path.join(DATASET_ROOT, "val/labels")

VAL_TARGET = 2000  # 目标数量

os.makedirs(VAL_IMG_DIR, exist_ok=True)
os.makedirs(VAL_LABEL_DIR, exist_ok=True)

def read_labels(label_file):
    """读取一个标签文件，返回其中的类别id列表"""
    if not os.path.exists(label_file):
        return []
    with open(label_file, "r") as f:
        lines = f.readlines()
    classes = [int(line.split()[0]) for line in lines]
    return classes

# 统计 test 中所有类别的分布
all_labels = []
img_to_classes = {}

for img_file in os.listdir(TEST_IMG_DIR):
    if not img_file.lower().endswith((".jpg", ".png", ".jpeg")):
        continue
    base = os.path.splitext(img_file)[0]
    label_file = os.path.join(TEST_LABEL_DIR, base + ".txt")
    classes = read_labels(label_file)
    if classes:
        img_to_classes[img_file] = classes
        all_labels.extend(classes)

total_count = Counter(all_labels)
print("原始类别分布:", total_count)

# 计算每个类别应该抽取多少张
total_images = len(img_to_classes)
target_count = {cls: int(total_count[cls] / sum(total_count.values()) * VAL_TARGET)
                for cls in total_count}
print("目标抽取数量:", target_count)

# 确保每类至少有1张
for cls in target_count:
    if target_count[cls] == 0:
        target_count[cls] = 1

# 抽样
selected = set()
per_class_images = defaultdict(list)
for img, classes in img_to_classes.items():
    for cls in set(classes):  # 避免重复算
        per_class_images[cls].append(img)

for cls, needed in target_count.items():
    if len(per_class_images[cls]) <= needed:
        chosen = per_class_images[cls]
    else:
        chosen = random.sample(per_class_images[cls], needed)
    selected.update(chosen)

# 如果总数不足/超出，再微调
selected = list(selected)
if len(selected) > VAL_TARGET + 200:
    print('裁剪中')
    selected = random.sample(selected, VAL_TARGET)

print(f"最终选择 {len(selected)} 张图像作为验证集")

# 拷贝文件
for img_file in selected:
    base = os.path.splitext(img_file)[0]
    label_file = base + ".txt"

    shutil.copy(os.path.join(TEST_IMG_DIR, img_file), VAL_IMG_DIR)
    if os.path.exists(os.path.join(TEST_LABEL_DIR, label_file)):
        shutil.copy(os.path.join(TEST_LABEL_DIR, label_file), VAL_LABEL_DIR)
