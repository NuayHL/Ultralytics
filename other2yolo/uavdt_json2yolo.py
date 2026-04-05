import argparse, json
from pathlib import Path

IMG_EXTS = (".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff")

def clamp01(x: float) -> float:
    return 0.0 if x < 0.0 else (1.0 if x > 1.0 else x)

def build_class_map_from_args(class_names):
    """
    根据命令行传入的类别生成映射，如:
    --classes car truck bus vehicle
    → {'car':0, 'truck':1, 'bus':2}
    """
    return {name: i for i, name in enumerate(class_names)}

def normalize_stem(json_path: Path) -> str:
    """
    将 'M0101_img000016.jpg.json' 或 'M0101_img000016.json'
    规范化为 'M0101_img000016'
    """
    s = json_path.name
    if s.lower().endswith(".json"):
        s = s[:-5]  # 去掉 .json
    for ext in IMG_EXTS:
        if s.lower().endswith(ext):
            s = s[: -len(ext)]
            break
    return s

def build_image_index(images_root: Path):
    """
    递归扫描 images_root 下所有图片，建立:
    规范化基名 -> 图片完整路径
    """
    index = {}
    for p in images_root.rglob("*"):
        if p.is_file() and p.suffix.lower() in IMG_EXTS:
            index[p.stem] = p
    return index

def convert_json_file(jp: Path, labels_dir: Path, class_to_id: dict, img_index: dict, strict_image_check=False):
    """
    将单个 JSON 转为 YOLO txt。返回 (保留框数, 总框数)。
    """
    data = json.loads(jp.read_text(encoding="utf-8"))
    try:
        W = float(data["size"]["width"])
        H = float(data["size"]["height"])
    except Exception:
        # 如果没有 size 字段，无法归一化，直接跳过
        print(f"[WARN] 'size' missing in {jp.name}, skipped.")
        return 0, 0

    base = normalize_stem(jp)        # e.g. "M0101_img000016"
    imgp = img_index.get(base, None) # 对应图片

    if strict_image_check and imgp is None:
        print(f"[WARN] No image found for {base} (recursive index lookup).")
        # 仍写一个空的 labels 文件，YOLO 允许无目标
        labels_dir.mkdir(parents=True, exist_ok=True)
        (labels_dir / f"{base}.txt").write_text("", encoding="utf-8")
        return 0, 0

    lines = []
    kept = 0
    total = 0

    for obj in data.get("objects", []):
        total += 1
        if obj.get("geometryType") != "rectangle":
            continue
        cls_name = obj.get("classTitle")
        if cls_name not in class_to_id:
            continue
        try:
            (x1, y1), (x2, y2) = obj["points"]["exterior"]
            x1, x2 = sorted((float(x1), float(x2)))
            y1, y2 = sorted((float(y1), float(y2)))
            bw, bh = max(0.0, x2 - x1), max(0.0, y2 - y1)
            if bw <= 0 or bh <= 0:
                continue
            xc = x1 + bw / 2.0
            yc = y1 + bh / 2.0

            xc /= W; yc /= H; bw /= W; bh /= H
            xc, yc, bw, bh = map(clamp01, (xc, yc, bw, bh))

            cid = class_to_id[cls_name]
            lines.append(f"{cid} {xc:.6f} {yc:.6f} {bw:.6f} {bh:.6f}")
            kept += 1
        except Exception as e:
            print(f"[WARN] Skipped one object in {jp.name}: {e}")

    labels_dir.mkdir(parents=True, exist_ok=True)
    (labels_dir / f"{base}.txt").write_text("\n".join(lines), encoding="utf-8")
    return kept, total

def convert_split(split_root: Path, class_to_id: dict,
                  images_sub="images", ann_sub="ann", labels_sub="labels",
                  strict=False):
    images_dir = split_root / images_sub
    ann_dir = split_root / ann_sub
    labels_dir = split_root / labels_sub

    assert images_dir.is_dir(), f"Images dir not found: {images_dir}"
    assert ann_dir.is_dir(), f"Ann dir not found: {ann_dir}"

    img_index = build_image_index(images_dir)  # 递归索引图片
    json_files = sorted(ann_dir.glob("*.json"))

    kept_boxes = 0
    total_boxes = 0
    for jp in json_files:
        k, t = convert_json_file(jp, labels_dir, class_to_id, img_index, strict_image_check=strict)
        kept_boxes += k
        total_boxes += t

    print(f"[{split_root.name}] Wrote labels to: {labels_dir}")
    print(f"[{split_root.name}] Boxes kept: {kept_boxes}/{total_boxes} across {len(json_files)} files.")

def main():
    ap = argparse.ArgumentParser(description="UAVDT JSON → YOLO labels converter")
    ap.add_argument("--root", required=True, help="Dataset root, e.g., datasets/UAVDT")
    ap.add_argument("--train", default="train", help="Train split folder name under root")
    ap.add_argument("--val", default="test", help="Val split folder name under root (you said test-as-val)")
    ap.add_argument("--images-sub", default="images", help="Images subfolder name under split (default: images)")
    ap.add_argument("--ann-sub", default="ann", help="Annotations subfolder name under split (default: ann)")
    ap.add_argument("--labels-sub", default="labels", help="Labels subfolder name to write (default: labels)")
    ap.add_argument("--classes", nargs="+", required=True, help="Class names in desired order, e.g. car truck bus vehicle")
    ap.add_argument("--strict", action="store_true", help="Warn if corresponding image is missing")
    args = ap.parse_args()

    root = Path(args.root)
    class_to_id = build_class_map_from_args(args.classes)
    print("[INFO] Class mapping:", class_to_id)

    convert_split(root / args.train, class_to_id, args.images_sub, args.ann_sub, args.labels_sub, args.strict)
    convert_split(root / args.val,   class_to_id, args.images_sub, args.ann_sub, args.labels_sub, args.strict)
    print("[DONE] Conversion finished.")

if __name__ == "__main__":
    main()


#执行代码：python uavdt_json2yolo.py --root datasets/UAVDT --classes car truck bus --strict