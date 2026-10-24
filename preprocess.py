"""Clean raw images with OpenCV and create stratified train/val/test splits.

Expected input layout:  data/raw/<class_name>/*.jpg
Output layout:          data/{train,val,test,pool}/<class_name>/*.jpg

"pool" is held back as the unlabelled review pool for the human-in-the-loop cycle, so the test set is never used for training.
"""
import argparse
import json
import random
import shutil
from pathlib import Path

import cv2

EXTS = {".jpg", ".jpeg", ".png", ".bmp"}


def blur_score(img) -> float:
    """Variance of the Laplacian: low values indicate blurry images."""
    return cv2.Laplacian(cv2.cvtColor(img, cv2.COLOR_BGR2GRAY), cv2.CV_64F).var()


def resize_max(img, max_side: int):
    h, w = img.shape[:2]
    scale = max_side / max(h, w)
    if scale >= 1:
        return img
    return cv2.resize(img, (int(w * scale), int(h * scale)), interpolation=cv2.INTER_AREA)


def main(raw, out, blur_thresh, max_side, seed, ratios):
    random.seed(seed)
    raw, out = Path(raw), Path(out)
    report = {"kept": {}, "dropped_unreadable": 0, "dropped_blurry": 0, "dropped_duplicate": 0}
    seen = set()
    for cls_dir in sorted(p for p in raw.iterdir() if p.is_dir()):
        kept = []
        for f in sorted(cls_dir.iterdir()):
            if f.suffix.lower() not in EXTS:
                continue
            img = cv2.imread(str(f))
            if img is None:
                report["dropped_unreadable"] += 1
                continue
            if blur_score(img) < blur_thresh:
                report["dropped_blurry"] += 1
                continue
            small = cv2.resize(img, (16, 16), interpolation=cv2.INTER_AREA)  # crude exact-dup hash
            key = small.tobytes()
            if key in seen:
                report["dropped_duplicate"] += 1
                continue
            seen.add(key)
            kept.append((f, resize_max(img, max_side)))
        random.shuffle(kept)
        n = len(kept)
        n_tr, n_va, n_te = (int(r * n) for r in ratios)
        parts = {"train": kept[:n_tr], "val": kept[n_tr:n_tr + n_va],
                 "test": kept[n_tr + n_va:n_tr + n_va + n_te], "pool": kept[n_tr + n_va + n_te:]}
        for split, items in parts.items():
            d = out / split / cls_dir.name
            d.mkdir(parents=True, exist_ok=True)
            for f, img in items:
                cv2.imwrite(str(d / f"{f.stem}.jpg"), img)
        report["kept"][cls_dir.name] = {k: len(v) for k, v in parts.items()}
    (out / "preprocess_report.json").write_text(json.dumps(report, indent=2))
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--raw", default="data/raw")
    ap.add_argument("--out", default="data")
    ap.add_argument("--blur-thresh", type=float, default=50.0)
    ap.add_argument("--max-side", type=int, default=640)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--ratios", type=float, nargs=3, default=(0.6, 0.15, 0.15), help="train, val, test ratios; the remainder becomes the review pool")
    a = ap.parse_args()
    main(a.raw, a.out, a.blur_thresh, a.max_side, a.seed, a.ratios)
