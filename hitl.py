"""Human-in-the-loop review cycle.

queue : run the model over the held-out pool (data/pool) and write low-confidence images to a CSV review queue.
        Never point this at data/test: reviewed images are added to training.
merge : read the CSV after YOU fill in 'reviewer_label', and copy corrected images into data/train/<label>/.
Retrain afterwards and compare metrics before/after (see README).
"""
import argparse
import csv
import shutil
from pathlib import Path

EXTS = {".jpg", ".jpeg", ".png"}


def queue(a):
    import torch
    from PIL import Image
    from common import build_model, build_transforms, get_device
    device = get_device()
    ckpt = torch.load(a.ckpt, map_location=device)
    model = build_model(ckpt["arch"], len(ckpt["classes"]), pretrained=False).to(device)
    model.load_state_dict(ckpt["model"]); model.eval()
    _, tf = build_transforms(ckpt["img_size"])
    rows = []
    for f in sorted(Path(a.images).rglob("*")):
        if f.suffix.lower() not in EXTS:
            continue
        x = tf(Image.open(f).convert("RGB")).unsqueeze(0).to(device)
        with torch.no_grad():
            prob = torch.softmax(model(x), 1)[0]
        conf, pred = prob.max(0)
        if conf.item() < a.threshold:
            rows.append({"image": str(f), "predicted": ckpt["classes"][pred.item()],
                         "confidence": f"{conf.item():.3f}",
                         "reviewer_label": f.parent.name if a.simulate else ""})
    with open(a.csv, "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=["image", "predicted", "confidence", "reviewer_label"])
        w.writeheader(); w.writerows(rows)
    print(f"{len(rows)} images below confidence {a.threshold} written to {a.csv}")


def merge(a):
    n_changed, n_confirmed = 0, 0
    with open(a.csv, newline="") as fh:
        for r in csv.DictReader(fh):
            label = r["reviewer_label"].strip()
            if not label:
                continue
            dest = Path(a.data) / "train" / label
            dest.mkdir(parents=True, exist_ok=True)
            shutil.copy(r["image"], dest / Path(r["image"]).name)
            if label == r["predicted"]:
                n_confirmed += 1
            else:
                n_changed += 1
    print(f"Added reviewed images to training set. Corrections: {n_changed}, confirmations: {n_confirmed}")
    return n_changed, n_confirmed


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)
    q = sub.add_parser("queue")
    q.add_argument("--ckpt", required=True); q.add_argument("--images", required=True)
    q.add_argument("--threshold", type=float, default=0.7); q.add_argument("--csv", default="outputs/review_queue.csv")
    q.add_argument("--simulate", action="store_true", help="fill reviewer_label from the folder name (simulated reviewer; say so in your write-up)")
    m = sub.add_parser("merge")
    m.add_argument("--csv", default="outputs/review_queue.csv"); m.add_argument("--data", default="data")
    a = ap.parse_args()
    queue(a) if a.cmd == "queue" else merge(a)
