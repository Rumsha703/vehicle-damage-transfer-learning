"""Evaluate a checkpoint on the held-out test set: precision, recall, F1, confusion matrix."""
import argparse
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import torch
from sklearn.metrics import classification_report, confusion_matrix, ConfusionMatrixDisplay

from common import build_loaders, build_model, get_device


def main(a):
    device = get_device()
    ckpt = torch.load(a.ckpt, map_location=device)
    sets, loaders = build_loaders(a.data, ckpt["img_size"], a.batch_size)
    model = build_model(ckpt["arch"], len(ckpt["classes"]), pretrained=False).to(device)
    model.load_state_dict(ckpt["model"]); model.eval()

    y_true, y_pred, wrong = [], [], []
    idx = 0
    with torch.no_grad():
        for x, y in loaders["test"]:
            p = model(x.to(device)).argmax(1).cpu()
            for yi, pi in zip(y.tolist(), p.tolist()):
                path = sets["test"].samples[idx][0]
                y_true.append(yi); y_pred.append(pi)
                if yi != pi:
                    wrong.append({"image": path, "true": ckpt["classes"][yi], "pred": ckpt["classes"][pi]})
                idx += 1

    rep = classification_report(y_true, y_pred, target_names=ckpt["classes"], output_dict=True, zero_division=0)
    out = Path(a.out); out.mkdir(parents=True, exist_ok=True)
    tag = Path(a.ckpt).stem
    (out / f"{tag}_test_report.json").write_text(json.dumps(rep, indent=2))
    (out / f"{tag}_misclassified.json").write_text(json.dumps(wrong, indent=2))
    print(classification_report(y_true, y_pred, target_names=ckpt["classes"], zero_division=0))
    cm = confusion_matrix(y_true, y_pred)
    fig, ax = plt.subplots(figsize=(7, 6))
    ConfusionMatrixDisplay(cm, display_labels=ckpt["classes"]).plot(ax=ax, xticks_rotation=45, colorbar=False)
    fig.tight_layout(); fig.savefig(out / f"{tag}_confusion_matrix.png", dpi=150)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--data", default="data")
    ap.add_argument("--out", default="outputs")
    ap.add_argument("--batch-size", type=int, default=32)
    main(ap.parse_args())
