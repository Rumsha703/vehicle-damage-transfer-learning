"""Grad-CAM overlays for ResNet-50 checkpoints: shows which image regions drove a prediction."""
import argparse
from pathlib import Path

import cv2
import numpy as np
import torch
from PIL import Image

from common import build_model, build_transforms, get_device


def main(a):
    device = get_device()
    ckpt = torch.load(a.ckpt, map_location=device)
    assert ckpt["arch"] == "resnet50", "This script targets ResNet-50 (layer4)."
    model = build_model("resnet50", len(ckpt["classes"]), pretrained=False).to(device)
    model.load_state_dict(ckpt["model"]); model.eval()
    _, tf = build_transforms(ckpt["img_size"])
    feats, grads = {}, {}
    model.layer4.register_forward_hook(lambda m, i, o: feats.update(v=o))
    model.layer4.register_full_backward_hook(lambda m, gi, go: grads.update(v=go[0]))
    out_dir = Path(a.out); out_dir.mkdir(parents=True, exist_ok=True)
    for f in sorted(Path(a.images).glob("*.jpg"))[: a.n]:
        img = Image.open(f).convert("RGB")
        x = tf(img).unsqueeze(0).to(device).requires_grad_(True)
        logits = model(x); cls = logits.argmax(1).item()
        model.zero_grad(); logits[0, cls].backward()
        w = grads["v"].mean(dim=(2, 3), keepdim=True)
        cam = torch.relu((w * feats["v"]).sum(1))[0].detach().cpu().numpy()
        cam = (cam - cam.min()) / (cam.max() - cam.min() + 1e-8)
        base = cv2.cvtColor(np.array(img), cv2.COLOR_RGB2BGR)
        cam = cv2.resize(cam, (base.shape[1], base.shape[0]))
        heat = cv2.applyColorMap(np.uint8(255 * cam), cv2.COLORMAP_JET)
        overlay = cv2.addWeighted(base, 0.6, heat, 0.4, 0)
        cv2.imwrite(str(out_dir / f"{f.stem}_{ckpt['classes'][cls]}.jpg"), overlay)
    print(f"Saved Grad-CAM overlays to {out_dir}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", required=True); ap.add_argument("--images", required=True)
    ap.add_argument("--out", default="outputs/gradcam"); ap.add_argument("--n", type=int, default=12)
    main(ap.parse_args())
