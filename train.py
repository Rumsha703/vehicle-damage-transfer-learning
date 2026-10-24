"""Transfer learning for vehicle damage classification (PyTorch).

Stage 1: freeze the pre-trained backbone, train the new head.
Stage 2: unfreeze the last block and fine-tune at a lower learning rate.
Use --scratch to train the same architecture from random weights (ablation baseline).
"""
import argparse
import json
from pathlib import Path

import torch
import torch.nn as nn
from sklearn.metrics import f1_score
from sklearn.utils.class_weight import compute_class_weight
import numpy as np

from common import (build_loaders, build_model, get_device, head_params,
                    last_block_params, set_seed)


def run_epoch(model, loader, criterion, device, optimizer=None):
    train = optimizer is not None
    model.train(train)
    losses, preds, labels = [], [], []
    with torch.set_grad_enabled(train):
        for x, y in loader:
            x, y = x.to(device), y.to(device)
            out = model(x)
            loss = criterion(out, y)
            if train:
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
            losses.append(loss.item() * x.size(0))
            preds += out.argmax(1).cpu().tolist()
            labels += y.cpu().tolist()
    n = len(labels)
    return sum(losses) / n, float(np.mean(np.array(preds) == np.array(labels))), \
        f1_score(labels, preds, average="macro")


def main(a):
    set_seed(a.seed)
    device = get_device()
    sets, loaders = build_loaders(a.data, a.img_size, a.batch_size)
    classes = sets["train"].classes
    model = build_model(a.arch, len(classes), pretrained=not a.scratch).to(device)

    cw = compute_class_weight("balanced", classes=np.arange(len(classes)),
                              y=np.array(sets["train"].targets))
    criterion = nn.CrossEntropyLoss(weight=torch.tensor(cw, dtype=torch.float, device=device))

    out_dir = Path(a.out); out_dir.mkdir(parents=True, exist_ok=True)
    tag = f"{a.arch}_{'scratch' if a.scratch else 'pretrained'}"
    history, best_f1 = [], -1.0

    stages = []
    if a.scratch:
        stages.append(("scratch", list(model.parameters()), a.lr, a.epochs_head + a.epochs_finetune))
    else:
        for p in model.parameters():
            p.requires_grad = False
        stages.append(("head", list(head_params(model, a.arch)), a.lr, a.epochs_head))
        stages.append(("finetune", None, a.lr / 10, a.epochs_finetune))

    for name, params, lr, epochs in stages:
        if name == "finetune":
            params = last_block_params(model, a.arch)
        for p in params:
            p.requires_grad = True
        opt = torch.optim.AdamW(params, lr=lr, weight_decay=1e-4)
        for ep in range(1, epochs + 1):
            tl, ta, tf = run_epoch(model, loaders["train"], criterion, device, opt)
            vl, va, vf = run_epoch(model, loaders["val"], criterion, device)
            history.append(dict(stage=name, epoch=ep, train_loss=tl, train_acc=ta, train_f1=tf,
                                val_loss=vl, val_acc=va, val_macro_f1=vf))
            print(f"[{name}] ep{ep} train_loss={tl:.3f} val_acc={va:.3f} val_macro_f1={vf:.3f}")
            if vf > best_f1:
                best_f1 = vf
                torch.save({"model": model.state_dict(), "classes": classes,
                            "arch": a.arch, "img_size": a.img_size}, out_dir / f"{tag}_best.pt")

    (out_dir / f"{tag}_history.json").write_text(json.dumps(history, indent=2))
    print(f"Best validation macro-F1: {best_f1:.4f}  (checkpoint: {tag}_best.pt)")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default="data")
    ap.add_argument("--out", default="outputs")
    ap.add_argument("--arch", default="resnet50", choices=["resnet50", "efficientnet_b0"])
    ap.add_argument("--img-size", type=int, default=224)
    ap.add_argument("--batch-size", type=int, default=32)
    ap.add_argument("--epochs-head", type=int, default=5)
    ap.add_argument("--epochs-finetune", type=int, default=10)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--scratch", action="store_true", help="no pre-trained weights (baseline)")
    ap.add_argument("--seed", type=int, default=42)
    main(ap.parse_args())
