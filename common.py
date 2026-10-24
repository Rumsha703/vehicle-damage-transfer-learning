"""Shared helpers: seeding, device, transforms, data loaders."""
import random
from pathlib import Path

import numpy as np

IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)


def set_seed(seed: int = 42) -> None:
    import torch
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def get_device():
    import torch
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def build_transforms(img_size: int = 224):
    from torchvision import transforms as T
    train_tf = T.Compose([
        T.RandomResizedCrop(img_size, scale=(0.7, 1.0)),
        T.RandomHorizontalFlip(),
        T.RandomRotation(10),
        T.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.1),
        T.ToTensor(),
        T.Normalize(IMAGENET_MEAN, IMAGENET_STD),
    ])
    eval_tf = T.Compose([
        T.Resize(int(img_size * 1.14)),
        T.CenterCrop(img_size),
        T.ToTensor(),
        T.Normalize(IMAGENET_MEAN, IMAGENET_STD),
    ])
    return train_tf, eval_tf


def build_loaders(data_dir, img_size=224, batch_size=32, num_workers=2):
    from torch.utils.data import DataLoader
    from torchvision import datasets
    data_dir = Path(data_dir)
    train_tf, eval_tf = build_transforms(img_size)
    sets = {
        "train": datasets.ImageFolder(data_dir / "train", train_tf),
        "val": datasets.ImageFolder(data_dir / "val", eval_tf),
        "test": datasets.ImageFolder(data_dir / "test", eval_tf),
    }
    loaders = {
        k: DataLoader(v, batch_size=batch_size, shuffle=(k == "train"),
                      num_workers=num_workers, pin_memory=True)
        for k, v in sets.items()
    }
    return sets, loaders


def build_model(arch: str, num_classes: int, pretrained: bool = True):
    """ResNet-50 or EfficientNet-B0 with a new classification head."""
    import torch.nn as nn
    from torchvision import models
    if arch == "resnet50":
        w = models.ResNet50_Weights.DEFAULT if pretrained else None
        m = models.resnet50(weights=w)
        m.fc = nn.Linear(m.fc.in_features, num_classes)
    elif arch == "efficientnet_b0":
        w = models.EfficientNet_B0_Weights.DEFAULT if pretrained else None
        m = models.efficientnet_b0(weights=w)
        m.classifier[1] = nn.Linear(m.classifier[1].in_features, num_classes)
    else:
        raise ValueError(f"Unsupported arch: {arch}")
    return m


def head_params(model, arch):
    return model.fc.parameters() if arch == "resnet50" else model.classifier.parameters()


def last_block_params(model, arch):
    """Parameters of the last backbone block plus the head (fine-tuning stage)."""
    if arch == "resnet50":
        return list(model.layer4.parameters()) + list(model.fc.parameters())
    return list(model.features[-2:].parameters()) + list(model.classifier.parameters())
