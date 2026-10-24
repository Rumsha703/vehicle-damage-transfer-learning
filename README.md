# Vehicle Damage Classification with Transfer Learning (PyTorch)

An end-to-end computer vision pipeline that classifies vehicle damage images using a pre-trained CNN fine-tuned with PyTorch. It covers OpenCV preprocessing, image augmentation, transfer learning, evaluation with precision, recall and F1-score, Grad-CAM explainability and a human-in-the-loop review cycle.

> **Status:** personal learning project. Results below are from my own runs on a public dataset. This is a research prototype, not a production system.

## Dataset

Dataset used: **<NAME, LINK, LICENCE – fill in>**. Images are not included in this repository. Place them as:

```
data/raw/<class_name>/*.jpg      e.g. dent, scratch, crack, undamaged
```

Check the dataset licence before use and cite it here. A classification-ready car-damage dataset from Kaggle works, or you can crop labelled regions from a detection dataset such as CarDD (check its access terms).

## Pipeline

| Step | Script | What it does |
|---|---|---|
| 1 | `src/preprocess.py` | OpenCV cleaning: drops unreadable, blurry (Laplacian variance) and duplicate images; resizes; makes stratified train/val/test splits plus a held-out `pool` for the review cycle |
| 2 | `src/train.py` | Transfer learning: frozen backbone, train head, then fine-tune last block at 10x lower learning rate. Augmentation, class-weighted loss, best checkpoint by validation macro-F1 |
| 3 | `src/evaluate.py` | Test-set precision, recall, F1 per class, confusion matrix, list of misclassified images |
| 4 | `src/gradcam.py` | Grad-CAM overlays showing which regions drive predictions |
| 5 | `src/hitl.py` | Human-in-the-loop: queue low-confidence pool images, review them, merge corrections into training, retrain |

## Quick start

```bash
pip install -r requirements.txt
python src/preprocess.py --raw data/raw --out data
python src/train.py --arch resnet50                 # transfer learning
python src/train.py --arch resnet50 --scratch       # baseline: no pre-trained weights
python src/evaluate.py --ckpt outputs/resnet50_pretrained_best.pt
python src/gradcam.py --ckpt outputs/resnet50_pretrained_best.pt --images data/test/<class>
```

A free Google Colab GPU is enough. Use `--arch efficientnet_b0` for a second architecture.

## Human-in-the-loop cycle

```bash
python src/hitl.py queue --ckpt outputs/resnet50_pretrained_best.pt --images data/pool --threshold 0.7
# open outputs/review_queue.csv and fill reviewer_label for each row
python src/hitl.py merge
python src/train.py --arch resnet50 --out outputs/round2
python src/evaluate.py --ckpt outputs/round2/resnet50_pretrained_best.pt
```

The pool is separate from the test set, so test results stay clean. If you use `--simulate` (reviewer label taken from the folder name), say so in your results.

## Results (fill in from your own runs)

| Model | Test accuracy | Macro precision | Macro recall | Macro F1 |
|---|---|---|---|---|
| ResNet-50 from scratch | | | | |
| ResNet-50 transfer learning | | | | |
| EfficientNet-B0 transfer learning | | | | |
| ResNet-50 after 1 review round | | | | |

Add: the confusion matrix, 3 to 4 Grad-CAM examples, and a short note on which classes are confused and why.

## Limitations (keep these honest)

- Image classification only. No object detection or segmentation.
- Small public dataset, single train/val/test split, one random seed.
- Not deployed or tested on real repair-shop images.
- Grad-CAM shows model attention, not proof of correct reasoning.

## Reproducibility

Fixed seed (42), pinned split ratios, `requirements.txt`, and saved training histories in `outputs/`.
