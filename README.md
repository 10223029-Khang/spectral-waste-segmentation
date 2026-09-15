# Spectral Compression as a Design Parameter for Multimodal Waste Segmentation

Undergraduate research project — Electrical and Computer Engineering Program,
Vietnamese-German University (VGU), Binh Duong, Vietnam.

Published work on the SpectralWaste dataset fixes the PCA compression depth at `k = 3` and
justifies that choice by retained variance. **We tested that justification and it does not hold.**

## Key results

We swept the number of retained principal components and measured segmentation accuracy
against the transmitted payload per frame.

| k  | Payload (MB) | Compression | mIoU (%) | Δ mIoU |
|----|--------------|-------------|----------|--------|
| 1  | 0.26         | 112.0×      | 31.05    | –      |
| 2  | 0.52         | 56.0×       | 34.06    | +3.01  |
| 3  | 0.79         | 37.3×       | 35.43    | +1.37  |
| 6  | 1.57         | 18.7×       | 37.27    | +1.84  |
| 12 | 3.15         | 9.3×        | 37.78    | +0.51  |

Raw hyperspectral cube: 29.4 MB per 256×256 frame.

**Retained variance does not predict accuracy.** The first three components carry 99.595% of
the training-set spectral variance, so components 4–12 carry at most 0.405% between them —
yet those components add 2.35 mIoU points, a 6.6% relative gain. Variance measures signal
energy, not class separability. Selecting `k` by a variance threshold optimises the wrong quantity.

**We would select k = 6** over the published `k = 3`: it recovers 1.84 of the 2.35 available
points while the spectral stream is still 18.7× smaller than the raw cube.

**A failure mode the aggregate metric hides.** On a test frame whose ground truth is empty, the
model labelled 14.80% of the area as material at 0.944 mean confidence. Correctly segmented
frames return mean confidences between 0.92 and 0.98, so no threshold separates the two
populations — confidence carries no information about correctness in this regime.

**Reproduction gap, reported openly.** Our CMX-B0 with RGB + PCA-3 reaches 35.50% mIoU against
56.6% published for the same configuration. The gap persists at `k = 12`, which rules out
compression depth. We attribute it primarily to training from random initialisation rather than
from ImageNet-pretrained encoders, supported by the per-class pattern: the largest deficits fall
on the classes that depend on fine texture.

## What is ours and what is not

This project builds directly on prior work. To be explicit:

**Not ours:**
- **CMX-B0** — the cross-modal fusion architecture, from Zhang et al., *CMX: Cross-Modal Fusion
  for RGB-X Semantic Segmentation with Transformers*, IEEE T-ITS 2023.
- **SpectralWaste** — the dataset, the acquisition setup and the baseline results we compare
  against, from Casao et al., IROS 2024.
- **MiT-B0 / SegFormer** decoder components, from Xie et al., NeurIPS 2021.

**Ours:**
- Treating `k` as an explicit design variable of the acquisition–server split, and measuring
  accuracy against transmitted payload across `k ∈ {1, 2, 3, 6, 12}`.
- The finding that retained variance fails to predict segmentation accuracy, and the resulting
  recommendation of `k = 6`.
- Reproducing CMX-B0 from random initialisation, quantifying the gap and isolating its cause.
- The observation that model confidence does not separate correct from incorrect predictions
  on unannotated frames.

## Reproduction setup

- Kaggle notebooks, Ubuntu, Python 3.12, PyTorch 2.9.0, CUDA 12.6, single NVIDIA Tesla T4
- Official SpectralWaste split: 514 train / 167 validation / 171 test
- PCA fitted on the training split only (incremental PCA; the full training matrix is ~30 GB in FP32)
- MiT-B0 ×2 encoders, widths [32, 64, 160, 256], decoder embedding 256, 11.186 M parameters at k=3
- SGD, momentum 0.9, weight decay 5e-4, lr 0.01 with polynomial decay (power 0.9)
- 100 epochs, batch size 8, random initialisation, weighted cross-entropy with γ = 0.12
- Input resolution 256×256; ~19 minutes per 100-epoch run

The dataset itself is not redistributed here — see the SpectralWaste authors for access.

## Contents
## Contents

- `spectral-compression-waste-segmentation.pdf` — the full write-up

The training and evaluation notebooks were run on Kaggle and are not archived here yet.

## Team

Five-person project supervised by Mr. Nguyen Vo That Thuyet and Mr. Chan Thai Nguyen Dai.

**My contribution (Cu Minh Khang):** ran the experimental programme — training and evaluation
across all `k` configurations, and the per-class and failure-mode analysis of the results.

## Limitations

One architecture, one input resolution, data from a single facility under fixed illumination,
and a model trained on 514 images. Each configuration was trained once, so the 0.51-point step
from `k = 6` to `k = 12` cannot be separated from run-to-run variance. We did not retrain the
baselines ourselves. The edge–server split is a design: we measured payload sizes, not
end-to-end latency on hardware.# spectral-waste-segmentation
