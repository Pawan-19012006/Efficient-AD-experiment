# EfficientAD Bottle Anomaly Detection

> An initial visual anomaly-detection experiment using EfficientAD-S and Anomalib on a partial MVTec AD bottle dataset subset, executed in Google Colab on an NVIDIA Tesla T4 GPU.

## Overview

This project explores **industrial visual anomaly detection**: learning the appearance of normal products and identifying unusual patterns in test images. The experiment uses the bottle category from the MVTec Anomaly Detection (MVTec AD) dataset and the EfficientAD implementation provided by Anomalib.

The current goal is to validate the end-to-end pipeline:

1. Prepare the environment and dataset.
2. Initialize and train EfficientAD on normal bottle images.
3. Run inference on normal and defective test images.
4. inspect image-level scores and spatial anomaly maps.
5. Evaluate image-level detection and pixel-level localization.

This is a **pipeline validation experiment**, not a reproduction of the full benchmark in the EfficientAD paper.

## Current Status

- [x] Colab GPU environment configured
- [x] Anomalib and EfficientAD imports verified
- [x] Bottle subset downloaded and inspected
- [x] Data module initialized
- [x] EfficientAD-S initialized
- [x] Training completed for 10 requested epochs
- [x] Model checkpoint saved
- [x] Predictions generated for the test set
- [x] Image-level metrics calculated
- [x] Pixel-level AUROC calculated
- [x] AU-PRO calculated at an FPR limit of 0.30
- [x] Sample anomaly maps visualized

## Environment

| Component | Value |
|---|---|
| Execution environment | Google Colab |
| GPU | NVIDIA Tesla T4 |
| Python | 3.13 (Colab runtime) |
| PyTorch | 2.11.0+cu130 |
| Torchvision | 0.26.0+cu130 |
| Anomalib | 2.7.0 |
| CUDA available | Yes |
| Model | EfficientAD-S (`EfficientAdModelSize.S`) |
| Training epochs requested | 10 |
| Training batch size | 1 |
| Evaluation batch size | 4 |
| Data loader workers | 2 |
| Random seed | 42 |

Versions describe the runtime used in this experiment. Recheck package compatibility if reproducing it in a different environment.

## Dataset

The experiment used the **bottle** category from a Hugging Face mirror of MVTec AD, downloaded with `huggingface_hub.snapshot_download`.

Observed files:

| Split / category | Number of images |
|---|---:|
| `train/good` | 209 |
| `test/good` | 20 |
| `test/broken_large` | 20 |
| `test/broken_small` | 22 |
| `test/contamination` | 21 |
| **Total test images** | **83** |
| Ground-truth masks for defective test images | 63 |

The training split contains normal (`good`) images. Defective test categories are used for evaluation. Ground-truth masks support pixel-level localization evaluation.

**Dataset limitation:** this was a partial mirror/subset, not the full MVTec AD bottle benchmark. Results should not be represented as official MVTec AD benchmark results or compared directly with published full-dataset numbers. Check the dataset's license and terms before any use, especially commercial use.

## Model and Training

EfficientAD is an industrial anomaly-detection method built around a pretrained teacher network, a student network, and an autoencoder branch. It produces image-level anomaly scores and spatial anomaly maps.

The model was initialized through Anomalib:

```python
from anomalib.models import EfficientAd

model = EfficientAd(
    model_size="small",
    imagenet_dir="/content/datasets/imagenette",
)
```

The Anomalib engine was configured for 10 epochs on one GPU. The implementation downloaded its pretrained teacher weights and auxiliary Imagenette data, calculated teacher feature statistics and validation quantiles, and then completed training.

The checkpoint was saved at:

```text
/content/results/efficientad_bottle/
  EfficientAd/MVTecAD/bottle/v1/
  weights/lightning/model.ckpt
```

The checkpoint was approximately **73.4 MB**. The path above is specific to the Colab runtime; the file must be copied to persistent storage if it needs to survive a runtime reset.

## Evaluation Results

### Summary

| Metric | Result |
|---|---:|
| Image-level classification accuracy | **97.59% (81/83)** |
| Balanced accuracy | **96.67%** |
| Image-level AUROC | **99.92%** |
| Pixel-level AUROC | **97.23%** |
| AU-PRO at FPR limit 0.30 | **90.15%** |

These values are the results recorded from this experiment. Pixel AUROC and AU-PRO were calculated using the prediction anomaly maps and ground-truth masks. The AU-PRO setting used a false-positive-rate limit of 0.30.

### Per-category image classification

| Test category | Images | Correct | Accuracy | Mean anomaly score |
|---|---:|---:|---:|---:|
| `broken_large` | 20 | 20 | 100.0% | 0.5118 |
| `broken_small` | 22 | 22 | 100.0% | 0.5143 |
| `contamination` | 21 | 20 | 95.2% | 0.5822 |
| `good` | 20 | 19 | 95.0% | 0.4983 |
| **Total** | **83** | **81** | **97.59%** | — |

The per-category mean scores are descriptive and should not be treated as calibrated probabilities.

### Confusion matrix

Rows are actual classes and columns are predicted classes, with class order `[normal, anomaly]`:

| Actual \ Predicted | Normal | Anomaly |
|---|---:|---:|
| Normal | 19 | 1 |
| Anomaly | 1 | 62 |

This corresponds to one false positive among the 20 normal test images and one false negative among the 63 defective test images.

### Interpretation

- The image-level AUROC indicates strong ranking separation between normal and defective images on this test subset.
- Pixel-level AUROC indicates that anomalous pixels generally receive higher scores than normal pixels.
- AU-PRO evaluates overlap with defect regions while accounting for false-positive pixels, using the specified FPR limit.
- Visual inspection of the first four `broken_large` samples showed responses around parts of the damaged bottle rims, but the heatmaps did not perfectly match every ground-truth defect region.

**Important:** these metrics do not establish production readiness. The test set is small and comes from a partial subset. More representative data, repeatable evaluation, threshold analysis, failure analysis, and controlled comparisons are required before drawing general conclusions.

## Prediction Outputs

Anomalib returned **21 prediction batches** for the 83 test images, using an evaluation batch size of 4.

Each prediction batch included fields such as:

- `image_path`
- `pred_score`
- `pred_label`
- `anomaly_map`
- `pred_mask`
- `gt_label`
- `gt_mask`

The anomaly map had shape `4 × 256 × 256` for the first batch. The image-level score is a scalar anomaly score; it is not a probability unless separately calibrated.

## Repository and Artifact Notes

The GitHub repository currently contains the experiment notebook and this README. The full experiment ZIP was too large for a normal repository upload.

Recommended artifact organization:

- Keep the notebook and documentation in GitHub.
- Store the full ZIP and checkpoint in persistent storage such as Google Drive, or attach the checkpoint to a GitHub Release if appropriate.
- Do not commit the entire dataset.
- If storing artifacts elsewhere, add a stable, access-controlled location here rather than publishing private links in a public repository.

Expected experiment bundle contents:

```text
efficientad_bottle_experiment/
├── README.md
├── metrics.json
├── predictions.csv
├── environment.json
├── model.ckpt
└── visualizations/
    └── first_batch.png
```

The `predictions.csv` contains per-image paths, ground-truth category, predicted label, and anomaly score. Paths may refer to the original Colab filesystem and will need adjustment outside that environment.

## Reproducing the Experiment

The original experiment was run interactively in Google Colab. For a reproduction:

1. Open `EfficientAD.ipynb` in Colab.
2. Select a GPU runtime.
3. Install compatible versions of Anomalib and dependencies as shown in the notebook.
4. Download the permitted MVTec AD bottle subset into the expected directory.
5. Run the notebook cells in order.
6. Preserve the checkpoint and generated artifacts outside the temporary Colab filesystem.
7. Recompute all metrics from the saved predictions and masks.

The original dataset download path used by Anomalib returned HTTP 404 during this run, so the experiment used a Hugging Face mirror. Dataset availability and layout may change; verify the data structure before running training.

## Research Paper

The experiment is motivated by:

**Batzner, K., Heckler, L., and König, R. — “EfficientAD: Accurate Visual Anomaly Detection at Millisecond-Level Latencies.”**

- Paper: https://arxiv.org/abs/2303.14535

The implementation used here is Anomalib's EfficientAD, and this notebook experiment should not be described as an exact reproduction of every experimental setting in the paper.

## Next Steps

- [ ] Preserve the checkpoint and experiment ZIP in persistent storage.
- [ ] Keep the notebook, README, and results organized in the private GitHub repository.
- [ ] Inspect false-positive and false-negative examples.
- [ ] Verify metric implementation and settings against the paper.
- [ ] Study EfficientAD's teacher–student architecture and autoencoder branch.
- [ ] Study training losses, feature-map normalization, anomaly-map fusion, and AU-PRO evaluation.
- [ ] Repeat evaluation on the complete dataset or another appropriately licensed representative dataset before making benchmark claims.
- [ ] Consider integration with a VLM or agent only after the actual downstream use case and evaluation requirements are established.

## License and Responsible Use

This repository documents an experiment and does not grant rights to the MVTec AD dataset, pretrained weights, or third-party dependencies. Review each resource's license and terms before redistribution or commercial use. The MVTec AD dataset is commonly distributed under non-commercial terms; obtain appropriate permission for any commercial application.
