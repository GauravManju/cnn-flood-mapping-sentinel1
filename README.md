**GitHub Repository:** https://github.com/GauravManju/cnn-flood-mapping-sentinel1


# CNN-Based Flood Extent Mapping Using Sentinel-1 SAR Imagery

**Module:** CO7113 — Artificial Intelligence (MSc)
**Institution:** University of Leicester
**Assignment:** Group Research Project — Assignment 2
**Submission Date:** 27 April 2026

---

This project was developed as part of CO7113 at the University of Leicester. We ran into several real challenges during implementation — including NaN-filled SAR files, GDAL installation issues on Mac, and AMP training errors — all of which are documented in the Common Issues section below. The commands in this README reflect exactly what we ran to produce the submitted results.


## What This Project Does

This project trains a deep learning model to automatically detect flooded areas
from satellite radar images. Given a Sentinel-1 SAR (Synthetic Aperture Radar)
image chip, the model predicts which pixels are flooded and which are not.

This is useful in real emergencies because radar satellites can see through clouds,
unlike normal optical cameras — meaning flood maps can be produced even when the
sky is overcast during a disaster.

---

## Our Results

The model was trained and evaluated on the Sen1Floods11 dataset using a
Tesla T4 GPU on Google Colab. Training ran for 54 epochs before early
stopping triggered (best checkpoint at epoch 39).

| Metric    | Score   |
|-----------|---------|
| IoU       | 0.5626  |
| F1 / Dice | 0.7201  |
| Precision | 0.6659  |
| Recall    | 0.7840  |
| Accuracy  | 93.37%  |

---

## Project Structure

```
project/
├── data/
│   └── sen1floods11/          <- dataset goes here (see Dataset Setup)
├── notebooks/
│   ├── 01_data_exploration.ipynb
│   └── 02_results_analysis.ipynb
├── src/
│   ├── __init__.py
│   ├── dataset.py             <- data loading, NaN handling, augmentation
│   ├── model.py               <- ResNet-34 U-Net, U-Net scratch, Vanilla CNN
│   ├── train.py               <- training loop, BCEWithLogitsLoss + Dice
│   ├── evaluate.py            <- IoU, Dice, precision, recall, confusion matrix
│   └── utils.py               <- logging, seeding, checkpointing, plotting
├── outputs/                   <- results saved here after training
│   ├── best_model.pth
│   ├── test_metrics.json
│   ├── training_curves.png
│   ├── confusion_matrix.png
│   ├── per_batch_metrics.csv
│   └── prediction_0000.png ... prediction_0007.png
├── main.py                    <- run this to train and evaluate
├── requirements.txt
└── README.md
```

---

## Important Note on the Dataset

The Sen1Floods11 S1Hand files store SAR values already in decibel (dB)
scale (approximately -50 to +5 dB), NOT linear scale. Many chips also
contain NaN nodata pixels due to SAR acquisition geometry.

Our preprocessing handles both automatically:
- NaN pixels are replaced with the channel mean before normalisation
- Values are clamped to [-50, +5] dB then Z-score normalised
- No log transformation is applied (data is already in dB)

---

## Dataset Setup

### Step 1 - Create the folders

```bash
mkdir -p data/sen1floods11/v1.1/data/flood_events/HandLabeled
mkdir -p data/sen1floods11/v1.1/splits
```

### Step 2 - Download the SAR images and labels (~4 GB, takes 10-30 mins)

```bash
gsutil cp -r "gs://sen1floods11/v1.1/data/flood_events/HandLabeled" \
  "data/sen1floods11/v1.1/data/flood_events/"
```

### Step 3 - Download the split CSV files

```bash
gsutil cp "gs://sen1floods11/v1.1/splits/flood_handlabeled/flood_train_data.csv" \
  "data/sen1floods11/v1.1/splits/flood_train_data.csv"

gsutil cp "gs://sen1floods11/v1.1/splits/flood_handlabeled/flood_valid_data.csv" \
  "data/sen1floods11/v1.1/splits/flood_valid_data.csv"

gsutil cp "gs://sen1floods11/v1.1/splits/flood_handlabeled/flood_test_data.csv" \
  "data/sen1floods11/v1.1/splits/flood_test_data.csv"
```

### Expected structure after download

```
data/sen1floods11/v1.1/
├── data/flood_events/HandLabeled/
│   ├── S1Hand/       <- SAR image chips (*_S1Hand.tif)
│   └── LabelHand/    <- flood mask labels (*_LabelHand.tif)
└── splits/
    ├── flood_train_data.csv   <- 252 training chips
    ├── flood_valid_data.csv   <- 89 validation chips
    └── flood_test_data.csv    <- 90 test chips
```

---

## Environment Setup

### Requirements
- Python 3.9 or higher
- GPU strongly recommended (Google Colab T4 GPU used for this project)

### Install packages

```bash
# PyTorch — CPU (Mac)
pip install torch torchvision

# PyTorch — CUDA 11.8 (Linux/Windows with NVIDIA GPU)
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu118

# Remaining packages
pip install segmentation-models-pytorch albumentations rasterio
```

Note: Do NOT install GDAL separately. Remove it from requirements.txt:
```bash
sed -i '' '/GDAL/d' requirements.txt
```
rasterio includes its own GDAL and is sufficient for this project.

---

## Running the Project

Always run commands from inside the project/ folder.

### Train the model (recommended — runs everything automatically)

```bash
python3 main.py
```

This loads the data, trains for up to 100 epochs, saves the best
checkpoint, evaluates on the test set, and saves all outputs.

Each epoch takes ~10-15 seconds on a T4 GPU.
Training stops automatically when val IoU stops improving (patience=15).

### Evaluate only (skip training, use saved checkpoint)

```bash
python3 main.py --eval_only --checkpoint outputs/best_model.pth
```

### Run on Google Colab (recommended — free T4 GPU)

```python
# Cell 1 — upload project zip and set working directory
from google.colab import files
import zipfile, os
uploaded = files.upload()  # upload CNN_Flood_Mapping_Code.zip
zipfile.ZipFile('CNN_Flood_Mapping_Code.zip').extractall('/content/')
os.chdir('/content/project')

# Cell 2 — install packages
!pip install segmentation-models-pytorch albumentations rasterio -q

# Cell 3 — download dataset
!mkdir -p data/sen1floods11/v1.1/data/flood_events/HandLabeled
!mkdir -p data/sen1floods11/v1.1/splits
!gsutil cp -r "gs://sen1floods11/v1.1/data/flood_events/HandLabeled" "data/sen1floods11/v1.1/data/flood_events/"
!gsutil cp "gs://sen1floods11/v1.1/splits/flood_handlabeled/flood_train_data.csv" "data/sen1floods11/v1.1/splits/flood_train_data.csv"
!gsutil cp "gs://sen1floods11/v1.1/splits/flood_handlabeled/flood_valid_data.csv" "data/sen1floods11/v1.1/splits/flood_valid_data.csv"
!gsutil cp "gs://sen1floods11/v1.1/splits/flood_handlabeled/flood_test_data.csv"  "data/sen1floods11/v1.1/splits/flood_test_data.csv"

# Cell 4 — run training
!python3 main.py
```

Enable GPU: Runtime -> Change runtime type -> T4 GPU

---

## Expected Outputs

| File | Description |
|------|-------------|
| best_model.pth | Model weights at best validation IoU |
| test_metrics.json | Final test results (IoU, Dice, precision, recall, accuracy) |
| history.json | Per-epoch loss and IoU for train and validation splits |
| training_curves.png | Loss and IoU learning curves |
| confusion_matrix.png | Pixel-level confusion matrix on test set |
| per_batch_metrics.csv | Per-chip IoU and Dice on test set |
| prediction_0000-0007.png | Sample predictions (VV input / ground truth / prediction) |
| config.json | Full training configuration |

---

## Reproducing Our Results

```bash
python3 main.py --model resnet34_unet --seed 42 --output_dir outputs/
```

Expected test metrics:
```
IoU       : 0.5626
Dice      : 0.7201
Precision : 0.6659
Recall    : 0.7840
Accuracy  : 0.9337
```

Minor variation (+/- 0.005 IoU) is possible due to hardware differences.

---

## Key Design Decisions

| Decision | Reason |
|----------|--------|
| Sentinel-1 SAR input | Sees through clouds — essential for flood emergencies |
| VV + VH dual-polarisation | Complementary sensitivity improves urban/vegetation discrimination |
| U-Net with skip connections | Preserves spatial detail lost during downsampling |
| ResNet-34 pretrained encoder | ImageNet features transfer well even to SAR data |
| BCEWithLogitsLoss not BCE | BCE causes CUDA assertion failures under AMP — logits version is stable |
| NaN imputation by channel mean | Prevents NaN propagation without introducing large constant bias |
| Positive class weight 4.5 | Compensates for ~18% flooded pixel fraction (class imbalance) |
| Encoder LR 10x lower than decoder | Preserves pretrained representations during fine-tuning |

---

## Common Issues and Fixes

**GDAL installation fails**
Do not install GDAL. Run: `sed -i '' '/GDAL/d' requirements.txt`

**NaN loss during training**
Caused by NaN pixels in SAR data entering the network.
The dataset.py in this submission handles this automatically.
Make sure you are using the latest dataset.py file.

**CUDA assertion error**
Caused by using BCE with sigmoid outputs under AMP.
The train.py in this submission uses BCEWithLogitsLoss which is AMP-safe.

**gsutil crashes on Mac**
Remove the -m flag: use `gsutil cp -r` not `gsutil -m cp -r`

**FileNotFoundError: flood_handlabeled_split.csv not found**
This project uses three separate CSV files, not one combined file.
Download flood_train_data.csv, flood_valid_data.csv, flood_test_data.csv
using the commands in Dataset Setup above.

---

## Group Contributions

| Member | Student ID | Contributions |
|--------|-----------|---------------|
| Gaurav Manju | gm420 | Project lead. Dataset acquisition and preprocessing pipeline, model architecture design, training loop implementation, debugging (NaN handling, BCEWithLogitsLoss fix, GDAL issues), Google Colab setup and execution, evaluation pipeline, report writing, README |
| Santhosh Kumar Venkatesan | skv14 | Dataset research and Sen1Floods11 documentation review, preprocessing validation, augmentation strategy, report sections on dataset and methodology |
| Mohammed Haaris | mhih1 | Literature review, model architecture research, baseline model comparison, report sections on literature review and model design |
| Dinakar Nayak Narsimhamurthy | dnnm1 | Training protocol research, hyperparameter selection, evaluation metrics research, report sections on training and evaluation |
| Sashi Dommaraju | sd683 | Results analysis, ablation study interpretation, limitations and future work sections, presentation preparation |




## AI Usage Declaration

Code structure and documentation were developed with assistance from
Claude (Anthropic). All content was reviewed and tested by group members.
The key technical fixes — NaN imputation strategy, BCEWithLogitsLoss
substitution, dB-scale data handling — were identified and implemented
by group members in response to real errors during development.
