# Laundry AI Assistant

An end-to-end computer vision pipeline that automates garment sorting (**Whites**, **Darks**, **Colors**) using a fine-tuned ResNet-50 network and performs real-time HSV saturation analysis to prevent dye bleed-risk.

---

## Highlights

* **ResNet-50 Transfer Learning:** Custom classification layer fine-tuned on single-garment images.
* **HSV Bleed-Risk Detection:** Real-time color space analysis to warn against high-saturation dye transfers.
* **Temporal Smoothing:** 15-frame rolling prediction buffer to eliminate webcam flicker.
* **Active Background Filter:** Variance-based detection that pauses predictions when no clothing is present.

---

## Repository Structure

```text
dataset_prepper.py   # Ingests H&M metadata, filters multi-patterns, and creates dataset splits
train_model.py       # Fine-tunes ResNet-50 with PyTorch Lightning and runs evaluation
live_sorter.py       # Real-time OpenCV execution engine with webcam feed & HSV risk check
README.md            # System documentation
```

> *Note: Raw datasets (`images_256_256/`, `articles.csv`) and model weights (`*.pth`) are ignored via `.gitignore` to keep the repository lightweight.*

---

## Quickstart

### 1. Requirements & Setup
Ensure you have **Python 3.8+** installed.

```bash
git clone https://github.com/shathahsaleem/Laundry_AI_Project.git
cd Laundry_AI_Project
pip install torch torchvision pytorch-lightning opencv-python pandas scikit-learn matplotlib seaborn Pillow torchmetrics
```

### 2. Dataset Preparation
Download the **H&M Personalized Fashion Recommendations** dataset and place `articles.csv` and `images_256_256/` into the root directory.

---

## Execution Pipeline

Execute the scripts sequentially:

### Step 1: Preprocess Dataset
Processes CSV metadata, strips multi-pattern items, balances classes, and generates `dataset/` (`train/`, `val/`, `test/`).

```bash
python dataset_prepper.py
```

### Step 2: Train & Evaluate Model
Trains ResNet-50, generates confusion matrices and classification reports, and saves model weights to `new_laundry_resnet50.pth`.

```bash
python train_model.py
```

### Step 3: Run Real-Time Classifier
Launches the live webcam sorting application.

```bash
python live_sorter.py
```

---

## Interactive Controls

While running `live_sorter.py`, hold garments up to the webcam and use these keys:

| Key | Mode / Action | Function |
| :---: | :--- | :--- |
| **`w`** | **Whites Mode** | Filters and verifies items for white loads |
| **`d`** | **Darks Mode** | Filters and verifies items for dark loads |
| **`c`** | **Colors Mode** | Activates active **HSV Bleed-Risk Detection** |
| **`r`** | **Reset** | Clears the temporal buffer and resets frame state |
| **`q`** | **Quit** | Exits the application |
