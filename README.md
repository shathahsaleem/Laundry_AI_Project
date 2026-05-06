# Laundry_AI_Project
🧺 AI-Powered Laundry Sorting Assistant

A repository containing all source code and files for a Computer Vision course project.

This project implements a real-time, computer-vision-based laundry sorting system. It uses a fine-tuned ResNet-50 deep learning model alongside OpenCV to classify clothing into three categories (Whites, Darks, Colors) and applies HSV (Hue, Saturation, Value) analysis to detect potential color-bleed risks before washing.

📂 Repository Structure
dataset_prepper.py parses the H&M dataset, applies rule-based filtering, and balances images into Train/Validation/Test directories.
train_model.py fine-tunes a pre-trained ResNet-50 model using PyTorch Lightning, evaluates performance, and generates confusion matrices. live_sorter.py Runs the real-time OpenCV application with webcam input, AI inference, temporal smoothing, and HSV bleed-risk detection. new_laundry_resnet50.pth saved PyTorch model weights (generated after training).

⚙️ Prerequisites & Installation

Make sure you have Python 3.8+ installed.

1. Open Terminal / Command Prompt

Navigate to your project folder:

cd path/to/your/project/folder
2. Install Dependencies
pip install torch torchvision pytorch-lightning opencv-python pandas scikit-learn matplotlib seaborn Pillow torchmetrics

📥 Dataset Setup

This project uses the H&M Personalized Fashion Recommendations dataset.

Before running the code, ensure the following are placed in the root directory:

articles.csv
Folder named exactly: images_256_256


🚀 How to Run the Project

Follow these steps in order.

🟢 Step 1: Prepare the Dataset
python dataset_prepper.py
What this does:
Reads articles.csv
Filters out multi-colored/patterned clothing
Categorizes items into Whites / Darks / Colors
Balances the dataset
Creates a new dataset/ folder with:
train/
val/
test/

🟢 Step 2: Train Model + Run Automated Testing
python train_model.py

⚠️ Important Note:

If running locally (not Google Colab), open train_model.py and comment out:

!pip install ...
!unzip ...
What this does:

1. Training Phase

Loads pre-trained ResNet-50
Applies data augmentation (color jitter, flipping)
Fine-tunes on training data

2. Model Saving

Saves trained model as:

new_laundry_resnet50.pth

3. Testing Phase

Automatically evaluates on the test dataset

4. Output Results

Prints Classification Report (Precision, Recall, F1-score)
Displays:
Confusion Matrix
Training & Validation curves

🟢 Step 3: Real-Time Live Testing (Webcam)
python live_sorter.py
Requirements:
Webcam connected
new_laundry_resnet50.pth in project directory

🎮 Interactive Controls

Hold clothing in front of the camera and press:

w → Whites Mode
d → Darks Mode
c → Colors Mode (includes bleed-risk detection)
r → Reset / Return to menu
q → Quit application

🧠 Key Features
Transfer Learning
Custom classification layer on a pre-trained ResNet-50 backbone.
Data Augmentation
Improves generalization using flipping and color jitter.
Temporal Smoothing
Uses a 15-frame buffer to stabilize predictions.
Background Detection
Pauses prediction if no clothing is detected (low variance threshold).
HSV Color Analysis
Detects high saturation levels to warn about possible dye bleeding.

📌 Notes
Ensure dataset paths are correct before running scripts
Training may take time depending on your hardware (GPU recommended)
Webcam performance depends on lighting conditions
