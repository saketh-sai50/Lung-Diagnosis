Lung CT Cancer Detection & Tumor Segmentation with U-Net and XAI

This project focuses on building a deep learning pipeline to detect lung cancer from CT scan images. It uses a **Convolutional Neural Network (CNN)** for classification and a **U-Net** model for precise tumor segmentation. Explainability is integrated using **Grad-CAM** and **SHAP**, allowing visual insights into model predictions for clinical trust.

---

 Project Objectives

- Classify CT scan images as **Benign**, **Malignant**, or **Normal**
- Perform **tumor segmentation** using U-Net
- Apply **Grad-CAM** for explainable AI (XAI)
- Generate visual diagnostic reports to support clinical decisions

---

 Models Used

- ✅ **Vision Transformer** for multi-class classification (3 classes)
- ✅ **U-Net** for segmentation of tumor regions
- ✅ **Grad-CAM** for class activation heatmaps
  

---

 Dataset

> The dataset contains CT images labeled as:
- `Benign` (non-cancerous)
- `Malignant` (cancerous)
- `Normal` (healthy lungs)

**Format:** `.png` / `.jpg` grayscale CT images  
**Labels:** Stored in `labels.csv` with fields: `filename`, `label`

(*Note: For licensing reasons, dataset is not included in this repo. Please refer to publicly available lung CT datasets such as the [LUNA16 dataset](https://luna16.grand-challenge.org/), [NSCLC Radiogenomics](https://wiki.cancerimagingarchive.net/display/Public/NSCLC+Radiogenomics), or others.*)

---

## 🛠️ Tech Stack

- Python
- Keras / TensorFlow
- OpenCV
- NumPy, Pandas, Matplotlib
- Grad-CAM, Scikit-learn
- Flask (optional for web interface)

---

## 🚀 How It Works

### 🩺 1. Preprocessing
- Resized CT images to 224×224
- Normalized pixel values
- Applied data augmentation for generalization

### 🧠 2. Classification (CNN)
- Trained a Vision Transformer (ViT) model on CT image data for 3-class classification.
- Applied a softmax output layer for multi-class prediction.
- Achieved improved feature representation by leveraging transformer-based attention mechanisms.
- Evaluated using accuracy, precision, recall, and F1-score

### 🧬 3. Segmentation (U-Net)
- Trained U-Net model on binary lung/tumor masks
- Output: segmented binary masks highlighting tumor areas

### 🔎 4. Explainability
- Applied **Grad-CAM** to CNN to generate activation heatmaps
- Combined outputs into downloadable diagnostic reports

---

## 📊 Results

| Metric | Value |
|--------|-------|
| Classification Accuracy | 94–97% |
| Segmentation Dice Score | 0.85–0.90 |
| XAI Quality | Verified via expert visual inspection |

---

## 🖼️ Sample Outputs

| Image | Prediction | Grad-CAM | U-Net Mask |
|-------|------------|----------|-------------|
| ![](samples/input.png) | Malignant | ![](samples/gradcam.png) | ![](samples/unet_mask.png) |

---
 Future Improvements

- Convert to **end-to-end Flask web app**
- Support for **3D CT scan volumes**
- Integration with **LUNA16 full dataset**
- Export to **PDF reports for doctors**

---

 License

This project is open-source for educational and non-commercial use. Attribution appreciated.



 Credits

Developed by KORUPOJU SAKETH SAI SRIKAR | AI & ML @ KITS Warangal  


---
