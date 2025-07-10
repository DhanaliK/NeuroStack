# 🍎 Lesion-Aware Explainability Validation for Apple Disease CNN

This repository implements a lightweight Grad-CAM-based explainability validation pipeline for apple leaf disease classification using a ResNet50 model. The goal is to visually assess whether the model focuses on the correct (lesion) regions when making predictions.


## 📌 Objective

To validate CNN explainability using Grad-CAM heatmaps and compare them with simulated lesion masks on real apple leaf images. IoU (Intersection over Union) is used to quantify attention-mask overlap.


## 📁 Project Structure

APPLE_CNN
/│

├── main.py # Core Python script

├── report.pdf # project report

├── Results/ # Output Grad-CAM visuals + IoU scores

│ └── visuals

│ └── iou_scores.csv

├── sample_dataset/ # Sample input images

│ └── images/ # 20 real apple leaf images (5 per class)

└── requirements.txt # Python dependencies


## 🧠 Model

- **ResNet50**, pre-trained on ImageNet
- Grad-CAM applied to the final convolutional layer
- Dummy lesion masks created using HSV thresholding


## 🛠️ How to Run

### 1. Clone the Repo

```
git clone https://github.com/your-username/apple_cnn.git
cd apple_cnn
```
### 2. Setup Virtual Environment (Optional but Recommended)
```
python -m venv venv
source venv/bin/activate  # or venv\Scripts\activate on Windows
pip install -r requirements.txt
```

### 3. Run the Main Script
```
python main.py
```

Output images will be saved in Results/visuals

IoU scores will be stored in Results/iou_scores.csv


## 📊 Sample Output

| Filename      | Predicted Class | IoU    |
| ------------- | --------------- | ------ |
| apple\_2.jpg  | leaf\_beetle    | 0.0477 |
| apple\_10.jpg | head\_cabbage   | 0.0001 |

 🖼️ Grad-CAM visualizations available in Results/visuals/


## 📚 Dataset Acknowledgement

This project uses data from the publicly available Apple Leaf Disease Dataset on Kaggle:

**Dataset:** Apple Leaf Disease Dataset  
**Author:** ludehsar  
**Source:** [https://www.kaggle.com/datasets/ludehsar/apple-disease-dataset](https://www.kaggle.com/datasets/ludehsar/apple-disease-dataset)

All rights and credits for the dataset belong to the original author. This project is for educational and research purposes only.



🙌 Author
---
Dhanali Khandagale
📧 dhanali26a@gmail.com
