# 🍃 Tea Leaf Disease Detection using Deep Learning

A deep learning-based image classification system to detect six types of tea leaf diseases using **VGG19** and **Transfer Learning**, with an interactive **Streamlit** web interface for real-time prediction, heatmap visualizations, and disease severity insights.

---

## Project Overview

- **Tech Stack**: Python, TensorFlow, Keras, Streamlit, OpenCV, Matplotlib  
- **Model**: Pre-trained VGG19 with fine-tuning  
- **Accuracy Achieved**: **95.07%** on test data  
- **Dataset Size**: 5,867 images  
- **Frontend**: Streamlit app for image upload, prediction, and visualization  

---

## ✅ Features

- Detects 6 common tea leaf diseases using image classification.
- Achieved 95.07% accuracy through data preprocessing and augmentation.
- Streamlit interface to upload leaf images and get predictions instantly.
- Integrated Grad-CAM heatmap to highlight infected regions on leaves.
- Displays class probabilities and estimated percentage of leaf damage.

---

## 📂 Dataset

- **Source**: [Tea Leaf Disease Dataset - Kaggle](https://www.kaggle.com/datasets/saikatdatta1994/tea-leaf-disease/data)  
- **Classes**:
  - Anthracnose
  - Algal Leaf
  - Bird Eye Spot
  - Brown Blight
  - Healthy
  - Grey Light

---

## 🚀 How to Run Locally

### 1️⃣ Clone the Repository
```bash
git clone https://github.com/vishalarunagiri24/Tea-Leaf-Disease-Detection-using-Deep-Learning.git
cd Tea-Leaf-Disease-Detection-using-Deep-Learning
```

### 2️⃣ Install Dependencies
```bash
pip install -r requirements.txt
```

### 3️⃣ Run the Streamlit App
```bash
streamlit run main.py
```


