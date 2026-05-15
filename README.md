# 👕 Clothing Type Predictor using Computer Vision & Image Processing

<p align="center">
  <img src="https://img.shields.io/badge/Python-3.11-blue?style=for-the-badge&logo=python">
  <img src="https://img.shields.io/badge/TensorFlow-DeepLearning-orange?style=for-the-badge&logo=tensorflow">
  <img src="https://img.shields.io/badge/Streamlit-WebApp-red?style=for-the-badge&logo=streamlit">
  <img src="https://img.shields.io/badge/MobileNetV2-ComputerVision-green?style=for-the-badge&logo=opencv">
</p>

---

## 📌 Project Overview

The **Clothing Type Predictor** is a Computer Vision and Deep Learning based application that classifies different categories of clothing from uploaded images.

This project uses **Convolutional Neural Networks (CNN)** along with **Image Processing techniques** to analyze clothing images and predict the clothing category accurately.

The application is deployed using **Streamlit**, allowing users to upload clothing images and receive real-time predictions with confidence scores.

---

## 🚀 Features

✅ Clothing image classification using CNN
✅ Real-time prediction through Streamlit
✅ Image preprocessing and normalization
✅ Confidence score visualization
✅ Deep Learning based image recognition
✅ User-friendly modern UI
✅ Fast and lightweight prediction system

---

## 🧠 Technologies Used

| Technology         | Purpose                   |
| ------------------ | ------------------------- |
| Python             | Core Programming Language |
| TensorFlow / Keras | Deep Learning Model       |
| ImageDataGenerator | Image Rescaling           |
| NumPy              | Numerical Operations      |
| PIL (Pillow)       | Image Handling            |
| Streamlit          | Web Application           |
| CNN                | Image Classification      |

---

## 📂 Project Structure

```bash
Clothing-Type-Predictor/
│
├── app.py
├── clothing_prediction.ipynb
├── clothing_model.h5
├── classes.json
├── requirements.txt
├── README.md
└── sample_images/
```

---

## 🔍 Project Workflow

### 1️⃣ Data Collection

* Clothing image dataset collected from image classification datasets.

### 2️⃣ Image Preprocessing

* Image resizing
* Normalization
* Conversion into arrays

### 3️⃣ Model Building

* CNN architecture developed using TensorFlow and Keras.

### 4️⃣ Model Training

* Model trained on multiple clothing categories.

### 5️⃣ Prediction

* Uploaded images are classified into clothing categories.

### 6️⃣ Deployment

* Streamlit web application created for real-time prediction.

---

## 🖼️ Image Processing Steps

✔ Image Upload
✔ RGB Conversion
✔ Image Resizing (224x224)
✔ Pixel Normalization
✔ Model Prediction
✔ Confidence Score Display

---

## 📊 Deep Learning Model

The project uses a **Convolutional Neural Network (CNN)** consisting of:

* Convolution Layers
* MaxPooling Layers
* Dense Layers
* Activation Functions
* Softmax Output Layer

The model extracts visual features such as:

* texture
* shape
* patterns
* clothing structure

to classify clothing images accurately.

---

## 💻 Run the Project Locally

### Clone Repository

```bash
git clone <https://github.com/anirray346/Clothing-Type-Prediction/tree/main>
```

### Install Dependencies

```bash
pip install -r requirements.txt
```

### Run Streamlit App

```bash
streamlit run app.py
```

---

## 🌐 Live Demo

🚀 Streamlit Deployment Link:
<https://clothing-type-prediction-26.streamlit.app/>

---

## 📈 Future Improvements

* Real-time webcam clothing prediction
* Transfer Learning implementation
* Mobile application integration
* Fashion recommendation system
* Multi-object clothing detection

---

## 🎯 Conclusion

This project demonstrates the practical implementation of:

* Computer Vision
* Image Processing
* Deep Learning
* CNN-based Image Classification

for building an intelligent clothing recognition system.

The developed system successfully predicts clothing categories from uploaded images through an interactive Streamlit application.

---

## 👨‍💻 Author

**Anirban Ray**

---
