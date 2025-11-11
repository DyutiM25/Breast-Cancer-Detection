# 🩺 Breast Cancer Detection using Random Forest

This project demonstrates the use of **Machine Learning**—specifically the **Random Forest Classifier**—to diagnose breast cancer with high accuracy.  
By analyzing medical diagnostic data, the model classifies tumors as **malignant** or **benign**, achieving an impressive **96.50% accuracy**.

---

## 📘 Table of Contents
- [Overview](#overview)
- [Dataset](#dataset)
- [Technology Stack](#technology-stack)
- [Methodology](#methodology)
- [Results](#results)
- [Installation](#installation)
- [Usage](#usage)
- [Future Work](#future-work)
- [References](#references)
- [Author](#author)

---

## 🔬 Overview
Cancer is a **heterogeneous disease** with many subtypes, making early diagnosis crucial for effective treatment and patient management.  
Machine Learning (ML) models can extract meaningful patterns from complex biomedical data, enabling accurate and automated diagnosis.

In this project, a **Random Forest Classifier**—an ensemble learning method—is applied to breast cancer diagnosis. Random Forests combine multiple decision trees to improve predictive performance and reduce overfitting, making them ideal for this kind of medical data analysis.

---

## 📊 Dataset
The dataset used is the **Breast Cancer Wisconsin (Diagnostic) Dataset**, available from the [UCI Machine Learning Repository](https://archive.ics.uci.edu/ml/datasets/Breast+Cancer+Wisconsin+(Diagnostic)).

### **Features**
- The dataset contains **30 numerical features** computed from digitized images of breast mass cell nuclei.
- Each sample is labeled as:
  - **M** → Malignant  
  - **B** → Benign  

---

## ⚙️ Technology Stack
- **Python 3.x**
- **Scikit-learn**
- **NumPy**
- **Pandas**
- **Matplotlib / Seaborn** (for visualization)
- **Jupyter Notebook** (for experimentation)

---

## 🧠 Methodology
1. **Data Preprocessing**
   - Loaded the dataset using Scikit-learn.
   - Handled missing values (if any).
   - Encoded labels (Malignant / Benign).
   - Normalized the feature set.

2. **Model Development**
   - Implemented the **Random Forest Classifier**.
   - Tuned hyperparameters such as:
     - Number of trees (`n_estimators`)
     - Maximum depth
     - Minimum samples split
   - Split data into **80% training** and **20% testing**.

3. **Model Evaluation**
   - Evaluated performance using:
     - Accuracy Score
     - Confusion Matrix
     - Precision, Recall, F1-Score
   - Visualized feature importance to understand model interpretability.

---

## 📈 Results
- **Model Used:** Random Forest Classifier  
- **Accuracy Achieved:** **96.50%**  
- The model efficiently distinguishes between malignant and benign tumors.  
- High accuracy and stability make it suitable for medical decision support systems.

---

## 💻 Installation

### 1️⃣ Clone the Repository
```bash
git clone https://github.com/DyutiM25/Breast-Cancer-Detection.git
cd Breast-Cancer-Detection
