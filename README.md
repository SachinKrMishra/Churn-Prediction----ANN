# 🧠 Bank Customer Churn Prediction using ANN

An end-to-end deep learning project that predicts whether a bank customer is likely to **churn** (leave the bank) using an **Artificial Neural Network (ANN)** built with TensorFlow and Keras.

## 🚀 Project Overview

Customer churn is a critical problem in banking and telecom sectors — identifying customers who are likely to leave enables targeted retention strategies.

This project builds an **Artificial Neural Network** to predict churn from customer data, demonstrating a complete **machine learning workflow** from preprocessing to model deployment.


## 🧩 Tech Stack

- **Language:** Python  
- **Libraries:** TensorFlow / Keras, scikit-learn, pandas, NumPy, matplotlib  
- **Tools:** TensorBoard, pickle, Jupyter Notebook  
- **Model Type:** Binary Classification (Churn / No Churn)


## 📊 Dataset

- **Source:** [Churn_Modelling.csv](https://www.kaggle.com/datasets/shrutimechlearn/churn-modelling)
- **Records:** 10,000 customers  
- **Features:**  
  - Numerical: CreditScore, Age, Tenure, Balance, EstimatedSalary, etc.  
  - Categorical: Geography, Gender  
  - Target: `Exited` (1 = Churn, 0 = Retained)


## ⚙️ Workflow

1. **Data Preprocessing**
   - Dropped irrelevant columns (`RowNumber`, `CustomerId`, `Surname`)
   - Label encoded `Gender` and one-hot encoded `Geography`
   - Feature scaling using `StandardScaler`
   - Saved preprocessing artifacts (`label_encoder_gender.pkl`, `onehot_encoder_geo.pkl`, `scaler.pkl`)

2. **Model Building (ANN)**
   - Input → Dense(64, ReLU) → Dense(32, ReLU) → Output(1, Sigmoid)
   - Optimizer: `Adam (lr = 0.01)`
   - Loss: `Binary Crossentropy`
   - Metrics: Accuracy

3. **Model Training**
   - Trained on 75% data with EarlyStopping (patience=10)
   - Validation on remaining 25%
   - TensorBoard used for visualization

4. **Model Evaluation**
   - **Training Accuracy:** ~87.8%  
   - **Validation Accuracy:** ~86.4%  
   - **Validation Loss:** ~0.34  

5. **Model Saving**
   - Trained model exported as `model.h5`
   - Ready for inference or deployment

