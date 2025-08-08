# 🪀 Heart Disease Classification

This project compares multiple machine learning classification algorithms on the **Heart Disease Prediction** dataset using Python.

---

## 📁 Dataset

**Source:** [Kaggle - Heart Failure Prediction](https://www.kaggle.com/datasets/fedesoriano/heart-failure-prediction)

The dataset contains features like age, sex, chest pain type, cholesterol level, and more to predict if a person has heart disease (binary classification).

---

## 🌐 Objective

To compare the performance, speed, and interpretability of the following models:

* Logistic Regression
* K-Nearest Neighbors (KNN)
* Random Forest Classifier

---

## ⚙️ Workflow

1. Data Loading & Exploration
2. Preprocessing (encoding + scaling)
3. Model Training & Evaluation
4. Performance Comparison
5. Final Conclusion

---

## 🔢 Models Used

| Model               | Accuracy | Time Taken | Comments                           |
| ------------------- | -------- | ---------- | ---------------------------------- |
| Logistic Regression | \~88.6%  | Fast       | Most interpretable                 |
| KNN                 | \~88.6%  | Fastest    | Simple, sensitive to scaling       |
| Random Forest       | \~88.6%  | Slowest    | Slightly better recall for class 1 |

---

## 📊 Evaluation Metrics

* Accuracy
* Precision, Recall, F1-Score
* Confusion Matrix
* Time Taken

---

## 📅 Conclusion

All models performed equally in accuracy:

* **KNN**: Fastest training time
* **Logistic Regression**: Most interpretable model
* **Random Forest**: Best recall for detecting positive cases

---

## 🛠️ Requirements

Install dependencies:

```bash
pip install pandas numpy matplotlib seaborn scikit-learn
```

---

## 📚 Files

* `HeartDisease-ML.ipynb`: Main Jupyter notebook with code & analysis
* `heart.csv`: Dataset file (or download from Kaggle link above)

---

## 🌐 Author

Asma Siddiqui
