# Breast Cancer Diagnosis with Bagging and Boosting  
This project applies Bagging and Boosting ensemble learning techniques to improve predictive accuracy for breast cancer diagnosis using the Breast Cancer Wisconsin (Diagnostic) dataset from Kaggle. The goal is to enhance model stability, reduce variance, and support more reliable clinical decision‑making.

---

## Project Overview

Ensemble learning methods such as Bagging and Boosting are powerful techniques for improving the performance of unstable base learners like Decision Trees.  

This project:
- Loads and preprocesses the Kaggle dataset  
- Trains a baseline Decision Tree
- Implements Bagging (Bootstrap Aggregation)  
- Implements AdaBoost and Gradient Boosting  
- Evaluates all models using standard classification metrics  
- Saves results to CSV  
- Generates a comparison chart of model performance  

---

## Dataset

**Source:**  
Kaggle – Breast Cancer Wisconsin (Diagnostic)  
[https://www.kaggle.com/datasets/uciml/breast-cancer-wisconsin-data](https://www.kaggle.com/datasets/uciml/breast-cancer-wisconsin-data)

**Details:**

- 569 samples  
- 30 numerical features describing cell nuclei  
- Diagnosis labels:  
  - **M** = Malignant → mapped to **1**  
  - **B** = Benign → mapped to **0**  
- No missing values  
- Some zero‑variance columns removed during preprocessing  

---

## Models Implemented

### 1. Baseline Model
- Decision Tree Classifier  
- Serves as the reference point for comparison  

### 2. Bagging (Bootstrap Aggregation)
- Decision Tree as base estimator  
- 100 estimators  
- Reduces variance and improves stability  

### 3. AdaBoost
- Decision stumps (max_depth=1)  
- Sequentially focuses on misclassified samples  
- Reduces bias  

### 4. Gradient Boosting
- 100 estimators  
- Learning rate = 0.1  
- Strong performance on structured datasets  

---

## Running the Project

### **1. Install dependencies**
```
pip install -r requirements.txt
```

### **2. Run the main script**
```
python main.py
```

### **3. Outputs generated automatically**
- `outputs/metrics.csv` – accuracy, precision, recall, and F1 for all models  
- `outputs/model_comparison.png` – bar chart comparing model performance  

---

## Results Summary

Your actual results may vary slightly, but typical performance:

| Model                  | Accuracy | Precision | Recall | F1 Score |
|-----------------------|----------|-----------|--------|----------|
| Decision Tree         | ~ 0.93   | ~ 0.93    | ~ 0.93  | ~ 0.93    |
| Bagging               | ~ 0.97   | ~ 0.97    | ~ 0.97  | ~ 0.97    |
| AdaBoost              | ~ 0.96   | ~ 0.96    | ~ 0.96  | ~ 0.96    |
| Gradient Boosting     | ~ 0.96   | ~ 0.96    | ~ 0.96  | ~ 0.96    |

Bagging performed best, offering the most stable and accurate predictions.

---

## Clinical Interpretation

- Bagging significantly reduces variance to make predictions more stable, which is a desirable property for medical diagnostics.  
- Boosting improves accuracy by focusing on difficult cases but may be more sensitive to noise.  
- Ensemble methods outperform a single Decision Tree, demonstrating their value in clinical decision‑support systems.  
- High recall for malignant cases is essential to minimize false negatives.  

---

## References

Demir, N. (2016, February 4). Ensemble methods: Elegant techniques to produce improved machine learning results. Toptal Engineering Blog. https://www.toptal.com/machine-learning/ensemble-methods-machine-learning

GeeksforGeeks. (2022, June 1). Bagging vs boosting in machine learning. https://www.geeksforgeeks.org/bagging-vs-boosting-in-machine-learning/

IBM Data and AI Team. (2023, October 16). Shedding light on AI bias with real world examples. IBM Blog. https://www.ibm.com/blog/shedding-light-on-ai-bias-with-real-world-examples/

Kumar, A., & Jain, M. (n.d.). Ensemble learning for AI developers: Learn bagging, stacking, and boosting methods with use cases. O’Reilly. https://www.oreilly.com/library/view/ensemble-learning-for/9781484259405/

Mwiti, D. (2023, August 21). A comprehensive guide to ensemble learning: What exactly do you need to know. Neptune.ai. https://neptune.ai/blog/ensemble-learning-guide

Rocca, J. (2021, December 9). Ensemble methods: Bagging, boosting, and stacking. Towards Data Science. https://towardsdatascience.com/ensemble-methods-bagging-boosting-and-stacking-c9214a10a205

UCI Machine Learning Repository. (2016). Breast Cancer Wisconsin (Diagnostic) dataset. Kaggle. https://www.kaggle.com/datasets/uciml/breast-cancer-wisconsin-data

UpGrad. (n.d.). Top 12 commerce project topics & ideas in 2023 [For freshers]. UpGrad Blog. https://www.upgrad.com/blog/bagging-vs-boosting/