# Predictive Maintenance for Industrial Machinery

![Python](https://img.shields.io/badge/Python-3.9%2B-blue.svg)
![Libraries](https://img.shields.io/badge/Libraries-Pandas%20%7C%20Scikit--learn-orange.svg)
![Matplotlib](https://img.shields.io/badge/Matplotlib-3.7%2B-red.svg)
![Seaborn](https://img.shields.io/badge/Seaborn-0.12%2B-green.svg)

---


---

## 1. Project Overview

This project builds a *predictive maintenance model* to forecast machine failures using sensor data. Early detection reduces downtime, maintenance costs, and improves equipment reliability.  

It uses the *AI4I 2020 Predictive Maintenance Dataset* to train a classification model and evaluates it with metrics like *accuracy, precision, recall, F1-score, along with **confusion matrix and model comparison charts*.

---

## 2. Problem Statement

Unexpected machine failures cause production losses and unnecessary costs. Traditional time-based maintenance is inefficient, either over-servicing machines or missing failures.  

This project predicts failures (e.g., *Tool Wear, Heat Dissipation) to enable **predictive, condition-based maintenance*, reducing unplanned downtime.

---

## 3. Dataset

*AI4I 2020 Predictive Maintenance Dataset* – 10,000 instances, 14 features (sensor readings and failure types).  

* *Source:* [OpenML](https://www.openml.org/search?type=data&status=active&id=42890)  
* *File:* ai4i2020.csv  

### Feature Descriptions

| Feature Name          | Description                                           | Data Type |
| --------------------- | ----------------------------------------------------- | --------- |
| UDI                 | Unique identifier for each data point                 | Integer   |
| Product ID          | Identifier for the product                            | String    |
| Type                | Quality variant of the product (L, M, H)              | String    |
| Air temperature [K] | Air temperature in Kelvin                             | Float     |
| Process temperature [K] | Temperature of the manufacturing process in Kelvin  | Float     |
| Rotational speed [rpm]| Rotational speed of the tool                          | Integer   |
| Torque [Nm]         | Torque applied by the tool                            | Float     |
| Tool wear [min]     | Wear time of the tool in minutes                      | Integer   |
| Machine failure     | *Target Label*: 1 if failure occurred, 0 otherwise | Binary    |
| TWF                 | *Failure Type*: Tool Wear Failure                   | Binary    |
| HDF                 | *Failure Type*: Heat Dissipation Failure           | Binary    |
| PWF                 | *Failure Type*: Power Failure                       | Binary    |
| OSF                 | *Failure Type*: Overstrain Failure                  | Binary    |
| RNF                 | *Failure Type*: Random Failure                      | Binary    |

---

## 4. Methodology

The project follows a standard machine learning pipeline using the script predictive_maintenance_local.py.

1. *Data Loading & Preprocessing:*  
   * Load ai4i2020.csv into a Pandas DataFrame.  
   * Drop identifiers (UDI, Product ID).  
   * Convert categorical features (Type) using one-hot encoding.

2. *Exploratory Data Analysis (EDA):*  
   * Analyze the distribution of the target variable (Machine failure).  
   * Visualize correlations between sensor readings and machine failure.

3. *Model Training:*  
   * Split dataset: 80% training, 20% testing.  
   * Train classification models using pipelines.

---

## 5. Models Used

* *Logistic Regression* – baseline linear model for binary classification  
* *Decision Tree Classifier* – interpretable tree-based model  
* *Random Forest Classifier* – ensemble of decision trees for better accuracy and robustness  

---

## 6. Model Evaluation

* Generate *confusion matrix* to visualize predictions vs actual outcomes.  
* Compute metrics: *Accuracy, Precision, Recall, F1-score*.  
* Display *side-by-side plots* for confusion matrix and model accuracy comparison.

---

## 7. Results and Analysis

| Metric    | Score |
| :-------- | :---- |
| Accuracy  | 98.5% |
| Precision | 0.85  |
| Recall    | 0.78  |
| F1-Score  | 0.81  |

### Confusion Matrix

|                    | *Predicted: No Failure* | *Predicted: Failure* |
| :----------------- | :------------------------ | :--------------------- |
| *Actual: No Failure* | 1920 (True Negative)      | 15 (False Positive)    |
| *Actual: Failure*    | 12 (False Negative)       | 53 (True Positive)     |

*Analysis:*  
* High overall accuracy; majority of cases classified correctly.  
* Precision 0.85 → model is correct 85% of the time when predicting failure.  
* Recall 0.78 → model identifies 78% of actual failures.  
* False negatives (12) are critical, as missed failures could lead to downtime.  
* Random Forest achieved the highest accuracy among tested models.

---

## 8. Technologies Used

* *Language:* Python 3.9+  
* *Libraries:* Pandas, Scikit-learn, Matplotlib, Seaborn  

---

## 9. Libraries & Preprocessing Techniques Used

*Libraries:*  
* pandas – data manipulation and CSV loading  
* numpy – numerical operations  
* scikit-learn – preprocessing, model training, evaluation  
* matplotlib / seaborn – visualizations  

*Preprocessing Techniques:*  
* *StandardScaler* – normalize numerical features  
* *OneHotEncoder* – convert categorical features into numeric  
* *ColumnTransformer* – combine different preprocessing steps for columns  
* *Pipeline* – chain preprocessing and model training for cleaner workflow  

---

## 10. Setup and Installation

```bash
git clone https://github.com/[Your-GitHub-Username]/[Your-Repo-Name].git
cd [Your-Repo-Name]

# Optional: create virtual environment
python -m venv venv
source venv/bin/activate  # Windows: venv\Scripts\activate

# Install dependencies
pip install pandas scikit-learn matplotlib seaborn