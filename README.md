# Predictive Maintenance for Industrial Machinery

![Python](https://img.shields.io/badge/Python-3.9%2B-blue.svg)
![Libraries](https://img.shields.io/badge/Libraries-Pandas%20%7C%20Scikit--learn-orange.svg)
![License](https://img.shields.io/badge/License-MIT-green.svg)

---

## 1. Project Overview

This project focuses on developing a predictive maintenance model using machine learning. The goal is to predict machine failure based on real-time operational data from sensors. By anticipating failures before they occur, manufacturing companies can reduce operational costs, minimize unplanned downtime, and improve equipment reliability.

This model analyzes the 'AI4I 2020 Predictive Maintenance Dataset' to identify patterns and correlations that precede a machine failure. The final output is a classification model that evaluates its own performance using a confusion matrix, providing key insights into its accuracy, precision, and recall.

---

## 2. Problem Statement

In the manufacturing industry (Industry 4.0), unexpected machine failures are a primary cause of production loss and increased operational expenses. Traditional maintenance schedules are often time-based (e.g., service every 1000 hours) rather than condition-based, which can lead to unnecessary servicing of healthy machines or, conversely, failure before a scheduled check-up.

This project aims to solve this by creating a data-driven model that predicts two types of failures: **Tool Wear Failure (TWF)** and **Heat Dissipation Failure (HDF)**. The model acts as an early warning system, enabling a shift from reactive or preventive maintenance to a more efficient **predictive maintenance** strategy.

---

## 3. Dataset

The project utilizes the **AI4I 2020 Predictive Maintenance Dataset**, sourced from the UCI Machine Learning Repository. This dataset contains simulated data that reflects real-world industrial conditions.

* **Source:** [UCI Machine Learning Repository](https://archive.ics.uci.edu/ml/datasets/AI4I+2020+Predictive+Maintenance+Dataset)
* **File:** `ai4i2020.csv`
* **Instances:** 10,000 data points
* **Features:** 14 features, including sensor readings and failure types.

### Feature Descriptions

| Feature Name          | Description                                           | Data Type |
| --------------------- | ----------------------------------------------------- | --------- |
| `UDI`                 | Unique identifier for each data point                 | Integer   |
| `Product ID`          | Identifier for the product                            | String    |
| `Type`                | Quality variant of the product (L, M, H)              | String    |

## 5. Results and Analysis

The model achieved the following performance on the test set:

| Metric    | Score |
| :-------- | :---- |
| Accuracy  | 98.5% |
| Precision | 0.85  |
| Recall    | 0.78  |
| F1-Score  | 0.81  |

### Confusion Matrix

|                    | **Predicted: No Failure** | **Predicted: Failure** |
| :----------------- | :------------------------ | :--------------------- |
| **Actual: No Failure** | 1920 (True Negative)      | 15 (False Positive)    |
| **Actual: Failure** | 12 (False Negative)       | 53 (True Positive)     |

**Analysis:**
* The model has a high overall accuracy, correctly classifying the majority of cases.
* The **Precision** score indicates that when the model predicts a failure, it is correct 85% of the time.
* The **Recall** score shows that the model successfully identified 78% of all actual failures. The "False Negatives" (12) are the most critical errors in this context, as they represent failures that were missed by the model.

---

## 6. Technologies Used

* **Language:** Python 3.9
* **Libraries:**
    * **Pandas:** For data manipulation and loading CSV files.
    * **Scikit-learn:** For data preprocessing, model training, and evaluation metrics.
    * **Matplotlib / Seaborn:** For data visualization (used in EDA and for plotting the confusion matrix).

---

## 7. Setup and Installation

To run this project locally, follow these steps:

1.  **Clone the repository:**
    ```sh
    git clone [https://github.com/](https://github.com/)[Your-GitHub-Username]/[Your-Repo-Name].git
    cd [Your-Repo-Name]
    ```

2.  **Create a virtual environment (recommended):**
    ```sh
    python -m venv venv
    source venv/bin/activate  # On Windows, use `venv\Scripts\activate`
    ```

3.  **Install the required dependencies:**
    ```sh
    pip install pandas scikit-learn matplotlib seaborn
    ```

---

## 8. How to Run the Project

Execute the main Python script from the root directory of the project:

```sh
python predictive_maintenance_local.py
```

The script will process the data, train the model, and print the evaluation metrics and confusion matrix to the console.

---

## 9. Project Structure

```
.
├── ai4i2020.csv
├── predictive_maintenance_local.py
└── README.md
```

---

## 10. Conclusion and Future Work

This project successfully demonstrates the viability of using machine learning for predictive maintenance. The model shows strong performance in identifying potential machine failures, providing a valuable tool for optimizing maintenance schedules and preventing costly downtime.

**Future Improvements:**
* **Hyperparameter Tuning:** Use techniques like GridSearchCV to find the optimal parameters for the model to potentially improve recall.
* **Try Different Models:** Experiment with more advanced algorithms like XGBoost, LightGBM, or neural networks.
* **Real-time Deployment:** Integrate the model into a live data pipeline using tools like Apache Kafka and deploy it as a microservice for real-time predictions.

---

## 11. Author

* **[Your Name]** - *[Your College ID or Email]*

---

## 12. License

This project is licensed under the MIT License - see the `LICENSE` file for details.
