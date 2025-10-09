import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, confusion_matrix, roc_auc_score, accuracy_score
import seaborn as sns
import matplotlib.pyplot as plt
import os
from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier

def run_predictive_maintenance_model_local():
    """
    Main function to run the predictive maintenance modeling pipeline
    using a local CSV file.
    """
    #load file from local
    local_file_path = 'ai4i2020.csv'

    try:
        print(f"Attempting to load the local dataset: '{local_file_path}'...")
        # check if it exists
        if not os.path.exists(local_file_path):
            raise FileNotFoundError(
                f"Error: The file '{local_file_path}' was not found in this directory.\n"
                "Please make sure your Python script and your CSV file are in the same folder."
            )
        
        #read csv
        df = pd.read_csv(local_file_path)
        print("Dataset loaded successfully from local file. ✅")
    except Exception as e:
        print(e)
        return

    # data exploration
    print("\n--- Data Exploration ---")
    print("Dataset shape:", df.shape)
    print("First 5 rows:\n", df.head())
    print("\nTarget variable distribution ('Machine failure'):")
    print(df['Machine failure'].value_counts(normalize=True) * 100)

    # Drop columns that are identifiers and not useful for prediction
    df = df.drop(['UDI', 'Product ID'], axis=1)

    X = df.drop(['Machine failure', 'TWF', 'HDF', 'PWF', 'OSF', 'RNF'], axis=1)
    y = df['Machine failure']

    categorical_features = ['Type']
    numerical_features = X.select_dtypes(include=np.number).columns.tolist()

    # preprocessing pipeline
    preprocessor = ColumnTransformer(
        transformers=[
            ('num', StandardScaler(), numerical_features),
            ('cat', OneHotEncoder(handle_unknown='ignore'), categorical_features)
        ])

    # training and testing dataset
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    print(f"\nTraining set contains {X_train.shape[0]} samples.")
    print(f"Test set contains {X_test.shape[0]} samples.")

    # train model
    model_pipeline = Pipeline(steps=[
        ('preprocessor', preprocessor),
        ('classifier', LogisticRegression(random_state=42, class_weight='balanced', max_iter=1000))
    ])

    print("\n--- Model Training ---")
    model_pipeline.fit(X_train, y_train)
    print("Model training is complete. 🦾")

    # eval performance
    print("\n--- Model Evaluation ---")
    y_pred = model_pipeline.predict(X_test)
    y_pred_proba = model_pipeline.predict_proba(X_test)[:, 1]

    print("\nClassification Report:\n")
    print(classification_report(y_test, y_pred, target_names=['No Failure', 'Failure']))
    print(f"ROC AUC Score: {roc_auc_score(y_test, y_pred_proba):.4f}")

    print("\nConfusion Matrix:")
    cm = confusion_matrix(y_test, y_pred)

    # Compare multiple models (Logistic Regression, Decision Tree, Random Forest)
    models = {
        'Logistic Regression': LogisticRegression(random_state=42, class_weight='balanced', max_iter=1000),
        'Decision Tree': DecisionTreeClassifier(random_state=42, class_weight='balanced'),
        'Random Forest': RandomForestClassifier(random_state=42, class_weight='balanced', n_estimators=100)
    }

    model_names = []
    accuracies = []

    for name, clf in models.items():
        pipe = Pipeline(steps=[
            ('preprocessor', preprocessor),
            ('classifier', clf)
        ])
        pipe.fit(X_train, y_train)
        acc = accuracy_score(y_test, pipe.predict(X_test))
        model_names.append(name)
        accuracies.append(acc)
        print(f"{name} Accuracy: {acc:.4f}")

    fig, axes = plt.subplots(1, 2, figsize=(14, 6))

    # confusion matrix
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=axes[0],
                xticklabels=['Predicted No Failure', 'Predicted Failure'],
                yticklabels=['Actual No Failure', 'Actual Failure'])
    axes[0].set_xlabel('Predicted Label')
    axes[0].set_ylabel('True Label')
    axes[0].set_title('Model Confusion Matrix (Local Data)')

    # model accuracy comparison
    axes[1].bar(model_names, accuracies, color=['#4C72B0', '#55A868', '#C44E52'])
    axes[1].set_title("Model Accuracy Comparison")
    axes[1].set_ylabel("Accuracy Score")
    axes[1].set_ylim(0, 1)
    for i, acc in enumerate(accuracies):
        axes[1].text(i, acc + 0.01, f"{acc:.2f}", ha='center', fontsize=10)

    plt.tight_layout()
    plt.show(block=True)


if __name__ == '__main__':
    run_predictive_maintenance_model_local()
