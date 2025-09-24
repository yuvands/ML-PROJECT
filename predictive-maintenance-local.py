import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, confusion_matrix, roc_auc_score
import seaborn as sns
import matplotlib.pyplot as plt
import os

def run_predictive_maintenance_model_local():
    """
    Main function to run the predictive maintenance modeling pipeline
    using a local CSV file.
    """
    # --- 1. Load Data From a Local File ---
    # Define the path to your local CSV file.
    # Because the script and the CSV are in the same folder, we just need the filename.
    local_file_path = 'ai4i2020.csv'

    try:
        print(f"Attempting to load the local dataset: '{local_file_path}'...")
        # Check if the file exists before trying to load it
        if not os.path.exists(local_file_path):
            raise FileNotFoundError(
                f"Error: The file '{local_file_path}' was not found in this directory.\n"
                "Please make sure your Python script and your CSV file are in the same folder."
            )
        
        # We use pandas to read the local CSV file.
        df = pd.read_csv(local_file_path)
        print("Dataset loaded successfully from local file. ✅")
    except Exception as e:
        print(e)
        return

    # --- 2. Data Exploration & Preprocessing ---
    print("\n--- Data Exploration ---")
    print("Dataset shape:", df.shape)
    print("First 5 rows:\n", df.head())
    print("\nTarget variable distribution ('Machine failure'):")
    print(df['Machine failure'].value_counts(normalize=True) * 100)

    # Drop columns that are identifiers and not useful for prediction
    df = df.drop(['UDI', 'Product ID'], axis=1)

    # Define features (X) and the target variable (y)
    X = df.drop(['Machine failure', 'TWF', 'HDF', 'PWF', 'OSF', 'RNF'], axis=1)
    y = df['Machine failure']

    # Separate columns into numerical and categorical types
    categorical_features = ['Type']
    numerical_features = X.select_dtypes(include=np.number).columns.tolist()

    # --- 3. Create Preprocessing Pipeline ---
    preprocessor = ColumnTransformer(
        transformers=[
            ('num', StandardScaler(), numerical_features),
            ('cat', OneHotEncoder(handle_unknown='ignore'), categorical_features)
        ])

    # --- 4. Split Data into Training and Testing Sets ---
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    print(f"\nTraining set contains {X_train.shape[0]} samples.")
    print(f"Test set contains {X_test.shape[0]} samples.")

    # --- 5. Define and Train the Machine Learning Model ---
    model_pipeline = Pipeline(steps=[
        ('preprocessor', preprocessor),
        ('classifier', LogisticRegression(random_state=42, class_weight='balanced', max_iter=1000))
    ])

    print("\n--- Model Training ---")
    model_pipeline.fit(X_train, y_train)
    print("Model training is complete. 🦾")

    # --- 6. Evaluate the Model's Performance ---
    print("\n--- Model Evaluation ---")
    y_pred = model_pipeline.predict(X_test)
    y_pred_proba = model_pipeline.predict_proba(X_test)[:, 1]

    print("\nClassification Report:\n")
    print(classification_report(y_test, y_pred, target_names=['No Failure', 'Failure']))
    print(f"ROC AUC Score: {roc_auc_score(y_test, y_pred_proba):.4f}")

    print("\nConfusion Matrix:")
    cm = confusion_matrix(y_test, y_pred)
    
    # Visualize the Confusion Matrix
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                xticklabels=['Predicted No Failure', 'Predicted Failure'],
                yticklabels=['Actual No Failure', 'Actual Failure'])
    plt.xlabel('Predicted Label')
    plt.ylabel('True Label')
    plt.title('Model Confusion Matrix (Local Data)')
    plt.show()


if __name__ == '__main__':
    run_predictive_maintenance_model_local()
