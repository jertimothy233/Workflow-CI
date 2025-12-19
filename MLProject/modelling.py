import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
import mlflow
import mlflow.sklearn

def main():
    mlflow.set_tracking_uri("file:./mlruns")
    mlflow.set_experiment("CI-Retraining")

    df = pd.read_csv("bank_preprocessed.csv")
    if "y" not in df.columns:
        raise ValueError("Kolom target 'y' tidak ditemukan")
    X = df.drop(columns=["y"])
    y = df["y"]

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

    model =  LogisticRegression(max_iter=2000)
    
    mlflow.sklearn.autolog()
    model.fit(X_train, y_train)

    y_pred = model.predict(X_test)
    y_proba = model.predict_proba(X_test)[:, 1]
    print("Accuracy:", accuracy_score(y_test, y_pred))
    print("F1:", f1_score(y_test, y_pred))
    print("ROC AUC:", roc_auc_score(y_test, y_proba))
    
if __name__ == "__main__":
    main()

      
