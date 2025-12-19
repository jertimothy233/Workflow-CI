import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
import mlflow
import mlflow.sklearn

def main()
    mlflow.set_experiment("CI-Retraining")

    df = pd.read_csv("bank_preprocessed.csv")
    if "y" not in df.columns:
        raise ValueError("Kolom target 'y' tidak ditemukan")
    X = df.drop(columns=["y"])
    y = df["y"]

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

    model =  LogisticRegression(max_iter=1000)
    mlflow.sklearn.autolog()

    with mlflow.start_run():
        model.fit(X_train, y_train)

if_name_== "__main__":
  main()
      
