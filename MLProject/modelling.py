import os
import mlflow
import mlflow.sklearn
import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score


def main():
    # Reset MLflow state (CI safe)
    if "MLFLOW_RUN_ID" in os.environ:
        del os.environ["MLFLOW_RUN_ID"]

    mlflow.end_run()

    mlflow.set_tracking_uri("file:./mlruns")
    mlflow.set_experiment("workflow-ci-experiment")

    df = pd.read_csv("dataset_preprocessed.csv")

    target_col = df.columns[-1]
    X = df.drop(columns=[target_col])
    y = df[target_col]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    with mlflow.start_run(run_name="ci-training"):
        model = RandomForestClassifier(
            n_estimators=100,
            random_state=42
        )
        model.fit(X_train, y_train)

        y_pred = model.predict(X_test)
        acc = accuracy_score(y_test, y_pred)

        mlflow.log_param("target_column", target_col)
        mlflow.log_metric("accuracy", acc)
        mlflow.sklearn.log_model(model, artifact_path="model")

        print(f"Training completed. Accuracy: {acc:.4f}")


if __name__ == "__main__":
    main()
