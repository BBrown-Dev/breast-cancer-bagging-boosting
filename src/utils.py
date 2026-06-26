import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score,
    f1_score, confusion_matrix, classification_report
)


def load_dataset(path: str = "data/data.csv"):
    """ Loads dataset from csv file """
    df = pd.read_csv(path)
    df["diagnosis"] = df["diagnosis"].map({"M": 1, "B": 0})
    X = df.drop(columns=["diagnosis", "id"])
    y = df["diagnosis"]
    return X, y, df


def train_test_split_scaled(X, y, test_size=0.2, random_state=42):
    """ Splits dataset into train and test sets """
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state, stratify=y
    )
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    return X_train_scaled, X_test_scaled, y_train, y_test, scaler


def evaluate_model(name: str, y_true, y_pred):
    """ Evaluates model performance """
    metrics = {
        "Model": name,
        "Accuracy": accuracy_score(y_true, y_pred),
        "Precision": precision_score(y_true, y_pred),
        "Recall": recall_score(y_true, y_pred),
        "F1": f1_score(y_true, y_pred),
    }

    print(f"\n{name} Performance")
    print("-" * 50)
    print(classification_report(y_true, y_pred))
    print("Confusion Matrix:")
    print(confusion_matrix(y_true, y_pred))

    return metrics


def save_metrics_to_csv(metrics_list, path="outputs/metrics.csv"):
    """ Saves metrics to csv file """
    df = pd.DataFrame(metrics_list)
    df.to_csv(path, index=False)
    print(f"\nSaved metrics to {path}")


def plot_comparison_chart(metrics_list, path="outputs/model_comparison.png"):
    """ Plots comparison chart """
    df = pd.DataFrame(metrics_list).set_index("Model")

    plt.figure(figsize=(10, 6))
    df.plot(kind="bar")
    plt.title("Model Performance Comparison")
    plt.ylabel("Score")
    plt.ylim(0.80, 1.00)
    plt.xticks(rotation=0)
    plt.grid(axis="y", linestyle="--", alpha=0.6)
    plt.tight_layout()
    plt.savefig(path)
    plt.close()

    print(f"Saved comparison chart to {path}")