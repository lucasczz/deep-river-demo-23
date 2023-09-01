from sklearn.metrics import roc_auc_score, precision_recall_fscore_support
from pathlib import Path
import pandas as pd
import numpy as np
import torch


def get_detection_report(labels, predictions):
    # Calculate ROC-AUC score

    # Calculate precision, recall, F1-score, and support
    precision, recall, fscore, support = precision_recall_fscore_support(
        labels, predictions, labels=[1]
    )
    result = ""
    # Print the evaluation metrics
    result += "{:<15} {:<10}".format("Metric", "Score")
    result += "\n-----------------------"
    result += "\n{:<15} {:<10.2%}".format("Precision", precision[0])
    result += "\n{:<15} {:<10.2%}".format("Recall", recall[0])
    result += "\n{:<15} {:<10.2%}".format("F1", fscore[0])
    result += "\n{:<15} {:<10}".format("Support", support[0])
    return result


def seed_everything(seed=42):
    np.random.seed(seed)
    torch.manual_seed(seed)


class StreamTracker:
    def __init__(self) -> None:
        self.results = None

    def update(self, *args):
        if self.results is None:
            self.results = [[arg] for arg in args]
        else:
            for result, arg in zip(self.results, args):
                result.append(arg)

    def get(self):
        return self.results


def get_taxi_data():
    data = pd.read_csv(Path(__file__).parent.joinpath("_nyc_taxi.csv"))
    return [
        ({"passengers": row["value"]}, row["is_anomaly"])
        for _, row in data.iterrows()
    ]


def get_turtlebot_data():
    features = [f"angular_vel_{i}" for i in ["x", "y", "z"]]
    data = pd.read_csv(Path(__file__).parent.joinpath("turtlebot_imu.csv"))
    return [
        (dict(zip(features, row[features])), row["anomaly_type"])
        for _, row in data.iterrows()
    ]
