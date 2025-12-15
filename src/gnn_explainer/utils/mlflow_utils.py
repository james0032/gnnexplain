"""MLflow utility functions for logging metrics, artifacts, and models.

This module provides helper functions to log various types of data to MLflow,
including training curves, evaluation results, and model comparisons.
"""

import logging
from pathlib import Path
from typing import Any, Dict, List, Optional, Union

import numpy as np
import pandas as pd

try:
    import mlflow
    MLFLOW_AVAILABLE = True
except ImportError:
    MLFLOW_AVAILABLE = False

logger = logging.getLogger(__name__)


def log_params_from_dict(params: Dict[str, Any], prefix: str = "") -> None:
    """Log parameters from a nested dictionary to MLflow.

    Args:
        params: Dictionary of parameters to log
        prefix: Prefix for parameter names (for nested dicts)
    """
    if not MLFLOW_AVAILABLE or not mlflow.active_run():
        return

    for key, value in params.items():
        param_name = f"{prefix}.{key}" if prefix else key

        if isinstance(value, dict):
            # Recursively log nested dictionaries
            log_params_from_dict(value, prefix=param_name)
        else:
            # Convert value to string for MLflow (MLflow params are strings)
            try:
                mlflow.log_param(param_name, value)
            except Exception as e:
                logger.warning(f"Failed to log parameter {param_name}: {e}")


def log_training_metrics(
    epoch: int,
    train_loss: float,
    val_loss: Optional[float] = None,
    metrics: Optional[Dict[str, float]] = None,
) -> None:
    """Log training metrics for a specific epoch.

    Args:
        epoch: Current epoch number
        train_loss: Training loss value
        val_loss: Validation loss value (optional)
        metrics: Additional metrics to log (e.g., accuracy, F1)
    """
    if not MLFLOW_AVAILABLE or not mlflow.active_run():
        return

    try:
        mlflow.log_metric("train_loss", train_loss, step=epoch)

        if val_loss is not None:
            mlflow.log_metric("val_loss", val_loss, step=epoch)

        if metrics:
            for metric_name, metric_value in metrics.items():
                mlflow.log_metric(metric_name, metric_value, step=epoch)
    except Exception as e:
        logger.warning(f"Failed to log training metrics for epoch {epoch}: {e}")


def log_evaluation_results(
    results: Dict[str, Any],
    prefix: str = "eval",
    step: Optional[int] = None,
) -> None:
    """Log evaluation results to MLflow.

    Args:
        results: Dictionary of evaluation metrics
        prefix: Prefix for metric names
        step: Optional step number for time-series logging
    """
    if not MLFLOW_AVAILABLE or not mlflow.active_run():
        return

    for metric_name, metric_value in results.items():
        if isinstance(metric_value, (int, float)):
            full_name = f"{prefix}.{metric_name}" if prefix else metric_name
            try:
                if step is not None:
                    mlflow.log_metric(full_name, float(metric_value), step=step)
                else:
                    mlflow.log_metric(full_name, float(metric_value))
            except Exception as e:
                logger.warning(f"Failed to log metric {full_name}: {e}")


def log_training_curve(
    history: List[Dict[str, float]],
    output_path: Optional[Union[str, Path]] = None,
) -> None:
    """Log training history as a CSV artifact and plot.

    Args:
        history: List of dictionaries containing metrics per epoch
        output_path: Optional path to save the CSV file
    """
    if not MLFLOW_AVAILABLE or not mlflow.active_run():
        return

    try:
        # Convert to DataFrame
        df = pd.DataFrame(history)

        # Save to CSV and log as artifact
        if output_path is None:
            output_path = "training_history.csv"

        df.to_csv(output_path, index=False)
        mlflow.log_artifact(output_path, artifact_path="training")

        logger.info(f"Logged training history to {output_path}")
    except Exception as e:
        logger.warning(f"Failed to log training curve: {e}")


def log_model_artifact(
    model_path: Union[str, Path],
    artifact_path: str = "models",
) -> None:
    """Log a model file as an MLflow artifact.

    Args:
        model_path: Path to the model file to log
        artifact_path: Subdirectory within the artifact store
    """
    if not MLFLOW_AVAILABLE or not mlflow.active_run():
        return

    try:
        mlflow.log_artifact(str(model_path), artifact_path=artifact_path)
        logger.info(f"Logged model artifact: {model_path}")
    except Exception as e:
        logger.warning(f"Failed to log model artifact {model_path}: {e}")


def log_predictions(
    predictions: Union[np.ndarray, List],
    labels: Union[np.ndarray, List],
    output_path: Optional[Union[str, Path]] = None,
    artifact_path: str = "predictions",
) -> None:
    """Log predictions and labels as a CSV artifact.

    Args:
        predictions: Model predictions
        labels: Ground truth labels
        output_path: Optional path to save the CSV file
        artifact_path: Subdirectory within the artifact store
    """
    if not MLFLOW_AVAILABLE or not mlflow.active_run():
        return

    try:
        # Create DataFrame with predictions and labels
        df = pd.DataFrame({
            "prediction": predictions,
            "label": labels,
        })

        # Save to CSV and log as artifact
        if output_path is None:
            output_path = "predictions.csv"

        df.to_csv(output_path, index=False)
        mlflow.log_artifact(output_path, artifact_path=artifact_path)

        logger.info(f"Logged predictions to {output_path}")
    except Exception as e:
        logger.warning(f"Failed to log predictions: {e}")


def log_confusion_matrix(
    y_true: Union[np.ndarray, List],
    y_pred: Union[np.ndarray, List],
    labels: Optional[List[str]] = None,
) -> None:
    """Log confusion matrix metrics to MLflow.

    Args:
        y_true: Ground truth labels
        y_pred: Predicted labels
        labels: Optional class labels
    """
    if not MLFLOW_AVAILABLE or not mlflow.active_run():
        return

    try:
        from sklearn.metrics import confusion_matrix, classification_report

        # Compute confusion matrix
        cm = confusion_matrix(y_true, y_pred, labels=labels)

        # Log confusion matrix as a metric
        mlflow.log_dict({"confusion_matrix": cm.tolist()}, "confusion_matrix.json")

        # Compute and log classification report
        report = classification_report(y_true, y_pred, labels=labels, output_dict=True)

        for class_name, metrics in report.items():
            if isinstance(metrics, dict):
                for metric_name, value in metrics.items():
                    mlflow.log_metric(f"{class_name}_{metric_name}", value)

        logger.info("Logged confusion matrix and classification report")
    except Exception as e:
        logger.warning(f"Failed to log confusion matrix: {e}")


def log_figure(
    fig,
    filename: str,
    artifact_path: str = "figures",
) -> None:
    """Log a matplotlib figure as an artifact.

    Args:
        fig: Matplotlib figure object
        filename: Filename for the saved figure
        artifact_path: Subdirectory within the artifact store
    """
    if not MLFLOW_AVAILABLE or not mlflow.active_run():
        return

    try:
        import matplotlib.pyplot as plt

        fig.savefig(filename, bbox_inches='tight', dpi=300)
        mlflow.log_artifact(filename, artifact_path=artifact_path)
        plt.close(fig)

        logger.info(f"Logged figure: {filename}")
    except Exception as e:
        logger.warning(f"Failed to log figure {filename}: {e}")


def log_dataset_info(
    dataset_name: str,
    num_samples: int,
    num_features: Optional[int] = None,
    num_classes: Optional[int] = None,
    additional_info: Optional[Dict[str, Any]] = None,
) -> None:
    """Log dataset information as MLflow parameters and tags.

    Args:
        dataset_name: Name of the dataset
        num_samples: Number of samples in the dataset
        num_features: Number of features (optional)
        num_classes: Number of classes (optional)
        additional_info: Additional dataset information
    """
    if not MLFLOW_AVAILABLE or not mlflow.active_run():
        return

    try:
        mlflow.set_tag("dataset_name", dataset_name)
        mlflow.log_param("num_samples", num_samples)

        if num_features is not None:
            mlflow.log_param("num_features", num_features)

        if num_classes is not None:
            mlflow.log_param("num_classes", num_classes)

        if additional_info:
            log_params_from_dict(additional_info, prefix="dataset")

        logger.info(f"Logged dataset info for {dataset_name}")
    except Exception as e:
        logger.warning(f"Failed to log dataset info: {e}")


def log_dict_as_artifact(
    data: Dict[str, Any],
    filename: str,
    artifact_path: str = "data",
) -> None:
    """Log a dictionary as a JSON artifact.

    Args:
        data: Dictionary to log
        filename: Filename for the saved JSON
        artifact_path: Subdirectory within the artifact store
    """
    if not MLFLOW_AVAILABLE or not mlflow.active_run():
        return

    try:
        mlflow.log_dict(data, f"{artifact_path}/{filename}")
        logger.info(f"Logged dictionary as artifact: {filename}")
    except Exception as e:
        logger.warning(f"Failed to log dictionary {filename}: {e}")


def is_mlflow_enabled() -> bool:
    """Check if MLflow is available and has an active run.

    Returns:
        True if MLflow is available and active, False otherwise
    """
    if not MLFLOW_AVAILABLE:
        return False

    try:
        return mlflow.active_run() is not None
    except Exception:
        return False
