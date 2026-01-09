"""Training pipeline definition."""

from kedro.pipeline import Pipeline, pipeline, node
from .nodes import train_model, compute_test_scores


def create_pipeline(**kwargs) -> Pipeline:
    """
    Create the training pipeline.

    NOTE: Changed to use pyg_data for both training and explanation.
    This ensures full compatibility with PyG explainers and enables
    subset optimization (100-1000× speedup).

    Returns:
        A Pipeline object for training
    """
    return pipeline([
        node(
            func=train_model,
            inputs={
                "pyg_data": "pyg_data",  # Changed from dgl_data to pyg_data
                "knowledge_graph": "knowledge_graph",
                "model_params": "params:model",
                "training_params": "params:training",
                "device_str": "params:device"
            },
            outputs="trained_model_artifact",
            name="train_model",
        ),
        node(
            func=compute_test_scores,
            inputs={
                "trained_model_artifact": "trained_model_artifact",
                "pyg_data": "pyg_data",  # Changed from dgl_data to pyg_data
                "knowledge_graph": "knowledge_graph",
                "device_str": "params:device"
            },
            outputs=["test_triple_scores", "test_triple_scores_csv", "top10_test"],
            name="compute_test_scores",
        ),
    ])
