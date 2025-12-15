"""Training pipeline definition."""

from kedro.pipeline import Pipeline, pipeline, node
from .nodes import train_model


def create_pipeline(**kwargs) -> Pipeline:
    """
    Create the training pipeline.

    Returns:
        A Pipeline object for training
    """
    return pipeline([
        node(
            func=train_model,
            inputs=[
                "pyg_data",
                "knowledge_graph",
                "params:model",
                "params:training",
                "params:device"
            ],
            outputs="trained_model_artifact",
            name="train_model",
        ),
    ])
