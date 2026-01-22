"""Project pipeline registry."""

from typing import Dict
from kedro.pipeline import Pipeline

from gnn_explainer.pipelines import (
    data_preparation,
    training,
    explanation,
    # evaluation,  # TODO: Implement
    # metrics,  # TODO: Implement
)


def register_pipelines() -> Dict[str, Pipeline]:
    """
    Register the project's pipelines.

    Returns:
        A mapping from pipeline names to ``Pipeline`` objects.
    """
    from kedro.framework.project import settings
    from kedro.config import OmegaConfigLoader

    # Load parameters to get enabled_explainers
    conf_loader = OmegaConfigLoader(
        conf_source=str(settings.CONF_SOURCE),
        base_env="base",
        default_run_env="local"
    )
    params = conf_loader["parameters"]
    enabled_explainers = params.get("explanation", {}).get("enabled_explainers", ["all"])

    # Individual pipelines
    data_prep_pipeline = data_preparation.create_pipeline()
    training_pipeline = training.create_pipeline()
    explain_pipeline = explanation.create_pipeline(enabled_explainers=enabled_explainers)
    pagelink_analysis_pipeline = explanation.create_pagelink_analysis_pipeline()
    # eval_pipeline = evaluation.create_pipeline()  # TODO
    # metrics_pipeline = metrics.create_pipeline()  # TODO

    # Combined pipelines
    data_and_train = data_prep_pipeline + training_pipeline
    # train_and_eval = training_pipeline + eval_pipeline  # TODO
    # explain_and_viz = explain_pipeline + metrics_pipeline  # TODO

    # Full pipeline (when all modules are implemented)
    # full_pipeline = (
    #     data_prep_pipeline +
    #     training_pipeline +
    #     eval_pipeline +
    #     explain_pipeline +
    #     metrics_pipeline
    # )

    return {
        # Atomic pipelines
        "data_prep": data_prep_pipeline,
        "training": training_pipeline,
        "explanation": explain_pipeline,
        "pagelink_analysis": pagelink_analysis_pipeline,  # Standalone analysis of existing results
        # "evaluation": eval_pipeline,  # TODO
        # "metrics": metrics_pipeline,  # TODO

        # Combined workflows
        "data_and_train": data_and_train,
        # "train_eval": train_and_eval,  # TODO
        # "explain_viz": explain_and_viz,  # TODO

        # Full pipeline
        # "gnn_explainer_full": full_pipeline,  # TODO

        # Default pipeline
        "__default__": data_and_train,
    }
