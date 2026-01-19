"""Pipeline for explaining GNN predictions."""

from kedro.pipeline import Pipeline, node
from .nodes import (
    prepare_model_for_explanation,
    select_triples_to_explain,
    run_gnnexplainer,
    run_pgexplainer,
    run_page_explainer,
    summarize_explanations
)


def get_enabled_explainers():
    """
    Get enabled explainers from Kedro configuration.

    This reads from conf/base/parameters.yml or conf/local/parameters.yml
    under explanation.enabled_explainers.

    Returns:
        List of enabled explainer names
    """
    try:
        from kedro.config import OmegaConfigLoader
        from kedro.framework.project import settings
        from pathlib import Path
        import os

        # Find project root (where conf/ directory is)
        # pipeline.py is at: src/gnn_explainer/pipelines/explanation/pipeline.py
        project_root = Path(__file__).resolve().parents[4]
        conf_path = project_root / "conf"

        if conf_path.exists():
            config_loader = OmegaConfigLoader(conf_source=str(conf_path))
            params = config_loader["parameters"]
            enabled = params.get("explanation", {}).get("enabled_explainers", ["all"])
            return enabled
    except Exception as e:
        print(f"Warning: Could not load config, using default explainers: {e}")

    return ["all"]


def create_pipeline(**kwargs) -> Pipeline:
    """
    Create explanation pipeline with configurable explainer selection.

    This pipeline:
    1. Loads the trained model and prepares it for explanation
    2. Selects triples (edges) to explain
    3. Runs selected explainers based on params:explanation.enabled_explainers
    4. Summarizes and compares explanations (if multiple explainers enabled)

    The enabled_explainers parameter controls which explainers run:
      - Set in conf/base/parameters.yml under explanation.enabled_explainers
      - Options: ["gnnexplainer"], ["pgexplainer"], ["page"], or any combination
      - Use ["all"] to run all explainers

    You can still use Kedro's tag filtering to override:
      - kedro run --pipeline=explanation --tags=page --exclude-tags=summary

    Returns:
        Kedro Pipeline
    """
    # Common nodes that always run
    common_nodes = [
        # Step 1: Prepare model for explanation
        node(
            func=prepare_model_for_explanation,
            inputs={
                "trained_model_dict": "trained_model",
                "pyg_data": "pyg_data",  # Changed from dgl_data to pyg_data
                "device_str": "params:device"
            },
            outputs="prepared_model",
            name="prepare_model_for_explanation"
        ),

        # Step 2: Select triples to explain
        node(
            func=select_triples_to_explain,
            inputs={
                "pyg_data": "pyg_data",  # Changed from dgl_data to pyg_data
                "knowledge_graph": "knowledge_graph",
                "selection_params": "params:explanation.triple_selection",
                "device_str": "params:device",
                "triple_file_content": "top10_test"  # Optional: file content for "from_file" strategy
            },
            outputs="selected_triples",
            name="select_triples_to_explain"
        ),
    ]

    # Get enabled explainers from parameters.yml configuration
    # First try kwargs (for programmatic override), then fall back to config file
    enabled_explainers = kwargs.get("enabled_explainers") or get_enabled_explainers()

    # If "all" is specified, enable all explainers
    if "all" in enabled_explainers:
        enabled_explainers = ["gnnexplainer", "pgexplainer", "page"]

    print(f"[Pipeline] Enabled explainers: {enabled_explainers}")

    # Build explainer nodes based on enabled_explainers parameter
    explainer_nodes = []

    # GNNExplainer node
    if "gnnexplainer" in enabled_explainers:
        explainer_nodes.append(
            node(
                func=run_gnnexplainer,
                inputs=[
                    "prepared_model",
                    "selected_triples",
                    "params:explanation"
                ],
                outputs="gnn_explanations",
                name="run_gnnexplainer",
                tags=["gnnexplainer", "explainer"]
            )
        )

    # PGExplainer node
    if "pgexplainer" in enabled_explainers:
        explainer_nodes.append(
            node(
                func=run_pgexplainer,
                inputs={
                    "model_dict": "prepared_model",
                    "selected_triples": "selected_triples",
                    "pyg_data": "pyg_data",  # Changed to use pyg_data directly
                    "explainer_params": "params:explanation"
                },
                outputs="pg_explanations",
                name="run_pgexplainer",
                tags=["pgexplainer", "explainer"]
            )
        )

    # PAGE Explainer node
    if "page" in enabled_explainers:
        explainer_nodes.append(
            node(
                func=run_page_explainer,
                inputs={
                    "model_dict": "prepared_model",
                    "selected_triples": "selected_triples",
                    "pyg_data": "pyg_data",  # Changed to use pyg_data directly
                    "explainer_params": "params:explanation"
                },
                outputs="page_explanations",
                name="run_page_explainer",
                tags=["page", "explainer"]
            )
        )

    # Summary node - only include if multiple explainers are enabled
    # The summary node requires outputs from all enabled explainers
    summary_nodes = []
    if len(enabled_explainers) > 1:
        # Build inputs dict based on which explainers are enabled
        # Using dict ensures correct argument mapping regardless of order
        summary_inputs = {
            "knowledge_graph": "knowledge_graph"  # Always needed
        }
        if "gnnexplainer" in enabled_explainers:
            summary_inputs["gnn_explanations"] = "gnn_explanations"
        if "pgexplainer" in enabled_explainers:
            summary_inputs["pg_explanations"] = "pg_explanations"
        if "page" in enabled_explainers:
            summary_inputs["page_explanations"] = "page_explanations"

        summary_nodes = [
            node(
                func=summarize_explanations,
                inputs=summary_inputs,
                outputs="explanation_summary",
                name="summarize_explanations",
                tags=["summary"]
            )
        ]

    # Build the full pipeline
    all_nodes = common_nodes + explainer_nodes + summary_nodes

    return Pipeline(all_nodes)
