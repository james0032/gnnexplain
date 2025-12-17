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
            inputs=[
                "trained_model",
                "pyg_data",
                "params:device"
            ],
            outputs="prepared_model",
            name="prepare_model_for_explanation"
        ),

        # Step 2: Select triples to explain
        node(
            func=select_triples_to_explain,
            inputs=[
                "pyg_data",
                "knowledge_graph",
                "params:explanation.triple_selection",
                "params:device",
                "top10_test"  # Optional: file content for "from_file" strategy
            ],
            outputs="selected_triples",
            name="select_triples_to_explain"
        ),
    ]

    # Get enabled explainers from parameters
    enabled_explainers = kwargs.get("enabled_explainers", ["all"])

    # If "all" is specified, enable all explainers
    if "all" in enabled_explainers:
        enabled_explainers = ["gnnexplainer", "pgexplainer", "page"]

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
                inputs=[
                    "prepared_model",
                    "selected_triples",
                    "pyg_data",
                    "params:explanation"
                ],
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
                inputs=[
                    "prepared_model",
                    "selected_triples",
                    "pyg_data",
                    "params:explanation"
                ],
                outputs="page_explanations",
                name="run_page_explainer",
                tags=["page", "explainer"]
            )
        )

    # Summary node - only include if multiple explainers are enabled
    # The summary node requires outputs from all enabled explainers
    summary_nodes = []
    if len(enabled_explainers) > 1:
        # Build inputs list based on which explainers are enabled
        summary_inputs = []
        if "gnnexplainer" in enabled_explainers:
            summary_inputs.append("gnn_explanations")
        if "pgexplainer" in enabled_explainers:
            summary_inputs.append("pg_explanations")
        summary_inputs.append("knowledge_graph")  # Always needed
        if "page" in enabled_explainers:
            summary_inputs.append("page_explanations")

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
