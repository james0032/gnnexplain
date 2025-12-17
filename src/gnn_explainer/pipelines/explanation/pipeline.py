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
    3. Runs selected explainers (GNNExplainer, PGExplainer, PAGE)
    4. Summarizes and compares explanations

    To run specific explainers, use Kedro's tag filtering:
      - Run only PAGE (no summary):
          kedro run --pipeline=explanation --tags=page --exclude-tags=summary
      - Run GNN and PG (no summary):
          kedro run --pipeline=explanation --tags=gnnexplainer,pgexplainer --exclude-tags=summary
      - Run all explainers + summary:
          kedro run --pipeline=explanation

    Note: The summary node requires all three explainer outputs. When running
    specific explainers, use --exclude-tags=summary to skip the summary step.

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

    # Explainer nodes with tags for selective execution
    explainer_nodes = [
        # GNNExplainer node
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
        ),

        # PGExplainer node
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
        ),

        # PAGE Explainer node
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
        ),
    ]

    # Summary node
    # Note: When using tag filtering to run specific explainers, the summary node
    # will fail if it's included. To run specific explainers without summarizing:
    #   kedro run --pipeline=explanation --tags=page --exclude-tags=summary
    # To run all explainers and summarize:
    #   kedro run --pipeline=explanation
    summary_nodes = [
        node(
            func=summarize_explanations,
            inputs=[
                "gnn_explanations",
                "pg_explanations",
                "knowledge_graph",
                "page_explanations"
            ],
            outputs="explanation_summary",
            name="summarize_explanations",
            tags=["summary"]
        )
    ]

    # Build the full pipeline
    all_nodes = common_nodes + explainer_nodes + summary_nodes

    return Pipeline(all_nodes)
