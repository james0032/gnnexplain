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
    Create explanation pipeline.

    This pipeline:
    1. Loads the trained model and prepares it for explanation
    2. Selects triples (edges) to explain
    3. Runs GNNExplainer to generate explanations
    4. Runs PGExplainer to generate explanations
    5. Runs PAGE explainer to generate explanations
    6. Summarizes and compares explanations

    Returns:
        Kedro Pipeline
    """
    return Pipeline([
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
                "params:device"
            ],
            outputs="selected_triples",
            name="select_triples_to_explain"
        ),

        # Step 3: Run GNNExplainer
        node(
            func=run_gnnexplainer,
            inputs=[
                "prepared_model",
                "selected_triples",
                "params:explanation.gnnexplainer"
            ],
            outputs="gnn_explanations",
            name="run_gnnexplainer"
        ),

        # Step 4: Run PGExplainer
        node(
            func=run_pgexplainer,
            inputs=[
                "prepared_model",
                "selected_triples",
                "pyg_data",
                "params:explanation.pgexplainer"
            ],
            outputs="pg_explanations",
            name="run_pgexplainer"
        ),

        # Step 5: Run PAGE Explainer
        node(
            func=run_page_explainer,
            inputs=[
                "prepared_model",
                "selected_triples",
                "pyg_data",
                "params:explanation.page"
            ],
            outputs="page_explanations",
            name="run_page_explainer"
        ),

        # Step 6: Summarize explanations
        node(
            func=summarize_explanations,
            inputs=[
                "gnn_explanations",
                "pg_explanations",
                "knowledge_graph",
                "page_explanations"
            ],
            outputs="explanation_summary",
            name="summarize_explanations"
        )
    ])
