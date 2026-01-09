"""Data preparation pipeline definition."""

from kedro.pipeline import Pipeline, pipeline, node
from .nodes import (
    load_triple_files,
    load_dictionaries,
    load_triples_from_files,
    create_knowledge_graph,
    convert_to_dgl_format,
    convert_to_pyg_format,
    generate_negative_samples_node,
)


def create_pipeline(**kwargs) -> Pipeline:
    """
    Create the data preparation pipeline.

    NOTE: Changed to generate pyg_data instead of dgl_data for full PyG pipeline compatibility.

    Returns:
        A Pipeline object for data preparation
    """
    return pipeline([
        node(
            func=load_triple_files,
            inputs=[
                "params:data.train_file",
                "params:data.val_file",
                "params:data.test_file"
            ],
            outputs="raw_triple_data",
            name="load_triple_files_node",
        ),
        node(
            func=load_dictionaries,
            inputs=[
                "params:data.node_dict",
                "params:data.rel_dict"
            ],
            outputs="dictionaries",
            name="load_dictionaries_node",
        ),
        node(
            func=load_triples_from_files,
            inputs=["raw_triple_data", "dictionaries"],
            outputs="triple_tensors",
            name="load_triples_node",
        ),
        node(
            func=create_knowledge_graph,
            inputs=["triple_tensors", "dictionaries"],
            outputs="knowledge_graph",
            name="create_kg_node",
        ),
        node(
            func=convert_to_pyg_format,  # Changed from convert_to_dgl_format
            inputs="knowledge_graph",
            outputs="pyg_data",  # Changed from dgl_data
            name="convert_to_pyg_node",  # Changed name
        ),
        node(
            func=generate_negative_samples_node,
            inputs=["pyg_data", "params:evaluation.num_neg_samples"],  # Changed from dgl_data
            outputs="negative_samples",
            name="generate_neg_samples_node",
        ),
    ])
