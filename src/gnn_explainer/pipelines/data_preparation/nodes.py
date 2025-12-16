"""Nodes for data preparation pipeline."""

import torch
import pandas as pd
from typing import Dict, Tuple
from ..utils import generate_negative_samples


def load_triple_files(
    train_file_path: str,
    val_file_path: str,
    test_file_path: str
) -> Dict:
    """
    Load train/val/test triple files.

    Args:
        train_file_path: Path to training triples
        val_file_path: Path to validation triples
        test_file_path: Path to test triples

    Returns:
        Dictionary containing triple data
    """
    print(f"Loading triple files...")
    print(f"  Train: {train_file_path}")
    print(f"  Val: {val_file_path}")
    print(f"  Test: {test_file_path}")

    return {
        'train_file': train_file_path,
        'val_file': val_file_path,
        'test_file': test_file_path
    }


def load_dictionaries(
    node_dict_path: str,
    rel_dict_path: str
) -> Dict:
    """
    Load node and relation dictionaries.

    Args:
        node_dict_path: Path to node dictionary file
        rel_dict_path: Path to relation dictionary file

    Returns:
        Dictionary containing node_dict and rel_dict
    """
    print(f"\nLoading dictionaries...")
    print(f"  Node dict: {node_dict_path}")
    print(f"  Rel dict: {rel_dict_path}")

    # Load node dictionary
    node_dict = {}
    with open(node_dict_path, 'r') as f:
        for line in f:
            parts = line.strip().split('\t')
            if len(parts) == 2:
                node_dict[parts[0]] = int(parts[1])

    # Load relation dictionary
    rel_dict = {}
    with open(rel_dict_path, 'r') as f:
        for line in f:
            parts = line.strip().split('\t')
            if len(parts) == 2:
                rel_dict[parts[0]] = int(parts[1])

    print(f"  Loaded {len(node_dict)} nodes")
    print(f"  Loaded {len(rel_dict)} relations")

    return {
        'node_dict': node_dict,
        'rel_dict': rel_dict,
        'num_nodes': len(node_dict),
        'num_relations': len(rel_dict)
    }


def load_triples_from_files(
    triple_files: Dict,
    dictionaries: Dict
) -> Dict:
    """
    Load triples from files and convert to tensor format.

    Args:
        triple_files: Dictionary with file paths
        dictionaries: Dictionary with node_dict and rel_dict

    Returns:
        Dictionary with loaded triples as tensors
    """
    node_dict = dictionaries['node_dict']
    rel_dict = dictionaries['rel_dict']

    def load_file(path: str) -> torch.Tensor:
        """Load a single triple file."""
        triples = []
        with open(path, 'r') as f:
            for line in f:
                parts = line.strip().split('\t')
                if len(parts) == 3:
                    head = node_dict.get(parts[0])
                    rel = rel_dict.get(parts[1])
                    tail = node_dict.get(parts[2])

                    if head is not None and rel is not None and tail is not None:
                        triples.append([head, rel, tail])

        return torch.tensor(triples, dtype=torch.long)

    print(f"\nLoading triples from files...")
    train_triples = load_file(triple_files['train_file'])
    val_triples = load_file(triple_files['val_file'])
    test_triples = load_file(triple_files['test_file'])

    print(f"  Train: {len(train_triples)} triples")
    print(f"  Val: {len(val_triples)} triples")
    print(f"  Test: {len(test_triples)} triples")

    return {
        'train_triples': train_triples,
        'val_triples': val_triples,
        'test_triples': test_triples
    }


def create_knowledge_graph(
    triple_data: Dict,
    dictionaries: Dict
) -> Dict:
    """
    Combine all KG data into single structure.

    Args:
        triple_data: Dictionary with train/val/test triples
        dictionaries: Dictionary with node_dict and rel_dict

    Returns:
        Complete knowledge graph data
    """
    print(f"\nCreating knowledge graph structure...")

    # Combine all triples for full graph
    all_triples = torch.cat([
        triple_data['train_triples'],
        triple_data['val_triples'],
        triple_data['test_triples']
    ], dim=0)

    # Create reverse mappings for easy lookup
    node_dict = dictionaries['node_dict']
    rel_dict = dictionaries['rel_dict']

    idx_to_entity = {v: k for k, v in node_dict.items()}
    idx_to_relation = {v: k for k, v in rel_dict.items()}

    kg_data = {
        **triple_data,
        **dictionaries,
        'all_triples': all_triples,
        'idx_to_entity': idx_to_entity,
        'idx_to_relation': idx_to_relation
    }

    print(f"  Total triples: {len(all_triples)}")
    print(f"  Nodes: {kg_data['num_nodes']}")
    print(f"  Relations: {kg_data['num_relations']}")

    return kg_data


def convert_to_pyg_format(
    knowledge_graph: Dict
) -> Dict:
    """
    Convert to PyTorch Geometric format.

    Args:
        knowledge_graph: Knowledge graph data

    Returns:
        PyG-compatible data structure
    """
    print(f"\nConverting to PyTorch Geometric format...")

    # Use training + validation triples for the graph structure
    # (test triples are held out for evaluation)
    graph_triples = torch.cat([
        knowledge_graph['train_triples'],
        knowledge_graph['val_triples']
    ], dim=0)

    # Create edge_index and edge_type
    # edge_index: (2, num_edges) - source and target node indices
    # edge_type: (num_edges,) - relation type for each edge
    edge_index = torch.stack([graph_triples[:, 0], graph_triples[:, 2]], dim=0)
    edge_type = graph_triples[:, 1]

    # Add reverse edges for undirected message passing
    reverse_edge_index = torch.stack([graph_triples[:, 2], graph_triples[:, 0]], dim=0)
    reverse_edge_type = graph_triples[:, 1]  # Same relation type

    # Combine forward and reverse edges
    full_edge_index = torch.cat([edge_index, reverse_edge_index], dim=1)
    full_edge_type = torch.cat([edge_type, reverse_edge_type], dim=0)

    print(f"  Created {full_edge_index.shape[1]} edges (including reverse)")

    return {
        'edge_index': full_edge_index,
        'edge_type': full_edge_type,
        'num_nodes': knowledge_graph['num_nodes'],
        'num_relations': knowledge_graph['num_relations'],
        'train_triples': knowledge_graph['train_triples'],
        'val_triples': knowledge_graph['val_triples'],
        'test_triples': knowledge_graph['test_triples'],
        'all_triples': knowledge_graph['all_triples']
    }


def generate_negative_samples_node(
    pyg_data: Dict,
    num_neg_samples: int
) -> torch.Tensor:
    """
    Generate negative samples for evaluation.

    Args:
        pyg_data: PyG data structure
        num_neg_samples: Number of negative samples per positive

    Returns:
        Negative samples tensor
    """
    print(f"\nGenerating negative samples for evaluation...")
    print(f"  Positive test triples: {len(pyg_data['test_triples'])}")
    print(f"  Negatives per positive: {num_neg_samples}")

    negative_samples = generate_negative_samples(
        pyg_data['test_triples'],
        pyg_data['num_nodes'],
        num_negatives=num_neg_samples
    )

    print(f"  Generated {len(negative_samples)} negative samples")

    return negative_samples
