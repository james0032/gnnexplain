"""Data utility functions."""

import json
import torch
from typing import Dict, Tuple


def load_edge_map(edge_map_path: str) -> Dict[int, str]:
    """
    Load edge mapping from JSON file and extract predicate names.

    Args:
        edge_map_path: Path to edge_map.json

    Returns:
        Dictionary mapping relation index to predicate name
    """
    edge_to_predicate = {}

    try:
        with open(edge_map_path, 'r') as f:
            edge_map = json.load(f)

        # edge_map format:
        # {"{\"predicate\": \"biolink:contributes_to\", ...}": "predicate:0", ...}
        for key_str, value in edge_map.items():
            try:
                # Parse the JSON key string
                key_dict = json.loads(key_str)

                # Extract the predicate from the key
                if 'predicate' in key_dict:
                    predicate_full = key_dict['predicate']

                    # Extract just the part after "biolink:" if present
                    if 'biolink:' in predicate_full:
                        predicate_name = predicate_full.split('biolink:')[1]
                    else:
                        predicate_name = predicate_full

                    # Extract the index from "predicate:0" -> 0
                    if isinstance(value, str) and ':' in value:
                        idx = int(value.split(':')[1])
                        edge_to_predicate[idx] = predicate_name
                    else:
                        print(f"Warning: Unexpected value format: {value}")

            except json.JSONDecodeError as e:
                print(f"Warning: Could not parse key: {key_str[:50]}... Error: {e}")
                continue
            except (ValueError, IndexError) as e:
                print(f"Warning: Could not extract index from value: {value}. Error: {e}")
                continue

    except FileNotFoundError:
        print(f"Warning: {edge_map_path} not found. Using default relation labels.")
        return {}
    except Exception as e:
        print(f"Warning: Error loading {edge_map_path}: {e}. Using default relation labels.")
        return {}

    print(f"Loaded {len(edge_to_predicate)} predicate mappings from {edge_map_path}")

    return edge_to_predicate


def generate_negative_samples(
    positive_triples: torch.Tensor,
    num_nodes: int,
    num_negatives: int = 1
) -> torch.Tensor:
    """
    Generate negative samples by corrupting head or tail entities.

    Args:
        positive_triples: Positive triples (num_pos, 3)
        num_nodes: Total number of nodes
        num_negatives: Number of negative samples per positive

    Returns:
        Negative triples (num_pos * num_negatives, 3)
    """
    num_pos = positive_triples.shape[0]
    negatives = []

    for _ in range(num_negatives):
        # Randomly corrupt head or tail
        corrupted = positive_triples.clone()
        corrupt_head = torch.rand(num_pos) < 0.5

        # Corrupt heads
        corrupted[corrupt_head, 0] = torch.randint(0, num_nodes,
                                                    (corrupt_head.sum(),))
        # Corrupt tails
        corrupted[~corrupt_head, 2] = torch.randint(0, num_nodes,
                                                     ((~corrupt_head).sum(),))
        negatives.append(corrupted)

    return torch.cat(negatives, dim=0)


def load_id_to_name_map(map_path: str) -> Dict[str, str]:
    """
    Load ID to name mapping from file.

    Args:
        map_path: Path to id_to_name.map file (TSV format: id\tname)

    Returns:
        Dictionary mapping node ID to readable name
    """
    id_to_name = {}

    try:
        with open(map_path, 'r', encoding='utf-8') as f:
            for line in f:
                parts = line.strip().split('\t')
                if len(parts) >= 2:
                    node_id = parts[0]
                    node_name = '\t'.join(parts[1:])  # In case name contains tabs
                    id_to_name[node_id] = node_name

    except FileNotFoundError:
        print(f"Warning: {map_path} not found. Using node IDs as labels.")
        return {}
    except Exception as e:
        print(f"Warning: Error loading {map_path}: {e}. Using node IDs as labels.")
        return {}

    print(f"Loaded {len(id_to_name)} node name mappings")

    return id_to_name
