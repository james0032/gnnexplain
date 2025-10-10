import matplotlib.pyplot as plt
from typing import Dict, Tuple, List
import torch
from torch_geometric.utils import k_hop_subgraph
import numpy as np
import networkx as nx

def link_prediction_explainer(model, edge_index, edge_type, triple, 
                              node_dict, rel_dict, device, k_hops=2, 
                              max_edges=2000, edge_map=None,
                              id_to_name=None): 
    """
    Custom explainer specifically for link prediction tasks.
    Uses edge perturbation to find important edges.
    
    Args:
        max_edges: Maximum number of edges to test. Skip if subgraph is larger.
    """
    from torch_geometric.utils import k_hop_subgraph
    
    head_idx = triple[0].item()
    rel_idx = triple[1].item()
    tail_idx = triple[2].item()
    
    # Get original prediction score
    model.eval()
    with torch.no_grad():
        original_score = model(edge_index, edge_type,
                              triple[0:1].to(device), 
                              triple[2:3].to(device), 
                              triple[1:2].to(device))
    
    # Extract k-hop subgraph around triple
    nodes_of_interest = torch.tensor([head_idx, tail_idx])
    subset, sub_edge_index, mapping, edge_mask_sub = k_hop_subgraph(
        nodes_of_interest,
        k_hops,
        edge_index,
        relabel_nodes=True,
        num_nodes=edge_index.max().item() + 1
    )
    
    sub_edge_type = edge_type[edge_mask_sub]
    original_edge_indices = torch.where(edge_mask_sub)[0]
    
    # CHECK: Skip if subgraph is too large
    if len(original_edge_indices) > max_edges:
        print(f"  ⚠️  Skipping: subgraph has {len(original_edge_indices)} edges (max: {max_edges})")
        # Return None to signal that explanation should be skipped
        return None
    
    # Compute importance by edge removal
    importance_scores = []
    
    print(f"  Testing {len(original_edge_indices)} edges in {k_hops}-hop subgraph...")
    
    for i, edge_idx in enumerate(original_edge_indices):
        # Create mask removing this edge
        mask = torch.ones(edge_index.shape[1], dtype=torch.bool)
        mask[edge_idx] = False
        
        # Forward pass without this edge
        with torch.no_grad():
            masked_edge_index = edge_index[:, mask]
            masked_edge_type = edge_type[mask]
            
            new_score = model(masked_edge_index, masked_edge_type,
                            triple[0:1].to(device), 
                            triple[2:3].to(device), 
                            triple[1:2].to(device))
        
        # Importance = change in prediction
        importance = abs(original_score.item() - new_score.item())
        importance_scores.append(importance)
        
        if (i + 1) % 50 == 0:
            print(f"    Processed {i+1}/{len(original_edge_indices)} edges...")
    
    # Normalize scores
    importance_scores = np.array(importance_scores)
    if importance_scores.max() > 0:
        importance_scores = importance_scores / importance_scores.max()
    
    # Create full-graph edge mask
    full_edge_mask = np.zeros(edge_index.shape[1])
    for i, edge_idx in enumerate(original_edge_indices):
        full_edge_mask[edge_idx] = importance_scores[i]
    
    # Create explanation dict
    idx_to_node_id = {v: k for k, v in node_dict.items()}
    idx_to_rel = {v: k for k, v in rel_dict.items()}
    
    # Function to get node name
    def get_node_name(node_idx):
        node_id = idx_to_node_id.get(node_idx, f"Node_{node_idx}")
        if id_to_name and node_id in id_to_name:
            return id_to_name[node_id]
        return node_id
    
    # Function to get relation name
    def get_relation_name(rel_idx):
        if edge_map and rel_idx in edge_map:
            return edge_map[rel_idx]
        return idx_to_rel.get(rel_idx, f"Rel_{rel_idx}")
    
    # Sort edges by importance
    sorted_indices = np.argsort(importance_scores)[::-1]
    top_k = min(10, len(sorted_indices))
    
    important_edges = []
    for idx in sorted_indices[:top_k]:
        edge_global_idx = original_edge_indices[idx].item()
        src = edge_index[0, edge_global_idx].item()
        dst = edge_index[1, edge_global_idx].item()
        rel = edge_type[edge_global_idx].item()
        score = importance_scores[idx]
        
        important_edges.append({
            'source': get_node_name(src),
            'target': get_node_name(dst),
            'relation': get_relation_name(rel),
            'importance': float(score)
        })
    
    explanation = {
        'triple': triple.tolist(),
        'head': get_node_name(head_idx),
        'relation': get_relation_name(rel_idx),
        'tail': get_node_name(tail_idx),
        'original_score': float(original_score.item()),
        'important_edges': important_edges,
        'edge_mask': torch.tensor(full_edge_mask)
    }
    
    return explanation


def simple_path_explanation(edge_index: torch.Tensor,
                            edge_type: torch.Tensor,
                            triple: torch.Tensor,
                            node_dict: Dict[str, int],
                            rel_dict: Dict[str, int],
                            k_hops: int = 2,
                            edge_map: Dict[int, str] = None,
                            id_to_name: Dict[str, str] = None) -> Dict: 
    """
    Simple path-based explanation: find paths connecting head to tail.
    """
    from collections import defaultdict
    
    head_idx = triple[0].item()
    tail_idx = triple[2].item()
    rel_idx = triple[1].item()
    
    # Reverse mappings
    idx_to_node_id = {v: k for k, v in node_dict.items()}
    idx_to_rel = {v: k for k, v in rel_dict.items()}
    
    # Function to get node name
    def get_node_name(node_idx):
        node_id = idx_to_node_id.get(node_idx, f"Node_{node_idx}")
        if id_to_name and node_id in id_to_name:
            return id_to_name[node_id]
        return node_id
    
    # Function to get relation name
    def get_relation_name(rel_idx):
        if edge_map and rel_idx in edge_map:
            return edge_map[rel_idx]
        return idx_to_rel.get(rel_idx, f"Rel_{rel_idx}")
    
    # Build adjacency for BFS
    adj = defaultdict(list)
    for i in range(edge_index.shape[1]):
        src = edge_index[0, i].item()
        dst = edge_index[1, i].item()
        rel = edge_type[i].item()
        adj[src].append((dst, rel, i))
    
    # BFS to find paths
    paths = []
    visited = set()
    queue = [(head_idx, [], [])]
    
    while queue and len(paths) < 5:
        current, path_nodes, path_edges = queue.pop(0)
        
        if len(path_nodes) > k_hops:
            continue
        
        if current == tail_idx and len(path_nodes) > 0:
            paths.append((path_nodes + [current], path_edges))
            continue
        
        if current in visited:
            continue
        visited.add(current)
        
        for neighbor, rel, edge_idx in adj.get(current, []):
            if neighbor not in visited:
                queue.append((
                    neighbor,
                    path_nodes + [current],
                    path_edges + [(current, neighbor, rel, edge_idx)]
                ))
    
    explanation = {
        'triple': triple.tolist(),
        'head': get_node_name(head_idx),
        'relation': get_relation_name(rel_idx),  # Use readable name
        'tail': get_node_name(tail_idx),
        'num_paths_found': len(paths),
        'paths': []
    }
    
    for path_nodes, path_edges in paths:
        path_desc = []
        for src, dst, rel, _ in path_edges:
            src_name = get_node_name(src)
            dst_name = get_node_name(dst) 
            rel_name = get_relation_name(rel)  # Use readable name
            path_desc.append(f"{src_name} -[{rel_name}]-> {dst_name}")
        
        explanation['paths'].append({
            'length': len(path_edges),
            'description': ' -> '.join([p.split(' -[')[0] for p in path_desc] + [path_desc[-1].split(']-> ')[1]]) if path_desc else '',
            'edges': path_desc
        })
    
    return explanation


