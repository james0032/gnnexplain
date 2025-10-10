import matplotlib.pyplot as plt
from typing import Dict, Tuple, List
import torch
from torch_geometric.utils import k_hop_subgraph
import numpy as np
import networkx as nx

def link_prediction_explainer(model, edge_index, edge_type, triple,
                              node_dict, rel_dict, device, k_hops=2,
                              max_edges=2000, edge_map=None,
                              id_to_name=None,
                              use_fast_mode=True):
    """
    Custom explainer specifically for link prediction tasks.
    Uses GPU-accelerated parallel edge perturbation to find important edges.

    Args:
        max_edges: Maximum number of edges to test. Skip if subgraph is larger.
        use_fast_mode: Use fast approximation by encoding once (default: True)
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
    # FIX: Ensure nodes_of_interest is on CPU for k_hop_subgraph
    nodes_of_interest = torch.tensor([head_idx, tail_idx], dtype=torch.long)

    # Move edge_index and edge_type to CPU for k_hop_subgraph
    edge_index_cpu = edge_index.cpu()
    edge_type_cpu = edge_type.cpu()

    subset, sub_edge_index, mapping, edge_mask_sub = k_hop_subgraph(
        nodes_of_interest,
        k_hops,
        edge_index_cpu,
        relabel_nodes=True,
        num_nodes=edge_index_cpu.max().item() + 1
    )

    sub_edge_type = edge_type_cpu[edge_mask_sub]
    original_edge_indices = torch.where(edge_mask_sub)[0]  # CPU tensor

    # CHECK: Skip if subgraph is too large
    if len(original_edge_indices) > max_edges:
        print(f"  ⚠️  Skipping: subgraph has {len(original_edge_indices)} edges (max: {max_edges})")
        return None

    num_edges_to_test = len(original_edge_indices)
    print(f"  Testing {num_edges_to_test} edges in {k_hops}-hop subgraph...")

    if use_fast_mode:
        # FAST MODE: Batch multiple edge perturbations together for GPU efficiency
        print(f"  Using GPU-accelerated FAST mode (batched graph encoding)...")

        with torch.no_grad():
            importance_scores = []
            batch_size = 500  # Process 50 edges at a time

            for batch_start in range(0, num_edges_to_test, batch_size):
                batch_end = min(batch_start + batch_size, num_edges_to_test)
                batch_indices = original_edge_indices[batch_start:batch_end]  # CPU tensor
                current_batch_size = len(batch_indices)

                # Create masks for this batch - all on GPU
                batch_masks = torch.ones((current_batch_size, edge_index.shape[1]),
                                        dtype=torch.bool, device=device)

                # FIX: Move batch indices to GPU for vectorized setting
                batch_indices_gpu = batch_indices.to(device)
                row_idx = torch.arange(current_batch_size, device=device)
                batch_masks[row_idx, batch_indices_gpu] = False

                # Process batch: compute scores for all masked graphs
                batch_scores = []

                for i in range(current_batch_size):
                    mask = batch_masks[i]
                    masked_edge_index = edge_index[:, mask]
                    masked_edge_type = edge_type[mask]

                    # Re-encode with masked graph (necessary for RGCN)
                    masked_node_emb = model.encode(masked_edge_index, masked_edge_type)

                    # Decode score for this triple - FIX: Use proper tensor indexing
                    head_idx_val = head_idx  # Already extracted as .item() above
                    tail_idx_val = tail_idx  # Already extracted as .item() above
                    masked_head_emb = masked_node_emb[head_idx_val:head_idx_val+1]
                    masked_tail_emb = masked_node_emb[tail_idx_val:tail_idx_val+1]

                    rel_idx_tensor = torch.tensor([rel_idx], device=device)
                    score = model.decoder(masked_head_emb, masked_tail_emb, rel_idx_tensor)
                    batch_scores.append(score.item())

                # Compute importance scores for this batch
                batch_importance = [abs(original_score.item() - s) for s in batch_scores]
                importance_scores.extend(batch_importance)

                if batch_end % 100 == 0 or batch_end == num_edges_to_test:
                    print(f"    Processed {batch_end}/{num_edges_to_test} edges...")

            importance_scores = np.array(importance_scores)

    else:
        # ORIGINAL MODE: Full forward pass for each edge (slower but more accurate)
        print(f"  Using standard mode (full forward pass per edge)...")
        importance_scores = []

        for i, edge_idx in enumerate(original_edge_indices):
            with torch.no_grad():
                mask = torch.ones(edge_index.shape[1], dtype=torch.bool, device=device)
                mask[edge_idx] = False

                masked_edge_index = edge_index[:, mask]
                masked_edge_type = edge_type[mask]

                new_score = model(masked_edge_index, masked_edge_type,
                                triple[0:1].to(device),
                                triple[2:3].to(device),
                                triple[1:2].to(device))

                importance = abs(original_score.item() - new_score.item())
                importance_scores.append(importance)

            if (i + 1) % 50 == 0:
                print(f"    Processed {i+1}/{num_edges_to_test} edges...")

        importance_scores = np.array(importance_scores)

    # Normalize scores
    if importance_scores.max() > 0:
        importance_scores = importance_scores / importance_scores.max()
    
    # Create full-graph edge mask - FIX: Convert tensor indices to int
    full_edge_mask = np.zeros(edge_index.shape[1])
    for i, edge_idx in enumerate(original_edge_indices):
        # Convert tensor to int to avoid index errors
        idx = edge_idx.item() if torch.is_tensor(edge_idx) else int(edge_idx)
        full_edge_mask[idx] = importance_scores[i]
    
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


