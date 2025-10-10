import matplotlib.pyplot as plt
from typing import Dict, Tuple, List
import torch
from torch_geometric.utils import k_hop_subgraph
import numpy as np
import networkx as nx

def visualize_explanation(explanation_data: Dict,
                         edge_index: torch.Tensor,
                         edge_type: torch.Tensor,
                         node_dict: Dict[str, int],
                         rel_dict: Dict[str, int],
                         save_path: str,
                         k_hops: int = 2,
                         edge_map: Dict[int, str] = None,
                         top_k_edges: int = 20,
                         id_to_name: Dict[str, str] = None):  
    """
    Visualize explanation subgraph and save as figure.
    
    Args:
        top_k_edges: Number of most important edges to display (default: 20)
    """
    triple = explanation_data['triple']
    edge_mask = explanation_data.get('edge_mask')
    
    head_idx, rel_idx, tail_idx = triple
    
    # Reverse dictionaries for labels
    idx_to_node_id = {v: k for k, v in node_dict.items()}
    idx_to_rel = {v: k for k, v in rel_dict.items()}
    
    # Function to get node name (prioritize readable name over ID)
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
    
    # Extract k-hop subgraph around head and tail
    nodes_of_interest = torch.tensor([head_idx, tail_idx])
    subset, sub_edge_index, mapping, edge_mask_sub = k_hop_subgraph(
        nodes_of_interest,
        k_hops,
        edge_index,
        relabel_nodes=True,
        num_nodes=edge_index.max().item() + 1
    )
    
    # Get edge types for subgraph
    sub_edge_type = edge_type[edge_mask_sub]
    
    # Get explanation scores for subgraph edges
    if edge_mask is not None:
        if isinstance(edge_mask, np.ndarray):
            full_edge_mask = edge_mask
        elif isinstance(edge_mask, torch.Tensor):
            full_edge_mask = edge_mask.cpu().numpy()
        else:
            full_edge_mask = np.array(edge_mask)
        
        explanation_scores = full_edge_mask[edge_mask_sub.cpu().numpy()]
    else:
        explanation_scores = np.ones(sub_edge_index.shape[1])
    
    # FILTER: Keep only top-K most important edges
    if len(explanation_scores) > top_k_edges:
        # Get indices of top-K edges
        top_k_indices = np.argsort(explanation_scores)[-top_k_edges:]
        
        # Filter subgraph to only include top-K edges
        sub_edge_index = sub_edge_index[:, top_k_indices]
        sub_edge_type = sub_edge_type[top_k_indices]
        explanation_scores = explanation_scores[top_k_indices]
        
        # Recompute which nodes are actually used
        used_nodes = torch.unique(sub_edge_index.flatten())
        
        # Create mapping from old to new node indices
        node_mapping = {old_idx.item(): new_idx for new_idx, old_idx in enumerate(used_nodes)}
        
        # Update subset to only include used nodes
        subset = subset[used_nodes]
        
        # Remap edge indices
        sub_edge_index_remapped = torch.zeros_like(sub_edge_index)
        for i in range(sub_edge_index.shape[1]):
            sub_edge_index_remapped[0, i] = node_mapping[sub_edge_index[0, i].item()]
            sub_edge_index_remapped[1, i] = node_mapping[sub_edge_index[1, i].item()]
        sub_edge_index = sub_edge_index_remapped
        
        # Update mapping for head/tail nodes
        if head_idx in subset:
            head_subgraph_idx = (subset == head_idx).nonzero(as_tuple=True)[0].item()
        else:
            head_subgraph_idx = None
            
        if tail_idx in subset:
            tail_subgraph_idx = (subset == tail_idx).nonzero(as_tuple=True)[0].item()
        else:
            tail_subgraph_idx = None
        
        print(f"  Filtered to top {top_k_edges} edges (from {len(edge_mask_sub)} total)")
    else:
        # Identify target nodes normally
        head_subgraph_idx = mapping[0].item() if head_idx in subset else None
        tail_subgraph_idx = mapping[1].item() if tail_idx in subset else None
    
    # Normalize scores for visualization
    if explanation_scores.max() > explanation_scores.min():
        explanation_scores = (explanation_scores - explanation_scores.min()) / \
                           (explanation_scores.max() - explanation_scores.min())
    else:
        explanation_scores = np.ones_like(explanation_scores) * 0.5
    
    # Create NetworkX graph
    G = nx.DiGraph()
    
    # Add nodes
    for i, node_idx in enumerate(subset.tolist()):
        node_label = get_node_name(node_idx) 
        if len(node_label) > 20:
            node_label = node_label[:17] + "..."
        G.add_node(i, label=node_label, original_idx=node_idx)
    
    # Add edges with explanation scores
    edge_colors = []
    edge_widths = []
    edge_labels = {}
    
    for i in range(sub_edge_index.shape[1]):
        src = sub_edge_index[0, i].item()
        dst = sub_edge_index[1, i].item()
        rel = sub_edge_type[i].item()
        score = explanation_scores[i]
        
        rel_label = get_relation_name(rel)
        if len(rel_label) > 20:
            rel_label = rel_label[:17] + "..."
        
        G.add_edge(src, dst, relation=rel_label, score=score)
        edge_labels[(src, dst)] = rel_label
        
        edge_colors.append(score)
        edge_widths.append(1 + score * 3)
    
    # Create figure
    fig, ax = plt.subplots(figsize=(16, 12))
    
    # Layout
    pos = nx.spring_layout(G, k=2, iterations=50, seed=42)
    
    # Draw nodes with different colors for head/tail
    node_colors = []
    node_sizes = []
    for node in G.nodes():
        if node == head_subgraph_idx:
            node_colors.append('#FF6B6B')
            node_sizes.append(2000)
        elif node == tail_subgraph_idx:
            node_colors.append('#FF9999')
            node_sizes.append(2000)
        else:
            node_colors.append('#95E1D3')
            node_sizes.append(1000)
    
    # Draw nodes
    nx.draw_networkx_nodes(G, pos, node_color=node_colors, 
                          node_size=node_sizes, alpha=0.9, ax=ax)
    
    # Draw edges with proper color mapping
    from matplotlib.colors import Normalize
    norm = Normalize(vmin=0, vmax=1)
    cmap = plt.cm.YlOrRd
    
    edges = list(G.edges())
    for (u, v), color_val, width in zip(edges, edge_colors, edge_widths):
        rgba = cmap(norm(color_val))
        nx.draw_networkx_edges(
            G, pos, [(u, v)], 
            edge_color=[rgba],
            width=width, alpha=0.7,
            arrows=True, arrowsize=20, arrowstyle='->',
            connectionstyle='arc3,rad=0.1',
            ax=ax
        )
    
    # Draw node labels
    node_labels_dict = nx.get_node_attributes(G, 'label')
    nx.draw_networkx_labels(G, pos, node_labels_dict, 
                           font_size=9, font_weight='bold', ax=ax)
    
    # Draw edge labels (relations)
    nx.draw_networkx_edge_labels(G, pos, edge_labels, 
                                 font_size=7, font_color='darkblue',
                                 bbox=dict(boxstyle='round,pad=0.3', 
                                         facecolor='white', alpha=0.7),
                                 ax=ax)
    
    # Add colorbar for edge importance
    sm = plt.cm.ScalarMappable(cmap=cmap, norm=norm)
    sm.set_array([])
    cbar = plt.colorbar(sm, ax=ax, fraction=0.046, pad=0.04)
    cbar.set_label('Edge Importance', rotation=270, labelpad=20, fontsize=12)
    
    # Title with readable relation name
    head_label = get_node_name(head_idx) 
    tail_label = get_node_name(tail_idx)

    rel_label = get_relation_name(rel_idx)
    
    # Truncate for title
    if len(head_label) > 25:
        head_label = head_label[:22] + "..."
    if len(tail_label) > 25:
        tail_label = tail_label[:22] + "..."
    if len(rel_label) > 30:
        rel_label = rel_label[:27] + "..."
    
    title = f"Explanation: ({head_label}) -[{rel_label}]-> ({tail_label})"
    ax.set_title(title, fontsize=14, fontweight='bold', pad=20)
    
    # Legend
    from matplotlib.patches import Patch
    legend_elements = [
        Patch(facecolor='#FF6B6B', label='Head Entity'),
        Patch(facecolor='#FF9999', label='Tail Entity'),
        Patch(facecolor='#95E1D3', label='Context Nodes')
    ]
    ax.legend(handles=legend_elements, loc='upper left', fontsize=10)
    
    ax.axis('off')
    plt.tight_layout()
    
    # Save figure
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"  Saved visualization to {save_path}")
        
def visualize_simple_explanation(explanation: Dict,
                                 edge_index: torch.Tensor,
                                 edge_type: torch.Tensor,
                                 node_dict: Dict[str, int],
                                 rel_dict: Dict[str, int],
                                 save_path: str,
                                 k_hops: int = 2,
                                 edge_map: Dict[int, str] = None,
                                 id_to_name: Dict[str, str] = None): 
    """Visualize explanation using path information."""
    
    triple = explanation['triple']
    head_idx, rel_idx, tail_idx = triple
    
    # Reverse dictionaries
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
    
    # Extract k-hop subgraph
    nodes_of_interest = torch.tensor([head_idx, tail_idx])
    subset, sub_edge_index, mapping, edge_mask_sub = k_hop_subgraph(
        nodes_of_interest,
        k_hops,
        edge_index,
        relabel_nodes=True,
        num_nodes=edge_index.max().item() + 1
    )
    
    sub_edge_type = edge_type[edge_mask_sub]
    
    # Create NetworkX graph
    G = nx.DiGraph()
    
    # Add nodes
    for i, node_idx in enumerate(subset.tolist()):
        node_label = get_node_name(node_idx) 
        if len(node_label) > 20:
            node_label = node_label[:17] + "..."
        G.add_node(i, label=node_label, original_idx=node_idx)
    
    # Add edges
    edge_labels = {}
    edge_colors = []
    edge_widths = []
    
    for i in range(sub_edge_index.shape[1]):
        src = sub_edge_index[0, i].item()
        dst = sub_edge_index[1, i].item()
        rel = sub_edge_type[i].item()
        
        # Use edge_map for relation names
        rel_label = get_relation_name(rel)
        if len(rel_label) > 20:
            rel_label = rel_label[:17] + "..."
        
        G.add_edge(src, dst, relation=rel_label)
        edge_labels[(src, dst)] = rel_label
        
        edge_colors.append(0.5)
        edge_widths.append(2.0)
    
    # Create figure
    fig, ax = plt.subplots(figsize=(16, 12))
    
    # Layout
    pos = nx.spring_layout(G, k=2, iterations=50, seed=42)
    
    # Identify target nodes
    head_subgraph_idx = mapping[0].item() if head_idx in subset else None
    tail_subgraph_idx = mapping[1].item() if tail_idx in subset else None
    
    # Node colors
    node_colors = []
    node_sizes = []
    for node in G.nodes():
        if node == head_subgraph_idx:
            node_colors.append('#FF6B6B')
            node_sizes.append(2000)
        elif node == tail_subgraph_idx:
            node_colors.append('#FF9999')
            node_sizes.append(2000)
        else:
            node_colors.append('#95E1D3')
            node_sizes.append(1000)
    
    # Draw
    nx.draw_networkx_nodes(G, pos, node_color=node_colors, 
                          node_size=node_sizes, alpha=0.9, ax=ax)
    
    nx.draw_networkx_edges(G, pos, edge_color=edge_colors,
                          width=edge_widths, alpha=0.7, 
                          edge_cmap=plt.cm.YlOrRd,
                          arrows=True, arrowsize=20, arrowstyle='->',
                          connectionstyle='arc3,rad=0.1', ax=ax)
    
    node_labels_dict = nx.get_node_attributes(G, 'label')
    nx.draw_networkx_labels(G, pos, node_labels_dict, 
                           font_size=9, font_weight='bold', ax=ax)
    
    nx.draw_networkx_edge_labels(G, pos, edge_labels, 
                                 font_size=7, font_color='darkblue',
                                 bbox=dict(boxstyle='round,pad=0.3', 
                                         facecolor='white', alpha=0.7),
                                 ax=ax)
    
    # Title
    title = f"Path Explanation: ({explanation.get('head', 'N/A')}) -[{explanation.get('relation', 'N/A')}]-> ({explanation.get('tail', 'N/A')})\n"
    title += f"Found {explanation.get('num_paths_found', 0)} connecting paths"
    ax.set_title(title, fontsize=14, fontweight='bold', pad=20)
    
    # Legend
    from matplotlib.patches import Patch
    legend_elements = [
        Patch(facecolor='#FF6B6B', label='Head Entity'),
        Patch(facecolor='#FF9999', label='Tail Entity'),
        Patch(facecolor='#95E1D3', label='Context Nodes')
    ]
    ax.legend(handles=legend_elements, loc='upper left', fontsize=10)
    
    ax.axis('off')
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()

