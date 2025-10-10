import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import RGCNConv
from torch_geometric.data import Data
from torch_geometric.explain import Explainer, GNNExplainer
from torch_geometric.utils import k_hop_subgraph
import numpy as np
from typing import Dict, Tuple, List
import pickle
import argparse
import matplotlib.pyplot as plt
import networkx as nx
import os
from utils import *
from triple_filter_prefix import *

class DistMult(nn.Module):
    """DistMult decoder for knowledge graph completion."""
    
    def __init__(self, num_relations: int, embedding_dim: int):
        super().__init__()
        self.relation_embeddings = nn.Parameter(
            torch.Tensor(num_relations, embedding_dim)
        )
        nn.init.xavier_uniform_(self.relation_embeddings)
    
    def forward(self, head_emb: torch.Tensor, tail_emb: torch.Tensor, 
                rel_idx: torch.Tensor) -> torch.Tensor:
        """
        Compute DistMult scores.
        
        Args:
            head_emb: Head entity embeddings (batch_size, embedding_dim)
            tail_emb: Tail entity embeddings (batch_size, embedding_dim)
            rel_idx: Relation indices (batch_size,)
        
        Returns:
            Scores for each triple (batch_size,)
        """
        rel_emb = self.relation_embeddings[rel_idx]
        # DistMult: <h, r, t> = sum(h * r * t)
        scores = torch.sum(head_emb * rel_emb * tail_emb, dim=1)
        return scores


class RGCNDistMultModel(nn.Module):
    """RGCN encoder with DistMult decoder for knowledge graph embedding."""
    
    def __init__(self, num_nodes: int, num_relations: int, 
                 embedding_dim: int = 128, num_layers: int = 2,
                 num_bases: int = 30, dropout: float = 0.2):
        super().__init__()
        self.num_nodes = num_nodes
        self.num_relations = num_relations
        self.embedding_dim = embedding_dim
        
        # Initial node embeddings
        self.node_embeddings = nn.Parameter(
            torch.Tensor(num_nodes, embedding_dim)
        )
        nn.init.xavier_uniform_(self.node_embeddings)
        
        # RGCN layers
        self.convs = nn.ModuleList()
        for i in range(num_layers):
            self.convs.append(
                RGCNConv(embedding_dim, embedding_dim, 
                        num_relations, num_bases=num_bases)
            )
        
        self.dropout = nn.Dropout(dropout)
        self.decoder = DistMult(num_relations, embedding_dim)
    
    def encode(self, edge_index: torch.Tensor, edge_type: torch.Tensor) -> torch.Tensor:
        """
        Encode nodes using RGCN.
        
        Args:
            edge_index: Edge indices (2, num_edges)
            edge_type: Edge types (num_edges,)
        
        Returns:
            Node embeddings (num_nodes, embedding_dim)
        """
        x = self.node_embeddings
        
        for conv in self.convs:
            x = conv(x, edge_index, edge_type)
            x = F.relu(x)
            x = self.dropout(x)
        
        return x
    
    def decode(self, node_emb: torch.Tensor, head_idx: torch.Tensor,
               tail_idx: torch.Tensor, rel_idx: torch.Tensor) -> torch.Tensor:
        """
        Decode triples using DistMult.
        
        Args:
            node_emb: Node embeddings (num_nodes, embedding_dim)
            head_idx: Head entity indices (batch_size,)
            tail_idx: Tail entity indices (batch_size,)
            rel_idx: Relation indices (batch_size,)
        
        Returns:
            Scores for each triple (batch_size,)
        """
        head_emb = node_emb[head_idx]
        tail_emb = node_emb[tail_idx]
        return self.decoder(head_emb, tail_emb, rel_idx)
    
    def forward(self, edge_index: torch.Tensor, edge_type: torch.Tensor,
                head_idx: torch.Tensor, tail_idx: torch.Tensor, 
                rel_idx: torch.Tensor) -> torch.Tensor:
        """Full forward pass."""
        node_emb = self.encode(edge_index, edge_type)
        scores = self.decode(node_emb, head_idx, tail_idx, rel_idx)
        return scores


class KGDataLoader:
    """Load and preprocess knowledge graph data."""
    
    def __init__(self, node_dict_path: str, rel_dict_path: str, edge_map_path: str = None):
        self.node_dict = self.load_dict(node_dict_path)
        self.rel_dict = self.load_dict(rel_dict_path)
        self.num_nodes = len(self.node_dict)
        self.num_relations = len(self.rel_dict)
        # Load edge map for predicate names
        if edge_map_path:
            self.edge_map = load_edge_map(edge_map_path)
        else:
            self.edge_map = {}
            
    @staticmethod
    def load_dict(path: str) -> Dict[str, int]:
        """Load entity or relation dictionary."""
        mapping = {}
        with open(path, 'r') as f:
            for line in f:
                parts = line.strip().split('\t')
                if len(parts) == 2:
                    mapping[parts[0]] = int(parts[1])
        return mapping
    
    def load_triples(self, path: str) -> torch.Tensor:
        """
        Load triples from file.
        
        Returns:
            Tensor of shape (num_triples, 3) with [head, relation, tail] indices
        """
        triples = []
        with open(path, 'r') as f:
            for line in f:
                parts = line.strip().split('\t')
                if len(parts) == 3:
                    head = self.node_dict.get(parts[0])
                    rel = self.rel_dict.get(parts[1])
                    tail = self.node_dict.get(parts[2])
                    
                    if head is not None and rel is not None and tail is not None:
                        triples.append([head, rel, tail])
        
        return torch.tensor(triples, dtype=torch.long)
    
    def create_pyg_data(self, triples: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Convert triples to PyG format.
        
        Returns:
            edge_index: (2, num_edges)
            edge_type: (num_edges,)
        """
        edge_index = torch.stack([triples[:, 0], triples[:, 2]], dim=0)
        edge_type = triples[:, 1]
        return edge_index, edge_type



def train_epoch(model: RGCNDistMultModel, 
                edge_index: torch.Tensor, 
                edge_type: torch.Tensor,
                train_triples: torch.Tensor,
                optimizer: torch.optim.Optimizer,
                device: torch.device,
                num_negatives: int = 5,
                batch_size: int = 1024) -> float:
    """Train for one epoch."""
    model.train()
    total_loss = 0
    num_batches = 0
    
    # Generate negative samples
    neg_triples = generate_negative_samples(train_triples, 
                                            model.num_nodes, 
                                            num_negatives)
    
    # Combine positive and negative samples
    all_triples = torch.cat([train_triples, neg_triples], dim=0)
    labels = torch.cat([
        torch.ones(len(train_triples)),
        torch.zeros(len(neg_triples))
    ]).to(device)
    
    # Shuffle
    perm = torch.randperm(len(all_triples))
    all_triples = all_triples[perm]
    labels = labels[perm]
    
    # Mini-batch training
    for i in range(0, len(all_triples), batch_size):
        batch_triples = all_triples[i:i+batch_size].to(device)
        batch_labels = labels[i:i+batch_size]
        
        optimizer.zero_grad()
        
        scores = model(edge_index, edge_type,
                      batch_triples[:, 0],
                      batch_triples[:, 2],
                      batch_triples[:, 1])
        
        loss = F.binary_cross_entropy_with_logits(scores, batch_labels)
        loss.backward()
        optimizer.step()
        
        total_loss += loss.item()
        num_batches += 1
    
    return total_loss / num_batches


@torch.no_grad()
def evaluate(model: RGCNDistMultModel,
            edge_index: torch.Tensor,
            edge_type: torch.Tensor,
            test_triples: torch.Tensor,
            all_triples: torch.Tensor,
            device: torch.device,
            batch_size: int = 1024,
            compute_mrr: bool = True,
            compute_hits: bool = True) -> Dict[str, float]:
    """
    Evaluate model on test set with MRR and Hit@10 metrics.
    Optimized with batch ranking, smart filtering, and count-based ranking.
    
    Args:
        model: Trained model
        edge_index: Graph edge indices
        edge_type: Graph edge types
        test_triples: Test triples to evaluate
        all_triples: All triples in the dataset (for filtering)
        device: Device
        batch_size: Batch size for evaluation
        compute_mrr: Whether to compute MRR metric
        compute_hits: Whether to compute Hit@10 metric
    
    Returns:
        Dictionary with accuracy, and optionally MRR and Hit@10 metrics
    """
    model.eval()
    
    results = {}
    all_predictions = []
    all_labels = []
    
    # Always compute accuracy with negative samples
    print("Computing accuracy metric...")
    
    # Encode all nodes once
    node_emb = model.encode(edge_index, edge_type)
    
    # Score positive samples
    for i in range(0, len(test_triples), batch_size):
        batch_triples = test_triples[i:i+batch_size].to(device)
        scores = model(edge_index, edge_type,
                      batch_triples[:, 0],
                      batch_triples[:, 2],
                      batch_triples[:, 1])
        all_predictions.extend(scores.cpu().tolist())
        all_labels.extend([1.0] * len(scores))
    
    # Generate negative samples for accuracy
    print("Generating negative samples for accuracy metric...")
    neg_triples = generate_negative_samples(test_triples, model.num_nodes, num_negatives=5)
    
    for i in range(0, len(neg_triples), batch_size):
        batch_triples = neg_triples[i:i+batch_size].to(device)
        scores = model(edge_index, edge_type,
                      batch_triples[:, 0],
                      batch_triples[:, 2],
                      batch_triples[:, 1])
        all_predictions.extend(scores.cpu().tolist())
        all_labels.extend([0.0] * len(scores))
    
    # Calculate accuracy
    all_predictions_tensor = torch.tensor(all_predictions)
    all_labels_tensor = torch.tensor(all_labels)
    predictions = (torch.sigmoid(all_predictions_tensor) > 0.5).float()
    accuracy = (predictions == all_labels_tensor).float().mean().item()
    results['accuracy'] = accuracy
    
    # Compute MRR and/or Hit@10 if requested
    if compute_mrr or compute_hits:
        print(f"Computing ranking metrics (MRR: {compute_mrr}, Hit@10: {compute_hits})...")
        
        print("Building filtered candidate sets...")
        # Pre-compute filtered candidates for each (head, rel) pair
        from collections import defaultdict
        invalid_tails = defaultdict(set)
        
        for triple in all_triples:
            h, r, t = triple[0].item(), triple[1].item(), triple[2].item()
            invalid_tails[(h, r)].add(t)
        
        print("Computing rankings...")
        
        mrr_sum = 0.0
        hits_at_10 = 0
        num_samples = 0
        
        # Process test triples in batches
        num_batches = (len(test_triples) + batch_size - 1) // batch_size
        
        for batch_idx in range(num_batches):
            start_idx = batch_idx * batch_size
            end_idx = min(start_idx + batch_size, len(test_triples))
            batch = test_triples[start_idx:end_idx]
            current_batch_size = len(batch)
            
            # Extract batch components
            batch_heads = batch[:, 0]
            batch_rels = batch[:, 1]
            batch_tails = batch[:, 2]
            
            # --- Tail Prediction: (h, r, ?) ---
            # Score all possible tails for all triples in batch simultaneously
            head_emb_expanded = node_emb[batch_heads]
            rel_emb_expanded = model.decoder.relation_embeddings[batch_rels]
            
            # Compute scores for all possible tails
            hr_product = head_emb_expanded * rel_emb_expanded
            all_tail_scores = torch.matmul(hr_product, node_emb.t())
            
            # Apply filtering: set invalid candidates to -inf
            for i in range(current_batch_size):
                h = batch_heads[i].item()
                r = batch_rels[i].item()
                t = batch_tails[i].item()
                
                # Get invalid tails for this (h, r) pair
                invalid_set = invalid_tails.get((h, r), set())
                
                # Mask out all invalid tails except the true one
                for invalid_t in invalid_set:
                    if invalid_t != t:
                        all_tail_scores[i, invalid_t] = float('-inf')
            
            # Count-based ranking (faster than sorting)
            true_tail_scores = all_tail_scores[torch.arange(current_batch_size), batch_tails]
            
            # For each triple, count how many scores are strictly greater
            ranks = (all_tail_scores > true_tail_scores.unsqueeze(1)).sum(dim=1) + 1
            
            # Accumulate metrics
            if compute_mrr:
                mrr_sum += (1.0 / ranks.float()).sum().item()
            if compute_hits:
                hits_at_10 += (ranks <= 10).sum().item()
            num_samples += current_batch_size
            
            if (batch_idx + 1) % 10 == 0 or batch_idx == num_batches - 1:
                print(f"  Processed {end_idx}/{len(test_triples)} test triples")
        
        # Calculate final metrics
        if compute_mrr:
            results['mrr'] = mrr_sum / num_samples
        if compute_hits:
            results['hit@10'] = hits_at_10 / num_samples
    
    return results

def visualize_explanation(explanation_data: Dict,
                         edge_index: torch.Tensor,
                         edge_type: torch.Tensor,
                         node_dict: Dict[str, int],
                         rel_dict: Dict[str, int],
                         save_path: str,
                         k_hops: int = 2,
                         edge_map: Dict[int, str] = None,
                         top_k_edges: int = 20):  
    """
    Visualize explanation subgraph and save as figure.
    
    Args:
        top_k_edges: Number of most important edges to display (default: 20)
    """
    triple = explanation_data['triple']
    edge_mask = explanation_data.get('edge_mask')
    
    head_idx, rel_idx, tail_idx = triple
    
    # Reverse dictionaries for labels
    idx_to_node = {v: k for k, v in node_dict.items()}
    idx_to_rel = {v: k for k, v in rel_dict.items()}
    
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
        node_label = idx_to_node.get(node_idx, f"Node_{node_idx}")
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
    head_label = idx_to_node.get(head_idx, f"Node_{head_idx}")
    tail_label = idx_to_node.get(tail_idx, f"Node_{tail_idx}")
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
                                 edge_map: Dict[int, str] = None): 
    """Visualize explanation using path information."""
    
    triple = explanation['triple']
    head_idx, rel_idx, tail_idx = triple
    
    # Reverse dictionaries
    idx_to_node = {v: k for k, v in node_dict.items()}
    idx_to_rel = {v: k for k, v in rel_dict.items()}
    
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
        node_label = idx_to_node.get(node_idx, f"Node_{node_idx}")
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

def link_prediction_explainer(model, edge_index, edge_type, triple, 
                              node_dict, rel_dict, device, k_hops=2, 
                              max_edges=2000, edge_map=None): 
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
    idx_to_node = {v: k for k, v in node_dict.items()}
    idx_to_rel = {v: k for k, v in rel_dict.items()}
    
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
            'source': idx_to_node.get(src, f"Node_{src}"),
            'target': idx_to_node.get(dst, f"Node_{dst}"),
            'relation': get_relation_name(rel),
            'importance': float(score)
        })
    
    explanation = {
        'triple': triple.tolist(),
        'head': idx_to_node.get(head_idx, f"Node_{head_idx}"),
        'relation': get_relation_name(rel_idx),
        'tail': idx_to_node.get(tail_idx, f"Node_{tail_idx}"),
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
                            edge_map: Dict[int, str] = None) -> Dict: 
    """
    Simple path-based explanation: find paths connecting head to tail.
    """
    from collections import defaultdict
    
    head_idx = triple[0].item()
    tail_idx = triple[2].item()
    rel_idx = triple[1].item()
    
    # Reverse mappings
    idx_to_node = {v: k for k, v in node_dict.items()}
    idx_to_rel = {v: k for k, v in rel_dict.items()}
    
    # ✅ Function to get relation name
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
        'head': idx_to_node.get(head_idx, f"Node_{head_idx}"),
        'relation': get_relation_name(rel_idx),  # Use readable name
        'tail': idx_to_node.get(tail_idx, f"Node_{tail_idx}"),
        'num_paths_found': len(paths),
        'paths': []
    }
    
    for path_nodes, path_edges in paths:
        path_desc = []
        for src, dst, rel, _ in path_edges:
            src_name = idx_to_node.get(src, f"Node_{src}")
            dst_name = idx_to_node.get(dst, f"Node_{dst}")
            rel_name = get_relation_name(rel)  # Use readable name
            path_desc.append(f"{src_name} -[{rel_name}]-> {dst_name}")
        
        explanation['paths'].append({
            'length': len(path_edges),
            'description': ' -> '.join([p.split(' -[')[0] for p in path_desc] + [path_desc[-1].split(']-> ')[1]]) if path_desc else '',
            'edges': path_desc
        })
    
    return explanation


def explain_triples(model: RGCNDistMultModel,
                   edge_index: torch.Tensor,
                   edge_type: torch.Tensor,
                   test_triples: torch.Tensor,
                   node_dict: Dict[str, int],
                   rel_dict: Dict[str, int],
                   device: torch.device,
                   num_samples: int = 10,
                   save_dir: str = 'explanations',
                   k_hops: int = 2,
                   use_simple_explanation: bool = False,
                   use_perturbation: bool = False,
                   max_edges: int = 2000,
                   edge_map: Dict[int, str] = None,
                   top_k_edges: int = 20,
                   subject_prefixes: List[str] = None,  
                   object_prefixes: List[str] = None) -> List[Dict]:
    """
    Explain test triples using different methods.
    
    Args:
        max_edges: Maximum edges for perturbation explainer (skip if larger)
    """
    model.eval()
    os.makedirs(save_dir, exist_ok=True)
    
    # FILTER: Apply prefix filtering if specified
    if subject_prefixes and object_prefixes:
        filtered_test_triples = filter_triples_by_prefix(
            test_triples,
            node_dict,
            subject_prefixes,
            object_prefixes
        )
        
        if len(filtered_test_triples) == 0:
            print("⚠️  No triples match the prefix criteria. Skipping explanations.")
            return []
        
        # Use filtered triples for sampling
        test_triples_to_sample = filtered_test_triples
    else:
        test_triples_to_sample = test_triples
        
    # Analyze graph connectivity
    print("\nAnalyzing graph connectivity...")
    degrees = torch.zeros(model.num_nodes, dtype=torch.long)
    for i in range(edge_index.shape[1]):
        src = edge_index[0, i].item()
        dst = edge_index[1, i].item()
        degrees[src] += 1
        degrees[dst] += 1
    
    avg_degree = degrees.float().mean().item()
    print(f"Average degree: {avg_degree:.2f}")
    
    explanations = []
    # ✅ SAMPLE: Use filtered triples
    num_samples = min(num_samples, len(test_triples_to_sample))
    sample_indices = torch.randperm(len(test_triples_to_sample))[:num_samples]
    
    method = "perturbation-based" if use_perturbation else "path-based"
    print(f"\nGenerating {method} explanations for {num_samples} test triples...")
    
    successful = 0
    failed = 0
    skipped = 0  # NEW COUNTER
    
    for idx_num, idx in enumerate(sample_indices):
        triple = test_triples_to_sample[idx]
        
        # Display triple with readable labels
        idx_to_node = {v: k for k, v in node_dict.items()}
        head_id = idx_to_node.get(triple[0].item(), f"Node_{triple[0].item()}")
        tail_id = idx_to_node.get(triple[2].item(), f"Node_{triple[2].item()}")
        
        print(f"\n[{idx_num+1}/{num_samples}] Explaining triple:")
        print(f"  {head_id} -> {tail_id}")
        print(f"  Indices: {triple.cpu().tolist()}")
        
        try:
            if use_perturbation:
                # Use perturbation-based explainer
                explanation_data = link_prediction_explainer(
                    model,
                    edge_index.to(device),
                    edge_type.to(device),
                    triple,
                    node_dict,
                    rel_dict,
                    device,
                    k_hops=k_hops,
                    max_edges=max_edges,
                    edge_map=edge_map
                )
                
                # CHECK: Handle None return (skipped)
                if explanation_data is None:
                    skipped += 1
                    continue
                
                print(f"  Original score: {explanation_data['original_score']:.4f}")
                print(f"  Top 5 important edges:")
                for i, edge in enumerate(explanation_data['important_edges'][:5]):
                    print(f"    {i+1}. {edge['source']} -[{edge['relation']}]-> {edge['target']} (importance: {edge['importance']:.4f})")
                
                explanations.append(explanation_data)
                
                # Visualize with edge importance
                save_path = os.path.join(save_dir, f'explanation_{idx_num+1}.png')
                visualize_explanation(
                    explanation_data,
                    edge_index.cpu(),
                    edge_type.cpu(),
                    node_dict,
                    rel_dict,
                    save_path,
                    k_hops=k_hops,
                    edge_map=edge_map,
                    top_k_edges=top_k_edges
                )
                print(f"  ✓ Saved to {save_path}")
                successful += 1
                
            else:
                # Use path-based explanation
                explanation_data = simple_path_explanation(
                    edge_index.cpu(),
                    edge_type.cpu(),
                    triple,
                    node_dict,
                    rel_dict,
                    k_hops=k_hops,
                    edge_map=edge_map
                )
                
                print(f"  Found {explanation_data['num_paths_found']} connecting paths")
                explanations.append(explanation_data)
                
                save_path = os.path.join(save_dir, f'explanation_{idx_num+1}.png')
                visualize_simple_explanation(
                    explanation_data,
                    edge_index.cpu(),
                    edge_type.cpu(),
                    node_dict,
                    rel_dict,
                    save_path,
                    k_hops=k_hops,
                    edge_map=edge_map
                )
                print(f"  ✓ Saved to {save_path}")
                successful += 1
                
        except Exception as e:
            print(f"  ✗ Error: {str(e)}")
            failed += 1
            import traceback
            traceback.print_exc()
    
    print(f"\n{'='*50}")
    print(f"Explanation Summary:")
    print(f"  Successful: {successful}/{num_samples}")
    print(f"  Skipped (too large): {skipped}/{num_samples}")  
    print(f"  Failed: {failed}/{num_samples}")
    print(f"{'='*50}")
    
    return explanations

def main():
    """Main training and explanation pipeline."""
    # Argument parser
    parser = argparse.ArgumentParser(description='Knowledge Graph RGCN-DistMult Training')
    
    # Model hyperparameters
    parser.add_argument('--embedding_dim', type=int, default=128,
                       help='Dimension of entity and relation embeddings')
    parser.add_argument('--num_layers', type=int, default=2,
                       help='Number of RGCN layers')
    parser.add_argument('--num_bases', type=int, default=30,
                       help='Number of bases for RGCN')
    parser.add_argument('--dropout', type=float, default=0.2,
                       help='Dropout rate')
    
    # Training hyperparameters
    parser.add_argument('--learning_rate', type=float, default=0.01,
                       help='Learning rate')
    parser.add_argument('--num_epochs', type=int, default=100,
                       help='Number of training epochs')
    parser.add_argument('--batch_size', type=int, default=1024,
                       help='Batch size for training')
    parser.add_argument('--num_negatives', type=int, default=5,
                       help='Number of negative samples per positive sample')
    parser.add_argument('--val_frequency', type=int, default=10,
                       help='Frequency (in epochs) to run validation')
    
    # Evaluation metrics
    parser.add_argument('--compute_mrr', action='store_true', default=True,
                       help='Compute MRR metric during evaluation')
    parser.add_argument('--no_mrr', action='store_false', dest='compute_mrr',
                       help='Skip MRR computation')
    parser.add_argument('--compute_hits', action='store_true', default=True,
                       help='Compute Hit@10 metric during evaluation')
    parser.add_argument('--no_hits', action='store_false', dest='compute_hits',
                       help='Skip Hit@10 computation')
    
    # Data paths
    parser.add_argument('--node_dict', type=str, default='node_dict',
                       help='Path to node dictionary file')
    parser.add_argument('--rel_dict', type=str, default='rel_dict',
                       help='Path to relation dictionary file')
    parser.add_argument('--edge_map', type=str, default='edge_map.json',
                       help='Path to edge mapping JSON file')
    parser.add_argument('--train_file', type=str, default='robo_train.txt',
                       help='Path to training triples file')
    parser.add_argument('--val_file', type=str, default='robo_val.txt',
                       help='Path to validation triples file')
    parser.add_argument('--test_file', type=str, default='robo_test.txt',
                       help='Path to test triples file')
    
    # Explanation parameters
    parser.add_argument('--num_explain', type=int, default=10,
                       help='Number of test triples to explain')
    parser.add_argument('--skip_explanation', action='store_true',
                       help='Skip explanation generation')
    parser.add_argument('--explanation_khops', type=int, default=2,
                       help='Number of hops for explanation subgraph visualization')
    parser.add_argument('--top_k_edges', type=int, default=20, 
                   help='Maximum number of edges to display in visualization (keeps most important)')
    parser.add_argument('--use_simple_explanation', action='store_true',
                       help='Use simple path-based explanation instead of GNNExplainer')
    parser.add_argument('--use_perturbation', action='store_true', 
                   help='Use edge perturbation explainer (slower but shows importance scores)')
    parser.add_argument('--max_edges', type=int, default=2000, 
                   help='Maximum edges for perturbation explainer (skip triples with more edges)')
    
    # NEW: Prefix filtering
    parser.add_argument('--subject_prefixes', type=str, nargs='+', 
                    default=['CHEBI', 'UNII', 'PUBCHEM.COMPOUND'],
                    help='Subject prefixes to filter (drug nodes)')
    parser.add_argument('--object_prefixes', type=str, nargs='+',
                    default=['MONDO'],
                    help='Object prefixes to filter (disease nodes)')
    parser.add_argument('--show_prefix_inventory', action='store_true',
                    help='Show inventory of node prefixes in test set')
    parser.add_argument('--no_prefix_filter', action='store_true',
                    help='Disable prefix filtering (use all test triples)')
    
    # Output paths
    parser.add_argument('--model_save_path', type=str, default='best_model.pt',
                       help='Path to save best model')
    parser.add_argument('--explanation_save_path', type=str, default='explanations.pkl',
                       help='Path to save explanations')
    parser.add_argument('--explanation_dir', type=str, default='explanations',
                       help='Directory to save explanation visualizations')
    
    args = parser.parse_args()
    
    # Configuration
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    print(f"\nHyperparameters:")
    for arg, value in vars(args).items():
        print(f"  {arg}: {value}")
    
    print("\n" + "="*50)
    print("Loading data...")
    print("="*50)
    data_loader = KGDataLoader(args.node_dict, args.rel_dict, args.edge_map)
    
    train_triples = data_loader.load_triples(args.train_file)
    val_triples = data_loader.load_triples(args.val_file)
    test_triples = data_loader.load_triples(args.test_file)
    
    print(f"Loaded {len(train_triples)} train, {len(val_triples)} val, {len(test_triples)} test triples")
    print(f"Num nodes: {data_loader.num_nodes}, Num relations: {data_loader.num_relations}")
    
    # All triples for filtered ranking
    all_triples = torch.cat([train_triples, val_triples, test_triples], dim=0)
    
    # Create graph structure from training data
    edge_index, edge_type = data_loader.create_pyg_data(train_triples)
    edge_index = edge_index.to(device)
    edge_type = edge_type.to(device)
    
    print("\n" + "="*50)
    print("Initializing model...")
    print("="*50)
    model = RGCNDistMultModel(
        num_nodes=data_loader.num_nodes,
        num_relations=data_loader.num_relations,
        embedding_dim=args.embedding_dim,
        num_layers=args.num_layers,
        num_bases=args.num_bases,
        dropout=args.dropout
    ).to(device)
    
    total_params = sum(p.numel() for p in model.parameters())
    print(f"Total parameters: {total_params:,}")
    
    optimizer = torch.optim.Adam(model.parameters(), lr=args.learning_rate)
    
    print("\n" + "="*50)
    print("Training...")
    print("="*50)
    best_val_mrr = 0
    for epoch in range(args.num_epochs):
        loss = train_epoch(model, edge_index, edge_type, train_triples,
                          optimizer, device, num_negatives=args.num_negatives,
                          batch_size=args.batch_size)
        
        print(f"Epoch {epoch+1}/{args.num_epochs}, Loss: {loss:.4f}")
        
        if (epoch + 1) % args.val_frequency == 0:
            print(f"\n--- Validation at Epoch {epoch+1} ---")
            val_metrics = evaluate(model, edge_index, edge_type, 
                                  val_triples, all_triples, device,
                                  batch_size=args.batch_size,
                                  compute_mrr=args.compute_mrr,
                                  compute_hits=args.compute_hits)
            print(f"Val Accuracy: {val_metrics['accuracy']:.4f}")
            if 'mrr' in val_metrics:
                print(f"Val MRR: {val_metrics['mrr']:.4f}")
            if 'hit@10' in val_metrics:
                print(f"Val Hit@10: {val_metrics['hit@10']:.4f}")
            print("-" * 35 + "\n")
            
            # Save best model based on available metrics
            if 'mrr' in val_metrics and val_metrics['mrr'] > best_val_mrr:
                best_val_mrr = val_metrics['mrr']
                torch.save(model.state_dict(), args.model_save_path)
                print(f"✓ Saved best model with MRR: {best_val_mrr:.4f}\n")
            elif 'mrr' not in val_metrics and val_metrics['accuracy'] > best_val_mrr:
                # Fall back to accuracy if MRR not computed
                best_val_mrr = val_metrics['accuracy']
                torch.save(model.state_dict(), args.model_save_path)
                print(f"✓ Saved best model with Accuracy: {best_val_mrr:.4f}\n")
    
    print("\n" + "="*50)
    print("Evaluating on test set...")
    print("="*50)
    model.load_state_dict(torch.load(args.model_save_path))
    test_metrics = evaluate(model, edge_index, edge_type, test_triples, 
                           all_triples, device, batch_size=args.batch_size,
                           compute_mrr=args.compute_mrr,
                           compute_hits=args.compute_hits)
    
    print("\n" + "="*50)
    print("FINAL TEST RESULTS")
    print("="*50)
    print(f"Test Accuracy: {test_metrics['accuracy']:.4f}")
    if 'mrr' in test_metrics:
        print(f"Test MRR:      {test_metrics['mrr']:.4f}")
    if 'hit@10' in test_metrics:
        print(f"Test Hit@10:   {test_metrics['hit@10']:.4f}")
    print("="*50)
    
    if not args.skip_explanation:
        print("\n" + "="*50)
        print("Generating explanations...")
        print("="*50)
        
        # Show prefix inventory if requested
        if args.show_prefix_inventory:
            print_prefix_inventory(test_triples, data_loader.node_dict, "Test Set")
        
        # Determine if we should filter
        if args.no_prefix_filter:
            subject_prefixes = None
            object_prefixes = None
        else:
            subject_prefixes = args.subject_prefixes
            object_prefixes = args.object_prefixes
            
        explanations = explain_triples(
            model, edge_index, edge_type, 
            test_triples, data_loader.node_dict, data_loader.rel_dict,
            device, num_samples=args.num_explain,
            save_dir=args.explanation_dir,
            k_hops=args.explanation_khops,
            use_simple_explanation=args.use_simple_explanation,
            use_perturbation=args.use_perturbation,
            max_edges=args.max_edges,
            edge_map=data_loader.edge_map,
            subject_prefixes=subject_prefixes, 
            object_prefixes=object_prefixes
        )
        
        # Save explanations
        with open(args.explanation_save_path, 'wb') as f:
            pickle.dump(explanations, f)
        
        print(f"\nGenerated {len(explanations)} explanations")
        print(f"Explanations saved to {args.explanation_save_path}")
        print(f"Visualizations saved to {args.explanation_dir}/")
    
    print("\n" + "="*50)
    print("Training completed!")
    print("="*50)


if __name__ == '__main__':
    main()