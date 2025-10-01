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
    
    def __init__(self, node_dict_path: str, rel_dict_path: str):
        self.node_dict = self.load_dict(node_dict_path)
        self.rel_dict = self.load_dict(rel_dict_path)
        self.num_nodes = len(self.node_dict)
        self.num_relations = len(self.rel_dict)
    
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


def generate_negative_samples(positive_triples: torch.Tensor, 
                              num_nodes: int, 
                              num_negatives: int = 1) -> torch.Tensor:
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


def visualize_simple_explanation(explanation: Dict,
                                 edge_index: torch.Tensor,
                                 edge_type: torch.Tensor,
                                 node_dict: Dict[str, int],
                                 rel_dict: Dict[str, int],
                                 save_path: str,
                                 k_hops: int = 2):
    """Visualize explanation using path information."""
    
    triple = explanation['triple']
    head_idx, rel_idx, tail_idx = triple
    
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
    
    # Reverse dictionaries
    idx_to_node = {v: k for k, v in node_dict.items()}
    idx_to_rel = {v: k for k, v in rel_dict.items()}
    
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
        
        rel_label = idx_to_rel.get(rel, f"Rel_{rel}")
        if len(rel_label) > 15:
            rel_label = rel_label[:12] + "..."
        
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
            node_colors.append('#4ECDC4')
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
        Patch(facecolor='#4ECDC4', label='Tail Entity'),
        Patch(facecolor='#95E1D3', label='Context Nodes')
    ]
    ax.legend(handles=legend_elements, loc='upper left', fontsize=10)
    
    ax.axis('off')
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()


def simple_path_explanation(edge_index: torch.Tensor,
                            edge_type: torch.Tensor,
                            triple: torch.Tensor,
                            node_dict: Dict[str, int],
                            rel_dict: Dict[str, int],
                            k_hops: int = 2) -> Dict:
    """
    Simple path-based explanation: find paths connecting head to tail.
    This doesn't rely on GNNExplainer and works even for sparse graphs.
    """
    from collections import defaultdict
    
    head_idx = triple[0].item()
    tail_idx = triple[2].item()
    rel_idx = triple[1].item()
    
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
    queue = [(head_idx, [], [])]  # (current_node, path_nodes, path_edges)
    
    while queue and len(paths) < 5:  # Find up to 5 paths
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
    
    # Reverse mappings
    idx_to_node = {v: k for k, v in node_dict.items()}
    idx_to_rel = {v: k for k, v in rel_dict.items()}
    
    explanation = {
        'triple': triple.tolist(),
        'head': idx_to_node.get(head_idx, f"Node_{head_idx}"),
        'relation': idx_to_rel.get(rel_idx, f"Rel_{rel_idx}"),
        'tail': idx_to_node.get(tail_idx, f"Node_{tail_idx}"),
        'num_paths_found': len(paths),
        'paths': []
    }
    
    for path_nodes, path_edges in paths:
        path_desc = []
        for src, dst, rel, _ in path_edges:
            src_name = idx_to_node.get(src, f"Node_{src}")
            dst_name = idx_to_node.get(dst, f"Node_{dst}")
            rel_name = idx_to_rel.get(rel, f"Rel_{rel}")
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
                   use_simple_explanation: bool = False) -> List[Dict]:
    """
    Use GNNExplainer to explain test triples and visualize them.
    Falls back to simple path-based explanation if GNNExplainer fails.
    
    Args:
        model: Trained model
        edge_index: Graph edge indices
        edge_type: Graph edge types
        test_triples: Test triples to explain
        node_dict: Node to index mapping
        rel_dict: Relation to index mapping
        device: Device
        num_samples: Number of test samples to explain
        save_dir: Directory to save visualizations
        k_hops: Number of hops for subgraph visualization
        use_simple_explanation: If True, skip GNNExplainer and use path-based method
    
    Returns:
        List of explanations
    """
    model.eval()
    
    # Create save directory
    os.makedirs(save_dir, exist_ok=True)
    
    # Analyze graph connectivity first
    print("\nAnalyzing graph connectivity...")
    degrees = torch.zeros(model.num_nodes, dtype=torch.long)
    for i in range(edge_index.shape[1]):
        src = edge_index[0, i].item()
        dst = edge_index[1, i].item()
        degrees[src] += 1
        degrees[dst] += 1
    
    avg_degree = degrees.float().mean().item()
    isolated_nodes = (degrees == 0).sum().item()
    
    print(f"Average degree: {avg_degree:.2f}")
    print(f"Isolated nodes: {isolated_nodes}")
    
    if avg_degree < 5:
        print("⚠️  WARNING: Graph has low connectivity (avg degree < 5)")
        print("   This may cause GNNExplainer to fail. Using simple path-based explanation.")
        use_simple_explanation = True
    
    explanations = []
    sample_indices = torch.randperm(len(test_triples))[:num_samples]
    
    print(f"\nGenerating explanations for {num_samples} test triples...")
    
    successful = 0
    failed = 0
    
    for idx_num, idx in enumerate(sample_indices):
        triple = test_triples[idx]
        
        print(f"\n[{idx_num+1}/{num_samples}] Explaining triple: {triple.cpu().tolist()}")
        
        # Check connectivity of this triple
        head_idx = triple[0].item()
        tail_idx = triple[2].item()
        head_degree = degrees[head_idx].item()
        tail_degree = degrees[tail_idx].item()
        
        print(f"  Head degree: {head_degree}, Tail degree: {tail_degree}")
        
        if head_degree == 0 or tail_degree == 0:
            print(f"  ⚠️  Isolated node detected, using simple explanation")
            use_simple_for_this = True
        else:
            use_simple_for_this = use_simple_explanation
        
        try:
            if use_simple_for_this:
                # Use simple path-based explanation
                explanation_data = simple_path_explanation(
                    edge_index.cpu(),
                    edge_type.cpu(),
                    triple,
                    node_dict,
                    rel_dict,
                    k_hops=k_hops
                )
                
                print(f"  Found {explanation_data['num_paths_found']} connecting paths")
                
                explanations.append(explanation_data)
                
                # Visualize
                save_path = os.path.join(save_dir, f'explanation_{idx_num+1}.png')
                visualize_simple_explanation(
                    explanation_data,
                    edge_index.cpu(),
                    edge_type.cpu(),
                    node_dict,
                    rel_dict,
                    save_path,
                    k_hops=k_hops
                )
                print(f"  ✓ Saved to {save_path}")
                successful += 1
            else:
                # Try GNNExplainer (may fail)
                triple_device = triple.to(device)
                
                # Create wrapper
                class ModelWrapper(nn.Module):
                    def __init__(self, base_model, edge_index, edge_type, target_triple):
                        super().__init__()
                        self.base_model = base_model
                        self.edge_index = edge_index
                        self.edge_type = edge_type
                        self.target_triple = target_triple
                    
                    def forward(self, x, edge_index, edge_attr=None):
                        node_emb = self.base_model.encode(edge_index, edge_attr)
                        score = self.base_model.decode(
                            node_emb,
                            self.target_triple[0:1],
                            self.target_triple[2:3],
                            self.target_triple[1:2]
                        )
                        return score
                
                wrapper = ModelWrapper(model, edge_index, edge_type, triple_device).to(device)
                
                explainer = Explainer(
                    model=wrapper,
                    algorithm=GNNExplainer(epochs=100),
                    explanation_type='model',
                    node_mask_type='attributes',
                    edge_mask_type='object',
                    model_config=dict(
                        mode='regression',
                        task_level='graph',
                        return_type='raw',
                    ),
                )
                
                data = Data(
                    x=model.node_embeddings.detach(),
                    edge_index=edge_index,
                    edge_attr=edge_type
                ).to(device)
                
                explanation = explainer(
                    x=data.x,
                    edge_index=data.edge_index,
                    edge_attr=data.edge_attr
                )
                
                explanation_data = {
                    'triple': triple.cpu().tolist(),
                    'edge_mask': explanation.edge_mask.cpu() if hasattr(explanation, 'edge_mask') else None,
                    'node_mask': explanation.node_mask.cpu() if hasattr(explanation, 'node_mask') else None,
                }
                
                explanations.append(explanation_data)
                
                # Visualize
                save_path = os.path.join(save_dir, f'explanation_{idx_num+1}.png')
                visualize_explanation(
                    explanation_data,
                    edge_index.cpu(),
                    edge_type.cpu(),
                    node_dict,
                    rel_dict,
                    save_path,
                    k_hops=k_hops
                )
                print(f"  ✓ Saved to {save_path}")
                successful += 1
                
        except Exception as e:
            print(f"  ✗ Error: {str(e)}")
            failed += 1
            
            # Try fallback to simple explanation
            try:
                print(f"  Trying fallback to simple path explanation...")
                explanation_data = simple_path_explanation(
                    edge_index.cpu(),
                    edge_type.cpu(),
                    triple,
                    node_dict,
                    rel_dict,
                    k_hops=k_hops
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
                    k_hops=k_hops
                )
                print(f"  ✓ Saved fallback explanation to {save_path}")
                successful += 1
                failed -= 1
            except Exception as e2:
                print(f"  ✗ Fallback also failed: {str(e2)}")
    
    print(f"\n{'='*50}")
    print(f"Explanation Summary:")
    print(f"  Successful: {successful}/{num_samples}")
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
    parser.add_argument('--node_dict', type=str, default='node_dict.txt',
                       help='Path to node dictionary file')
    parser.add_argument('--rel_dict', type=str, default='rel_dict.txt',
                       help='Path to relation dictionary file')
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
    parser.add_argument('--use_simple_explanation', action='store_true',
                       help='Use simple path-based explanation instead of GNNExplainer')
    
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
    data_loader = KGDataLoader(args.node_dict, args.rel_dict)
    
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
        explanations = explain_triples(
            model, edge_index, edge_type, 
            test_triples, data_loader.node_dict, data_loader.rel_dict,
            device, num_samples=args.num_explain,
            save_dir=args.explanation_dir,
            k_hops=args.explanation_khops,
            use_simple_explanation=args.use_simple_explanation
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