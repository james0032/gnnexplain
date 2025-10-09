"""
Standalone script for generating explanations from trained RGCN-DistMult models.
This script loads a trained model and uses GNNExplainer to explain test triples.

Usage:
    python explain_model.py --model_path best_model.pt --num_samples 10
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import RGCNConv
from torch_geometric.data import Data
from torch_geometric.explain import Explainer, GNNExplainer, PGExplainer
from torch_geometric.utils import k_hop_subgraph
import numpy as np
import matplotlib.pyplot as plt
import networkx as nx
import argparse
import pickle
import os
from typing import Dict, List
from collections import defaultdict


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
        rel_emb = self.relation_embeddings[rel_idx]
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
        
        self.node_embeddings = nn.Parameter(
            torch.Tensor(num_nodes, embedding_dim)
        )
        nn.init.xavier_uniform_(self.node_embeddings)
        
        self.convs = nn.ModuleList()
        for i in range(num_layers):
            self.convs.append(
                RGCNConv(embedding_dim, embedding_dim, 
                        num_relations, num_bases=num_bases)
            )
        
        self.dropout = nn.Dropout(dropout)
        self.decoder = DistMult(num_relations, embedding_dim)
    
    def encode(self, edge_index: torch.Tensor, edge_type: torch.Tensor) -> torch.Tensor:
        x = self.node_embeddings
        
        for conv in self.convs:
            x = conv(x, edge_index, edge_type)
            x = F.relu(x)
            x = self.dropout(x)
        
        return x
    
    def decode(self, node_emb: torch.Tensor, head_idx: torch.Tensor,
               tail_idx: torch.Tensor, rel_idx: torch.Tensor) -> torch.Tensor:
        head_emb = node_emb[head_idx]
        tail_emb = node_emb[tail_idx]
        return self.decoder(head_emb, tail_emb, rel_idx)
    
    def forward(self, edge_index: torch.Tensor, edge_type: torch.Tensor,
                head_idx: torch.Tensor, tail_idx: torch.Tensor, 
                rel_idx: torch.Tensor) -> torch.Tensor:
        node_emb = self.encode(edge_index, edge_type)
        scores = self.decode(node_emb, head_idx, tail_idx, rel_idx)
        return scores


def load_dict(path: str) -> Dict[str, int]:
    """Load entity or relation dictionary."""
    mapping = {}
    with open(path, 'r') as f:
        for line in f:
            parts = line.strip().split('\t')
            if len(parts) == 2:
                mapping[parts[0]] = int(parts[1])
    return mapping


def load_triples(path: str, node_dict: Dict, rel_dict: Dict) -> torch.Tensor:
    """Load triples from file."""
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


def analyze_graph_connectivity(edge_index: torch.Tensor, 
                               test_triples: torch.Tensor,
                               num_nodes: int) -> Dict:
    """Analyze graph connectivity statistics."""
    print("\n" + "="*50)
    print("GRAPH CONNECTIVITY ANALYSIS")
    print("="*50)
    
    # Compute degree statistics
    degrees = torch.zeros(num_nodes, dtype=torch.long)
    for i in range(edge_index.shape[1]):
        src = edge_index[0, i].item()
        dst = edge_index[1, i].item()
        degrees[src] += 1
        degrees[dst] += 1
    
    print(f"Total nodes: {num_nodes}")
    print(f"Total edges: {edge_index.shape[1]}")
    print(f"Average degree: {degrees.float().mean().item():.2f}")
    print(f"Max degree: {degrees.max().item()}")
    print(f"Min degree: {degrees.min().item()}")
    print(f"Nodes with degree 0: {(degrees == 0).sum().item()}")
    
    # Check test triple connectivity
    test_heads = test_triples[:, 0]
    test_tails = test_triples[:, 2]
    
    head_degrees = degrees[test_heads]
    tail_degrees = degrees[test_tails]
    
    print(f"\nTest Triple Statistics:")
    print(f"Avg head degree: {head_degrees.float().mean().item():.2f}")
    print(f"Avg tail degree: {tail_degrees.float().mean().item():.2f}")
    print(f"Test triples with isolated heads: {(head_degrees == 0).sum().item()}")
    print(f"Test triples with isolated tails: {(tail_degrees == 0).sum().item()}")
    
    # Find poorly connected test triples
    poorly_connected = (head_degrees < 5) | (tail_degrees < 5)
    print(f"Test triples with low connectivity (<5 edges): {poorly_connected.sum().item()}")
    
    return {
        'avg_degree': degrees.float().mean().item(),
        'isolated_nodes': (degrees == 0).sum().item(),
        'poorly_connected_test': poorly_connected.sum().item()
    }


def simple_path_explanation(edge_index: torch.Tensor,
                            edge_type: torch.Tensor,
                            triple: torch.Tensor,
                            node_dict: Dict,
                            rel_dict: Dict,
                            k_hops: int = 2) -> Dict:
    """
    Simple path-based explanation: find paths connecting head to tail.
    This doesn't rely on GNNExplainer and works even for sparse graphs.
    """
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
            'description': ' -> '.join([p.split(' -[')[0] for p in path_desc] + [path_desc[-1].split(']-> ')[1]]),
            'edges': path_desc
        })
    
    return explanation


def visualize_simple_explanation(explanation: Dict,
                                 edge_index: torch.Tensor,
                                 edge_type: torch.Tensor,
                                 node_dict: Dict,
                                 rel_dict: Dict,
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
    
    # Add edges (highlight paths if found)
    path_edges = set()
    if explanation['num_paths_found'] > 0:
        # Mark edges that are part of discovered paths
        for path_info in explanation['paths']:
            # Parse path to extract edge info
            # This is a simplified version - you might need to enhance this
            pass
    
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
        
        # Color based on whether it's in a path (simplified)
        edge_colors.append(0.5)  # Neutral color
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
    title = f"Path Explanation: ({explanation['head']}) -[{explanation['relation']}]-> ({explanation['tail']})\n"
    title += f"Found {explanation['num_paths_found']} connecting paths"
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


def main():
    parser = argparse.ArgumentParser(description='Explain trained RGCN-DistMult model')
    
    # Model and data paths
    parser.add_argument('--model_path', type=str, required=True,
                       help='Path to trained model checkpoint')
    parser.add_argument('--node_dict', type=str, default='node_dict',
                       help='Path to node dictionary')
    parser.add_argument('--rel_dict', type=str, default='rel_dict',
                       help='Path to relation dictionary')
    parser.add_argument('--train_file', type=str, default='robo_train.txt',
                       help='Path to training triples')
    parser.add_argument('--test_file', type=str, default='robo_test.txt',
                       help='Path to test triples')
    
    # Model hyperparameters (must match training)
    parser.add_argument('--embedding_dim', type=int, default=128)
    parser.add_argument('--num_layers', type=int, default=2)
    parser.add_argument('--num_bases', type=int, default=30)
    parser.add_argument('--dropout', type=float, default=0.2)
    
    # Explanation parameters
    parser.add_argument('--num_samples', type=int, default=10,
                       help='Number of test triples to explain')
    parser.add_argument('--k_hops', type=int, default=2,
                       help='Number of hops for subgraph extraction')
    parser.add_argument('--output_dir', type=str, default='explanations',
                       help='Directory to save explanations')
    parser.add_argument('--use_path_explanation', action='store_true',
                       help='Use simple path-based explanation instead of GNNExplainer')
    
    args = parser.parse_args()
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    # Load data
    print("\nLoading data...")
    node_dict = load_dict(args.node_dict)
    rel_dict = load_dict(args.rel_dict)
    
    train_triples = load_triples(args.train_file, node_dict, rel_dict)
    test_triples = load_triples(args.test_file, node_dict, rel_dict)
    
    print(f"Nodes: {len(node_dict)}, Relations: {len(rel_dict)}")
    print(f"Train triples: {len(train_triples)}, Test triples: {len(test_triples)}")
    
    # Create graph structure
    edge_index = torch.stack([train_triples[:, 0], train_triples[:, 2]], dim=0).to(device)
    edge_type = train_triples[:, 1].to(device)
    
    # Analyze connectivity
    stats = analyze_graph_connectivity(edge_index.cpu(), test_triples, len(node_dict))
    
    if stats['poorly_connected_test'] > len(test_triples) * 0.5:
        print("\n⚠️  WARNING: More than 50% of test triples have low connectivity!")
        print("   Consider using --use_path_explanation for better results.")
    
    # Load model
    print("\nLoading model...")
    model = RGCNDistMultModel(
        num_nodes=len(node_dict),
        num_relations=len(rel_dict),
        embedding_dim=args.embedding_dim,
        num_layers=args.num_layers,
        num_bases=args.num_bases,
        dropout=args.dropout
    ).to(device)
    
    model.load_state_dict(torch.load(args.model_path, map_location=device))
    model.eval()
    print("✓ Model loaded successfully")
    
    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Sample test triples
    sample_indices = torch.randperm(len(test_triples))[:args.num_samples]
    
    print(f"\n{'='*50}")
    print(f"Generating explanations for {args.num_samples} test triples")
    print(f"{'='*50}")
    
    explanations = []
    
    for idx_num, idx in enumerate(sample_indices):
        triple = test_triples[idx]
        print(f"\n[{idx_num+1}/{args.num_samples}] Triple: {triple.tolist()}")
        
        try:
            if args.use_path_explanation:
                # Use simple path-based explanation
                explanation = simple_path_explanation(
                    edge_index.cpu(),
                    edge_type.cpu(),
                    triple,
                    node_dict,
                    rel_dict,
                    k_hops=args.k_hops
                )
                
                print(f"  Found {explanation['num_paths_found']} connecting paths")
                if explanation['num_paths_found'] > 0:
                    print(f"  Shortest path length: {min(p['length'] for p in explanation['paths'])}")
                
                explanations.append(explanation)
                
                # Visualize
                save_path = os.path.join(args.output_dir, f'explanation_{idx_num+1}.png')
                visualize_simple_explanation(
                    explanation,
                    edge_index.cpu(),
                    edge_type.cpu(),
                    node_dict,
                    rel_dict,
                    save_path,
                    k_hops=args.k_hops
                )
                print(f"  ✓ Saved to {save_path}")
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
                    algorithm=PGExplainer(epochs=30, lr=0.003),
                    explanation_type='model',
                    #node_mask_type='attributes',
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
                save_path = os.path.join(args.output_dir, f'explanation_{idx_num+1}.png')
                visualize_simple_explanation(
                    explanation_data,
                    edge_index.cpu(),
                    edge_type.cpu(),
                    node_dict,
                    rel_dict,
                    save_path,
                    k_hops=args.k_hops
                )
                print(f"  ✓ Saved to {save_path}")
                successful += 1
                
        except Exception as e:
            print(f"  ✗ Error: {str(e)}")
            import traceback
            traceback.print_exc()
    
    # Save explanations
    output_file = os.path.join(args.output_dir, 'explanations.pkl')
    with open(output_file, 'wb') as f:
        pickle.dump(explanations, f)
    
    print(f"\n{'='*50}")
    print(f"✓ Generated {len(explanations)} explanations")
    print(f"✓ Saved to {output_file}")
    print(f"{'='*50}")


if __name__ == '__main__':
    main()