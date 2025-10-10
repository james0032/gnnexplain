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
from explainers import *
from visualize_explanation import *

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
                   object_prefixes: List[str] = None,
                   id_to_name=None) -> List[Dict]:
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
                    edge_map=edge_map,
                    id_to_name=id_to_name
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
                    top_k_edges=top_k_edges,
                    id_to_name=id_to_name
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
                    edge_map=edge_map,
                    id_to_name=id_to_name
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
                    edge_map=edge_map,
                    id_to_name=id_to_name
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
    parser.add_argument('--id_to_name_map', type=str, default='id_to_name.map',
                   help='Path to ID to name mapping file')
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
    id_to_name = load_id_to_name_map(args.id_to_name_map)
    
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
            object_prefixes=object_prefixes,
            id_to_name=id_to_name
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