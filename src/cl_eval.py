"""
Comprehensive Evaluation Script for RGCN-DistMult Knowledge Graph Model

This script:
1. Loads a trained model from best_model.pt
2. Evaluates on test triples with metrics: accuracy, MRR, Hit@k
3. Generates explanations using all available explainers:
   - Path-based explainer (simple_path_explanation)
   - Perturbation-based explainer (link_prediction_explainer)
4. Saves explanations and visualizations

Usage:
    python cl_eval.py --model_path best_model.pt --test_file robo_test.txt
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import RGCNConv
import numpy as np
import argparse
import pickle
import os
from typing import Dict, List, Tuple
from collections import defaultdict

# Import utilities and explainers
from utils import load_edge_map, generate_negative_samples, load_id_to_name_map
from explainers import link_prediction_explainer, simple_path_explanation
from visualize_explanation import visualize_explanation, visualize_simple_explanation
from triple_filter_prefix import filter_triples_by_prefix, print_prefix_inventory


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
        """Compute DistMult scores."""
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
        """Encode nodes using RGCN."""
        x = self.node_embeddings

        for conv in self.convs:
            x = conv(x, edge_index, edge_type)
            x = F.relu(x)
            x = self.dropout(x)

        return x

    def decode(self, node_emb: torch.Tensor, head_idx: torch.Tensor,
               tail_idx: torch.Tensor, rel_idx: torch.Tensor) -> torch.Tensor:
        """Decode triples using DistMult."""
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
        """Load triples from file."""
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
        """Convert triples to PyG format."""
        edge_index = torch.stack([triples[:, 0], triples[:, 2]], dim=0)
        edge_type = triples[:, 1]
        return edge_index, edge_type


@torch.no_grad()
def evaluate(model: RGCNDistMultModel,
            edge_index: torch.Tensor,
            edge_type: torch.Tensor,
            test_triples: torch.Tensor,
            all_triples: torch.Tensor,
            device: torch.device,
            batch_size: int = 1024,
            compute_mrr: bool = True,
            compute_hits: bool = True,
            hit_k_values: List[int] = [1, 3, 10]) -> Dict[str, float]:
    """
    Comprehensive evaluation with accuracy, MRR, and Hit@K metrics.

    Args:
        model: Trained model
        edge_index: Graph edge indices
        edge_type: Graph edge types
        test_triples: Test triples to evaluate
        all_triples: All triples in dataset (for filtering)
        device: Device
        batch_size: Batch size for evaluation
        compute_mrr: Whether to compute MRR metric
        compute_hits: Whether to compute Hit@K metrics
        hit_k_values: List of K values for Hit@K (default: [1, 3, 10])

    Returns:
        Dictionary with evaluation metrics
    """
    model.eval()
    results = {}

    print("\n" + "="*60)
    print("EVALUATION METRICS")
    print("="*60)

    # 1. ACCURACY METRIC
    print("\n[1/3] Computing Accuracy...")
    all_predictions = []
    all_labels = []

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
    print(f"  ✓ Accuracy: {accuracy:.4f}")

    # 2. MRR AND HIT@K METRICS
    if compute_mrr or compute_hits:
        print(f"\n[2/3] Computing Ranking Metrics...")
        print("  Building filtered candidate sets...")

        # Pre-compute filtered candidates for each (head, rel) pair
        invalid_tails = defaultdict(set)
        for triple in all_triples:
            h, r, t = triple[0].item(), triple[1].item(), triple[2].item()
            invalid_tails[(h, r)].add(t)

        print("  Computing rankings...")
        mrr_sum = 0.0
        hit_counts = {k: 0 for k in hit_k_values}
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

            # Tail Prediction: (h, r, ?)
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

            # Count-based ranking
            true_tail_scores = all_tail_scores[torch.arange(current_batch_size), batch_tails]

            # For each triple, count how many scores are strictly greater
            ranks = (all_tail_scores > true_tail_scores.unsqueeze(1)).sum(dim=1) + 1

            # Accumulate metrics
            if compute_mrr:
                mrr_sum += (1.0 / ranks.float()).sum().item()

            if compute_hits:
                for k in hit_k_values:
                    hit_counts[k] += (ranks <= k).sum().item()

            num_samples += current_batch_size

            if (batch_idx + 1) % 10 == 0 or batch_idx == num_batches - 1:
                print(f"    Processed {end_idx}/{len(test_triples)} test triples")

        # Calculate final metrics
        if compute_mrr:
            results['mrr'] = mrr_sum / num_samples
            print(f"  ✓ MRR: {results['mrr']:.4f}")

        if compute_hits:
            for k in hit_k_values:
                metric_name = f'hit@{k}'
                results[metric_name] = hit_counts[k] / num_samples
                print(f"  ✓ Hit@{k}: {results[metric_name]:.4f}")

    print("\n" + "="*60)
    print("EVALUATION COMPLETE")
    print("="*60)

    return results


def explain_triples_all_methods(model: RGCNDistMultModel,
                                edge_index: torch.Tensor,
                                edge_type: torch.Tensor,
                                test_triples: torch.Tensor,
                                node_dict: Dict[str, int],
                                rel_dict: Dict[str, int],
                                device: torch.device,
                                num_samples: int = 10,
                                save_dir: str = 'explanations',
                                k_hops: int = 2,
                                max_edges: int = 2000,
                                edge_map: Dict[int, str] = None,
                                top_k_edges: int = 20,
                                subject_prefixes: List[str] = None,
                                object_prefixes: List[str] = None,
                                id_to_name: Dict[str, str] = None,
                                use_fast_mode: bool = True) -> Dict[str, List[Dict]]:
    """
    Generate explanations using ALL available explainer methods.

    Returns:
        Dictionary with explanations from each method:
        {
            'path_based': [...],
            'perturbation_based': [...]
        }
    """
    model.eval()
    os.makedirs(save_dir, exist_ok=True)

    # Apply prefix filtering if specified
    if subject_prefixes and object_prefixes:
        filtered_test_triples = filter_triples_by_prefix(
            test_triples,
            node_dict,
            subject_prefixes,
            object_prefixes
        )

        if len(filtered_test_triples) == 0:
            print("⚠️  No triples match the prefix criteria. Using all test triples.")
            test_triples_to_sample = test_triples
        else:
            test_triples_to_sample = filtered_test_triples
            print(f"✓ Filtered to {len(filtered_test_triples)} triples matching prefix criteria")
    else:
        test_triples_to_sample = test_triples

    # Sample test triples
    num_samples = min(num_samples, len(test_triples_to_sample))
    sample_indices = torch.randperm(len(test_triples_to_sample))[:num_samples]

    print("\n" + "="*60)
    print(f"GENERATING EXPLANATIONS FOR {num_samples} TEST TRIPLES")
    print("="*60)

    all_explanations = {
        'path_based': [],
        'perturbation_based': []
    }

    stats = {
        'path_based': {'successful': 0, 'failed': 0},
        'perturbation_based': {'successful': 0, 'failed': 0, 'skipped': 0}
    }

    # Generate explanations with BOTH methods
    for idx_num, idx in enumerate(sample_indices):
        triple = test_triples_to_sample[idx]

        # Display triple with readable labels
        idx_to_node = {v: k for k, v in node_dict.items()}
        head_id = idx_to_node.get(triple[0].item(), f"Node_{triple[0].item()}")
        tail_id = idx_to_node.get(triple[2].item(), f"Node_{triple[2].item()}")

        print(f"\n{'='*60}")
        print(f"[{idx_num+1}/{num_samples}] Explaining triple:")
        print(f"  {head_id} -> {tail_id}")
        print(f"  Indices: {triple.cpu().tolist()}")
        print(f"{'='*60}")

        # METHOD 1: Path-based explanation
        print("\n[Method 1] Path-based explanation...")
        try:
            path_explanation = simple_path_explanation(
                edge_index.cpu(),
                edge_type.cpu(),
                triple,
                node_dict,
                rel_dict,
                k_hops=k_hops,
                edge_map=edge_map,
                id_to_name=id_to_name
            )

            print(f"  ✓ Found {path_explanation['num_paths_found']} connecting paths")
            all_explanations['path_based'].append(path_explanation)
            stats['path_based']['successful'] += 1

            # Visualize
            save_path = os.path.join(save_dir, f'path_explanation_{idx_num+1}.png')
            visualize_simple_explanation(
                path_explanation,
                edge_index.cpu(),
                edge_type.cpu(),
                node_dict,
                rel_dict,
                save_path,
                k_hops=k_hops,
                edge_map=edge_map,
                id_to_name=id_to_name
            )
            print(f"  ✓ Saved visualization to {save_path}")

        except Exception as e:
            print(f"  ✗ Error: {str(e)}")
            stats['path_based']['failed'] += 1

        # METHOD 2: Perturbation-based explanation
        print("\n[Method 2] Perturbation-based explanation...")
        try:
            pert_explanation = link_prediction_explainer(
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
                id_to_name=id_to_name,
                use_fast_mode=use_fast_mode
            )

            if pert_explanation is None:
                print(f"  ⚠️  Skipped: subgraph too large (> {max_edges} edges)")
                stats['perturbation_based']['skipped'] += 1
            else:
                print(f"  ✓ Original score: {pert_explanation['original_score']:.4f}")
                print(f"  ✓ Top 5 important edges:")
                for i, edge in enumerate(pert_explanation['important_edges'][:5]):
                    print(f"    {i+1}. {edge['source']} -[{edge['relation']}]-> {edge['target']} (importance: {edge['importance']:.4f})")

                all_explanations['perturbation_based'].append(pert_explanation)
                stats['perturbation_based']['successful'] += 1

                # Visualize
                save_path = os.path.join(save_dir, f'perturbation_explanation_{idx_num+1}.png')
                visualize_explanation(
                    pert_explanation,
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
                print(f"  ✓ Saved visualization to {save_path}")

        except Exception as e:
            print(f"  ✗ Error: {str(e)}")
            stats['perturbation_based']['failed'] += 1

    # Print summary
    print("\n" + "="*60)
    print("EXPLANATION SUMMARY")
    print("="*60)
    print(f"\nPath-based explanations:")
    print(f"  Successful: {stats['path_based']['successful']}/{num_samples}")
    print(f"  Failed: {stats['path_based']['failed']}/{num_samples}")

    print(f"\nPerturbation-based explanations:")
    print(f"  Successful: {stats['perturbation_based']['successful']}/{num_samples}")
    print(f"  Skipped (too large): {stats['perturbation_based']['skipped']}/{num_samples}")
    print(f"  Failed: {stats['perturbation_based']['failed']}/{num_samples}")
    print("="*60)

    return all_explanations


def main():
    """Main evaluation and explanation pipeline."""
    parser = argparse.ArgumentParser(description='Comprehensive Evaluation for RGCN-DistMult Model')

    # Model and data paths
    parser.add_argument('--model_path', type=str, default='best_model.pt',
                       help='Path to trained model checkpoint')
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

    # Model hyperparameters (must match training)
    parser.add_argument('--embedding_dim', type=int, default=128,
                       help='Dimension of entity and relation embeddings')
    parser.add_argument('--num_layers', type=int, default=2,
                       help='Number of RGCN layers')
    parser.add_argument('--num_bases', type=int, default=30,
                       help='Number of bases for RGCN')
    parser.add_argument('--dropout', type=float, default=0.2,
                       help='Dropout rate')

    # Evaluation parameters
    parser.add_argument('--batch_size', type=int, default=1024,
                       help='Batch size for evaluation')
    parser.add_argument('--compute_mrr', action='store_true', default=True,
                       help='Compute MRR metric')
    parser.add_argument('--no_mrr', action='store_false', dest='compute_mrr',
                       help='Skip MRR computation')
    parser.add_argument('--compute_hits', action='store_true', default=True,
                       help='Compute Hit@K metrics')
    parser.add_argument('--no_hits', action='store_false', dest='compute_hits',
                       help='Skip Hit@K computation')
    parser.add_argument('--hit_k_values', type=int, nargs='+', default=[1, 3, 10],
                       help='K values for Hit@K metric (default: 1 3 10)')

    # Explanation parameters
    parser.add_argument('--num_explain', type=int, default=10,
                       help='Number of test triples to explain')
    parser.add_argument('--skip_explanation', action='store_true',
                       help='Skip explanation generation')
    parser.add_argument('--explanation_khops', type=int, default=2,
                       help='Number of hops for explanation subgraph')
    parser.add_argument('--top_k_edges', type=int, default=20,
                       help='Maximum edges to display in visualization')
    parser.add_argument('--max_edges', type=int, default=2000,
                       help='Maximum edges for perturbation explainer')
    parser.add_argument('--use_fast_explainer', action='store_true', default=True,
                       help='Use fast GPU-accelerated explainer (default: True)')
    parser.add_argument('--use_slow_explainer', action='store_false', dest='use_fast_explainer',
                       help='Use slower but potentially more accurate explainer')

    # Prefix filtering
    parser.add_argument('--subject_prefixes', type=str, nargs='+',
                       default=['CHEBI', 'UNII', 'PUBCHEM.COMPOUND'],
                       help='Subject prefixes to filter (drug nodes)')
    parser.add_argument('--object_prefixes', type=str, nargs='+',
                       default=['MONDO'],
                       help='Object prefixes to filter (disease nodes)')
    parser.add_argument('--show_prefix_inventory', action='store_true',
                       help='Show inventory of node prefixes in test set')
    parser.add_argument('--no_prefix_filter', action='store_true',
                       help='Disable prefix filtering')

    # Output paths
    parser.add_argument('--explanation_save_path', type=str, default='explanations_all.pkl',
                       help='Path to save explanations')
    parser.add_argument('--explanation_dir', type=str, default='explanations',
                       help='Directory to save explanation visualizations')
    parser.add_argument('--metrics_save_path', type=str, default='evaluation_metrics.pkl',
                       help='Path to save evaluation metrics')

    args = parser.parse_args()

    # Setup
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"\n{'='*60}")
    print(f"COMPREHENSIVE MODEL EVALUATION")
    print(f"{'='*60}")
    print(f"Device: {device}")
    print(f"Model: {args.model_path}")

    # Load data
    print("\n" + "="*60)
    print("LOADING DATA")
    print("="*60)
    data_loader = KGDataLoader(args.node_dict, args.rel_dict, args.edge_map)
    id_to_name = load_id_to_name_map(args.id_to_name_map)

    train_triples = data_loader.load_triples(args.train_file)
    val_triples = data_loader.load_triples(args.val_file)
    test_triples = data_loader.load_triples(args.test_file)

    print(f"✓ Loaded triples:")
    print(f"  Train: {len(train_triples)}")
    print(f"  Val:   {len(val_triples)}")
    print(f"  Test:  {len(test_triples)}")
    print(f"✓ Graph statistics:")
    print(f"  Nodes:     {data_loader.num_nodes}")
    print(f"  Relations: {data_loader.num_relations}")

    # All triples for filtered ranking
    all_triples = torch.cat([train_triples, val_triples, test_triples], dim=0)

    # Create graph structure from training data
    edge_index, edge_type = data_loader.create_pyg_data(train_triples)
    edge_index = edge_index.to(device)
    edge_type = edge_type.to(device)

    # Load model
    print("\n" + "="*60)
    print("LOADING MODEL")
    print("="*60)
    model = RGCNDistMultModel(
        num_nodes=data_loader.num_nodes,
        num_relations=data_loader.num_relations,
        embedding_dim=args.embedding_dim,
        num_layers=args.num_layers,
        num_bases=args.num_bases,
        dropout=args.dropout
    ).to(device)

    model.load_state_dict(torch.load(args.model_path, map_location=device))
    model.eval()

    total_params = sum(p.numel() for p in model.parameters())
    print(f"✓ Model loaded successfully")
    print(f"  Total parameters: {total_params:,}")

    # Evaluate on test set
    test_metrics = evaluate(
        model, edge_index, edge_type, test_triples,
        all_triples, device, batch_size=args.batch_size,
        compute_mrr=args.compute_mrr,
        compute_hits=args.compute_hits,
        hit_k_values=args.hit_k_values
    )

    # Save metrics
    with open(args.metrics_save_path, 'wb') as f:
        pickle.dump(test_metrics, f)
    print(f"\n✓ Saved metrics to {args.metrics_save_path}")

    # Generate explanations
    if not args.skip_explanation:
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

        all_explanations = explain_triples_all_methods(
            model, edge_index, edge_type,
            test_triples, data_loader.node_dict, data_loader.rel_dict,
            device, num_samples=args.num_explain,
            save_dir=args.explanation_dir,
            k_hops=args.explanation_khops,
            max_edges=args.max_edges,
            edge_map=data_loader.edge_map,
            top_k_edges=args.top_k_edges,
            subject_prefixes=subject_prefixes,
            object_prefixes=object_prefixes,
            id_to_name=id_to_name,
            use_fast_mode=args.use_fast_explainer
        )

        # Save explanations
        with open(args.explanation_save_path, 'wb') as f:
            pickle.dump(all_explanations, f)

        print(f"\n✓ Saved explanations to {args.explanation_save_path}")
        print(f"✓ Saved visualizations to {args.explanation_dir}/")

    # Final summary
    print("\n" + "="*60)
    print("FINAL RESULTS SUMMARY")
    print("="*60)
    print("\nTest Metrics:")
    for metric, value in test_metrics.items():
        print(f"  {metric:12s}: {value:.4f}")

    if not args.skip_explanation:
        print(f"\nExplanations Generated:")
        print(f"  Path-based:        {len(all_explanations['path_based'])}")
        print(f"  Perturbation-based: {len(all_explanations['perturbation_based'])}")

    print("\n" + "="*60)
    print("EVALUATION COMPLETE!")
    print("="*60)


if __name__ == '__main__':
    main()
