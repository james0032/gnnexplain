"""
Evaluation Script for Kedro-trained GNN Models

This script evaluates models trained via the Kedro pipeline.
It handles the Kedro model artifact format (pickle with model_state_dict and model_config).

Usage:
    python src/eval_kedro_model.py --model_path data/06_models/trained_model.pkl
"""

import torch
import torch.nn.functional as F
import pickle
import argparse
import sys
from pathlib import Path
from typing import Dict, List
from collections import defaultdict

# Import data loader from cl_eval
from cl_eval import KGDataLoader

# Import model classes
sys.path.append(str(Path(__file__).parent / "gnn_explainer" / "pipelines" / "training"))
from gnn_explainer.pipelines.training.model import RGCNDistMultModel
from gnn_explainer.pipelines.training.kg_models import CompGCNKGModel
from gnn_explainer.pipelines.utils import generate_negative_samples


def load_kedro_model(model_path: str, device: torch.device):
    """
    Load a model trained via Kedro pipeline.

    The Kedro pipeline saves models as pickle files with this structure:
    {
        'model_state_dict': {...},
        'model_config': {
            'num_nodes': ...,
            'num_relations': ...,
            'model_type': 'rgcn' or 'compgcn',
            'decoder_type': 'distmult', 'complex', 'rotate', or 'conve',
            'embedding_dim': ...,
            'num_layers': ...,
            'dropout': ...,
            ...
        },
        'training_info': {...}
    }
    """
    print(f"\nLoading Kedro model from: {model_path}")

    with open(model_path, 'rb') as f:
        model_artifact = pickle.load(f)

    # Extract components
    model_state_dict = model_artifact['model_state_dict']
    model_config = model_artifact['model_config']

    print(f"Model configuration:")
    print(f"  Model type: {model_config.get('model_type', 'rgcn')}")
    print(f"  Decoder type: {model_config.get('decoder_type', 'distmult')}")
    print(f"  Nodes: {model_config['num_nodes']}")
    print(f"  Relations: {model_config['num_relations']}")
    print(f"  Embedding dim: {model_config['embedding_dim']}")
    print(f"  Num layers: {model_config['num_layers']}")
    print(f"  Dropout: {model_config['dropout']}")

    # Initialize model based on type
    model_type = model_config.get('model_type', 'rgcn')

    if model_type == 'compgcn':
        decoder_type = model_config.get('decoder_type', 'distmult')

        # ConvE-specific parameters
        conve_kwargs = None
        if decoder_type == 'conve':
            conve_kwargs = {
                'input_drop': model_config.get('conve_input_drop', 0.2),
                'hidden_drop': model_config.get('conve_hidden_drop', 0.3),
                'feature_drop': model_config.get('conve_feature_drop', 0.2),
                'num_filters': model_config.get('conve_num_filters', 32),
                'kernel_size': model_config.get('conve_kernel_size', 3),
            }

        model = CompGCNKGModel(
            num_nodes=model_config['num_nodes'],
            num_relations=model_config['num_relations'],
            embedding_dim=model_config['embedding_dim'],
            decoder_type=decoder_type,
            num_layers=model_config['num_layers'],
            comp_fn=model_config.get('comp_fn', 'sub'),
            dropout=model_config['dropout'],
            conve_kwargs=conve_kwargs
        ).to(device)

    else:  # rgcn
        model = RGCNDistMultModel(
            num_nodes=model_config['num_nodes'],
            num_relations=model_config['num_relations'],
            embedding_dim=model_config['embedding_dim'],
            num_layers=model_config['num_layers'],
            num_bases=model_config.get('num_bases', 30),
            dropout=model_config['dropout']
        ).to(device)

    # Load state dict
    model.load_state_dict(model_state_dict)
    model.eval()

    total_params = sum(p.numel() for p in model.parameters())
    print(f"✓ Model loaded successfully")
    print(f"  Total parameters: {total_params:,}")

    return model, model_config


@torch.no_grad()
def evaluate_kedro_model(
    model,
    edge_index: torch.Tensor,
    edge_type: torch.Tensor,
    test_triples: torch.Tensor,
    all_triples: torch.Tensor,
    device: torch.device,
    batch_size: int = 1024,
    compute_mrr: bool = True,
    compute_hits: bool = True,
    hit_k_values: List[int] = [1, 3, 10]
) -> Dict[str, float]:
    """
    Comprehensive evaluation for Kedro-trained models.

    Works with both RGCN and CompGCN models.
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

    # Score positive samples
    for i in range(0, len(test_triples), batch_size):
        batch_triples = test_triples[i:i+batch_size].to(device)
        scores = model(
            edge_index, edge_type,
            batch_triples[:, 0],
            batch_triples[:, 2],
            batch_triples[:, 1]
        )
        all_predictions.extend(scores.cpu().tolist())
        all_labels.extend([1.0] * len(scores))

    # Generate negative samples for accuracy
    num_nodes = edge_index.max().item() + 1
    neg_triples = generate_negative_samples(test_triples, num_nodes, num_negatives=5)

    for i in range(0, len(neg_triples), batch_size):
        batch_triples = neg_triples[i:i+batch_size].to(device)
        scores = model(
            edge_index, edge_type,
            batch_triples[:, 0],
            batch_triples[:, 2],
            batch_triples[:, 1]
        )
        all_predictions.extend(scores.cpu().tolist())
        all_labels.extend([0.0] * len(scores))

    # Calculate accuracy
    all_predictions_tensor = torch.tensor(all_predictions)
    all_labels_tensor = torch.tensor(all_labels)
    predictions = (torch.sigmoid(all_predictions_tensor) > 0.5).float()
    accuracy = (predictions == all_labels_tensor).float().mean().item()
    results['accuracy'] = accuracy
    print(f"  ✓ Accuracy: {accuracy:.4f}")

    # 2. MRR AND HIT@K METRICS (using scoring function)
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
            batch_heads = batch[:, 0].to(device)
            batch_rels = batch[:, 1].to(device)
            batch_tails = batch[:, 2].to(device)

            # Score all possible tails for each (head, rel) pair
            # We'll score each candidate tail one by one (inefficient but compatible)
            all_tail_scores = torch.zeros(current_batch_size, num_nodes, device=device)

            for tail_candidate in range(num_nodes):
                # Create batch of triples with this tail candidate
                candidate_tails = torch.full((current_batch_size,), tail_candidate,
                                            dtype=torch.long, device=device)

                # Score these triples
                scores = model(edge_index, edge_type, batch_heads, candidate_tails, batch_rels)
                all_tail_scores[:, tail_candidate] = scores

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


def main():
    """Main evaluation pipeline."""
    parser = argparse.ArgumentParser(description='Evaluate Kedro-trained GNN Model')

    # Model and data paths
    parser.add_argument('--model_path', type=str, required=True,
                       help='Path to Kedro trained model pickle file')
    parser.add_argument('--node_dict', type=str, required=True,
                       help='Path to node dictionary file')
    parser.add_argument('--rel_dict', type=str, required=True,
                       help='Path to relation dictionary file')
    parser.add_argument('--train_file', type=str, required=True,
                       help='Path to training triples file')
    parser.add_argument('--val_file', type=str, required=True,
                       help='Path to validation triples file')
    parser.add_argument('--test_file', type=str, required=True,
                       help='Path to test triples file')

    # Evaluation parameters
    parser.add_argument('--batch_size', type=int, default=1024,
                       help='Batch size for evaluation')
    parser.add_argument('--compute_mrr', action='store_true', default=True,
                       help='Compute MRR metric')
    parser.add_argument('--compute_hits', action='store_true', default=True,
                       help='Compute Hit@K metrics')
    parser.add_argument('--hit_k_values', type=int, nargs='+', default=[1, 3, 10],
                       help='K values for Hit@K metric')

    # Output paths
    parser.add_argument('--metrics_save_path', type=str, default='evaluation_metrics.pkl',
                       help='Path to save evaluation metrics')

    args = parser.parse_args()

    # Setup
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"\n{'='*60}")
    print(f"KEDRO MODEL EVALUATION")
    print(f"{'='*60}")
    print(f"Device: {device}")
    print(f"Model: {args.model_path}")

    # Load data
    print("\n" + "="*60)
    print("LOADING DATA")
    print("="*60)
    data_loader = KGDataLoader(args.node_dict, args.rel_dict, edge_map_path=None)

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
    model, model_config = load_kedro_model(args.model_path, device)

    # Evaluate on test set
    test_metrics = evaluate_kedro_model(
        model, edge_index, edge_type, test_triples,
        all_triples, device, batch_size=args.batch_size,
        compute_mrr=args.compute_mrr,
        compute_hits=args.compute_hits,
        hit_k_values=args.hit_k_values
    )

    # Save metrics
    with open(args.metrics_save_path, 'wb') as f:
        pickle.dump({
            'metrics': test_metrics,
            'model_config': model_config,
        }, f)
    print(f"\n✓ Saved metrics to {args.metrics_save_path}")

    # Final summary
    print("\n" + "="*60)
    print("FINAL RESULTS SUMMARY")
    print("="*60)
    print("\nTest Metrics:")
    for metric, value in test_metrics.items():
        print(f"  {metric:12s}: {value:.4f}")

    print("\n" + "="*60)
    print("EVALUATION COMPLETE!")
    print("="*60)


if __name__ == '__main__':
    main()
