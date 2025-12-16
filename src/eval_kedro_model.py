"""
Evaluation Script for Kedro-trained GNN Models

This script evaluates models trained via the Kedro pipeline.
It handles the Kedro model artifact format (pickle with model_state_dict and model_config).

Usage:
    python src/eval_kedro_model.py --model_path data/06_models/trained_model.pkl
"""

import torch
import pickle
import argparse
import sys
from pathlib import Path

# Import the evaluation logic from cl_eval
from cl_eval import (
    evaluate,
    KGDataLoader,
)

# Import model classes
sys.path.append(str(Path(__file__).parent / "gnn_explainer" / "pipelines" / "training"))
from gnn_explainer.pipelines.training.model import RGCNDistMultModel
from gnn_explainer.pipelines.training.kg_models import CompGCNKGModel


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
    test_metrics = evaluate(
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
