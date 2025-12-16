"""Nodes for training pipeline."""

import logging
import torch
import torch.nn.functional as F
import torch.optim as optim
from typing import Dict
from .model import RGCNDistMultModel
from .kg_models import CompGCNKGModel
from ..utils import generate_negative_samples
from gnn_explainer.utils.mlflow_utils import (
    log_params_from_dict,
    log_training_metrics,
    is_mlflow_enabled,
)

logger = logging.getLogger(__name__)


def train_model(
    pyg_data: Dict,
    knowledge_graph: Dict,
    model_params: Dict,
    training_params: Dict,
    device_str: str
) -> Dict:
    """
    Train KG embedding model with early stopping.

    Supports multiple model architectures:
    - RGCN + DistMult
    - CompGCN + (DistMult | ComplEx | RotatE | ConvE)

    Args:
        pyg_data: PyG format graph data
        knowledge_graph: Knowledge graph with dictionaries
        model_params: Model hyperparameters
        training_params: Training configuration
        device_str: Device string ("cuda" or "cpu")

    Returns:
        Dictionary with trained model state and metadata
    """
    print("\n" + "="*60)
    print("STARTING MODEL TRAINING")
    print("="*60)

    device = torch.device(device_str if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # Get model configuration
    model_type = model_params.get('model_type', 'rgcn')
    decoder_type = model_params.get('decoder_type', 'distmult')

    # Initialize model based on type
    print(f"\nInitializing {model_type.upper()} model...")
    print(f"  Model type: {model_type}")
    print(f"  Decoder type: {decoder_type}")
    print(f"  Embedding dim: {model_params['embedding_dim']}")
    print(f"  Num layers: {model_params['num_layers']}")
    print(f"  Dropout: {model_params['dropout']}")

    if model_type == 'compgcn':
        # CompGCN with selected decoder
        print(f"  Composition function: {model_params.get('comp_fn', 'sub')}")

        # ConvE-specific parameters
        conve_kwargs = {}
        if decoder_type == 'conve':
            conve_kwargs = {
                'input_drop': model_params.get('conve_input_drop', 0.2),
                'hidden_drop': model_params.get('conve_hidden_drop', 0.3),
                'feature_drop': model_params.get('conve_feature_drop', 0.2),
                'num_filters': model_params.get('conve_num_filters', 32),
                'kernel_size': model_params.get('conve_kernel_size', 3),
            }

        model = CompGCNKGModel(
            num_nodes=knowledge_graph['num_nodes'],
            num_relations=knowledge_graph['num_relations'],
            embedding_dim=model_params['embedding_dim'],
            decoder_type=decoder_type,
            num_layers=model_params['num_layers'],
            comp_fn=model_params.get('comp_fn', 'sub'),
            dropout=model_params['dropout'],
            conve_kwargs=conve_kwargs if conve_kwargs else None
        ).to(device)

    elif model_type == 'rgcn':
        # Original RGCN + DistMult
        print(f"  Num bases: {model_params.get('num_bases', 30)}")

        model = RGCNDistMultModel(
            num_nodes=knowledge_graph['num_nodes'],
            num_relations=knowledge_graph['num_relations'],
            embedding_dim=model_params['embedding_dim'],
            num_layers=model_params['num_layers'],
            num_bases=model_params.get('num_bases', 30),
        dropout=model_params['dropout']
    ).to(device)

    else:
        raise ValueError(f"Unknown model_type: {model_type}. Choose 'rgcn' or 'compgcn'")

    # Count parameters
    num_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"  Total parameters: {num_params:,}")

    # Log model parameters to MLflow
    if is_mlflow_enabled():
        log_params_from_dict(model_params, prefix="model")
        log_params_from_dict(training_params, prefix="training")

    # Optimizer
    optimizer = optim.Adam(
        model.parameters(),
        lr=training_params['learning_rate'],
        weight_decay=training_params.get('weight_decay', 0.0)
    )

    # Move graph data to device
    edge_index = pyg_data['edge_index'].to(device)
    edge_type = pyg_data['edge_type'].to(device)
    train_triples = pyg_data['train_triples']
    val_triples = pyg_data['val_triples']

    # Training loop
    print(f"\nStarting training for {training_params['num_epochs']} epochs...")
    print(f"  Learning rate: {training_params['learning_rate']}")
    print(f"  Batch size: {training_params['batch_size']}")
    print(f"  Patience: {training_params['patience']}")

    best_val_loss = float('inf')
    best_model_state = None
    patience_counter = 0
    batch_size = training_params['batch_size']
    num_epochs = training_params['num_epochs']
    patience = training_params['patience']
    gradient_clip = training_params.get('gradient_clip', None)

    for epoch in range(num_epochs):
        # Train epoch
        model.train()
        total_loss = 0
        num_batches = 0

        # Generate negative samples
        neg_triples = generate_negative_samples(
            train_triples,
            knowledge_graph['num_nodes'],
            num_negatives=5
        )

        # Combine positive and negative
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
        total_batches = (len(all_triples) + batch_size - 1) // batch_size
        for batch_idx, i in enumerate(range(0, len(all_triples), batch_size), 1):
            batch_triples = all_triples[i:i+batch_size].to(device)
            batch_labels = labels[i:i+batch_size]

            optimizer.zero_grad()

            scores = model(
                edge_index, edge_type,
                batch_triples[:, 0],
                batch_triples[:, 2],
                batch_triples[:, 1]
            )

            loss = F.binary_cross_entropy_with_logits(scores, batch_labels)
            loss.backward()

            if gradient_clip:
                torch.nn.utils.clip_grad_norm_(model.parameters(), gradient_clip)

            optimizer.step()

            total_loss += loss.item()
            num_batches += 1

            # Print batch progress every 10 batches or on last batch
            if batch_idx % 10 == 0 or batch_idx == total_batches:
                avg_loss = total_loss / num_batches
                msg = f"  Epoch {epoch+1}/{num_epochs} - Batch {batch_idx}/{total_batches} - Avg Loss: {avg_loss:.4f}"
                logger.info(msg)
                print(msg, flush=True)  # Also print to ensure visibility

        train_loss = total_loss / num_batches

        # Validation
        model.eval()
        val_loss = 0
        val_batches = 0

        with torch.no_grad():
            # Generate validation negatives
            val_neg_triples = generate_negative_samples(
                val_triples,
                knowledge_graph['num_nodes'],
                num_negatives=5
            )

            val_all_triples = torch.cat([val_triples, val_neg_triples], dim=0)
            val_labels = torch.cat([
                torch.ones(len(val_triples)),
                torch.zeros(len(val_neg_triples))
            ]).to(device)

            for i in range(0, len(val_all_triples), batch_size):
                batch_triples = val_all_triples[i:i+batch_size].to(device)
                batch_labels = val_labels[i:i+batch_size]

                scores = model(
                    edge_index, edge_type,
                    batch_triples[:, 0],
                    batch_triples[:, 2],
                    batch_triples[:, 1]
                )

                loss = F.binary_cross_entropy_with_logits(scores, batch_labels)
                val_loss += loss.item()
                val_batches += 1

        val_loss = val_loss / val_batches

        # Log metrics to MLflow
        if is_mlflow_enabled():
            log_training_metrics(
                epoch=epoch + 1,
                train_loss=train_loss,
                val_loss=val_loss,
            )

        # Print progress every epoch
        msg = f"Epoch {epoch+1:3d}/{num_epochs}: Train Loss = {train_loss:.4f}, Val Loss = {val_loss:.4f}"
        logger.info(msg)
        print(msg, flush=True)

        # Early stopping
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_model_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
            patience_counter = 0
            msg = f"  ✓ New best model (val_loss: {val_loss:.4f})"
            logger.info(msg)
            print(msg, flush=True)
        else:
            patience_counter += 1

        if patience_counter >= patience:
            print(f"\nEarly stopping triggered after {epoch+1} epochs")
            break

    print("\n" + "="*60)
    print("TRAINING COMPLETED")
    print("="*60)
    print(f"Best validation loss: {best_val_loss:.4f}")
    print(f"Total epochs trained: {epoch+1}")

    return {
        'model_state_dict': best_model_state,
        'model_config': {
            'num_nodes': knowledge_graph['num_nodes'],
            'num_relations': knowledge_graph['num_relations'],
            **model_params
        },
        'training_info': {
            'final_val_loss': best_val_loss,
            'num_epochs_trained': epoch + 1,
            'best_epoch': epoch + 1 - patience_counter,
        }
    }


def compute_test_scores(
    trained_model_artifact: Dict,
    pyg_data: Dict,
    device_str: str,
    batch_size: int = 1024
) -> Dict:
    """
    Compute prediction scores for test triples.

    Args:
        trained_model_artifact: Trained model artifact from training
        pyg_data: PyG format graph data
        device_str: Device string ("cuda" or "cpu")
        batch_size: Batch size for scoring

    Returns:
        Dictionary with test triples and their prediction scores
    """
    print("\n" + "="*60)
    print("COMPUTING TEST TRIPLE SCORES")
    print("="*60)

    device = torch.device(device_str if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # Extract model configuration
    model_state_dict = trained_model_artifact['model_state_dict']
    model_config = trained_model_artifact['model_config']

    # Reconstruct model
    model_type = model_config.get('model_type', 'rgcn')
    decoder_type = model_config.get('decoder_type', 'distmult')

    print(f"\nLoading model...")
    print(f"  Model type: {model_type}")
    print(f"  Decoder type: {decoder_type}")

    if model_type == 'compgcn':
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

    # Move graph data to device
    edge_index = pyg_data['edge_index'].to(device)
    edge_type = pyg_data['edge_type'].to(device)
    test_triples = pyg_data['test_triples']

    print(f"\nScoring {len(test_triples)} test triples...")

    # Compute scores
    all_scores = []
    num_batches = (len(test_triples) + batch_size - 1) // batch_size

    with torch.no_grad():
        for batch_idx, i in enumerate(range(0, len(test_triples), batch_size), 1):
            batch_triples = test_triples[i:i+batch_size].to(device)

            scores = model(
                edge_index, edge_type,
                batch_triples[:, 0],  # heads
                batch_triples[:, 2],  # tails
                batch_triples[:, 1]   # relations
            )

            all_scores.extend(scores.cpu().tolist())

            # Print progress every 10 batches
            if batch_idx % 10 == 0 or batch_idx == num_batches:
                print(f"  Processed {batch_idx}/{num_batches} batches", flush=True)

    # Convert to tensor
    all_scores_tensor = torch.tensor(all_scores)

    print(f"\n✓ Scoring complete")
    print(f"  Total triples scored: {len(all_scores)}")
    print(f"  Score range: [{all_scores_tensor.min():.4f}, {all_scores_tensor.max():.4f}]")
    print(f"  Mean score: {all_scores_tensor.mean():.4f}")

    print("\n" + "="*60)

    return {
        'test_triples': test_triples.cpu(),
        'scores': all_scores_tensor,
        'model_type': model_type,
        'decoder_type': decoder_type,
    }
