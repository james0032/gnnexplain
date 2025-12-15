"""Nodes for training pipeline."""

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
        for i in range(0, len(all_triples), batch_size):
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

        # Print progress
        if (epoch + 1) % 10 == 0 or epoch == 0:
            print(f"Epoch {epoch+1:3d}/{num_epochs}: "
                  f"Train Loss = {train_loss:.4f}, "
                  f"Val Loss = {val_loss:.4f}")

        # Early stopping
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_model_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
            patience_counter = 0
            print(f"  ✓ New best model (val_loss: {val_loss:.4f})")
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
