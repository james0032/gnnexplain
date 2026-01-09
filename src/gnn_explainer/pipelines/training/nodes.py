"""Nodes for training pipeline."""

import logging
import torch
import torch.nn.functional as F
import torch.optim as optim
import dgl
from typing import Dict
from .model import RGCNDistMultModel
from .kg_models import CompGCNKGModel
from .kg_models_dgl import CompGCNKGModelDGL
from ..utils import generate_negative_samples
from gnn_explainer.utils.mlflow_utils import (
    log_params_from_dict,
    log_training_metrics,
    is_mlflow_enabled,
)

logger = logging.getLogger(__name__)


def train_model(
    dgl_data: Dict = None,
    pyg_data: Dict = None,
    knowledge_graph: Dict = None,
    model_params: Dict = None,
    training_params: Dict = None,
    device_str: str = "cuda"
) -> Dict:
    """
    Train KG embedding model with early stopping.

    Supports multiple model architectures:
    - RGCN + DistMult (PyG only)
    - CompGCN + (DistMult | ComplEx | RotatE | ConvE) (DGL or PyG)

    Args:
        dgl_data: DGL format graph data (preferred)
        pyg_data: PyG format graph data (legacy, for backward compatibility)
        knowledge_graph: Knowledge graph with dictionaries
        model_params: Model hyperparameters
        training_params: Training configuration
        device_str: Device string ("cuda" or "cpu")

    Returns:
        Dictionary with trained model state and metadata
    """
    # Determine which data format to use
    use_dgl = dgl_data is not None
    graph_data = dgl_data if use_dgl else pyg_data

    if graph_data is None:
        raise ValueError("Either dgl_data or pyg_data must be provided")
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
        print(f"  Using {'DGL' if use_dgl else 'PyG'} backend")

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

        # Choose DGL or PyG model
        ModelClass = CompGCNKGModelDGL if use_dgl else CompGCNKGModel

        model = ModelClass(
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
    if use_dgl:
        # DGL graph
        g = graph_data['graph'].to(device)
        edge_index = graph_data['edge_index'].to(device)
        edge_type = graph_data['edge_type'].to(device)
    else:
        # PyG format
        edge_index = graph_data['edge_index'].to(device)
        edge_type = graph_data['edge_type'].to(device)
        g = None

    train_triples = graph_data['train_triples']
    val_triples = graph_data['val_triples']

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

            # Forward pass (DGL or PyG)
            if use_dgl:
                scores = model(
                    g=g,
                    head_idx=batch_triples[:, 0],
                    tail_idx=batch_triples[:, 2],
                    rel_idx=batch_triples[:, 1]
                )
            else:
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

                # Forward pass (DGL or PyG)
                if use_dgl:
                    scores = model(
                        g=g,
                        head_idx=batch_triples[:, 0],
                        tail_idx=batch_triples[:, 2],
                        rel_idx=batch_triples[:, 1]
                    )
                else:
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
    dgl_data: Dict = None,
    pyg_data: Dict = None,
    knowledge_graph: Dict = None,
    device_str: str = "cuda",
    batch_size: int = 1024
) -> Dict:
    """
    Compute prediction scores for test triples.

    Args:
        trained_model_artifact: Trained model artifact from training
        dgl_data: DGL format graph data (preferred)
        pyg_data: PyG format graph data (legacy)
        knowledge_graph: Knowledge graph with dictionaries
        device_str: Device string ("cuda" or "cpu")
        batch_size: Batch size for scoring

    Returns:
        Dictionary with test triples and their prediction scores
    """
    # Determine which data format to use
    use_dgl = dgl_data is not None
    graph_data = dgl_data if use_dgl else pyg_data

    if graph_data is None:
        raise ValueError("Either dgl_data or pyg_data must be provided")
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

        # Choose DGL or PyG model
        ModelClass = CompGCNKGModelDGL if use_dgl else CompGCNKGModel

        model = ModelClass(
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
    if use_dgl:
        # DGL graph
        g = graph_data['graph'].to(device)
        edge_index = graph_data['edge_index'].to(device)
        edge_type = graph_data['edge_type'].to(device)
    else:
        # PyG format
        edge_index = graph_data['edge_index'].to(device)
        edge_type = graph_data['edge_type'].to(device)
        g = None

    test_triples = graph_data['test_triples']

    print(f"\nScoring {len(test_triples)} test triples...")

    # Compute scores
    all_scores = []
    num_batches = (len(test_triples) + batch_size - 1) // batch_size

    with torch.no_grad():
        for batch_idx, i in enumerate(range(0, len(test_triples), batch_size), 1):
            batch_triples = test_triples[i:i+batch_size].to(device)

            # Forward pass (DGL or PyG)
            if use_dgl:
                scores = model(
                    g=g,
                    head_idx=batch_triples[:, 0],
                    tail_idx=batch_triples[:, 2],
                    rel_idx=batch_triples[:, 1]
                )
            else:
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

    # Convert to tensor and apply sigmoid
    all_scores_tensor = torch.tensor(all_scores)
    sigmoid_scores = torch.sigmoid(all_scores_tensor)

    print(f"\n✓ Scoring complete")
    print(f"  Total triples scored: {len(all_scores)}")
    print(f"  Raw score range: [{all_scores_tensor.min():.4f}, {all_scores_tensor.max():.4f}]")
    print(f"  Sigmoid score range: [{sigmoid_scores.min():.4f}, {sigmoid_scores.max():.4f}]")
    print(f"  Mean sigmoid score: {sigmoid_scores.mean():.4f}")

    # Create CSV DataFrame with entity/relation names
    import pandas as pd

    # Get mapping dictionaries from knowledge_graph
    # First try to get pre-computed reverse mappings
    idx_to_entity = knowledge_graph.get('idx_to_entity', {})
    idx_to_relation = knowledge_graph.get('idx_to_relation', {})

    # If reverse mappings don't exist, create them from node_dict and rel_dict
    if not idx_to_entity or not idx_to_relation:
        print("Creating reverse mappings from node_dict and rel_dict...")
        node_dict = knowledge_graph.get('node_dict', {})
        rel_dict = knowledge_graph.get('rel_dict', {})

        # Create reverse mappings: index -> name
        idx_to_entity = {v: k for k, v in node_dict.items()}
        idx_to_relation = {v: k for k, v in rel_dict.items()}

        print(f"  Created idx_to_entity with {len(idx_to_entity)} entities")
        print(f"  Created idx_to_relation with {len(idx_to_relation)} relations")
    else:
        print(f"Using pre-computed reverse mappings:")
        print(f"  idx_to_entity: {len(idx_to_entity)} entities")
        print(f"  idx_to_relation: {len(idx_to_relation)} relations")

    # Create lists for entity and relation names
    head_ids = []
    head_names = []
    relation_names = []
    tail_ids = []
    tail_names = []

    for i in range(len(test_triples)):
        head_idx = test_triples[i, 0].item()
        rel_idx = test_triples[i, 1].item()
        tail_idx = test_triples[i, 2].item()

        head_ids.append(head_idx)
        tail_ids.append(tail_idx)
        head_names.append(idx_to_entity.get(head_idx, f"node_{head_idx}"))
        tail_names.append(idx_to_entity.get(tail_idx, f"node_{tail_idx}"))
        relation_names.append(idx_to_relation.get(rel_idx, f"rel_{rel_idx}"))

    df = pd.DataFrame({
        'head_id': head_ids,
        'head': head_names,
        'relation': relation_names,
        'tail_id': tail_ids,
        'tail': tail_names,
        'raw_score': all_scores_tensor.tolist(),
        'sigmoid_score': sigmoid_scores.tolist(),
    })

    # Sort by sigmoid_score in descending order (highest scores first)
    df = df.sort_values('sigmoid_score', ascending=False).reset_index(drop=True)

    print(f"\n✓ Created DataFrame with {len(df)} rows")
    print(f"  Sorted by sigmoid_score (descending)")
    print(f"\nTop 5 predicted triples:")
    for i in range(min(5, len(df))):
        print(f"  {i+1}. ({df.iloc[i]['head']}, {df.iloc[i]['relation']}, {df.iloc[i]['tail']}) - score: {df.iloc[i]['sigmoid_score']:.4f}")

    # Create top 10 triples string in test file format (tab-separated: head\trelation\ttail)
    top10_df = df.head(10)
    top10_lines = []

    for i in range(len(top10_df)):
        head = top10_df.iloc[i]['head']
        relation = top10_df.iloc[i]['relation']
        tail = top10_df.iloc[i]['tail']
        top10_lines.append(f"{head}\t{relation}\t{tail}")

    top10_text = "\n".join(top10_lines)

    print(f"\n✓ Created top 10 triples file")
    print("  (Tab-separated format: head\\trelation\\ttail)")
    print("\n" + "="*60)

    # Return pickle data, CSV DataFrame, and top10 text
    scores_dict = {
        'test_triples': test_triples.cpu(),
        'scores': all_scores_tensor,
        'sigmoid_scores': sigmoid_scores,
        'model_type': model_type,
        'decoder_type': decoder_type,
    }

    return scores_dict, df, top10_text
