"""Nodes for training pipeline."""

import logging
import torch
import torch.nn.functional as F
import torch.optim as optim
import numpy as np
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

        model_kwargs = dict(
            num_nodes=knowledge_graph['num_nodes'],
            num_relations=knowledge_graph['num_relations'],
            embedding_dim=model_params['embedding_dim'],
            decoder_type=decoder_type,
            num_layers=model_params['num_layers'],
            comp_fn=model_params.get('comp_fn', 'sub'),
            dropout=model_params['dropout'],
            conve_kwargs=conve_kwargs if conve_kwargs else None,
        )
        if not use_dgl:
            # Gradient checkpointing for PyG encoder (bounds GPU memory)
            model_kwargs['use_checkpoint'] = True

        model = ModelClass(**model_kwargs).to(device)

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

    # Separate optimizers for encoder (stepped once/epoch) and decoder (stepped per batch)
    lr = training_params['learning_rate']
    wd = training_params.get('weight_decay', 0.0)
    enc_optimizer = optim.Adam(model.encoder.parameters(), lr=lr, weight_decay=wd)
    dec_optimizer = optim.Adam(model.decoder.parameters(), lr=lr, weight_decay=wd)

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

    # Checkpoint configuration
    from pathlib import Path
    import os

    # Check for DATA_DIR environment variable first, then fall back to project root
    # DATA_DIR allows specifying an external data directory (e.g., on server)
    data_dir_env = os.environ.get('DATA_DIR')
    if data_dir_env:
        base_dir = Path(data_dir_env)
        print(f"  Using DATA_DIR from environment: {base_dir}")
    else:
        # Fall back to project root for local development
        # nodes.py is at: src/gnn_explainer/pipelines/training/nodes.py
        # project root is 4 levels up
        base_dir = Path(__file__).resolve().parents[4]
        print(f"  Using project root (DATA_DIR not set): {base_dir}")

    checkpoint_interval = training_params.get('checkpoint_interval', 2)
    checkpoint_dir_config = training_params.get('checkpoint_dir', 'data/04_model_checkpoints')

    # Resolve checkpoint directory path
    # If checkpoint_dir_config is absolute, use it directly
    # If relative and starts with "data/", strip "data/" prefix when DATA_DIR is set
    if Path(checkpoint_dir_config).is_absolute():
        checkpoint_dir = Path(checkpoint_dir_config)
    elif data_dir_env and checkpoint_dir_config.startswith('data/'):
        # Strip "data/" prefix since DATA_DIR already points to data folder
        checkpoint_dir = base_dir / checkpoint_dir_config[5:]  # Remove "data/" prefix
    else:
        checkpoint_dir = base_dir / checkpoint_dir_config

    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    checkpoint_path = checkpoint_dir / 'compgcn_training_checkpoint.pt'
    print(f"  Checkpoint directory: {checkpoint_dir}")
    print(f"  Checkpoint interval: every {checkpoint_interval} epochs")

    # Check for existing checkpoint to resume from
    start_epoch = 0
    if checkpoint_path.exists():
        print(f"\n  Found existing checkpoint at {checkpoint_path}")
        try:
            checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
            # Load current model state (for resuming training)
            # Support both old format ('model_state_dict') and new format ('current_model_state_dict')
            current_state = checkpoint.get('current_model_state_dict', checkpoint.get('model_state_dict'))
            model.load_state_dict(current_state)
            if 'enc_optimizer_state_dict' in checkpoint:
                enc_optimizer.load_state_dict(checkpoint['enc_optimizer_state_dict'])
                dec_optimizer.load_state_dict(checkpoint['dec_optimizer_state_dict'])
            else:
                # Legacy single-optimizer checkpoint — skip optimizer restore
                print("  (Legacy checkpoint: optimizer state skipped)")
            start_epoch = checkpoint['epoch'] + 1
            best_val_loss = checkpoint.get('best_val_loss', float('inf'))
            best_model_state = checkpoint.get('best_model_state', None)
            patience_counter = checkpoint.get('patience_counter', 0)
            print(f"  ✓ Resuming from epoch {start_epoch}/{num_epochs}")
            print(f"  Best val_loss so far: {best_val_loss:.4f}")
        except Exception as e:
            print(f"  ⚠ Failed to load checkpoint: {e}")
            print(f"  Starting from scratch...")

    for epoch in range(start_epoch, num_epochs):
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

        # === Two-phase training with gradient checkpointing ===
        #
        # Phase 1 (decoder): Encode once with no_grad. For each mini-batch,
        #   compute decoder loss, backward, step decoder optimizer — so
        #   decoder params (ConvE) get updated at full learning rate every
        #   batch. Embedding gradients accumulate across all batches.
        #
        # Phase 2 (encoder): Re-run encode WITH grad (using checkpointing
        #   to bound memory), backward using accumulated embedding gradients.
        #   Step encoder optimizer once per epoch.

        # Phase 1: Encode once, decode per batch
        with torch.no_grad():
            if use_dgl:
                node_emb, rel_emb = model.encode(g=g)
            else:
                node_emb, rel_emb = model.encode(edge_index, edge_type)

        node_emb_leaf = node_emb.detach().requires_grad_(True)
        rel_emb_leaf = rel_emb.detach().requires_grad_(True)

        total_batches = (len(all_triples) + batch_size - 1) // batch_size
        for batch_idx, i in enumerate(range(0, len(all_triples), batch_size), 1):
            batch_triples = all_triples[i:i+batch_size].to(device)
            batch_labels = labels[i:i+batch_size]

            dec_optimizer.zero_grad()

            scores = model.decode(
                node_emb_leaf, rel_emb_leaf,
                batch_triples[:, 0],
                batch_triples[:, 2],
                batch_triples[:, 1]
            )

            loss = F.binary_cross_entropy_with_logits(scores, batch_labels)
            loss.backward()

            # Step decoder only; embedding leaf grads accumulate
            if gradient_clip:
                torch.nn.utils.clip_grad_norm_(model.decoder.parameters(), gradient_clip)
            dec_optimizer.step()

            total_loss += loss.item()
            num_batches += 1

            if batch_idx % 10 == 0 or batch_idx == total_batches:
                avg_loss = total_loss / num_batches
                msg = f"  Epoch {epoch+1}/{num_epochs} - Batch {batch_idx}/{total_batches} - Avg Loss: {avg_loss:.4f}"
                logger.info(msg)
                print(msg, flush=True)

        # Phase 2: Backward through encoder with gradient checkpointing
        # Average the accumulated embedding gradients across batches
        if node_emb_leaf.grad is not None:
            node_emb_leaf.grad.div_(total_batches)
        if rel_emb_leaf.grad is not None:
            rel_emb_leaf.grad.div_(total_batches)

        if node_emb_leaf.grad is not None or rel_emb_leaf.grad is not None:
            enc_optimizer.zero_grad()

            if use_dgl:
                node_emb_ckpt, rel_emb_ckpt = model.encode(g=g)
            else:
                node_emb_ckpt, rel_emb_ckpt = model.encode(edge_index, edge_type)

            node_grad = node_emb_leaf.grad if node_emb_leaf.grad is not None else torch.zeros_like(node_emb_ckpt)
            rel_grad = rel_emb_leaf.grad if rel_emb_leaf.grad is not None else torch.zeros_like(rel_emb_ckpt)
            torch.autograd.backward(
                [node_emb_ckpt, rel_emb_ckpt],
                [node_grad, rel_grad]
            )

            if gradient_clip:
                torch.nn.utils.clip_grad_norm_(model.encoder.parameters(), gradient_clip)
            enc_optimizer.step()

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

            # Encode once for validation
            if use_dgl:
                val_node_emb, val_rel_emb = model.encode(g=g)
            else:
                val_node_emb, val_rel_emb = model.encode(edge_index, edge_type)

            for i in range(0, len(val_all_triples), batch_size):
                batch_triples = val_all_triples[i:i+batch_size].to(device)
                batch_labels = val_labels[i:i+batch_size]

                scores = model.decode(
                    val_node_emb, val_rel_emb,
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

        # Save checkpoint every checkpoint_interval epochs
        if (epoch + 1) % checkpoint_interval == 0:
            # Build model_config compatible with both compute_test_scores and explanation pipeline
            model_config_for_checkpoint = {
                'num_nodes': knowledge_graph['num_nodes'],
                'num_relations': knowledge_graph['num_relations'],
                **model_params
            }
            # Add conve_kwargs for explanation pipeline compatibility
            if model_params.get('decoder_type') == 'conve':
                model_config_for_checkpoint['conve_kwargs'] = {
                    'input_drop': model_params.get('conve_input_drop', 0.2),
                    'hidden_drop': model_params.get('conve_hidden_drop', 0.3),
                    'feature_drop': model_params.get('conve_feature_drop', 0.2),
                    'num_filters': model_params.get('conve_num_filters', 32),
                    'kernel_size': model_params.get('conve_kernel_size', 3),
                }

            checkpoint = {
                'epoch': epoch,
                'current_model_state_dict': model.state_dict(),  # Current state for resume
                'enc_optimizer_state_dict': enc_optimizer.state_dict(),
                'dec_optimizer_state_dict': dec_optimizer.state_dict(),
                'best_val_loss': best_val_loss,
                'best_model_state': best_model_state,
                'patience_counter': patience_counter,
                'train_loss': train_loss,
                'val_loss': val_loss,
                'model_config': model_config_for_checkpoint,
                'training_params': training_params,
                # Also save in trained_model_artifact format for direct use with compute_test_scores
                'model_state_dict': best_model_state,  # Best model for inference
            }
            # Save versioned checkpoint (by epoch number)
            versioned_checkpoint_path = checkpoint_dir / f'compgcn_checkpoint_epoch_{epoch+1:04d}.pt'
            torch.save(checkpoint, versioned_checkpoint_path)
            # Also save as 'latest' for easy resume
            torch.save(checkpoint, checkpoint_path)
            print(f"  [Checkpoint saved: {versioned_checkpoint_path.name} and {checkpoint_path.name}]", flush=True)

        if patience_counter >= patience:
            print(f"\nEarly stopping triggered after {epoch+1} epochs")
            break

    print("\n" + "="*60)
    print("TRAINING COMPLETED")
    print("="*60)
    print(f"Best validation loss: {best_val_loss:.4f}")
    print(f"Total epochs trained: {epoch+1}")

    # Build model_config compatible with both compute_test_scores and explanation pipeline
    final_model_config = {
        'num_nodes': knowledge_graph['num_nodes'],
        'num_relations': knowledge_graph['num_relations'],
        **model_params
    }
    # Add conve_kwargs for explanation pipeline compatibility
    if model_params.get('decoder_type') == 'conve':
        final_model_config['conve_kwargs'] = {
            'input_drop': model_params.get('conve_input_drop', 0.2),
            'hidden_drop': model_params.get('conve_hidden_drop', 0.3),
            'feature_drop': model_params.get('conve_feature_drop', 0.2),
            'num_filters': model_params.get('conve_num_filters', 32),
            'kernel_size': model_params.get('conve_kernel_size', 3),
        }

    return {
        'model_state_dict': best_model_state,
        'model_config': final_model_config,
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
    batch_size: int = 1024,
    top_k_triples: int = 10,
    custom_test_file: str = None
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
        top_k_triples: Number of top triples to output to file (default: 10)
        custom_test_file: Optional path to custom test file (tab-separated: head\trelation\ttail)
                         If provided, scores triples from this file instead of pyg_data['test_triples']

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

    # Determine output file prefix from custom_test_file
    output_prefix = None
    if custom_test_file:
        from pathlib import Path
        output_prefix = Path(custom_test_file).stem  # filename without extension
        print(f"Custom test file: {custom_test_file}")
        print(f"Output prefix: {output_prefix}")

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

    # Load test triples from custom file or use default from graph data
    if custom_test_file:
        print(f"\nLoading test triples from custom file: {custom_test_file}")
        # Get entity and relation mappings from knowledge_graph
        node_dict = knowledge_graph.get('node_dict', {})
        rel_dict = knowledge_graph.get('rel_dict', {})

        # Parse custom test file (tab-separated: head\trelation\ttail)
        custom_triples = []
        skipped = 0
        with open(custom_test_file, 'r') as f:
            for line_num, line in enumerate(f, 1):
                line = line.strip()
                if not line:
                    continue
                parts = line.split('\t')
                if len(parts) != 3:
                    print(f"  Warning: Skipping line {line_num}, expected 3 tab-separated values, got {len(parts)}")
                    skipped += 1
                    continue
                head, relation, tail = parts

                # Convert names to indices
                head_idx = node_dict.get(head)
                rel_idx = rel_dict.get(relation)
                tail_idx = node_dict.get(tail)

                if head_idx is None:
                    print(f"  Warning: Unknown entity '{head}' at line {line_num}")
                    skipped += 1
                    continue
                if rel_idx is None:
                    print(f"  Warning: Unknown relation '{relation}' at line {line_num}")
                    skipped += 1
                    continue
                if tail_idx is None:
                    print(f"  Warning: Unknown entity '{tail}' at line {line_num}")
                    skipped += 1
                    continue

                custom_triples.append([head_idx, rel_idx, tail_idx])

        if not custom_triples:
            raise ValueError(f"No valid triples found in {custom_test_file}")

        test_triples = torch.tensor(custom_triples, dtype=torch.long)
        print(f"  Loaded {len(test_triples)} triples from custom file")
        if skipped > 0:
            print(f"  Skipped {skipped} invalid lines")
    else:
        test_triples = graph_data['test_triples']

    num_triples = len(test_triples)

    print(f"\nScoring {num_triples:,} test triples...")
    print(f"  Batch size: {batch_size:,}")

    # Pre-allocate tensor for efficiency (important for 30M+ rows)
    all_scores_tensor = torch.zeros(num_triples, dtype=torch.float32)
    num_batches = (num_triples + batch_size - 1) // batch_size

    # Progress reporting interval - adaptive based on number of batches
    # For 30M rows with batch_size=8192: ~3,662 batches -> report every ~366 batches (10 reports)
    progress_interval = max(1, num_batches // 10)

    with torch.no_grad():
        for batch_idx, i in enumerate(range(0, num_triples, batch_size), 1):
            batch_end = min(i + batch_size, num_triples)
            batch_triples = test_triples[i:batch_end].to(device)

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

            # Store directly in pre-allocated tensor (avoids list append overhead)
            all_scores_tensor[i:batch_end] = scores.cpu()

            # Print progress at adaptive intervals (roughly 10 progress reports)
            if batch_idx % progress_interval == 0 or batch_idx == num_batches:
                pct = 100.0 * batch_idx / num_batches
                print(f"  Processed {batch_idx:,}/{num_batches:,} batches ({pct:.1f}%)", flush=True)

    # Apply sigmoid
    sigmoid_scores = torch.sigmoid(all_scores_tensor)

    print(f"\n✓ Scoring complete")
    print(f"  Total triples scored: {num_triples:,}")
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

    # Vectorized DataFrame creation (much faster for 30M+ rows)
    print(f"\nCreating DataFrame (vectorized for {num_triples:,} rows)...")

    # Extract indices as numpy arrays (fast)
    head_ids = test_triples[:, 0].numpy()
    rel_ids = test_triples[:, 1].numpy()
    tail_ids = test_triples[:, 2].numpy()

    # Vectorized name lookup using numpy vectorize (faster than Python loop)
    def get_entity_name(idx):
        return idx_to_entity.get(int(idx), f"node_{idx}")

    def get_relation_name(idx):
        return idx_to_relation.get(int(idx), f"rel_{idx}")

    vec_entity_lookup = np.vectorize(get_entity_name)
    vec_relation_lookup = np.vectorize(get_relation_name)

    print(f"  Mapping entity names...")
    head_names = vec_entity_lookup(head_ids)
    tail_names = vec_entity_lookup(tail_ids)

    print(f"  Mapping relation names...")
    relation_names = vec_relation_lookup(rel_ids)

    df = pd.DataFrame({
        'head_id': head_ids,
        'head': head_names,
        'relation': relation_names,
        'tail_id': tail_ids,
        'tail': tail_names,
        'raw_score': all_scores_tensor.numpy(),
        'sigmoid_score': sigmoid_scores.numpy(),
    })

    # Sort by sigmoid_score in descending order (highest scores first)
    df = df.sort_values('sigmoid_score', ascending=False).reset_index(drop=True)

    print(f"\n✓ Created DataFrame with {len(df)} rows")
    print(f"  Sorted by sigmoid_score (descending)")
    print(f"\nTop 5 predicted triples:")
    for i in range(min(5, len(df))):
        print(f"  {i+1}. ({df.iloc[i]['head']}, {df.iloc[i]['relation']}, {df.iloc[i]['tail']}) - score: {df.iloc[i]['sigmoid_score']:.4f}")

    # Create top-k triples string in test file format (tab-separated: head\trelation\ttail)
    topk_df = df.head(top_k_triples)
    topk_lines = []

    for i in range(len(topk_df)):
        head = topk_df.iloc[i]['head']
        relation = topk_df.iloc[i]['relation']
        tail = topk_df.iloc[i]['tail']
        topk_lines.append(f"{head}\t{relation}\t{tail}")

    topk_text = "\n".join(topk_lines)

    print(f"\n✓ Created top {top_k_triples} triples file")
    print("  (Tab-separated format: head\\trelation\\ttail)")
    if output_prefix:
        print(f"  Output prefix: {output_prefix}")
    print("\n" + "="*60)

    # Return pickle data, CSV DataFrame, and topk text
    scores_dict = {
        'test_triples': test_triples.cpu() if hasattr(test_triples, 'cpu') else test_triples,
        'scores': all_scores_tensor,
        'sigmoid_scores': sigmoid_scores,
        'model_type': model_type,
        'decoder_type': decoder_type,
        'output_prefix': output_prefix,  # For custom output file naming
    }

    return scores_dict, df, topk_text
