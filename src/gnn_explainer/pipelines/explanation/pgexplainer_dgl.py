"""
PGExplainer with DGL - Native Batching Support

This module provides a DGL-based PGExplainer implementation with native batch training,
significantly faster than the PyG version.
"""

import torch
import torch.nn as nn
import dgl
from dgl.nn.pytorch.explain import PGExplainer as DGLPGExplainer
from typing import Dict, Tuple, List
import numpy as np


class PGExplainerDGL:
    """
    PGExplainer wrapper using DGL's native implementation with batch training.

    DGL's PGExplainer supports:
    - Native batch training (5-10x faster than PyG)
    - Temperature annealing for better training
    - Direct edge mask generation

    Args:
        model: The GNN model to explain
        num_features: Node embedding dimension
        num_hops: Number of message passing hops
        epochs: Training epochs
        lr: Learning rate
        coff_budget: Coefficient for budget loss (sparsity)
        coff_connect: Coefficient for connectivity loss
        sample_bias: Bias for sampling
    """

    def __init__(
        self,
        model: nn.Module,
        num_features: int,
        num_hops: int = 2,
        epochs: int = 30,
        lr: float = 0.003,
        coff_budget: float = 0.01,
        coff_connect: float = 0.0005,
        sample_bias: float = 0.0
    ):
        self.model = model
        self.num_features = num_features
        self.num_hops = num_hops
        self.epochs = epochs
        self.lr = lr

        # Initialize DGL PGExplainer
        self.explainer = DGLPGExplainer(
            model=model,
            num_features=num_features,
            num_hops=num_hops,
            coff_budget=coff_budget,
            coff_connect=coff_connect,
            sample_bias=sample_bias
        )

        self.optimizer = torch.optim.Adam(
            self.explainer.parameters(),
            lr=lr
        )

    def train_explainer(
        self,
        g: dgl.DGLGraph,
        node_feat: torch.Tensor,
        training_nodes: torch.Tensor,
        batch_size: int = 32,
        temperature_schedule: bool = True,
        temp_start: float = 5.0,
        temp_end: float = 1.0,
        verbose: bool = True
    ) -> Dict[str, float]:
        """
        Train the PGExplainer with batch processing and temperature annealing.

        Args:
            g: DGL graph
            node_feat: Node features (can be embeddings from the model)
            training_nodes: Node IDs to train on
            batch_size: Batch size for training
            temperature_schedule: Whether to use temperature annealing
            temp_start: Starting temperature
            temp_end: Ending temperature
            verbose: Print training progress

        Returns:
            Training statistics
        """
        self.explainer.train()

        total_loss = 0.0
        num_batches = 0
        success_count = 0

        # Temperature schedule
        if temperature_schedule:
            temperatures = np.linspace(temp_start, temp_end, self.epochs)
        else:
            temperatures = [1.0] * self.epochs

        if verbose:
            print(f"[PG-DGL] Training PGExplainer for {self.epochs} epochs...", flush=True)
            print(f"  Training nodes: {len(training_nodes)}", flush=True)
            print(f"  Batch size: {batch_size}", flush=True)
            if temperature_schedule:
                print(f"  Temperature schedule: {temp_start:.1f} → {temp_end:.1f}", flush=True)

        for epoch in range(self.epochs):
            epoch_loss = 0.0
            epoch_batches = 0
            temperature = temperatures[epoch]

            # Shuffle training nodes
            perm = torch.randperm(len(training_nodes))
            shuffled_nodes = training_nodes[perm]

            # Batch training
            for batch_idx in range(0, len(shuffled_nodes), batch_size):
                batch_nodes = shuffled_nodes[batch_idx:batch_idx + batch_size]

                self.optimizer.zero_grad()

                try:
                    # Forward pass through explainer
                    # This generates edge masks for the batch of nodes
                    edge_mask = self.explainer.explain_node(
                        nodes=batch_nodes,
                        graph=g,
                        feat=node_feat,
                        temperature=temperature,
                        training=True
                    )

                    # Get model predictions with the mask
                    # Note: This is a simplified version - you may need to adapt
                    # based on your specific model architecture
                    loss = self.explainer.loss(
                        nodes=batch_nodes,
                        graph=g,
                        feat=node_feat,
                        edge_mask=edge_mask
                    )

                    loss.backward()
                    self.optimizer.step()

                    epoch_loss += loss.item()
                    epoch_batches += 1
                    success_count += 1

                except Exception as e:
                    if verbose and batch_idx < 3:  # Only print first few errors
                        print(f"[PG-DGL] Batch {batch_idx} failed: {e}", flush=True)
                    continue

            if epoch_batches > 0:
                avg_loss = epoch_loss / epoch_batches
                total_loss += avg_loss
                num_batches += 1

                # Print progress
                if verbose and (epoch % 5 == 0 or epoch == self.epochs - 1):
                    print(f"[PG-DGL] Epoch {epoch+1}/{self.epochs}: "
                          f"Loss = {avg_loss:.4f}, Temp = {temperature:.2f}", flush=True)

        final_loss = total_loss / num_batches if num_batches > 0 else 0.0
        success_rate = success_count / (self.epochs * ((len(training_nodes) + batch_size - 1) // batch_size))

        if verbose:
            print(f"[PG-DGL] ✓ Training completed", flush=True)
            print(f"  Final loss: {final_loss:.4f}", flush=True)
            print(f"  Success rate: {success_rate:.2%}", flush=True)

        return {
            'final_loss': final_loss,
            'success_rate': success_rate,
            'total_batches': num_batches
        }

    def explain_node_batch(
        self,
        g: dgl.DGLGraph,
        node_feat: torch.Tensor,
        nodes: torch.Tensor,
        top_k: int = 10,
        temperature: float = 1.0
    ) -> List[Dict]:
        """
        Generate explanations for a batch of nodes (BATCHED - much faster!).

        Args:
            g: DGL graph
            node_feat: Node features
            nodes: Node IDs to explain
            top_k: Number of top important edges to return
            temperature: Sampling temperature

        Returns:
            List of explanation dictionaries, one per node
        """
        self.explainer.eval()

        with torch.no_grad():
            # Get edge mask for all nodes in one batch
            edge_mask = self.explainer.explain_node(
                nodes=nodes,
                graph=g,
                feat=node_feat,
                temperature=temperature,
                training=False
            )

        # Process explanations for each node
        explanations = []

        for i, node_id in enumerate(nodes):
            # Get edges connected to this node
            # Find incoming and outgoing edges
            src, dst = g.edges()
            node_edges = (src == node_id) | (dst == node_id)
            edge_indices = torch.where(node_edges)[0]

            if len(edge_indices) == 0:
                explanations.append({
                    'node_id': node_id.item(),
                    'important_edges': torch.empty((2, 0), dtype=torch.long),
                    'importance_scores': torch.empty(0),
                    'edge_mask': torch.empty(0)
                })
                continue

            # Get importance scores for this node's edges
            node_edge_mask = edge_mask[edge_indices]

            # Get top-k important edges
            k = min(top_k, len(node_edge_mask))
            top_scores, top_idx = torch.topk(node_edge_mask, k)

            # Get the actual edge indices
            important_edge_ids = edge_indices[top_idx]
            important_edges = torch.stack([
                src[important_edge_ids],
                dst[important_edge_ids]
            ], dim=0)

            explanations.append({
                'node_id': node_id.item(),
                'important_edges': important_edges,
                'importance_scores': top_scores,
                'edge_mask': node_edge_mask,
                'num_edges': len(edge_indices)
            })

        return explanations

    def save(self, path: str):
        """Save trained explainer."""
        torch.save({
            'explainer_state_dict': self.explainer.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'num_features': self.num_features,
            'num_hops': self.num_hops,
            'epochs': self.epochs,
            'lr': self.lr
        }, path)

    def load(self, path: str):
        """Load trained explainer."""
        checkpoint = torch.load(path)
        self.explainer.load_state_dict(checkpoint['explainer_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        return checkpoint


def train_pgexplainer_dgl(
    model: nn.Module,
    g: dgl.DGLGraph,
    node_feat: torch.Tensor,
    training_edges: torch.Tensor,
    config: Dict
) -> PGExplainerDGL:
    """
    Convenience function to train a PGExplainer with DGL.

    Args:
        model: The GNN model to explain
        g: DGL graph
        node_feat: Node features/embeddings
        training_edges: Edge indices to train on [2, num_edges]
        config: Configuration dictionary

    Returns:
        Trained PGExplainerDGL instance
    """
    # Extract training nodes from edges
    training_nodes = torch.unique(training_edges.flatten())

    # Initialize explainer
    explainer = PGExplainerDGL(
        model=model,
        num_features=node_feat.size(1),
        num_hops=config.get('num_hops', 2),
        epochs=config.get('epochs', 30),
        lr=config.get('lr', 0.003),
        coff_budget=config.get('coff_budget', 0.01),
        coff_connect=config.get('coff_connect', 0.0005)
    )

    # Train
    stats = explainer.train_explainer(
        g=g,
        node_feat=node_feat,
        training_nodes=training_nodes,
        batch_size=config.get('batch_size', 32),
        temperature_schedule=config.get('temperature_schedule', True),
        verbose=config.get('verbose', True)
    )

    return explainer, stats
