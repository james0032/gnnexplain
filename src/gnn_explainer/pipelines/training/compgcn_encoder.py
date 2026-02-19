"""
CompGCN Encoder

Learns both node and relation embeddings jointly using composition operations.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from .compgcn_layer import CompGCNConv


class CompGCNEncoder(nn.Module):
    """
    CompGCN Encoder for Knowledge Graphs.

    Learns node and relation embeddings jointly through multi-layer
    composition-based graph convolutions.

    Args:
        num_nodes: Number of entities
        num_relations: Number of relations
        embedding_dim: Embedding dimension
        num_layers: Number of CompGCN layers
        comp_fn: Composition function ('sub', 'mult', 'corr')
        dropout: Dropout rate
        aggr: Aggregation scheme ('add', 'mean')
    """

    def __init__(
        self,
        num_nodes: int,
        num_relations: int,
        embedding_dim: int,
        num_layers: int = 2,
        comp_fn: str = 'sub',
        dropout: float = 0.2,
        aggr: str = 'add'
    ):
        super().__init__()

        self.num_nodes = num_nodes
        self.num_relations = num_relations
        self.embedding_dim = embedding_dim
        self.num_layers = num_layers
        self.comp_fn = comp_fn

        # Initial node embeddings
        self.node_emb = nn.Parameter(
            torch.Tensor(num_nodes, embedding_dim)
        )

        # Initial relation embeddings
        # Note: num_relations * 2 to account for inverse relations
        self.rel_emb = nn.Parameter(
            torch.Tensor(num_relations, embedding_dim)
        )

        # CompGCN layers
        self.convs = nn.ModuleList()
        for i in range(num_layers):
            self.convs.append(
                CompGCNConv(
                    embedding_dim,
                    embedding_dim,
                    num_relations,
                    comp_fn=comp_fn,
                    aggr=aggr
                )
            )

        self.dropout = nn.Dropout(dropout)

        self.reset_parameters()

    def reset_parameters(self):
        """Initialize parameters with Xavier uniform."""
        nn.init.xavier_uniform_(self.node_emb)
        nn.init.xavier_uniform_(self.rel_emb)

    def forward(
        self,
        edge_index: torch.Tensor,
        edge_type: torch.Tensor,
        edge_weight: torch.Tensor = None
    ) -> tuple:
        """
        Forward pass through CompGCN layers.

        Args:
            edge_index: Edge indices (2, num_edges)
            edge_type: Edge types (num_edges,)
            edge_weight: Optional edge weights (num_edges,) for weighted message passing.
                         Used by PaGE-Link explainer to learn edge masks.

        Returns:
            Tuple of (node_embeddings, relation_embeddings)
        """
        x = self.node_emb
        rel = self.rel_emb

        # Pass through CompGCN layers
        for i, conv in enumerate(self.convs):
            x, rel = conv(x, edge_index, edge_type, rel, edge_weight=edge_weight)

            # Apply activation and dropout (except last layer)
            if i < len(self.convs) - 1:
                x = F.relu(x)
                x = self.dropout(x)

                rel = F.relu(rel)
                rel = self.dropout(rel)

        return x, rel

    def get_embeddings(
        self,
        edge_index: torch.Tensor,
        edge_type: torch.Tensor,
        edge_weight: torch.Tensor = None
    ) -> dict:
        """
        Get final embeddings as a dictionary.

        Args:
            edge_index: Edge indices
            edge_type: Edge types
            edge_weight: Optional edge weights for weighted message passing

        Returns:
            Dictionary with 'node_emb' and 'rel_emb'
        """
        node_emb, rel_emb = self.forward(edge_index, edge_type, edge_weight=edge_weight)

        return {
            'node_emb': node_emb,
            'rel_emb': rel_emb
        }

    def __repr__(self):
        return (f'{self.__class__.__name__}('
                f'num_nodes={self.num_nodes}, '
                f'num_relations={self.num_relations}, '
                f'embedding_dim={self.embedding_dim}, '
                f'num_layers={self.num_layers}, '
                f'comp_fn={self.comp_fn})')
