"""RGCN-DistMult model architecture."""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import RGCNConv


class DistMult(nn.Module):
    """DistMult decoder for knowledge graph completion."""

    def __init__(self, num_relations: int, embedding_dim: int):
        super().__init__()
        self.relation_embeddings = nn.Parameter(
            torch.Tensor(num_relations, embedding_dim)
        )
        nn.init.xavier_uniform_(self.relation_embeddings)

    def forward(
        self,
        head_emb: torch.Tensor,
        tail_emb: torch.Tensor,
        rel_idx: torch.Tensor
    ) -> torch.Tensor:
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

    def __init__(
        self,
        num_nodes: int,
        num_relations: int,
        embedding_dim: int = 128,
        num_layers: int = 2,
        num_bases: int = 30,
        dropout: float = 0.2
    ):
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
                RGCNConv(
                    embedding_dim,
                    embedding_dim,
                    num_relations,
                    num_bases=num_bases
                )
            )

        self.dropout = nn.Dropout(dropout)
        self.decoder = DistMult(num_relations, embedding_dim)

    def encode(
        self,
        edge_index: torch.Tensor,
        edge_type: torch.Tensor
    ) -> torch.Tensor:
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

    def decode(
        self,
        node_emb: torch.Tensor,
        head_idx: torch.Tensor,
        tail_idx: torch.Tensor,
        rel_idx: torch.Tensor
    ) -> torch.Tensor:
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

    def forward(
        self,
        edge_index: torch.Tensor,
        edge_type: torch.Tensor,
        head_idx: torch.Tensor,
        tail_idx: torch.Tensor,
        rel_idx: torch.Tensor
    ) -> torch.Tensor:
        """Full forward pass: encode + decode."""
        node_emb = self.encode(edge_index, edge_type)
        scores = self.decode(node_emb, head_idx, tail_idx, rel_idx)
        return scores
