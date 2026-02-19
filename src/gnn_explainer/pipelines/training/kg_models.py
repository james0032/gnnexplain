"""
Knowledge Graph Embedding Models

Unified framework supporting multiple encoder-decoder combinations:
- Encoders: RGCN, CompGCN
- Decoders: DistMult, ComplEx, RotatE, ConvE
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Dict

from .model import RGCNDistMultModel  # Existing RGCN model
from .compgcn_encoder import CompGCNEncoder
from .conve_decoder import ConvE


class CompGCNKGModel(nn.Module):
    """
    Unified CompGCN-based Knowledge Graph Embedding Model.

    Supports multiple decoders:
    - 'distmult': DistMult (simple bilinear)
    - 'complex': ComplEx (complex-valued embeddings)
    - 'rotate': RotatE (rotations in complex space)
    - 'conve': ConvE (2D convolutions)

    Args:
        num_nodes: Number of entities
        num_relations: Number of relations
        embedding_dim: Embedding dimension
        decoder_type: Type of decoder ('distmult', 'complex', 'rotate', 'conve')
        num_layers: Number of CompGCN layers
        comp_fn: Composition function ('sub', 'mult', 'corr')
        dropout: Dropout rate
        conve_kwargs: Additional kwargs for ConvE decoder
    """

    def __init__(
        self,
        num_nodes: int,
        num_relations: int,
        embedding_dim: int = 200,
        decoder_type: str = 'complex',
        num_layers: int = 2,
        comp_fn: str = 'sub',
        dropout: float = 0.2,
        conve_kwargs: Optional[Dict] = None
    ):
        super().__init__()

        self.num_nodes = num_nodes
        self.num_relations = num_relations
        self.embedding_dim = embedding_dim
        self.decoder_type = decoder_type.lower()

        # CompGCN Encoder
        self.encoder = CompGCNEncoder(
            num_nodes=num_nodes,
            num_relations=num_relations,
            embedding_dim=embedding_dim,
            num_layers=num_layers,
            comp_fn=comp_fn,
            dropout=dropout
        )

        # Initialize decoder based on type
        if self.decoder_type == 'distmult':
            from torch_geometric.nn.kge import DistMult
            self.decoder = DistMult(num_nodes, num_relations, embedding_dim)

        elif self.decoder_type == 'complex':
            from torch_geometric.nn.kge import ComplEx
            self.decoder = ComplEx(num_nodes, num_relations, embedding_dim)

        elif self.decoder_type == 'rotate':
            from torch_geometric.nn.kge import RotatE
            self.decoder = RotatE(num_nodes, num_relations, embedding_dim)

        elif self.decoder_type == 'conve':
            conve_kwargs = conve_kwargs or {}
            self.decoder = ConvE(
                num_nodes=num_nodes,
                num_relations=num_relations,
                embedding_dim=embedding_dim,
                **conve_kwargs
            )

        else:
            raise ValueError(f"Unknown decoder type: {decoder_type}. "
                           f"Choose from: distmult, complex, rotate, conve")

    def encode(
        self,
        edge_index: torch.Tensor,
        edge_type: torch.Tensor,
        edge_weight: torch.Tensor = None
    ) -> tuple:
        """
        Encode graph to get node and relation embeddings.

        Args:
            edge_index: Edge indices (2, num_edges)
            edge_type: Edge types (num_edges,)
            edge_weight: Optional edge weights (num_edges,) for weighted message passing.
                         Used by PaGE-Link explainer to learn edge masks.

        Returns:
            Tuple of (node_embeddings, relation_embeddings)
        """
        return self.encoder(edge_index, edge_type, edge_weight=edge_weight)

    def decode(
        self,
        node_emb: torch.Tensor,
        rel_emb: torch.Tensor,
        head_idx: torch.Tensor,
        tail_idx: torch.Tensor,
        rel_idx: torch.Tensor
    ) -> torch.Tensor:
        """
        Decode triples to get scores.

        Args:
            node_emb: Node embeddings from encoder
            rel_emb: Relation embeddings from encoder
            head_idx: Head entity indices
            tail_idx: Tail entity indices
            rel_idx: Relation indices

        Returns:
            Triple scores
        """
        if self.decoder_type == 'conve':
            # ConvE uses embeddings directly
            head_emb = node_emb[head_idx]
            tail_emb = node_emb[tail_idx]
            rel_emb_batch = rel_emb[rel_idx]

            scores = self.decoder(head_emb, rel_emb_batch, tail_emb)

        else:
            # PyG decoders (DistMult, ComplEx, RotatE) use indices
            # Update their internal embeddings first
            with torch.no_grad():
                # For PyG KGE models, we need to handle embeddings carefully
                # They have their own node_emb and rel_emb parameters
                pass

            # For PyG decoders, use the forward method
            # Note: This is a simplified version - may need adjustment
            if hasattr(self.decoder, 'forward'):
                # Update decoder embeddings (this is a workaround)
                # In practice, you might want to use the decoder's loss function
                head_emb = node_emb[head_idx]
                tail_emb = node_emb[tail_idx]
                rel_emb_batch = rel_emb[rel_idx]

                # Calculate scores based on decoder type
                if self.decoder_type == 'distmult':
                    # DistMult: <h, r, t> = sum(h * r * t)
                    scores = torch.sum(head_emb * rel_emb_batch * tail_emb, dim=1)

                elif self.decoder_type == 'complex':
                    # ComplEx: Re(<h, r, conj(t)>)
                    # Split into real and imaginary parts
                    dim = self.embedding_dim // 2
                    h_real, h_img = head_emb[:, :dim], head_emb[:, dim:]
                    r_real, r_img = rel_emb_batch[:, :dim], rel_emb_batch[:, dim:]
                    t_real, t_img = tail_emb[:, :dim], tail_emb[:, dim:]

                    # ComplEx scoring
                    scores = torch.sum(
                        h_real * r_real * t_real +
                        h_real * r_img * t_img +
                        h_img * r_real * t_img -
                        h_img * r_img * t_real,
                        dim=1
                    )

                elif self.decoder_type == 'rotate':
                    # RotatE: -||h ∘ r - t||
                    # Relations are rotations in complex space
                    dim = self.embedding_dim // 2
                    h_real, h_img = head_emb[:, :dim], head_emb[:, dim:]

                    # Relation as rotation (phase)
                    phase = rel_emb_batch / (self.embedding_dim / (2 * torch.pi))

                    # Apply rotation
                    r_cos, r_sin = torch.cos(phase), torch.sin(phase)
                    rotated_real = h_real * r_cos - h_img * r_sin
                    rotated_img = h_real * r_sin + h_img * r_cos

                    t_real, t_img = tail_emb[:, :dim], tail_emb[:, dim:]

                    # Distance
                    dist = torch.sqrt(
                        (rotated_real - t_real) ** 2 +
                        (rotated_img - t_img) ** 2
                    )
                    scores = -torch.sum(dist, dim=1)

        return scores

    def forward(
        self,
        edge_index: torch.Tensor,
        edge_type: torch.Tensor,
        head_idx: torch.Tensor,
        tail_idx: torch.Tensor,
        rel_idx: torch.Tensor,
        edge_weight: torch.Tensor = None
    ) -> torch.Tensor:
        """
        Full forward pass: encode + decode.

        Args:
            edge_index: Edge indices (2, num_edges)
            edge_type: Edge types (num_edges,)
            head_idx: Head entity indices (batch_size,)
            tail_idx: Tail entity indices (batch_size,)
            rel_idx: Relation indices (batch_size,)
            edge_weight: Optional edge weights (num_edges,) for weighted message passing

        Returns:
            Triple scores (batch_size,)
        """
        # Encode
        node_emb, rel_emb = self.encode(edge_index, edge_type, edge_weight=edge_weight)

        # Decode
        scores = self.decode(node_emb, rel_emb, head_idx, tail_idx, rel_idx)

        return scores

    def __repr__(self):
        return (f'{self.__class__.__name__}('
                f'encoder=CompGCN, '
                f'decoder={self.decoder_type.upper()}, '
                f'embedding_dim={self.embedding_dim})')
