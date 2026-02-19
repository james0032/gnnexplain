"""
Knowledge Graph Embedding Models - DGL Implementation

Unified framework supporting multiple encoder-decoder combinations:
- Encoder: CompGCN (DGL)
- Decoders: DistMult, ComplEx, RotatE, ConvE
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import dgl
from typing import Optional, Dict

from .compgcn_encoder_dgl import CompGCNEncoderDGL
from .conve_decoder import ConvE


class CompGCNKGModelDGL(nn.Module):
    """
    Unified CompGCN-based Knowledge Graph Embedding Model (DGL version).

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

        # CompGCN Encoder (DGL)
        self.encoder = CompGCNEncoderDGL(
            num_nodes=num_nodes,
            num_relations=num_relations,
            embedding_dim=embedding_dim,
            num_layers=num_layers,
            comp_fn=comp_fn,
            dropout=dropout,
            aggr='sum'  # DGL uses 'sum' instead of 'add'
        )

        # Initialize decoder based on type
        if self.decoder_type == 'conve':
            conve_kwargs = conve_kwargs or {}
            self.decoder = ConvE(
                num_nodes=num_nodes,
                num_relations=num_relations,
                embedding_dim=embedding_dim,
                **conve_kwargs
            )
        # For other decoders, we implement them directly (no PyG dependency)
        else:
            self.decoder = None  # Decoder logic is in decode() method

    def encode(
        self,
        edge_index: torch.Tensor = None,
        edge_type: torch.Tensor = None,
        g: dgl.DGLGraph = None
    ) -> tuple:
        """
        Encode graph to get node and relation embeddings.

        Args:
            edge_index: Edge indices (2, num_edges) - for backward compatibility
            edge_type: Edge types (num_edges,) - for backward compatibility
            g: DGL graph (preferred)

        Returns:
            Tuple of (node_embeddings, relation_embeddings)
        """
        if g is not None:
            return self.encoder(g)
        elif edge_index is not None and edge_type is not None:
            return self.encoder.forward_with_edge_index(edge_index, edge_type)
        else:
            raise ValueError("Either provide DGL graph 'g' or edge_index + edge_type")

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

            scores = self.decoder(head_emb, rel_emb_batch, tail_emb, tail_idx=tail_idx)

        elif self.decoder_type == 'distmult':
            # DistMult: <h, r, t> = sum(h * r * t)
            head_emb = node_emb[head_idx]
            tail_emb = node_emb[tail_idx]
            rel_emb_batch = rel_emb[rel_idx]
            scores = torch.sum(head_emb * rel_emb_batch * tail_emb, dim=1)

        elif self.decoder_type == 'complex':
            # ComplEx: Re(<h, r, conj(t)>)
            head_emb = node_emb[head_idx]
            tail_emb = node_emb[tail_idx]
            rel_emb_batch = rel_emb[rel_idx]

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
            head_emb = node_emb[head_idx]
            tail_emb = node_emb[tail_idx]
            rel_emb_batch = rel_emb[rel_idx]

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

        else:
            raise ValueError(f"Unknown decoder type: {self.decoder_type}")

        return scores

    def forward(
        self,
        edge_index: torch.Tensor = None,
        edge_type: torch.Tensor = None,
        g: dgl.DGLGraph = None,
        head_idx: torch.Tensor = None,
        tail_idx: torch.Tensor = None,
        rel_idx: torch.Tensor = None
    ) -> torch.Tensor:
        """
        Full forward pass: encode + decode.

        Args:
            edge_index: Edge indices (2, num_edges) - for backward compatibility
            edge_type: Edge types (num_edges,) - for backward compatibility
            g: DGL graph (preferred)
            head_idx: Head entity indices (batch_size,)
            tail_idx: Tail entity indices (batch_size,)
            rel_idx: Relation indices (batch_size,)

        Returns:
            Triple scores (batch_size,)
        """
        # Encode
        node_emb, rel_emb = self.encode(edge_index, edge_type, g)

        # Decode
        scores = self.decode(node_emb, rel_emb, head_idx, tail_idx, rel_idx)

        return scores

    def __repr__(self):
        return (f'{self.__class__.__name__}('
                f'encoder=CompGCN-DGL, '
                f'decoder={self.decoder_type.upper()}, '
                f'embedding_dim={self.embedding_dim})')
