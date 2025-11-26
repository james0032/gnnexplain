"""
CompGCN Convolution Layer

Based on: "Composition-based Multi-Relational Graph Convolutional Networks"
Paper: https://arxiv.org/abs/1911.03082
Official Implementation: https://github.com/malllabiisc/CompGCN
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import MessagePassing
from torch_geometric.utils import degree
from typing import Optional, Callable


class CompGCNConv(MessagePassing):
    """
    Composition-based Graph Convolutional Layer.

    This layer jointly embeds both nodes and relations by using composition
    operations to combine node and relation embeddings during message passing.

    Args:
        in_channels: Input feature dimension
        out_channels: Output feature dimension
        num_relations: Number of relations (edge types)
        comp_fn: Composition function ('sub', 'mult', 'corr')
        aggr: Aggregation scheme ('add', 'mean', 'max')
        bias: Whether to add bias

    Composition Functions:
        - 'sub': Subtraction (h - r)
        - 'mult': Multiplication (h * r)
        - 'corr': Circular correlation
    """

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        num_relations: int,
        comp_fn: str = 'sub',
        aggr: str = 'add',
        bias: bool = True,
        **kwargs
    ):
        super().__init__(aggr=aggr, **kwargs)

        self.in_channels = in_channels
        self.out_channels = out_channels
        self.num_relations = num_relations
        self.comp_fn = comp_fn

        # Weight matrices for self-loop, forward edges, and backward edges
        self.W_self = nn.Linear(in_channels, out_channels, bias=False)
        self.W_forward = nn.Linear(in_channels, out_channels, bias=False)
        self.W_backward = nn.Linear(in_channels, out_channels, bias=False)

        # Weight for relation transformation
        self.W_rel = nn.Linear(in_channels, out_channels, bias=False)

        if bias:
            self.bias = nn.Parameter(torch.Tensor(out_channels))
        else:
            self.register_parameter('bias', None)

        self.reset_parameters()

    def reset_parameters(self):
        """Initialize parameters."""
        nn.init.xavier_uniform_(self.W_self.weight)
        nn.init.xavier_uniform_(self.W_forward.weight)
        nn.init.xavier_uniform_(self.W_backward.weight)
        nn.init.xavier_uniform_(self.W_rel.weight)
        if self.bias is not None:
            nn.init.zeros_(self.bias)

    def forward(
        self,
        x: torch.Tensor,
        edge_index: torch.Tensor,
        edge_type: torch.Tensor,
        rel_emb: torch.Tensor
    ) -> tuple:
        """
        Forward pass.

        Args:
            x: Node features (num_nodes, in_channels)
            edge_index: Edge indices (2, num_edges)
            edge_type: Edge types (num_edges,)
            rel_emb: Relation embeddings (num_relations, in_channels)

        Returns:
            Tuple of (updated node embeddings, updated relation embeddings)
        """
        # Separate forward and backward edges
        # Forward: original edges
        # Backward: reverse edges (for directed graphs)
        num_edges = edge_index.size(1)

        # Self-loop contribution
        out_self = self.W_self(x)

        # Forward edges message passing
        out_forward = self.propagate(
            edge_index,
            x=x,
            edge_type=edge_type,
            rel_emb=rel_emb,
            mode='forward'
        )

        # Combine self-loop and forward messages
        out = out_self + out_forward

        if self.bias is not None:
            out = out + self.bias

        # Update relation embeddings
        rel_emb_updated = self.W_rel(rel_emb)

        return out, rel_emb_updated

    def message(
        self,
        x_j: torch.Tensor,
        edge_type: torch.Tensor,
        rel_emb: torch.Tensor,
        mode: str
    ) -> torch.Tensor:
        """
        Construct messages.

        Args:
            x_j: Source node features
            edge_type: Edge types
            rel_emb: Relation embeddings
            mode: 'forward' or 'backward'

        Returns:
            Messages after composition
        """
        # Get relation embeddings for each edge
        rel = rel_emb[edge_type]

        # Apply composition function
        if self.comp_fn == 'sub':
            # Subtraction: h - r
            msg = x_j - rel
        elif self.comp_fn == 'mult':
            # Multiplication: h * r
            msg = x_j * rel
        elif self.comp_fn == 'corr':
            # Circular correlation
            msg = self.circular_correlation(x_j, rel)
        else:
            raise ValueError(f"Unknown composition function: {self.comp_fn}")

        # Apply weight transformation
        if mode == 'forward':
            msg = self.W_forward.weight @ msg.t()
            msg = msg.t()
        else:
            msg = self.W_backward.weight @ msg.t()
            msg = msg.t()

        return msg

    def circular_correlation(
        self,
        h: torch.Tensor,
        r: torch.Tensor
    ) -> torch.Tensor:
        """
        Circular correlation operation.

        Uses FFT for efficient computation:
        corr(h, r) = IFFT(FFT(h) ⊙ conj(FFT(r)))

        Args:
            h: Head embeddings
            r: Relation embeddings

        Returns:
            Circular correlation result
        """
        # Use FFT for circular correlation
        h_fft = torch.fft.rfft(h, dim=-1)
        r_fft = torch.fft.rfft(r, dim=-1)

        # Element-wise multiplication in frequency domain
        corr_fft = h_fft * torch.conj(r_fft)

        # Inverse FFT
        corr = torch.fft.irfft(corr_fft, n=h.size(-1), dim=-1)

        return corr

    def __repr__(self):
        return (f'{self.__class__.__name__}({self.in_channels}, '
                f'{self.out_channels}, num_relations={self.num_relations}, '
                f'comp_fn={self.comp_fn})')
