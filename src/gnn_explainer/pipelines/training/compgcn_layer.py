"""
CompGCN Convolution Layer

Based on: "Composition-based Multi-Relational Graph Convolutional Networks"
Paper: https://arxiv.org/abs/1911.03082
Official Implementation: https://github.com/malllabiisc/CompGCN
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional

try:
    from torch_scatter import scatter_add
except ImportError:
    # PyG >= 2.1 bundles scatter; fall back to torch_geometric or native
    try:
        from torch_geometric.utils import scatter
        def scatter_add(src, index, dim=0, out=None, dim_size=None):
            result = scatter(src, index, dim=dim, dim_size=dim_size, reduce='sum')
            if out is not None:
                out.add_(result)
                return out
            return result
    except ImportError:
        # Pure PyTorch fallback
        def scatter_add(src, index, dim=0, out=None, dim_size=None):
            if out is None:
                size = list(src.shape)
                size[dim] = dim_size or (index.max().item() + 1)
                out = torch.zeros(size, dtype=src.dtype, device=src.device)
            return out.scatter_add_(dim, index.unsqueeze(-1).expand_as(src), src)


# Default chunk size for message passing (edges per chunk).
# Keeps peak memory per chunk to ~2 GiB with dim=32.
_DEFAULT_CHUNK_SIZE = 4_000_000


class CompGCNConv(nn.Module):
    """
    Composition-based Graph Convolutional Layer.

    Uses manual chunked message passing instead of PyG's propagate() to
    keep peak GPU memory bounded for large graphs (tens of millions of edges).

    Args:
        in_channels: Input feature dimension
        out_channels: Output feature dimension
        num_relations: Number of relations (edge types)
        comp_fn: Composition function ('sub', 'mult', 'corr')
        aggr: Aggregation scheme ('add' or 'mean')
        bias: Whether to add bias
        chunk_size: Number of edges to process per chunk (controls peak memory)
    """

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        num_relations: int,
        comp_fn: str = 'sub',
        aggr: str = 'add',
        bias: bool = True,
        chunk_size: int = _DEFAULT_CHUNK_SIZE,
        **kwargs
    ):
        super().__init__()

        self.in_channels = in_channels
        self.out_channels = out_channels
        self.num_relations = num_relations
        self.comp_fn = comp_fn
        self.aggr = aggr
        self.chunk_size = chunk_size

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
        rel_emb: torch.Tensor,
        edge_weight: Optional[torch.Tensor] = None
    ) -> tuple:
        """
        Forward pass with chunked message passing.

        Args:
            x: Node features (num_nodes, in_channels)
            edge_index: Edge indices (2, num_edges)
            edge_type: Edge types (num_edges,)
            rel_emb: Relation embeddings (num_relations, in_channels)
            edge_weight: Optional edge weights (num_edges,) for weighted message passing.

        Returns:
            Tuple of (updated node embeddings, updated relation embeddings)
        """
        num_nodes = x.size(0)

        # Self-loop contribution
        out = self.W_self(x)

        # Chunked message passing for forward edges
        out_forward = self._chunked_propagate(
            edge_index, x, edge_type, rel_emb,
            mode='forward', edge_weight=edge_weight,
            num_nodes=num_nodes
        )
        out = out + out_forward

        if self.bias is not None:
            out = out + self.bias

        # Update relation embeddings
        rel_emb_updated = self.W_rel(rel_emb)

        return out, rel_emb_updated

    def _chunked_propagate(
        self,
        edge_index: torch.Tensor,
        x: torch.Tensor,
        edge_type: torch.Tensor,
        rel_emb: torch.Tensor,
        mode: str,
        edge_weight: Optional[torch.Tensor],
        num_nodes: int,
    ) -> torch.Tensor:
        """
        Message passing in chunks to bound peak memory.

        Instead of computing messages for all edges simultaneously (which
        creates ~6 tensors of (num_edges, dim) for FFT-based correlation),
        we process edges in chunks and scatter-add results incrementally.
        """
        num_edges = edge_index.size(1)
        out = torch.zeros(num_nodes, self.out_channels, device=x.device, dtype=x.dtype)

        src_idx = edge_index[0]  # source nodes
        dst_idx = edge_index[1]  # destination nodes

        W = self.W_forward if mode == 'forward' else self.W_backward

        for start in range(0, num_edges, self.chunk_size):
            end = min(start + self.chunk_size, num_edges)

            # Gather source features and relation embeddings for this chunk
            chunk_src = src_idx[start:end]
            chunk_dst = dst_idx[start:end]
            chunk_etype = edge_type[start:end]

            x_j = x[chunk_src]               # (chunk, in_channels)
            rel = rel_emb[chunk_etype]        # (chunk, in_channels)

            # Composition
            if self.comp_fn == 'sub':
                msg = x_j - rel
            elif self.comp_fn == 'mult':
                msg = x_j * rel
            elif self.comp_fn == 'corr':
                msg = self._circular_correlation(x_j, rel)
            else:
                raise ValueError(f"Unknown composition function: {self.comp_fn}")

            # Weight transform: (out_channels, in_channels) @ (chunk, in_channels).T
            msg = F.linear(msg, W.weight)     # (chunk, out_channels)

            # Edge weight scaling (PaGE-Link)
            if edge_weight is not None:
                chunk_weight = edge_weight[start:end]
                msg = msg * chunk_weight.unsqueeze(-1)

            # Scatter-add messages to destination nodes
            scatter_add(msg, chunk_dst, dim=0, out=out)

        if self.aggr == 'mean':
            # Compute degree for normalization
            deg = torch.zeros(num_nodes, device=x.device, dtype=x.dtype)
            ones = torch.ones(num_edges, device=x.device, dtype=x.dtype)
            scatter_add(ones, dst_idx, dim=0, out=deg)
            deg = deg.clamp(min=1).unsqueeze(-1)
            out = out / deg

        return out

    @staticmethod
    def _circular_correlation(h: torch.Tensor, r: torch.Tensor) -> torch.Tensor:
        """
        Circular correlation: corr(h, r) = IFFT(FFT(h) * conj(FFT(r)))
        """
        h_fft = torch.fft.rfft(h, dim=-1)
        r_fft = torch.fft.rfft(r, dim=-1)
        corr_fft = h_fft * torch.conj(r_fft)
        return torch.fft.irfft(corr_fft, n=h.size(-1), dim=-1)

    def __repr__(self):
        return (f'{self.__class__.__name__}({self.in_channels}, '
                f'{self.out_channels}, num_relations={self.num_relations}, '
                f'comp_fn={self.comp_fn}, chunk_size={self.chunk_size})')
