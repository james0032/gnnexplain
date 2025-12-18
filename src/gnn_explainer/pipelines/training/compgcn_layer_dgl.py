"""
CompGCN Convolution Layer - DGL Implementation

Based on: "Composition-based Multi-Relational Graph Convolutional Networks"
Paper: https://arxiv.org/abs/1911.03082
Official Implementation: https://github.com/malllabiisc/CompGCN

This is a DGL port of the PyG CompGCN layer for better batching and performance.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import dgl
import dgl.function as fn
from typing import Optional


class CompGCNConvDGL(nn.Module):
    """
    Composition-based Graph Convolutional Layer (DGL version).

    This layer jointly embeds both nodes and relations by using composition
    operations to combine node and relation embeddings during message passing.

    Args:
        in_channels: Input feature dimension
        out_channels: Output feature dimension
        num_relations: Number of relations (edge types)
        comp_fn: Composition function ('sub', 'mult', 'corr')
        aggr: Aggregation scheme ('sum', 'mean', 'max')
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
        aggr: str = 'sum',
        bias: bool = True,
    ):
        super().__init__()

        self.in_channels = in_channels
        self.out_channels = out_channels
        self.num_relations = num_relations
        self.comp_fn = comp_fn
        self.aggr = aggr

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
        g: dgl.DGLGraph,
        node_feat: torch.Tensor,
        rel_emb: torch.Tensor
    ) -> tuple:
        """
        Forward pass.

        Args:
            g: DGL graph with edge data 'etype' containing edge types
            node_feat: Node features (num_nodes, in_channels)
            rel_emb: Relation embeddings (num_relations, in_channels)

        Returns:
            Tuple of (updated node embeddings, updated relation embeddings)
        """
        with g.local_scope():
            # Store node features in graph
            g.ndata['h'] = node_feat

            # Get edge types from graph
            edge_type = g.edata['etype']

            # Get relation embeddings for each edge
            g.edata['rel_emb'] = rel_emb[edge_type]

            # Apply composition function
            if self.comp_fn == 'sub':
                # Message function: subtraction composition
                def message_func(edges):
                    # h_src - rel
                    msg = edges.src['h'] - edges.data['rel_emb']
                    return {'msg': msg}
            elif self.comp_fn == 'mult':
                # Message function: multiplication composition
                def message_func(edges):
                    # h_src * rel
                    msg = edges.src['h'] * edges.data['rel_emb']
                    return {'msg': msg}
            elif self.comp_fn == 'corr':
                # Message function: circular correlation composition
                def message_func(edges):
                    msg = self.circular_correlation(edges.src['h'], edges.data['rel_emb'])
                    return {'msg': msg}
            else:
                raise ValueError(f"Unknown composition function: {self.comp_fn}")

            # Reduce function based on aggregation type
            if self.aggr == 'sum':
                reduce_func = fn.sum('msg_transformed', 'h_neigh')
            elif self.aggr == 'mean':
                reduce_func = fn.mean('msg_transformed', 'h_neigh')
            elif self.aggr == 'max':
                reduce_func = fn.max('msg_transformed', 'h_neigh')
            else:
                raise ValueError(f"Unknown aggregation: {self.aggr}")

            # Separate forward and backward edges based on direction
            # Assuming edge direction is stored in g.edata['direction']
            # 0 = forward, 1 = backward

            # Get edge direction (if not available, treat all as forward)
            if 'direction' in g.edata:
                edge_direction = g.edata['direction']
                forward_mask = edge_direction == 0
                backward_mask = edge_direction == 1
            else:
                # If no direction info, assume first half forward, second half backward
                num_edges = g.num_edges()
                forward_mask = torch.arange(num_edges, device=node_feat.device) < num_edges // 2
                backward_mask = ~forward_mask

            # Create edge IDs
            edge_ids = torch.arange(g.num_edges(), device=node_feat.device)
            forward_eids = edge_ids[forward_mask]
            backward_eids = edge_ids[backward_mask]

            # Message passing on forward edges
            g.apply_edges(message_func, edges=forward_eids)
            g.edata['msg_transformed'] = torch.zeros_like(g.ndata['h'][g.edges()[1]])

            # Transform messages with forward weights
            forward_msg = g.edata['msg'][forward_eids]
            g.edata['msg_transformed'][forward_eids] = self.W_forward(forward_msg)

            # Aggregate messages
            g.update_all(lambda edges: {'msg_transformed': edges.data['msg_transformed']}, reduce_func)

            # Get aggregated neighbor features (default to zero if no neighbors)
            if 'h_neigh' in g.ndata:
                h_neigh = g.ndata['h_neigh']
            else:
                h_neigh = torch.zeros_like(node_feat)

            # Self-loop contribution
            out_self = self.W_self(node_feat)

            # Combine self-loop and neighbor messages
            out = out_self + h_neigh

            if self.bias is not None:
                out = out + self.bias

            # Update relation embeddings
            rel_emb_updated = self.W_rel(rel_emb)

            return out, rel_emb_updated

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
