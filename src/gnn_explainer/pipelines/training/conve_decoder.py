"""
ConvE Decoder

2D Convolutional Knowledge Graph Embeddings

Based on: "Convolutional 2D Knowledge Graph Embeddings"
Paper: https://arxiv.org/abs/1707.01476
Official Implementation: https://github.com/TimDettmers/ConvE
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional


class ConvE(nn.Module):
    """
    ConvE: 2D Convolutional Knowledge Graph Embedding Decoder.

    Uses 2D convolutions over reshaped embeddings for link prediction.
    Highly parameter-efficient compared to other KG embedding models.

    Args:
        num_nodes: Number of entities
        num_relations: Number of relations
        embedding_dim: Embedding dimension (must be factorizable for 2D reshape)
        input_drop: Dropout for input layer
        hidden_drop: Dropout for hidden layers
        feature_drop: Dropout for convolutional features
        embedding_height: Height of reshaped embedding (embedding_dim = height * width)
        num_filters: Number of convolutional filters
        kernel_size: Size of convolutional kernel
    """

    def __init__(
        self,
        num_nodes: int,
        num_relations: int,
        embedding_dim: int = 200,
        input_drop: float = 0.2,
        hidden_drop: float = 0.3,
        feature_drop: float = 0.2,
        embedding_height: Optional[int] = None,
        num_filters: int = 32,
        kernel_size: int = 3
    ):
        super().__init__()

        self.num_nodes = num_nodes
        self.num_relations = num_relations
        self.embedding_dim = embedding_dim

        # Calculate 2D reshaping dimensions
        if embedding_height is None:
            # Auto-calculate height (try to make square-ish)
            embedding_height = self._find_embedding_height(embedding_dim)

        self.embedding_height = embedding_height
        self.embedding_width = embedding_dim // embedding_height

        assert embedding_dim == embedding_height * self.embedding_width, \
            f"embedding_dim must be divisible by embedding_height: {embedding_dim} != {embedding_height} * {self.embedding_width}"

        # Dropout layers
        self.inp_drop = nn.Dropout(input_drop)
        self.hidden_drop = nn.Dropout(hidden_drop)
        self.feature_drop = nn.Dropout(feature_drop)

        # Batch normalization
        self.bn0 = nn.BatchNorm2d(1)
        self.bn1 = nn.BatchNorm2d(num_filters)
        self.bn2 = nn.BatchNorm1d(embedding_dim)

        # 2D Convolution
        # Input: stacked (head, relation) embeddings
        # Output: feature maps
        self.conv1 = nn.Conv2d(
            in_channels=1,
            out_channels=num_filters,
            kernel_size=kernel_size,
            stride=1,
            padding=1,
            bias=True
        )

        # Calculate flattened size after convolution
        # After conv with padding=1, size should be similar
        flat_sz_h = 2 * self.embedding_height
        flat_sz_w = self.embedding_width
        self.flat_sz = num_filters * flat_sz_h * flat_sz_w

        # Fully connected layer
        self.fc = nn.Linear(self.flat_sz, embedding_dim)

        # For scoring
        self.bias = nn.Parameter(torch.zeros(num_nodes))

    def _find_embedding_height(self, embedding_dim: int) -> int:
        """Find a good height for 2D reshaping."""
        # Try to find factors close to square root
        import math
        sqrt_dim = int(math.sqrt(embedding_dim))

        # Try factors starting from sqrt
        for h in range(sqrt_dim, 0, -1):
            if embedding_dim % h == 0:
                return h

        # Fallback to 1
        return 1

    def forward(
        self,
        head_emb: torch.Tensor,
        rel_emb: torch.Tensor,
        tail_emb: Optional[torch.Tensor] = None,
        all_node_emb: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """
        Forward pass.

        Args:
            head_emb: Head entity embeddings (batch_size, embedding_dim)
            rel_emb: Relation embeddings (batch_size, embedding_dim)
            tail_emb: Tail entity embeddings (batch_size, embedding_dim) or None
            all_node_emb: All node embeddings (num_nodes, embedding_dim) for scoring

        Returns:
            Scores for triples
        """
        batch_size = head_emb.size(0)

        # Reshape embeddings to 2D
        head_2d = head_emb.view(-1, 1, self.embedding_height, self.embedding_width)
        rel_2d = rel_emb.view(-1, 1, self.embedding_height, self.embedding_width)

        # Stack head and relation vertically
        stacked = torch.cat([head_2d, rel_2d], dim=2)  # (batch, 1, 2*height, width)

        # Apply input dropout and batch norm
        x = self.bn0(stacked)
        x = self.inp_drop(x)

        # Convolution
        x = self.conv1(x)  # (batch, num_filters, 2*height, width)
        x = self.bn1(x)
        x = F.relu(x)
        x = self.feature_drop(x)

        # Flatten
        x = x.view(batch_size, -1)  # (batch, flat_sz)

        # Fully connected
        x = self.fc(x)  # (batch, embedding_dim)
        x = self.hidden_drop(x)
        x = self.bn2(x)
        x = F.relu(x)

        if tail_emb is not None:
            # Score specific triples: x · tail
            scores = torch.sum(x * tail_emb, dim=1) + self.bias[torch.arange(batch_size)]
        elif all_node_emb is not None:
            # Score against all entities: x · all_nodes^T
            scores = torch.mm(x, all_node_emb.t())
            scores = scores + self.bias.expand_as(scores)
        else:
            raise ValueError("Either tail_emb or all_node_emb must be provided")

        return scores

    def __repr__(self):
        return (f'{self.__class__.__name__}('
                f'num_nodes={self.num_nodes}, '
                f'num_relations={self.num_relations}, '
                f'embedding_dim={self.embedding_dim}, '
                f'reshape=({self.embedding_height}x{self.embedding_width}))')
