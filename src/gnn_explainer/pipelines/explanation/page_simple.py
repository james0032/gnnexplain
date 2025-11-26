"""
Simplified PAGE (Parametric Generative Explainer) for Knowledge Graph Link Prediction.

This is a simplified implementation that adapts PAGE's VGAE approach for link prediction.
It uses a variational graph auto-encoder to learn latent representations of subgraphs
and identify important edges for explaining link predictions.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, Tuple, Optional
import numpy as np


class GraphConvolution(nn.Module):
    """
    Simple Graph Convolution layer.
    """

    def __init__(self, in_features, out_features, dropout=0.0, act=F.relu):
        super(GraphConvolution, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.dropout = dropout
        self.act = act
        self.weight = nn.Parameter(torch.FloatTensor(in_features, out_features))
        self.bias = nn.Parameter(torch.FloatTensor(out_features))
        self.reset_parameters()

    def reset_parameters(self):
        torch.nn.init.xavier_uniform_(self.weight)
        torch.nn.init.zeros_(self.bias)

    def forward(self, input, adj):
        """
        Args:
            input: (batch_size, num_nodes, in_features)
            adj: (batch_size, num_nodes, num_nodes)
        """
        input = F.dropout(input, self.dropout, training=self.training)
        support = torch.matmul(input, self.weight)
        output = torch.bmm(adj, support)
        output = output + self.bias
        return self.act(output) if self.act is not None else output


class VGAEEncoder(nn.Module):
    """
    Variational Graph Auto-Encoder Encoder.

    Encodes subgraph into latent space (mu, logvar).
    """

    def __init__(self, input_dim, hidden_dim1, hidden_dim2, output_dim, dropout=0.0):
        super(VGAEEncoder, self).__init__()
        self.gc1 = GraphConvolution(input_dim, hidden_dim1, dropout, act=F.relu)
        self.gc2 = GraphConvolution(hidden_dim1, hidden_dim2, dropout, act=F.relu)
        self.gc_mu = GraphConvolution(hidden_dim2, output_dim, dropout, act=lambda x: x)
        self.gc_logvar = GraphConvolution(hidden_dim2, output_dim, dropout, act=lambda x: x)

    def forward(self, x, adj):
        """
        Args:
            x: Node features (batch_size, num_nodes, input_dim)
            adj: Adjacency matrix (batch_size, num_nodes, num_nodes)

        Returns:
            mu: Mean of latent distribution (batch_size, num_nodes, output_dim)
            logvar: Log variance of latent distribution (batch_size, num_nodes, output_dim)
        """
        h1 = self.gc1(x, adj)
        h2 = self.gc2(h1, adj)
        mu = self.gc_mu(h2, adj)
        logvar = self.gc_logvar(h2, adj)
        return mu, logvar


class VGAEDecoder(nn.Module):
    """
    Variational Graph Auto-Encoder Decoder with MLP.

    Decodes latent representation back to adjacency matrix.
    """

    def __init__(self, input_dim, hidden_dim1, hidden_dim2, dropout=0.0):
        super(VGAEDecoder, self).__init__()
        self.fc1 = nn.Linear(input_dim, hidden_dim1)
        self.fc2 = nn.Linear(hidden_dim1, hidden_dim2)
        self.dropout = dropout

    def forward(self, z):
        """
        Args:
            z: Latent representation (batch_size, num_nodes, input_dim)

        Returns:
            adj_reconstructed: Reconstructed adjacency (batch_size, num_nodes, num_nodes)
        """
        z = F.relu(self.fc1(z))
        z = torch.sigmoid(self.fc2(z))
        z = F.dropout(z, self.dropout, training=self.training)

        # Inner product decoder
        adj_reconstructed = torch.bmm(z, z.transpose(1, 2))
        adj_reconstructed = torch.sigmoid(adj_reconstructed)

        return adj_reconstructed


class VGAE(nn.Module):
    """
    Complete Variational Graph Auto-Encoder for subgraph representation learning.
    """

    def __init__(
        self,
        input_dim,
        encoder_hidden1=32,
        encoder_hidden2=16,
        latent_dim=16,
        decoder_hidden1=16,
        decoder_hidden2=16,
        dropout=0.0
    ):
        super(VGAE, self).__init__()
        self.encoder = VGAEEncoder(input_dim, encoder_hidden1, encoder_hidden2, latent_dim, dropout)
        self.decoder = VGAEDecoder(latent_dim, decoder_hidden1, decoder_hidden2, dropout)
        self.latent_dim = latent_dim

    def reparameterize(self, mu, logvar):
        """Reparameterization trick for VAE."""
        if self.training:
            std = torch.exp(0.5 * logvar)
            eps = torch.randn_like(std)
            return mu + eps * std
        else:
            return mu

    def forward(self, x, adj):
        """
        Args:
            x: Node features (batch_size, num_nodes, input_dim)
            adj: Adjacency matrix (batch_size, num_nodes, num_nodes)

        Returns:
            adj_reconstructed: Reconstructed adjacency
            mu: Latent mean
            logvar: Latent log variance
            z: Sampled latent representation
        """
        mu, logvar = self.encoder(x, adj)
        z = self.reparameterize(mu, logvar)
        adj_reconstructed = self.decoder(z)

        return adj_reconstructed, mu, logvar, z

    def encode(self, x, adj):
        """Encode without sampling."""
        mu, logvar = self.encoder(x, adj)
        return mu, logvar


def vgae_loss(adj_reconstructed, adj_true, mu, logvar, kl_weight=0.2):
    """
    Compute VGAE loss: reconstruction loss + KL divergence.

    Args:
        adj_reconstructed: Reconstructed adjacency
        adj_true: True adjacency
        mu: Latent mean
        logvar: Latent log variance
        kl_weight: Weight for KL divergence term

    Returns:
        Total loss, reconstruction loss, KL divergence
    """
    # Reconstruction loss (binary cross-entropy)
    recon_loss = F.binary_cross_entropy(adj_reconstructed, adj_true, reduction='mean')

    # KL divergence
    kl_div = -0.5 * torch.mean(1 + logvar - mu.pow(2) - logvar.exp())

    # Total loss
    total_loss = recon_loss + kl_weight * kl_div

    return total_loss, recon_loss, kl_div


class PAGEExplainer:
    """
    Simplified PAGE Explainer for Knowledge Graph Link Prediction.

    This explainer:
    1. Trains a VGAE to reconstruct subgraphs around triples
    2. Uses latent representations to identify important edges
    3. Generates explanations based on reconstruction importance
    """

    def __init__(
        self,
        input_dim,
        encoder_hidden1=32,
        encoder_hidden2=16,
        latent_dim=16,
        decoder_hidden1=16,
        decoder_hidden2=16,
        dropout=0.0,
        device='cpu'
    ):
        self.device = device
        self.vgae = VGAE(
            input_dim=input_dim,
            encoder_hidden1=encoder_hidden1,
            encoder_hidden2=encoder_hidden2,
            latent_dim=latent_dim,
            decoder_hidden1=decoder_hidden1,
            decoder_hidden2=decoder_hidden2,
            dropout=dropout
        ).to(device)

    def train_on_subgraphs(
        self,
        subgraphs: list,
        epochs=100,
        lr=0.003,
        kl_weight=0.2,
        verbose=True
    ):
        """
        Train VGAE on extracted subgraphs.

        Args:
            subgraphs: List of (x, adj) tuples
            epochs: Number of training epochs
            lr: Learning rate
            kl_weight: Weight for KL divergence
            verbose: Print training progress
        """
        optimizer = torch.optim.Adam(self.vgae.parameters(), lr=lr)

        self.vgae.train()

        for epoch in range(epochs):
            total_loss = 0
            total_recon = 0
            total_kl = 0

            for x, adj in subgraphs:
                x = x.to(self.device)
                adj = adj.to(self.device)

                optimizer.zero_grad()

                adj_reconstructed, mu, logvar, z = self.vgae(x, adj)

                loss, recon_loss, kl_div = vgae_loss(adj_reconstructed, adj, mu, logvar, kl_weight)

                loss.backward()
                optimizer.step()

                total_loss += loss.item()
                total_recon += recon_loss.item()
                total_kl += kl_div.item()

            if verbose and (epoch + 1) % 20 == 0:
                avg_loss = total_loss / len(subgraphs)
                avg_recon = total_recon / len(subgraphs)
                avg_kl = total_kl / len(subgraphs)
                print(f"  Epoch {epoch+1}/{epochs}: Loss={avg_loss:.4f}, "
                      f"Recon={avg_recon:.4f}, KL={avg_kl:.4f}")

    def explain(self, x, adj, target_edges=None):
        """
        Generate explanation for a subgraph.

        Args:
            x: Node features (batch_size, num_nodes, input_dim)
            adj: Adjacency matrix (batch_size, num_nodes, num_nodes)
            target_edges: Optional specific edges to focus on

        Returns:
            edge_importance: Importance scores for each edge
        """
        self.vgae.eval()

        with torch.no_grad():
            x = x.to(self.device)
            adj = adj.to(self.device)

            # Encode to latent space
            mu, logvar = self.vgae.encode(x, adj)
            z = mu  # Use mean for explanation

            # Decode
            adj_reconstructed = self.vgae.decoder(z)

            # Compute edge importance as reconstruction error
            # Higher reconstruction = more important for the structure
            edge_importance = torch.abs(adj_reconstructed - adj)

            # Alternative: use reconstruction quality (higher = more important)
            # edge_importance = adj_reconstructed * adj

        return edge_importance

    def save(self, path):
        """Save trained VGAE model."""
        torch.save(self.vgae.state_dict(), path)

    def load(self, path):
        """Load trained VGAE model."""
        self.vgae.load_state_dict(torch.load(path, map_location=self.device))


def extract_k_hop_subgraph(
    edge_index: torch.Tensor,
    node_idx: int,
    num_hops: int,
    num_nodes: int
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Extract k-hop subgraph around a node.

    Args:
        edge_index: Edge index (2, num_edges)
        node_idx: Center node
        num_hops: Number of hops
        num_nodes: Total number of nodes in graph

    Returns:
        subgraph_nodes: Indices of nodes in subgraph
        subgraph_edges: Edge index for subgraph
    """
    # Start with center node
    subgraph_nodes = {node_idx}

    # Expand k hops
    for _ in range(num_hops):
        new_nodes = set()
        for node in subgraph_nodes:
            # Find neighbors
            mask = (edge_index[0] == node) | (edge_index[1] == node)
            neighbors = torch.cat([edge_index[0, mask], edge_index[1, mask]]).unique()
            new_nodes.update(neighbors.tolist())
        subgraph_nodes.update(new_nodes)

    subgraph_nodes = sorted(list(subgraph_nodes))
    subgraph_nodes_tensor = torch.tensor(subgraph_nodes)

    # Extract edges within subgraph
    node_set = set(subgraph_nodes)
    mask = torch.tensor([
        (edge_index[0, i].item() in node_set) and (edge_index[1, i].item() in node_set)
        for i in range(edge_index.size(1))
    ])

    subgraph_edges = edge_index[:, mask]

    # Remap node indices
    node_map = {old_idx: new_idx for new_idx, old_idx in enumerate(subgraph_nodes)}
    subgraph_edges_remapped = torch.tensor([
        [node_map[edge_index[0, i].item()], node_map[edge_index[1, i].item()]]
        for i in range(edge_index.size(1)) if mask[i]
    ]).t() if mask.any() else torch.zeros((2, 0), dtype=torch.long)

    return subgraph_nodes_tensor, subgraph_edges_remapped


def extract_link_subgraph(
    edge_index: torch.Tensor,
    head_idx: int,
    tail_idx: int,
    num_hops: int,
    num_nodes: int
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    Extract subgraph around both head and tail nodes for link prediction explanation.

    Args:
        edge_index: Edge index (2, num_edges)
        head_idx: Head node index
        tail_idx: Tail node index
        num_hops: Number of hops
        num_nodes: Total number of nodes

    Returns:
        subgraph_nodes: Nodes in subgraph
        subgraph_edges: Edges in subgraph
        adj_matrix: Adjacency matrix for subgraph
    """
    # Get subgraphs around head and tail
    head_nodes, _ = extract_k_hop_subgraph(edge_index, head_idx, num_hops, num_nodes)
    tail_nodes, _ = extract_k_hop_subgraph(edge_index, tail_idx, num_hops, num_nodes)

    # Union of nodes
    subgraph_nodes = torch.cat([head_nodes, tail_nodes]).unique()
    num_subgraph_nodes = len(subgraph_nodes)

    # Extract edges within subgraph
    node_set = set(subgraph_nodes.tolist())
    mask = torch.tensor([
        (edge_index[0, i].item() in node_set) and (edge_index[1, i].item() in node_set)
        for i in range(edge_index.size(1))
    ])

    subgraph_edges = edge_index[:, mask]

    # Create adjacency matrix
    adj_matrix = torch.zeros((num_subgraph_nodes, num_subgraph_nodes))
    node_map = {old_idx.item(): new_idx for new_idx, old_idx in enumerate(subgraph_nodes)}

    for i in range(subgraph_edges.size(1)):
        src = subgraph_edges[0, i].item()
        dst = subgraph_edges[1, i].item()
        if src in node_map and dst in node_map:
            adj_matrix[node_map[src], node_map[dst]] = 1.0

    return subgraph_nodes, subgraph_edges, adj_matrix
