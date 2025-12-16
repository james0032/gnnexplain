"""
Improved PAGE (Parametric Generative Explainer) for Knowledge Graph Link Prediction.

This implementation combines:
- Option 3: Integrated CompGCN-VGAE (uses frozen encoder features)
- Option 2: Prediction-Aware Training (weighted by model predictions)

Goal: Explain "Why did the CompGCN model predict this triple?"
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, Tuple, Optional, List
import numpy as np


class CompGCNFeatureExtractor(nn.Module):
    """
    Wrapper to extract frozen CompGCN features for subgraphs.

    Uses the trained CompGCN encoder to get contextualized embeddings,
    then extracts subgraph features.
    """

    def __init__(self, compgcn_model, edge_index, edge_type, device='cpu'):
        super().__init__()
        self.compgcn_model = compgcn_model
        self.edge_index = edge_index.to(device)
        self.edge_type = edge_type.to(device)
        self.device = device

        # Freeze CompGCN parameters
        for param in self.compgcn_model.parameters():
            param.requires_grad = False

        self.compgcn_model.eval()

    @torch.no_grad()
    def extract_full_embeddings(self):
        """Extract embeddings for all nodes using CompGCN encoder."""
        node_emb, rel_emb = self.compgcn_model.encode(self.edge_index, self.edge_type)
        return node_emb, rel_emb

    def extract_subgraph_features(self, subgraph_nodes, node_emb):
        """
        Extract features for a subgraph using pre-computed embeddings.

        Args:
            subgraph_nodes: Indices of nodes in subgraph
            node_emb: Full graph node embeddings

        Returns:
            Subgraph node features (num_subgraph_nodes, embedding_dim)
        """
        return node_emb[subgraph_nodes]


class IntegratedVGAE(nn.Module):
    """
    VGAE that uses CompGCN embeddings as input features.

    Architecture:
    - Input: CompGCN node embeddings (frozen, pre-computed)
    - Encoder: Simple GCN layers to encode into latent space
    - Decoder: MLP-based decoder to reconstruct adjacency
    """

    def __init__(
        self,
        input_dim,  # CompGCN embedding dimension
        hidden_dim=32,
        latent_dim=16,
        decoder_hidden_dim=16,
        dropout=0.0
    ):
        super().__init__()

        self.input_dim = input_dim
        self.latent_dim = latent_dim

        # Encoder: CompGCN embeddings -> Latent space
        self.fc_hidden = nn.Linear(input_dim, hidden_dim)
        self.fc_mu = nn.Linear(hidden_dim, latent_dim)
        self.fc_logvar = nn.Linear(hidden_dim, latent_dim)

        # Decoder: Latent space -> Adjacency reconstruction
        self.fc_decode1 = nn.Linear(latent_dim, decoder_hidden_dim)
        self.fc_decode2 = nn.Linear(decoder_hidden_dim, decoder_hidden_dim)

        self.dropout = dropout

    def encode(self, x):
        """
        Encode CompGCN features into latent space.

        Args:
            x: CompGCN embeddings (batch_size, num_nodes, input_dim)

        Returns:
            mu, logvar: Latent distribution parameters
        """
        # x is already contextualized from CompGCN, so we just need to map to latent space
        h = F.relu(self.fc_hidden(x))
        h = F.dropout(h, self.dropout, training=self.training)

        mu = self.fc_mu(h)
        logvar = self.fc_logvar(h)

        return mu, logvar

    def reparameterize(self, mu, logvar):
        """Reparameterization trick."""
        if self.training:
            std = torch.exp(0.5 * logvar)
            eps = torch.randn_like(std)
            return mu + eps * std
        else:
            return mu

    def decode(self, z):
        """
        Decode latent representation to adjacency matrix.

        Args:
            z: Latent representation (batch_size, num_nodes, latent_dim)

        Returns:
            Reconstructed adjacency (batch_size, num_nodes, num_nodes)
        """
        # Pass through MLP
        h = F.relu(self.fc_decode1(z))
        h = torch.sigmoid(self.fc_decode2(h))
        h = F.dropout(h, self.dropout, training=self.training)

        # Inner product for adjacency
        adj_reconstructed = torch.bmm(h, h.transpose(1, 2))
        adj_reconstructed = torch.sigmoid(adj_reconstructed)

        return adj_reconstructed

    def forward(self, x):
        """
        Full forward pass.

        Args:
            x: CompGCN embeddings (batch_size, num_nodes, input_dim)

        Returns:
            adj_reconstructed, mu, logvar, z
        """
        mu, logvar = self.encode(x)
        z = self.reparameterize(mu, logvar)
        adj_reconstructed = self.decode(z)

        return adj_reconstructed, mu, logvar, z


def prediction_aware_vgae_loss(
    adj_reconstructed,
    adj_true,
    mu,
    logvar,
    prediction_score,
    kl_weight=0.2,
    prediction_weight=1.0
):
    """
    Compute prediction-aware VGAE loss.

    The key idea: High-confidence predictions should have more accurate
    reconstructions. We weight the reconstruction loss by the prediction score.

    Args:
        adj_reconstructed: Reconstructed adjacency
        adj_true: True adjacency
        mu: Latent mean
        logvar: Latent log variance
        prediction_score: CompGCN's prediction score for this triple
        kl_weight: Weight for KL divergence
        prediction_weight: Weight for prediction-awareness

    Returns:
        total_loss, recon_loss, kl_div, weighted_recon_loss
    """
    # Standard reconstruction loss (BCE)
    recon_loss = F.binary_cross_entropy(adj_reconstructed, adj_true, reduction='mean')

    # KL divergence
    kl_div = -0.5 * torch.mean(1 + logvar - mu.pow(2) - logvar.exp())

    # Prediction-aware weighting
    # Higher scores -> more weight on reconstruction
    # This makes the model focus on explaining high-confidence predictions
    score_weight = torch.sigmoid(prediction_score)  # Normalize to [0, 1]
    weighted_recon_loss = recon_loss * (1.0 + prediction_weight * score_weight)

    # Total loss
    total_loss = weighted_recon_loss + kl_weight * kl_div

    return total_loss, recon_loss, kl_div, weighted_recon_loss


class ImprovedPAGEExplainer:
    """
    Improved PAGE Explainer that uses CompGCN features and prediction scores.

    Key improvements:
    1. Uses frozen CompGCN encoder embeddings (Option 3)
    2. Prediction-aware training (Option 2)
    3. Faithful to model predictions

    Goal: Explain "Why did CompGCN predict this triple?"
    """

    def __init__(
        self,
        compgcn_model,
        edge_index,
        edge_type,
        embedding_dim,  # CompGCN embedding dimension
        hidden_dim=32,
        latent_dim=16,
        decoder_hidden_dim=16,
        dropout=0.0,
        device='cpu'
    ):
        self.device = device
        self.embedding_dim = embedding_dim

        # Feature extractor (frozen CompGCN)
        self.feature_extractor = CompGCNFeatureExtractor(
            compgcn_model=compgcn_model,
            edge_index=edge_index,
            edge_type=edge_type,
            device=device
        )

        # Extract embeddings once (frozen)
        print("Extracting CompGCN embeddings for full graph...")
        self.node_emb, self.rel_emb = self.feature_extractor.extract_full_embeddings()
        print(f"✓ Extracted embeddings: nodes={self.node_emb.shape}, relations={self.rel_emb.shape}")

        # VGAE (trainable)
        self.vgae = IntegratedVGAE(
            input_dim=embedding_dim,
            hidden_dim=hidden_dim,
            latent_dim=latent_dim,
            decoder_hidden_dim=decoder_hidden_dim,
            dropout=dropout
        ).to(device)

        # Store CompGCN model for scoring
        self.compgcn_model = compgcn_model

    def get_subgraph_features(self, subgraph_nodes):
        """Get CompGCN features for a subgraph."""
        return self.feature_extractor.extract_subgraph_features(
            subgraph_nodes,
            self.node_emb
        )

    def get_triple_score(self, head_idx, tail_idx, rel_idx):
        """Get CompGCN's prediction score for a triple."""
        with torch.no_grad():
            head_idx_t = torch.tensor([head_idx], device=self.device)
            tail_idx_t = torch.tensor([tail_idx], device=self.device)
            rel_idx_t = torch.tensor([rel_idx], device=self.device)

            score = self.compgcn_model.decode(
                self.node_emb,
                self.rel_emb,
                head_idx_t,
                tail_idx_t,
                rel_idx_t
            )

            return score.item()

    def train_on_subgraphs(
        self,
        subgraphs_data: List[Dict],
        epochs=100,
        lr=0.003,
        kl_weight=0.2,
        prediction_weight=1.0,
        verbose=True
    ):
        """
        Train VGAE on subgraphs with prediction-aware loss.

        Args:
            subgraphs_data: List of dicts with:
                - 'features': CompGCN embeddings for subgraph
                - 'adj': Adjacency matrix
                - 'prediction_score': CompGCN score for this triple
            epochs: Training epochs
            lr: Learning rate
            kl_weight: KL divergence weight
            prediction_weight: Weight for prediction-awareness
            verbose: Print progress
        """
        optimizer = torch.optim.Adam(self.vgae.parameters(), lr=lr)

        self.vgae.train()

        for epoch in range(epochs):
            total_loss = 0
            total_recon = 0
            total_kl = 0
            total_weighted_recon = 0

            for data in subgraphs_data:
                x = data['features'].to(self.device)
                adj = data['adj'].to(self.device)
                pred_score = torch.tensor(data['prediction_score'], device=self.device)

                optimizer.zero_grad()

                # Forward pass
                adj_reconstructed, mu, logvar, z = self.vgae(x)

                # Prediction-aware loss
                loss, recon_loss, kl_div, weighted_recon = prediction_aware_vgae_loss(
                    adj_reconstructed,
                    adj,
                    mu,
                    logvar,
                    pred_score,
                    kl_weight,
                    prediction_weight
                )

                # Backward pass
                loss.backward()
                optimizer.step()

                total_loss += loss.item()
                total_recon += recon_loss.item()
                total_kl += kl_div.item()
                total_weighted_recon += weighted_recon.item()

            if verbose and (epoch + 1) % 10 == 0:
                avg_loss = total_loss / len(subgraphs_data)
                avg_recon = total_recon / len(subgraphs_data)
                avg_weighted_recon = total_weighted_recon / len(subgraphs_data)
                avg_kl = total_kl / len(subgraphs_data)

                print(f"  Epoch [{epoch+1}/{epochs}]: "
                      f"Loss={avg_loss:.4f}, "
                      f"Recon={avg_recon:.4f}, "
                      f"WeightedRecon={avg_weighted_recon:.4f}, "
                      f"KL={avg_kl:.4f}", flush=True)

    def explain(self, x, adj):
        """
        Generate explanation for a subgraph.

        Args:
            x: CompGCN features (batch_size, num_nodes, embedding_dim)
            adj: Adjacency matrix (batch_size, num_nodes, num_nodes)

        Returns:
            edge_importance: Importance scores for each edge
            latent_representation: Latent space representation
        """
        self.vgae.eval()

        with torch.no_grad():
            x = x.to(self.device)
            adj = adj.to(self.device)

            # Encode to latent space
            mu, logvar = self.vgae.encode(x)
            z = mu  # Use mean for explanation

            # Decode
            adj_reconstructed = self.vgae.decode(z)

            # Edge importance = reconstruction quality
            # Higher reconstruction accuracy = more important edge
            edge_importance = adj_reconstructed * adj  # Only score existing edges

            # Alternative: use reconstruction error for existing edges
            # edge_importance = torch.abs(adj_reconstructed - adj) * adj

        return edge_importance, z

    def save(self, path):
        """Save trained VGAE model."""
        torch.save(self.vgae.state_dict(), path)

    def load(self, path):
        """Load trained VGAE model."""
        self.vgae.load_state_dict(torch.load(path, map_location=self.device))


# Utility functions (reuse from page_simple.py)

def extract_k_hop_subgraph(
    edge_index: torch.Tensor,
    node_idx: int,
    num_hops: int,
    num_nodes: int
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Extract k-hop subgraph around a node."""
    subgraph_nodes = {node_idx}

    for _ in range(num_hops):
        new_nodes = set()
        for node in subgraph_nodes:
            mask = (edge_index[0] == node) | (edge_index[1] == node)
            neighbors = torch.cat([edge_index[0, mask], edge_index[1, mask]]).unique()
            new_nodes.update(neighbors.tolist())
        subgraph_nodes.update(new_nodes)

    subgraph_nodes = sorted(list(subgraph_nodes))
    subgraph_nodes_tensor = torch.tensor(subgraph_nodes)

    node_set = set(subgraph_nodes)
    mask = torch.tensor([
        (edge_index[0, i].item() in node_set) and (edge_index[1, i].item() in node_set)
        for i in range(edge_index.size(1))
    ])

    subgraph_edges = edge_index[:, mask]

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
    num_nodes: int,
    method: str = 'khop',
    edge_type: Optional[torch.Tensor] = None,
    max_path_length: int = 3
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    Extract subgraph around both head and tail nodes.

    Args:
        edge_index: Full graph edge index
        head_idx: Head node index
        tail_idx: Tail node index
        num_hops: Number of hops (for khop method)
        num_nodes: Total number of nodes in graph
        method: 'khop' or 'paths'
        edge_type: Edge types tensor (needed for paths method)
        max_path_length: Maximum path length (for paths method)

    Returns:
        subgraph_nodes, subgraph_edges, adj_matrix
    """
    if method == 'paths' and edge_type is not None:
        # Use path-based extraction (same as in nodes.py)
        try:
            from ..explanation.nodes import extract_path_based_subgraph

            # Extract path-based subgraph
            device = edge_index.device
            subset, sub_edge_index, mapping, edge_mask = extract_path_based_subgraph(
                head_idx, tail_idx, edge_index, edge_type, max_path_length, device
            )

            subgraph_nodes = subset
            num_subgraph_nodes = len(subgraph_nodes)

            # Extract subgraph edges from original edge_index
            subgraph_edges = edge_index[:, edge_mask]

            # Build adjacency matrix
            adj_matrix = torch.zeros((num_subgraph_nodes, num_subgraph_nodes))
            node_map = {old_idx.item(): new_idx for new_idx, old_idx in enumerate(subgraph_nodes)}

            for i in range(subgraph_edges.size(1)):
                src = subgraph_edges[0, i].item()
                dst = subgraph_edges[1, i].item()
                if src in node_map and dst in node_map:
                    adj_matrix[node_map[src], node_map[dst]] = 1.0

            return subgraph_nodes, subgraph_edges, adj_matrix

        except Exception as e:
            print(f"Warning: Path-based extraction failed ({e}), falling back to k-hop")
            # Fall through to k-hop method

    # K-hop method (default)
    head_nodes, _ = extract_k_hop_subgraph(edge_index, head_idx, num_hops, num_nodes)
    tail_nodes, _ = extract_k_hop_subgraph(edge_index, tail_idx, num_hops, num_nodes)

    subgraph_nodes = torch.cat([head_nodes, tail_nodes]).unique()
    num_subgraph_nodes = len(subgraph_nodes)

    node_set = set(subgraph_nodes.tolist())
    mask = torch.tensor([
        (edge_index[0, i].item() in node_set) and (edge_index[1, i].item() in node_set)
        for i in range(edge_index.size(1))
    ])

    subgraph_edges = edge_index[:, mask]

    adj_matrix = torch.zeros((num_subgraph_nodes, num_subgraph_nodes))
    node_map = {old_idx.item(): new_idx for new_idx, old_idx in enumerate(subgraph_nodes)}

    for i in range(subgraph_edges.size(1)):
        src = subgraph_edges[0, i].item()
        dst = subgraph_edges[1, i].item()
        if src in node_map and dst in node_map:
            adj_matrix[node_map[src], node_map[dst]] = 1.0

    return subgraph_nodes, subgraph_edges, adj_matrix
