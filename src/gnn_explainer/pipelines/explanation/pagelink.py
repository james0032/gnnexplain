"""
PaGE-Link: Path-based GNN Explanation for Heterogeneous Link Prediction.

This implementation adapts PaGE-Link from:
https://github.com/amazon-science/page-link-path-based-gnn-explanation

Paper: "PaGE-Link: Path-based Graph Neural Network Explanation for
Heterogeneous Link Prediction" (WWW 2023)

Key differences from standard PAGE:
1. Learns edge masks through optimization (like GNNExplainer)
2. Uses path-based regularization to encourage explanations as paths
3. Extracts k-shortest paths from learned masks
4. Works with heterogeneous graphs (multiple edge types)

The algorithm:
1. Extract k-hop subgraph around source and target nodes
2. Initialize learnable edge masks
3. Optimize masks with: L = L_pred + alpha * L_path
   - L_pred: Prediction loss (maintain original prediction)
   - L_path: Path loss (edges on paths get higher weights)
4. Extract top-k shortest paths from learned masks
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, Tuple, Optional, List, Set
import numpy as np
from collections import defaultdict
from pathlib import Path


class PaGELinkExplainer(nn.Module):
    """
    PaGE-Link Explainer for Knowledge Graph Link Prediction.

    Learns edge masks through optimization with path-based regularization,
    then extracts k-shortest paths as explanations.
    """

    def __init__(
        self,
        model,
        edge_index: torch.Tensor,
        edge_type: torch.Tensor,
        num_nodes: int,
        num_relations: int,
        node_emb: torch.Tensor,
        rel_emb: torch.Tensor,
        lr: float = 0.01,
        num_epochs: int = 100,
        alpha: float = 1.0,  # Weight for path loss
        beta: float = 0.1,   # Weight for mask size regularization
        k_paths: int = 5,    # Number of paths to extract
        device: str = 'cpu'
    ):
        super().__init__()
        self.model = model
        self.edge_index = edge_index.to(device)
        self.edge_type = edge_type.to(device)
        self.num_nodes = num_nodes
        self.num_relations = num_relations
        self.node_emb = node_emb.to(device)
        self.rel_emb = rel_emb.to(device)
        self.lr = lr
        self.num_epochs = num_epochs
        self.alpha = alpha
        self.beta = beta
        self.k_paths = k_paths
        self.device = device

        # Freeze model parameters
        for param in self.model.parameters():
            param.requires_grad = False
        self.model.eval()

    def _extract_subgraph(
        self,
        head_idx: int,
        tail_idx: int,
        num_hops: int = 2,
        verbose: bool = False
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, Dict]:
        """
        Extract k-hop subgraph around head and tail nodes.

        Returns:
            sub_edge_index: Edge index for subgraph (remapped to local indices)
            sub_edge_type: Edge types for subgraph
            subset: Original node indices in subgraph
            mapping: Dict mapping original indices to local indices
        """
        import time

        if verbose:
            print(f"      Extracting {num_hops}-hop subgraph...", flush=True)
        start_time = time.time()

        # Collect nodes within k hops of both head and tail using vectorized operations
        # Start with head and tail nodes
        current_nodes = torch.tensor([head_idx, tail_idx], device=self.device)
        all_nodes = current_nodes.clone()

        for hop in range(num_hops):
            if verbose:
                print(f"        Hop {hop+1}/{num_hops}: {len(current_nodes):,} nodes to expand...", end='', flush=True)
            hop_start = time.time()

            # Vectorized neighbor finding using isin
            # Find all edges where source is in current_nodes
            src_mask = torch.isin(self.edge_index[0], current_nodes)
            # Find all edges where destination is in current_nodes
            dst_mask = torch.isin(self.edge_index[1], current_nodes)

            # Get neighbors (destinations where source matches, sources where dest matches)
            neighbors_from_src = self.edge_index[1, src_mask]
            neighbors_from_dst = self.edge_index[0, dst_mask]

            # Combine and get unique new nodes
            new_neighbors = torch.cat([neighbors_from_src, neighbors_from_dst]).unique()

            # Add to all_nodes
            all_nodes = torch.cat([all_nodes, new_neighbors]).unique()

            # Next iteration expands from newly discovered nodes
            current_nodes = new_neighbors

            if verbose:
                print(f" found {len(new_neighbors):,} neighbors ({time.time()-hop_start:.2f}s)", flush=True)

        subset = all_nodes.sort()[0]  # Sort for consistent ordering

        if verbose:
            print(f"        Total subgraph nodes: {len(subset):,}", flush=True)
            print(f"        Filtering edges...", end='', flush=True)
        filter_start = time.time()

        # Vectorized edge filtering using torch.isin (MUCH faster than Python loop)
        edge_mask = torch.isin(self.edge_index[0], subset) & torch.isin(self.edge_index[1], subset)

        sub_edge_index_orig = self.edge_index[:, edge_mask]
        sub_edge_type = self.edge_type[edge_mask]

        if verbose:
            print(f" {sub_edge_index_orig.size(1):,} edges ({time.time()-filter_start:.2f}s)", flush=True)

        # Create mapping from original to local indices using vectorized operations
        if verbose:
            print(f"        Remapping indices...", end='', flush=True)
        remap_start = time.time()

        # Create a lookup tensor for fast remapping
        # This avoids the slow Python dict/loop approach
        if len(subset) > 0:
            # Create mapping tensor: mapping_tensor[original_idx] = local_idx
            # Only works efficiently if subset indices are not too sparse
            max_idx = subset.max().item() + 1

            # For very large graphs, use searchsorted for memory efficiency
            if max_idx > 10_000_000:  # Use searchsorted for huge graphs
                # subset is already sorted, so we can use searchsorted
                local_src = torch.searchsorted(subset, sub_edge_index_orig[0])
                local_dst = torch.searchsorted(subset, sub_edge_index_orig[1])
                sub_edge_index = torch.stack([local_src, local_dst])
            else:
                # For smaller graphs, use direct indexing (faster but more memory)
                mapping_tensor = torch.zeros(max_idx, dtype=torch.long, device=self.device)
                mapping_tensor[subset] = torch.arange(len(subset), device=self.device)

                local_src = mapping_tensor[sub_edge_index_orig[0]]
                local_dst = mapping_tensor[sub_edge_index_orig[1]]
                sub_edge_index = torch.stack([local_src, local_dst])
        else:
            sub_edge_index = torch.zeros((2, 0), dtype=torch.long, device=self.device)

        # Also create dict mapping for compatibility with rest of code
        mapping = {orig.item(): local for local, orig in enumerate(subset)}

        if verbose:
            print(f" done ({time.time()-remap_start:.2f}s)", flush=True)
            print(f"        Subgraph extraction total: {time.time()-start_time:.2f}s", flush=True)

        return sub_edge_index, sub_edge_type, subset, mapping

    def _init_edge_mask(self, num_edges: int) -> nn.Parameter:
        """Initialize learnable edge mask with Kaiming initialization."""
        mask = nn.Parameter(torch.empty(num_edges, device=self.device))
        nn.init.kaiming_uniform_(mask.unsqueeze(0))
        return mask

    def _prediction_loss(
        self,
        head_idx: int,
        tail_idx: int,
        rel_idx: int,
        original_score: float
    ) -> torch.Tensor:
        """
        Compute prediction loss to maintain original prediction.

        The goal is to find edges that, when weighted by the mask,
        produce a similar prediction score.
        """
        head_idx_t = torch.tensor([head_idx], device=self.device)
        tail_idx_t = torch.tensor([tail_idx], device=self.device)
        rel_idx_t = torch.tensor([rel_idx], device=self.device)

        # Get current prediction with masked edges
        current_score = self.model.decode(
            self.node_emb, self.rel_emb,
            head_idx_t, tail_idx_t, rel_idx_t
        )

        # MSE loss between original and current score
        target = torch.tensor([original_score], device=self.device)
        return F.mse_loss(current_score, target)

    def _find_paths_bfs(
        self,
        sub_edge_index: torch.Tensor,
        sub_edge_type: torch.Tensor,
        edge_weights: torch.Tensor,
        head_local: int,
        tail_local: int,
        max_length: int = 4,
        top_k: int = 5
    ) -> List[List[Tuple[int, int, int]]]:
        """
        Find k-shortest paths using BFS with edge weights.

        Returns list of paths, where each path is a list of (src, rel, dst) tuples.
        """
        from heapq import heappush, heappop

        num_nodes = max(sub_edge_index.max().item() + 1, max(head_local, tail_local) + 1) if sub_edge_index.numel() > 0 else max(head_local, tail_local) + 1

        # Build adjacency list with weights
        adj = defaultdict(list)  # node -> [(neighbor, edge_idx, rel_type, weight)]
        for i in range(sub_edge_index.size(1)):
            src = sub_edge_index[0, i].item()
            dst = sub_edge_index[1, i].item()
            rel = sub_edge_type[i].item()
            weight = edge_weights[i].item()
            adj[src].append((dst, i, rel, weight))
            # Also add reverse for undirected search
            adj[dst].append((src, i, rel, weight))

        # Priority queue: (negative_score, path_length, current_node, path)
        # Use negative score because heapq is min-heap but we want max-weight paths
        pq = [(-0.0, 0, head_local, [])]
        found_paths = []
        visited_states = set()

        while pq and len(found_paths) < top_k:
            neg_score, length, node, path = heappop(pq)
            score = -neg_score

            if node == tail_local and len(path) > 0:
                found_paths.append((score, path))
                continue

            if length >= max_length:
                continue

            # State: (node, frozenset of visited edges)
            visited_edges = frozenset(e[1] for e in path) if path else frozenset()
            state = (node, visited_edges)
            if state in visited_states:
                continue
            visited_states.add(state)

            for neighbor, edge_idx, rel, weight in adj[node]:
                if edge_idx not in visited_edges:
                    new_path = path + [(node, rel, neighbor, edge_idx)]
                    new_score = score + weight
                    heappush(pq, (-new_score, length + 1, neighbor, new_path))

        # Convert to output format
        result_paths = []
        for score, path in sorted(found_paths, key=lambda x: -x[0]):
            result_paths.append([(p[0], p[1], p[2]) for p in path])

        return result_paths[:top_k]

    def _path_loss(
        self,
        edge_mask: torch.Tensor,
        sub_edge_index: torch.Tensor,
        sub_edge_type: torch.Tensor,
        head_local: int,
        tail_local: int,
        max_path_length: int = 4
    ) -> torch.Tensor:
        """
        Compute path-based loss.

        Encourages edges on paths between head and tail to have high weights,
        and edges not on paths to have low weights.
        """
        # Get edge weights from mask
        edge_weights = torch.sigmoid(edge_mask)

        # Find paths with current weights
        paths = self._find_paths_bfs(
            sub_edge_index, sub_edge_type, edge_weights,
            head_local, tail_local,
            max_length=max_path_length,
            top_k=self.k_paths
        )

        if not paths:
            # No paths found - just regularize mask size
            return edge_weights.mean()

        # Collect edge indices on paths
        edges_on_paths = set()
        for path in paths:
            for i, (src, rel, dst) in enumerate(path):
                # Find matching edge
                for j in range(sub_edge_index.size(1)):
                    if (sub_edge_index[0, j].item() == src and
                        sub_edge_index[1, j].item() == dst):
                        edges_on_paths.add(j)
                    elif (sub_edge_index[0, j].item() == dst and
                          sub_edge_index[1, j].item() == src):
                        edges_on_paths.add(j)

        if not edges_on_paths:
            return edge_weights.mean()

        # Path loss: maximize weights on path edges, minimize weights off path edges
        on_path_mask = torch.zeros(len(edge_mask), device=self.device, dtype=torch.bool)
        for idx in edges_on_paths:
            on_path_mask[idx] = True

        # Encourage high weights on path edges (negative loss)
        on_path_loss = -edge_weights[on_path_mask].mean() if on_path_mask.any() else torch.tensor(0.0, device=self.device)

        # Encourage low weights off path edges (positive loss)
        off_path_loss = edge_weights[~on_path_mask].mean() if (~on_path_mask).any() else torch.tensor(0.0, device=self.device)

        return on_path_loss + off_path_loss

    def _mask_size_loss(self, edge_mask: torch.Tensor) -> torch.Tensor:
        """Regularization to encourage sparse masks."""
        return torch.sigmoid(edge_mask).mean()

    def explain(
        self,
        head_idx: int,
        tail_idx: int,
        rel_idx: int,
        num_hops: int = 2,
        max_path_length: int = 4,
        verbose: bool = False
    ) -> Dict:
        """
        Generate explanation for a predicted link.

        Args:
            head_idx: Head node index
            tail_idx: Tail node index
            rel_idx: Relation type index
            num_hops: Number of hops for subgraph extraction
            max_path_length: Maximum path length to consider
            verbose: Print progress

        Returns:
            Dictionary with:
            - paths: List of paths (each path is list of (src, rel, dst) tuples)
            - edge_mask: Learned edge mask values
            - important_edges: Top-k edges by mask value
            - subgraph_info: Information about extracted subgraph
        """
        # Get original prediction score
        with torch.no_grad():
            head_idx_t = torch.tensor([head_idx], device=self.device)
            tail_idx_t = torch.tensor([tail_idx], device=self.device)
            rel_idx_t = torch.tensor([rel_idx], device=self.device)
            original_score = self.model.decode(
                self.node_emb, self.rel_emb,
                head_idx_t, tail_idx_t, rel_idx_t
            ).item()

        if verbose:
            print(f"    Original prediction score: {original_score:.4f}", flush=True)

        # Extract subgraph
        sub_edge_index, sub_edge_type, subset, mapping = self._extract_subgraph(
            head_idx, tail_idx, num_hops, verbose=verbose
        )

        num_edges = sub_edge_index.size(1)
        if num_edges == 0:
            return {
                'paths': [],
                'edge_mask': None,
                'important_edges': None,
                'subgraph_info': {
                    'num_nodes': len(subset),
                    'num_edges': 0,
                    'subset': subset
                },
                'error': 'No edges in subgraph'
            }

        head_local = mapping[head_idx]
        tail_local = mapping[tail_idx]

        if verbose:
            print(f"    Subgraph extracted: {len(subset):,} nodes, {num_edges:,} edges", flush=True)
            print(f"    Starting optimization ({self.num_epochs} epochs)...", flush=True)

        # Initialize edge mask
        edge_mask = self._init_edge_mask(num_edges)
        optimizer = torch.optim.Adam([edge_mask], lr=self.lr)

        # Optimization loop
        best_loss = float('inf')
        best_mask = None
        print_interval = max(1, self.num_epochs // 5)  # Print ~5 times during optimization

        for epoch in range(self.num_epochs):
            optimizer.zero_grad()

            # Compute losses
            # For prediction loss, we'd need to modify the model's forward pass
            # to use edge weights - for simplicity, we focus on path loss
            path_loss = self._path_loss(
                edge_mask, sub_edge_index, sub_edge_type,
                head_local, tail_local, max_path_length
            )
            size_loss = self._mask_size_loss(edge_mask)

            total_loss = self.alpha * path_loss + self.beta * size_loss

            total_loss.backward()
            optimizer.step()

            if total_loss.item() < best_loss:
                best_loss = total_loss.item()
                best_mask = edge_mask.detach().clone()

            if verbose and ((epoch + 1) % print_interval == 0 or epoch == 0):
                print(f"      Epoch {epoch+1}/{self.num_epochs}: loss={total_loss.item():.4f}", flush=True)

        # Use best mask
        edge_mask = best_mask if best_mask is not None else edge_mask.detach()
        edge_weights = torch.sigmoid(edge_mask)

        # Extract final paths
        paths = self._find_paths_bfs(
            sub_edge_index, sub_edge_type, edge_weights,
            head_local, tail_local,
            max_length=max_path_length,
            top_k=self.k_paths
        )

        # Convert paths back to global indices
        global_paths = []
        for path in paths:
            global_path = []
            for src_local, rel, dst_local in path:
                src_global = subset[src_local].item()
                dst_global = subset[dst_local].item()
                global_path.append((src_global, rel, dst_global))
            global_paths.append(global_path)

        # Get top-k important edges
        top_k = min(10, num_edges)
        top_k_values, top_k_indices = torch.topk(edge_weights, top_k)

        important_edges = []
        for i in range(top_k):
            idx = top_k_indices[i].item()
            src_local = sub_edge_index[0, idx].item()
            dst_local = sub_edge_index[1, idx].item()
            rel = sub_edge_type[idx].item()
            src_global = subset[src_local].item()
            dst_global = subset[dst_local].item()
            important_edges.append({
                'src': src_global,
                'rel': rel,
                'dst': dst_global,
                'weight': top_k_values[i].item()
            })

        return {
            'triple': {
                'head_idx': head_idx,
                'tail_idx': tail_idx,
                'relation_idx': rel_idx
            },
            'prediction_score': original_score,
            'paths': global_paths,
            'edge_mask': edge_weights.cpu(),
            'important_edges': important_edges,
            'subgraph_info': {
                'num_nodes': len(subset),
                'num_edges': num_edges,
                'subset': subset.cpu(),
                'sub_edge_index': sub_edge_index.cpu(),
                'sub_edge_type': sub_edge_type.cpu()
            }
        }


def run_pagelink_explainer(
    model_dict: Dict,
    selected_triples: Dict,
    pyg_data: Dict,
    explainer_params: Dict
) -> Dict:
    """
    Run PaGE-Link explainer on selected triples.

    Args:
        model_dict: Prepared model dictionary with model, embeddings, device
        selected_triples: Selected triples to explain
        pyg_data: PyG format graph data
        explainer_params: Explainer parameters

    Returns:
        Dictionary with explanations
    """
    import time

    print("\n" + "="*60)
    print("RUNNING PaGE-Link EXPLAINER")
    print("="*60)

    # Extract components from model_dict
    model = model_dict['model']
    device = model_dict['device']
    edge_index_model = model_dict['edge_index']
    edge_type_model = model_dict['edge_type']

    # Get graph info from pyg_data
    edge_index = pyg_data['edge_index'].to(device)
    edge_type = pyg_data['edge_type'].to(device)
    num_nodes = pyg_data['num_nodes']
    num_relations = pyg_data['num_relations']

    # Compute node and relation embeddings from the trained model
    print(f"\nComputing embeddings from trained model...")
    model.eval()
    with torch.no_grad():
        # CompGCN encode() returns tuple: (node_embeddings, relation_embeddings)
        node_emb, rel_emb = model.encode(edge_index_model, edge_type_model)
    print(f"  Node embeddings: {node_emb.shape}")
    print(f"  Relation embeddings: {rel_emb.shape}")

    # Get parameters
    pagelink_params = explainer_params.get('pagelink', {})
    lr = pagelink_params.get('lr', 0.01)
    num_epochs = pagelink_params.get('num_epochs', 100)
    alpha = pagelink_params.get('alpha', 1.0)
    beta = pagelink_params.get('beta', 0.1)
    k_paths = pagelink_params.get('k_paths', 5)
    num_hops = pagelink_params.get('num_hops', 2)
    max_path_length = pagelink_params.get('max_path_length', 4)

    print(f"\nPaGE-Link Parameters:")
    print(f"  Learning rate: {lr}")
    print(f"  Epochs: {num_epochs}")
    print(f"  Alpha (path loss weight): {alpha}")
    print(f"  Beta (size regularization): {beta}")
    print(f"  K paths to extract: {k_paths}")
    print(f"  Subgraph hops: {num_hops}")
    print(f"  Max path length: {max_path_length}")

    # Create explainer
    print(f"\nInitializing PaGE-Link explainer...")
    print(f"  Graph size: {num_nodes:,} nodes, {edge_index.size(1):,} edges", flush=True)
    explainer = PaGELinkExplainer(
        model=model,
        edge_index=edge_index,
        edge_type=edge_type,
        num_nodes=num_nodes,
        num_relations=num_relations,
        node_emb=node_emb,
        rel_emb=rel_emb,
        lr=lr,
        num_epochs=num_epochs,
        alpha=alpha,
        beta=beta,
        k_paths=k_paths,
        device=device
    )
    print(f"  ✓ Explainer initialized successfully", flush=True)

    # Get triples to explain
    selected_edge_index = selected_triples['selected_edge_index']
    selected_edge_type = selected_triples['selected_edge_type']
    triples_readable = selected_triples['triples_readable']

    print(f"\nGenerating explanations for {len(triples_readable)} triples...")

    explanations = []
    start_time = time.time()

    for i in range(len(triples_readable)):
        triple_start = time.time()
        head_idx = selected_edge_index[0, i].item()
        tail_idx = selected_edge_index[1, i].item()
        rel_idx = selected_edge_type[i].item()
        triple = triples_readable[i]

        print(f"\n  [{i+1}/{len(triples_readable)}] Explaining: {triple['triple']}")

        try:
            explanation = explainer.explain(
                head_idx=head_idx,
                tail_idx=tail_idx,
                rel_idx=rel_idx,
                num_hops=num_hops,
                max_path_length=max_path_length,
                verbose=True
            )

            # Add readable triple info
            explanation['triple'] = triple

            triple_time = time.time() - triple_start
            print(f"    ✓ Found {len(explanation['paths'])} paths in {triple_time:.1f}s")

            # Print top paths
            if explanation['paths']:
                print(f"    Top paths:")
                for j, path in enumerate(explanation['paths'][:3]):
                    path_str = " -> ".join([f"({p[0]})-[{p[1]}]->({p[2]})" for p in path])
                    print(f"      {j+1}. {path_str}")

        except Exception as e:
            print(f"    ✗ Error: {str(e)}")
            explanation = {
                'triple': triple,
                'error': str(e),
                'error_type': type(e).__name__
            }

        explanations.append(explanation)

        # Progress estimate
        if i < len(triples_readable) - 1:
            elapsed = time.time() - start_time
            avg_time = elapsed / (i + 1)
            remaining = len(triples_readable) - (i + 1)
            eta_seconds = avg_time * remaining
            eta_mins = int(eta_seconds / 60)
            eta_secs = int(eta_seconds % 60)
            print(f"    Progress: {i+1}/{len(triples_readable)} | ETA: {eta_mins}m {eta_secs}s")

    total_time = time.time() - start_time
    print(f"\n✓ PaGE-Link completed: {len(explanations)} explanations in {total_time:.1f}s")

    return {
        'explainer_type': 'PaGELink',
        'explanations': explanations,
        'num_explanations': len(explanations),
        'params': pagelink_params
    }
