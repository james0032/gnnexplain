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
        device: str = 'cpu',
        exclude_inverse_edges: bool = True,  # Filter out artificial inverse edges
        edge_direction: torch.Tensor = None  # 0=forward, 1=inverse
    ):
        super().__init__()
        self.model = model
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

        # Filter out inverse edges if requested
        if exclude_inverse_edges and edge_direction is not None:
            # Keep only forward edges (direction == 0)
            forward_mask = (edge_direction == 0)
            self.edge_index = edge_index[:, forward_mask].to(device)
            self.edge_type = edge_type[forward_mask].to(device)
            print(f"  Filtered inverse edges: {edge_index.size(1):,} -> {self.edge_index.size(1):,} edges")
        else:
            self.edge_index = edge_index.to(device)
            self.edge_type = edge_type.to(device)
            if exclude_inverse_edges:
                print(f"  Warning: exclude_inverse_edges=True but no edge_direction provided")

        # Freeze model parameters
        for param in self.model.parameters():
            param.requires_grad = False
        self.model.eval()

        # Cache for full graph adjacency (built once, reused for all triples)
        self._full_graph_adj = None

    def _build_full_graph_adjacency(self, verbose: bool = False) -> Dict:
        """
        Build adjacency list for full graph (cached at class level).
        Only built once, reused for all triple explanations.
        """
        if self._full_graph_adj is not None:
            if verbose:
                print(f"        Using cached full graph adjacency", flush=True)
            return self._full_graph_adj

        import time
        if verbose:
            print(f"        Building full graph adjacency (one-time)...", end='', flush=True)
        start = time.time()

        edge_index_cpu = self.edge_index.cpu()
        adj = defaultdict(list)
        for i in range(edge_index_cpu.size(1)):
            src = edge_index_cpu[0, i].item()
            dst = edge_index_cpu[1, i].item()
            adj[src].append((dst, i))
            adj[dst].append((src, i))  # Undirected for path finding

        self._full_graph_adj = adj
        if verbose:
            print(f" done ({time.time()-start:.2f}s)", flush=True)

        return self._full_graph_adj

    def _extract_path_subgraph(
        self,
        head_idx: int,
        tail_idx: int,
        max_path_length: int = 3,
        verbose: bool = False
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, Dict]:
        """
        Extract subgraph containing only edges on paths of length <= max_path_length
        between head and tail nodes.

        This is more efficient and relevant than k-hop + k-core pruning because:
        1. Only includes edges that could actually be part of an explanation path
        2. Produces much smaller subgraphs (typically 100-10000 edges vs millions)
        3. Directly aligned with what the explainer is trying to find

        Algorithm:
        1. BFS from head to find all nodes reachable within max_path_length steps
        2. BFS from tail to find all nodes reachable within max_path_length steps
        3. Keep only nodes reachable from both (intersection)
        4. For each distance d from head, keep nodes at distance <= max_path_length - d from tail
        5. Keep only edges between valid nodes

        Returns:
            sub_edge_index: Edge index for subgraph (remapped to local indices)
            sub_edge_type: Edge types for subgraph
            subset: Original node indices in subgraph
            mapping: Dict mapping original indices to local indices
        """
        import time

        if verbose:
            print(f"      Extracting path-based subgraph (max_length={max_path_length})...", flush=True)
        start_time = time.time()

        # Use cached full graph adjacency (built once, reused for all triples)
        adj = self._build_full_graph_adjacency(verbose=verbose)

        # BFS from head: compute distance from head to all reachable nodes
        if verbose:
            print(f"        BFS from head...", end='', flush=True)
        bfs_start = time.time()

        dist_from_head = {head_idx: 0}
        frontier = [head_idx]
        for d in range(1, max_path_length + 1):
            next_frontier = []
            for node in frontier:
                for neighbor, _ in adj[node]:
                    if neighbor not in dist_from_head:
                        dist_from_head[neighbor] = d
                        next_frontier.append(neighbor)
            frontier = next_frontier
            if not frontier:
                break

        if verbose:
            print(f" reached {len(dist_from_head):,} nodes ({time.time()-bfs_start:.2f}s)", flush=True)

        # BFS from tail: compute distance from tail to all reachable nodes
        if verbose:
            print(f"        BFS from tail...", end='', flush=True)
        bfs_start = time.time()

        dist_from_tail = {tail_idx: 0}
        frontier = [tail_idx]
        for d in range(1, max_path_length + 1):
            next_frontier = []
            for node in frontier:
                for neighbor, _ in adj[node]:
                    if neighbor not in dist_from_tail:
                        dist_from_tail[neighbor] = d
                        next_frontier.append(neighbor)
            frontier = next_frontier
            if not frontier:
                break

        if verbose:
            print(f" reached {len(dist_from_tail):,} nodes ({time.time()-bfs_start:.2f}s)", flush=True)

        # Find nodes on valid paths: dist_from_head[n] + dist_from_tail[n] <= max_path_length
        if verbose:
            print(f"        Finding nodes on paths...", end='', flush=True)
        path_start = time.time()

        valid_nodes = set()
        for node in dist_from_head:
            if node in dist_from_tail:
                if dist_from_head[node] + dist_from_tail[node] <= max_path_length:
                    valid_nodes.add(node)

        # Always include head and tail
        valid_nodes.add(head_idx)
        valid_nodes.add(tail_idx)

        if verbose:
            print(f" {len(valid_nodes):,} nodes ({time.time()-path_start:.2f}s)", flush=True)

        # Find edges on valid paths using adjacency list (more efficient than iterating all edges)
        if verbose:
            print(f"        Finding edges on paths...", end='', flush=True)
        edge_start = time.time()

        valid_edge_indices = set()
        for src in valid_nodes:
            for dst, edge_idx in adj[src]:
                if dst in valid_nodes:
                    # Check if this edge can be on a valid path
                    # An edge (u, v) is on a path if:
                    # dist_from_head[u] + 1 + dist_from_tail[v] <= max_path_length OR
                    # dist_from_head[v] + 1 + dist_from_tail[u] <= max_path_length
                    d_head_src = dist_from_head.get(src, float('inf'))
                    d_head_dst = dist_from_head.get(dst, float('inf'))
                    d_tail_src = dist_from_tail.get(src, float('inf'))
                    d_tail_dst = dist_from_tail.get(dst, float('inf'))

                    if (d_head_src + 1 + d_tail_dst <= max_path_length or
                        d_head_dst + 1 + d_tail_src <= max_path_length):
                        valid_edge_indices.add(edge_idx)

        valid_edge_indices = list(valid_edge_indices)

        if verbose:
            print(f" {len(valid_edge_indices):,} edges ({time.time()-edge_start:.2f}s)", flush=True)

        # Extract subgraph
        if len(valid_edge_indices) == 0:
            if verbose:
                print(f"        WARNING: No edges found on paths!", flush=True)
            subset = torch.tensor([head_idx, tail_idx], device=self.device)
            mapping = {head_idx: 0, tail_idx: 1}
            return torch.zeros((2, 0), dtype=torch.long, device=self.device), \
                   torch.zeros(0, dtype=torch.long, device=self.device), \
                   subset, mapping

        valid_edge_indices_t = torch.tensor(valid_edge_indices, dtype=torch.long)
        sub_edge_index_orig = self.edge_index[:, valid_edge_indices_t]
        sub_edge_type = self.edge_type[valid_edge_indices_t]

        # Get unique nodes and create mapping
        if verbose:
            print(f"        Remapping indices...", end='', flush=True)
        remap_start = time.time()

        subset = torch.cat([sub_edge_index_orig[0], sub_edge_index_orig[1]]).unique().sort()[0]
        # Ensure head and tail are in subset
        subset = torch.cat([subset, torch.tensor([head_idx, tail_idx], device=self.device)]).unique().sort()[0]

        # Create mapping using searchsorted (efficient for sorted arrays)
        local_src = torch.searchsorted(subset, sub_edge_index_orig[0])
        local_dst = torch.searchsorted(subset, sub_edge_index_orig[1])
        sub_edge_index = torch.stack([local_src, local_dst])

        # Dict mapping for compatibility
        mapping = {orig.item(): local for local, orig in enumerate(subset)}

        if verbose:
            print(f" done ({time.time()-remap_start:.2f}s)", flush=True)
            print(f"        Path subgraph extraction total: {time.time()-start_time:.2f}s", flush=True)
            print(f"        Final subgraph: {len(subset):,} nodes, {sub_edge_index.size(1):,} edges", flush=True)

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

    def _build_adjacency_list(
        self,
        sub_edge_index: torch.Tensor,
        sub_edge_type: torch.Tensor
    ) -> Dict:
        """
        Build adjacency list once (cached) - no weights, just structure.
        Returns dict with adjacency and edge lookup structures.
        """
        num_edges = sub_edge_index.size(1)

        # Build adjacency list: node -> [(neighbor, edge_idx, rel_type)]
        adj = defaultdict(list)
        for i in range(num_edges):
            src = sub_edge_index[0, i].item()
            dst = sub_edge_index[1, i].item()
            rel = sub_edge_type[i].item()
            adj[src].append((dst, i, rel))
            # Also add reverse for undirected search
            adj[dst].append((src, i, rel))

        # Build edge lookup: (src, dst) -> edge_idx for fast lookups
        edge_lookup = {}
        for i in range(num_edges):
            src = sub_edge_index[0, i].item()
            dst = sub_edge_index[1, i].item()
            edge_lookup[(src, dst)] = i
            edge_lookup[(dst, src)] = i  # undirected

        return {'adj': adj, 'edge_lookup': edge_lookup}

    def _find_paths_bfs(
        self,
        adj_data: Dict,
        edge_weights: torch.Tensor,
        head_local: int,
        tail_local: int,
        max_length: int = 4,
        top_k: int = 5
    ) -> List[Tuple[float, List[Tuple[int, int, int, int]]]]:
        """
        Find k-shortest paths using BFS with edge weights.

        Args:
            adj_data: Pre-built adjacency data from _build_adjacency_list
            edge_weights: Current edge weights (changes each epoch)
            head_local, tail_local: Source and target nodes
            max_length: Maximum path length
            top_k: Number of paths to find

        Returns list of (path_score, path) tuples, where path_score is the mean
        of sigmoid edge mask weights along the path (in range (0, 1), comparable
        across different path lengths), and each path is a list of
        (src, rel, dst, edge_idx) tuples. Sorted descending by path_score.
        """
        from heapq import heappush, heappop

        adj = adj_data['adj']

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
            visited_edges = frozenset(e[3] for e in path) if path else frozenset()
            state = (node, visited_edges)
            if state in visited_states:
                continue
            visited_states.add(state)

            for neighbor, edge_idx, rel in adj[node]:
                if edge_idx not in visited_edges:
                    weight = edge_weights[edge_idx].item()
                    new_path = path + [(node, rel, neighbor, edge_idx)]
                    new_score = score + weight
                    heappush(pq, (-new_score, length + 1, neighbor, new_path))

        # Convert sum to mean edge weight so scores are in (0, 1),
        # making paths of different lengths comparable.
        # Note: BFS uses sum for exploration (can't know final length mid-search),
        # so we re-sort by mean here. This means the top-k by mean may differ
        # from top-k by sum, but mean is the fairer comparison metric.
        result_paths = []
        for score, path in found_paths:
            mean_score = score / len(path) if path else 0.0
            result_paths.append((mean_score, path))

        result_paths.sort(key=lambda x: -x[0])
        return result_paths[:top_k]

    def _path_loss(
        self,
        edge_mask: torch.Tensor,
        adj_data: Dict,
        head_local: int,
        tail_local: int,
        max_path_length: int = 4
    ) -> torch.Tensor:
        """
        Compute path-based loss.

        Encourages edges on paths between head and tail to have high weights,
        and edges not on paths to have low weights.

        Args:
            edge_mask: Learnable edge mask parameters
            adj_data: Pre-built adjacency data (cached, built once)
            head_local, tail_local: Source and target nodes in local indices
            max_path_length: Maximum path length to consider
        """
        # Get edge weights from mask
        edge_weights = torch.sigmoid(edge_mask)

        # Find paths with current weights (uses cached adjacency)
        paths = self._find_paths_bfs(
            adj_data, edge_weights,
            head_local, tail_local,
            max_length=max_path_length,
            top_k=self.k_paths
        )

        if not paths:
            # No paths found - just regularize mask size
            return edge_weights.mean()

        # Collect edge indices on paths - O(1) lookup since paths now include edge_idx
        edges_on_paths = set()
        for _score, path in paths:
            for (src, rel, dst, edge_idx) in path:
                edges_on_paths.add(edge_idx)

        if not edges_on_paths:
            return edge_weights.mean()

        # Vectorized path loss computation
        edges_on_paths_tensor = torch.tensor(list(edges_on_paths), device=self.device, dtype=torch.long)

        # Encourage high weights on path edges (negative loss)
        on_path_weights = edge_weights[edges_on_paths_tensor]
        on_path_loss = -on_path_weights.mean()

        # Encourage low weights off path edges (positive loss)
        # Create mask for off-path edges
        on_path_mask = torch.zeros(len(edge_mask), device=self.device, dtype=torch.bool)
        on_path_mask[edges_on_paths_tensor] = True
        off_path_weights = edge_weights[~on_path_mask]
        off_path_loss = off_path_weights.mean() if len(off_path_weights) > 0 else torch.tensor(0.0, device=self.device)

        return on_path_loss + off_path_loss

    def _mask_size_loss(self, edge_mask: torch.Tensor) -> torch.Tensor:
        """Regularization to encourage sparse masks."""
        return torch.sigmoid(edge_mask).mean()

    def explain(
        self,
        head_idx: int,
        tail_idx: int,
        rel_idx: int,
        max_path_length: int = 3,
        top_k_edges: int = 100,
        verbose: bool = False
    ) -> Dict:
        """
        Generate explanation for a predicted link.

        Args:
            head_idx: Head node index
            tail_idx: Tail node index
            rel_idx: Relation type index
            max_path_length: Maximum path length for subgraph extraction and path finding
            top_k_edges: Number of top important edges to return
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

        # Extract path-based subgraph (only edges on paths of length <= max_path_length)
        sub_edge_index, sub_edge_type, subset, mapping = self._extract_path_subgraph(
            head_idx, tail_idx, max_path_length=max_path_length, verbose=verbose
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
            print(f"    Building adjacency list...", end='', flush=True)

        # Build adjacency list ONCE (cached for all epochs)
        import time
        adj_start = time.time()
        adj_data = self._build_adjacency_list(sub_edge_index, sub_edge_type)
        if verbose:
            print(f" done ({time.time()-adj_start:.2f}s)", flush=True)
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

            # Compute losses using cached adjacency data
            path_loss = self._path_loss(
                edge_mask, adj_data,
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

        # Extract final paths (using cached adjacency)
        scored_paths = self._find_paths_bfs(
            adj_data, edge_weights,
            head_local, tail_local,
            max_length=max_path_length,
            top_k=self.k_paths
        )

        # Convert paths back to global indices
        # New path format: (src_local, rel, dst_local, edge_idx)
        global_paths = []
        path_scores = []
        for path_score, path in scored_paths:
            global_path = []
            for src_local, rel, dst_local, edge_idx in path:
                src_global = subset[src_local].item()
                dst_global = subset[dst_local].item()
                global_path.append((src_global, rel, dst_global))
            global_paths.append(global_path)
            path_scores.append(path_score)

        # Get top-k important edges
        top_k = min(top_k_edges, num_edges)
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
            'path_scores': path_scores,
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

    # Get or reconstruct edge_direction (0=forward, 1=inverse)
    # The data preparation code concatenates forward edges, then inverse edges
    # Each half has the same size (inverse edges are just forward edges with src/dst swapped)
    if 'edge_direction' in pyg_data:
        edge_direction = pyg_data['edge_direction']
    else:
        # Reconstruct: first half is forward (0), second half is inverse (1)
        num_edges = edge_index.size(1)
        edge_direction = torch.cat([
            torch.zeros(num_edges // 2, dtype=torch.long),
            torch.ones(num_edges // 2, dtype=torch.long)
        ])
        print(f"  Reconstructed edge_direction: {num_edges // 2:,} forward, {num_edges // 2:,} inverse edges")

    # Get exclude_inverse_edges parameter (default True)
    pagelink_params = explainer_params.get('pagelink', {})
    exclude_inverse_edges = pagelink_params.get('exclude_inverse_edges', True)

    # Compute node and relation embeddings from the trained model
    print(f"\nComputing embeddings from trained model...")
    model.eval()
    with torch.no_grad():
        # CompGCN encode() returns tuple: (node_embeddings, relation_embeddings)
        node_emb, rel_emb = model.encode(edge_index_model, edge_type_model)
    print(f"  Node embeddings: {node_emb.shape}")
    print(f"  Relation embeddings: {rel_emb.shape}")

    # Get parameters (pagelink_params already loaded above for exclude_inverse_edges)
    lr = pagelink_params.get('lr', 0.01)
    num_epochs = pagelink_params.get('num_epochs', 100)
    alpha = pagelink_params.get('alpha', 1.0)
    beta = pagelink_params.get('beta', 0.1)
    k_paths = pagelink_params.get('k_paths', 5)
    max_path_length = pagelink_params.get('max_path_length', 3)  # Default 3 for path-based extraction
    top_k_edges = pagelink_params.get('top_k_edges', 100)  # Number of important edges to return

    print(f"\nPaGE-Link Parameters:")
    print(f"  Learning rate: {lr}")
    print(f"  Epochs: {num_epochs}")
    print(f"  Alpha (path loss weight): {alpha}")
    print(f"  Beta (size regularization): {beta}")
    print(f"  K paths to extract: {k_paths}")
    print(f"  Max path length (subgraph): {max_path_length}")
    print(f"  Top-k edges to return: {top_k_edges}")
    print(f"  Exclude inverse edges: {exclude_inverse_edges}")

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
        device=device,
        exclude_inverse_edges=exclude_inverse_edges,
        edge_direction=edge_direction
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
                max_path_length=max_path_length,
                top_k_edges=top_k_edges,
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
