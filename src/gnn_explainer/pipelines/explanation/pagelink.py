"""
PaGE-Link: Path-based GNN Explanation for Heterogeneous Link Prediction.

Aligned with the original publication:
https://github.com/amazon-science/page-link-path-based-gnn-explanation

Paper: "PaGE-Link: Path-based Graph Neural Network Explanation for
Heterogeneous Link Prediction" (WWW 2023)

The algorithm:
1. Extract k-hop subgraph around source and target nodes
2. Prune subgraph: remove high-degree node edges, extract k-core
3. Initialize per-edge-type learnable edge masks
4. Optimize masks with: L = L_pred + L_path
   - L_pred: Prediction loss (maintain original prediction direction)
   - L_path: Path loss (alpha * on_path + beta * off_path)
   The GNN is re-run each epoch with edge-weighted message passing.
5. Extract top-k shortest paths from learned masks using Yen's algorithm
   with degree-penalized edge weights
"""

import torch
import torch.nn as nn
import numpy as np
from typing import Dict, Tuple, Optional, List
from collections import defaultdict
from heapq import heappush, heappop

from torch_geometric.utils import k_hop_subgraph


class PaGELinkExplainer(nn.Module):
    """
    PaGE-Link Explainer for Knowledge Graph Link Prediction.

    Learns edge masks through optimization with prediction loss and
    path-based regularization, then extracts k-shortest paths as explanations.
    Matches the original PaGE-Link publication algorithm.
    """

    def __init__(
        self,
        model,
        edge_index: torch.Tensor,
        edge_type: torch.Tensor,
        edge_index_model: torch.Tensor,
        edge_type_model: torch.Tensor,
        num_nodes: int,
        num_relations: int,
        lr: float = 0.001,
        num_epochs: int = 100,
        alpha: float = 1.0,
        beta: float = 1.0,
        k_paths: int = 5,
        num_hops: int = 2,
        prune_max_degree: int = -1,
        k_core: int = 2,
        device: str = 'cpu',
        exclude_inverse_edges: bool = True,
        edge_direction: torch.Tensor = None
    ):
        """
        Args:
            model: Trained CompGCN KG model (frozen during explanation)
            edge_index: Graph edges for subgraph extraction (2, num_edges)
            edge_type: Edge types for subgraph extraction (num_edges,)
            edge_index_model: Model's training graph edges (may include inverses)
            edge_type_model: Model's training graph edge types
            num_nodes: Number of entities
            num_relations: Number of relations
            lr: Learning rate for Adam optimizer
            num_epochs: Number of optimization epochs
            alpha: Weight for on-path loss (encourage high weights on paths)
            beta: Weight for off-path loss (encourage low weights off paths)
            k_paths: Number of paths to extract
            num_hops: Number of hops for k-hop subgraph extraction
            prune_max_degree: Remove edges of nodes with degree > this (-1 to disable)
            k_core: k for k-core subgraph extraction
            device: Device to use
            exclude_inverse_edges: Whether to filter inverse edges from subgraph
            edge_direction: Edge direction labels (0=forward, 1=inverse)
        """
        super().__init__()
        self.model = model
        self.num_nodes = num_nodes
        self.num_relations = num_relations
        self.lr = lr
        self.num_epochs = num_epochs
        self.alpha = alpha
        self.beta = beta
        self.k_paths = k_paths
        self.num_hops = num_hops
        self.prune_max_degree = prune_max_degree
        self.k_core = k_core
        self.device = device

        # Store model's training graph (needed for re-running encoder each epoch)
        self.edge_index_model = edge_index_model.to(device)
        self.edge_type_model = edge_type_model.to(device)

        # Filter out inverse edges for subgraph extraction if requested
        if exclude_inverse_edges and edge_direction is not None:
            forward_mask = (edge_direction == 0)
            self.edge_index = edge_index[:, forward_mask].to(device)
            self.edge_type = edge_type[forward_mask].to(device)
            # Build mapping from filtered edge indices to model graph indices
            self._forward_edge_indices = forward_mask.nonzero(as_tuple=True)[0].to(device)
            print(f"  Filtered inverse edges: {edge_index.size(1):,} -> {self.edge_index.size(1):,} edges")
        else:
            self.edge_index = edge_index.to(device)
            self.edge_type = edge_type.to(device)
            self._forward_edge_indices = torch.arange(edge_index.size(1), device=device)
            if exclude_inverse_edges:
                print(f"  Warning: exclude_inverse_edges=True but no edge_direction provided")

        # Freeze model parameters
        for param in self.model.parameters():
            param.requires_grad = False
        self.model.eval()

    # ------------------------------------------------------------------
    # Subgraph extraction (k-hop, matching original)
    # ------------------------------------------------------------------

    def _extract_khop_subgraph(
        self,
        head_idx: int,
        tail_idx: int,
        verbose: bool = False
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, Dict, torch.Tensor]:
        """
        Extract k-hop subgraph around head and tail nodes.

        Returns:
            sub_edge_index: Subgraph edge index (local indices)
            sub_edge_type: Subgraph edge types
            subset: Original node indices in subgraph
            mapping: Dict mapping original -> local node indices
            full_edge_indices: Indices into self.edge_index for each subgraph edge
        """
        import time
        if verbose:
            print(f"      Extracting {self.num_hops}-hop subgraph...", end='', flush=True)
        start = time.time()

        # Get k-hop subgraph (union of head and tail neighborhoods)
        seed_nodes = torch.tensor([head_idx, tail_idx], device=self.device)
        subset, sub_edge_index, mapping_tensor, edge_mask = k_hop_subgraph(
            seed_nodes,
            self.num_hops,
            self.edge_index,
            relabel_nodes=True,
            num_nodes=self.num_nodes
        )

        # edge_mask is a boolean mask over self.edge_index
        full_edge_indices = edge_mask.nonzero(as_tuple=True)[0]
        sub_edge_type = self.edge_type[full_edge_indices]

        # Build dict mapping
        mapping = {}
        for local_idx, orig_idx in enumerate(subset.tolist()):
            mapping[orig_idx] = local_idx

        if verbose:
            print(f" {len(subset):,} nodes, {sub_edge_index.size(1):,} edges ({time.time()-start:.2f}s)", flush=True)

        return sub_edge_index, sub_edge_type, subset, mapping, full_edge_indices

    # ------------------------------------------------------------------
    # Graph pruning (matching original: high-degree removal + k-core)
    # ------------------------------------------------------------------

    def _remove_high_degree_edges(
        self,
        sub_edge_index: torch.Tensor,
        num_sub_nodes: int,
        max_degree: int,
        preserve_nodes: set
    ) -> torch.Tensor:
        """
        Remove edges incident to high-degree nodes (except preserved ones).

        Returns:
            Boolean mask over edges to keep.
        """
        # Compute degree (undirected)
        src, dst = sub_edge_index[0], sub_edge_index[1]
        deg = torch.zeros(num_sub_nodes, dtype=torch.long, device=self.device)
        deg.scatter_add_(0, src, torch.ones_like(src))
        deg.scatter_add_(0, dst, torch.ones_like(dst))

        high_degree = deg > max_degree
        # Don't prune preserved nodes
        for n in preserve_nodes:
            high_degree[n] = False

        high_degree_nodes = high_degree.nonzero(as_tuple=True)[0]
        # Remove edges where either endpoint is high-degree
        src_high = torch.isin(src, high_degree_nodes)
        dst_high = torch.isin(dst, high_degree_nodes)
        keep_mask = ~(src_high | dst_high)

        return keep_mask

    def _extract_k_core(
        self,
        sub_edge_index: torch.Tensor,
        num_sub_nodes: int,
        k: int,
        preserve_nodes: set
    ) -> torch.Tensor:
        """
        Iteratively remove edges of nodes with degree < k (except preserved ones).

        Returns:
            Boolean mask over edges to keep.
        """
        keep_mask = torch.ones(sub_edge_index.size(1), dtype=torch.bool, device=self.device)

        while True:
            current_edges = sub_edge_index[:, keep_mask]
            if current_edges.size(1) == 0:
                break

            src, dst = current_edges[0], current_edges[1]
            deg = torch.zeros(num_sub_nodes, dtype=torch.long, device=self.device)
            deg.scatter_add_(0, src, torch.ones_like(src))
            deg.scatter_add_(0, dst, torch.ones_like(dst))

            # Nodes with 0 < degree < k (and not preserved)
            low_degree = (deg > 0) & (deg < k)
            for n in preserve_nodes:
                low_degree[n] = False

            if not low_degree.any():
                break

            low_degree_nodes = low_degree.nonzero(as_tuple=True)[0]
            all_src = sub_edge_index[0]
            all_dst = sub_edge_index[1]
            src_low = torch.isin(all_src, low_degree_nodes)
            dst_low = torch.isin(all_dst, low_degree_nodes)
            remove = src_low | dst_low
            keep_mask = keep_mask & ~remove

        return keep_mask

    def _prune_graph(
        self,
        sub_edge_index: torch.Tensor,
        sub_edge_type: torch.Tensor,
        num_sub_nodes: int,
        head_local: int,
        tail_local: int,
        full_edge_indices: torch.Tensor,
        verbose: bool = False
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Apply high-degree pruning and k-core extraction to subgraph.

        Returns:
            pruned_sub_edge_index, pruned_sub_edge_type, pruned_full_edge_indices
        """
        import time
        if verbose:
            print(f"      Pruning subgraph (max_degree={self.prune_max_degree}, k_core={self.k_core})...", end='', flush=True)
        start = time.time()

        preserve = {head_local, tail_local}
        keep_mask = torch.ones(sub_edge_index.size(1), dtype=torch.bool, device=self.device)

        if self.prune_max_degree > 0:
            degree_mask = self._remove_high_degree_edges(
                sub_edge_index, num_sub_nodes, self.prune_max_degree, preserve
            )
            keep_mask = keep_mask & degree_mask

        # Apply k-core on the (possibly degree-pruned) edges
        k_core_mask = self._extract_k_core(
            sub_edge_index, num_sub_nodes, self.k_core, preserve
        )

        # If k-core results in no edges, fall back to degree-pruned (or original)
        combined = keep_mask & k_core_mask
        if combined.sum() > 0:
            keep_mask = combined
        # else: keep_mask stays as degree-pruned (or all edges if no degree pruning)

        pruned_edge_index = sub_edge_index[:, keep_mask]
        pruned_edge_type = sub_edge_type[keep_mask]
        pruned_full_indices = full_edge_indices[keep_mask]

        if verbose:
            orig = sub_edge_index.size(1)
            pruned = pruned_edge_index.size(1)
            print(f" {orig:,} -> {pruned:,} edges ({time.time()-start:.2f}s)", flush=True)

        return pruned_edge_index, pruned_edge_type, pruned_full_indices

    # ------------------------------------------------------------------
    # Edge mask initialization (per-type, matching original)
    # ------------------------------------------------------------------

    def _init_edge_masks(
        self,
        sub_edge_index: torch.Tensor,
        sub_edge_type: torch.Tensor
    ) -> Dict[int, nn.Parameter]:
        """
        Initialize per-edge-type masks matching the original PaGE-Link.

        Uses: std = calculate_gain('relu') * sqrt(2 / (2 * num_type_nodes))
        """
        edge_masks = {}
        unique_types = sub_edge_type.unique()

        for etype in unique_types:
            etype_val = etype.item()
            type_mask = (sub_edge_type == etype_val)
            num_type_edges = type_mask.sum().item()

            # Count nodes involved in this edge type
            type_edges = sub_edge_index[:, type_mask]
            num_type_nodes = torch.cat([type_edges[0], type_edges[1]]).unique().numel()
            num_type_nodes = max(num_type_nodes, 1)  # avoid division by zero

            std = nn.init.calculate_gain('relu') * np.sqrt(2.0 / (2 * num_type_nodes))
            edge_masks[etype_val] = nn.Parameter(
                torch.randn(num_type_edges, device=self.device) * std
            )

        return edge_masks

    def _assemble_edge_weights(
        self,
        edge_masks: Dict[int, nn.Parameter],
        sub_edge_type: torch.Tensor
    ) -> torch.Tensor:
        """Assemble per-type sigmoid masks into a single edge weight tensor."""
        weights = torch.empty(sub_edge_type.size(0), device=self.device)
        for etype, mask in edge_masks.items():
            type_mask = (sub_edge_type == etype)
            weights[type_mask] = mask.sigmoid()
        return weights

    def _build_full_edge_weight(
        self,
        edge_weights: torch.Tensor,
        pruned_full_edge_indices: torch.Tensor
    ) -> torch.Tensor:
        """
        Build full-graph edge weight tensor for the model's training graph.
        Non-subgraph edges get weight 1.0, subgraph edges get learned weights.
        """
        full_weight = torch.ones(self.edge_index_model.size(1), device=self.device)

        # Map pruned subgraph edges to positions in the model's training graph
        # self._forward_edge_indices maps from filtered (forward-only) graph to full graph
        model_edge_indices = self._forward_edge_indices[pruned_full_edge_indices]
        full_weight[model_edge_indices] = edge_weights

        return full_weight

    # ------------------------------------------------------------------
    # Path finding: Yen's k-shortest paths with degree penalty
    # ------------------------------------------------------------------

    def _build_adjacency_list(
        self,
        sub_edge_index: torch.Tensor
    ) -> Dict[int, List[Tuple[int, int]]]:
        """Build adjacency list: node -> [(neighbor, edge_idx)]."""
        adj = defaultdict(list)
        num_edges = sub_edge_index.size(1)
        for i in range(num_edges):
            src = sub_edge_index[0, i].item()
            dst = sub_edge_index[1, i].item()
            adj[src].append((dst, i))
            adj[dst].append((src, i))  # undirected for path finding
        return adj

    def _get_neg_path_score_func(
        self,
        adj: Dict,
        edge_weights: torch.Tensor,
        sub_edge_index: torch.Tensor,
        num_sub_nodes: int,
        head_local: int,
        tail_local: int
    ):
        """
        Build the degree-penalized weight function matching the original:
        weight(u, v) = log(in_degree[v]) - log(edge_weight(u,v))

        Lower weight = better path (for shortest path algorithms).
        High edge weight and low degree destination are preferred.
        """
        # Compute in-degrees
        in_deg = torch.zeros(num_sub_nodes, dtype=torch.float, device=self.device)
        dst = sub_edge_index[1]
        in_deg.scatter_add_(0, dst, torch.ones_like(dst, dtype=torch.float))
        # Also count reverse (undirected)
        src = sub_edge_index[0]
        in_deg.scatter_add_(0, src, torch.ones_like(src, dtype=torch.float))

        log_in_deg = in_deg.clamp(min=1).log()
        # Set head/tail degree to 0 so they are preferred on paths
        log_in_deg[head_local] = 0
        log_in_deg[tail_local] = 0

        log_eweights = edge_weights.clamp(min=1e-8).log()

        # Build (u, v) -> weight mapping using both directions
        neg_score_map = {}
        for i in range(sub_edge_index.size(1)):
            u = sub_edge_index[0, i].item()
            v = sub_edge_index[1, i].item()
            # Forward direction
            w = log_in_deg[v].item() - log_eweights[i].item()
            neg_score_map[(u, v)] = w
            # Reverse direction (undirected)
            w_rev = log_in_deg[u].item() - log_eweights[i].item()
            neg_score_map[(v, u)] = w_rev

        def weight_func(u, v):
            return neg_score_map.get((u, v), float('inf'))

        return weight_func

    def _bidirectional_dijkstra(
        self,
        adj: Dict,
        weight_func,
        source: int,
        target: int,
        ignore_nodes: set = None,
        ignore_edges: set = None
    ) -> Tuple[float, List[int]]:
        """
        Bidirectional Dijkstra shortest path.

        Returns:
            (distance, path) or (float('inf'), []) if no path found.
        """
        if source == target:
            return (0, [source])

        if ignore_nodes is None:
            ignore_nodes = set()
        if ignore_edges is None:
            ignore_edges = set()

        # Forward search from source
        dists_f = {source: 0}
        parents_f = {source: None}
        pq_f = [(0, source)]

        # Backward search from target
        dists_b = {target: 0}
        parents_b = {target: None}
        pq_b = [(0, target)]

        best_dist = float('inf')
        best_node = None

        finalized_f = set()
        finalized_b = set()

        while pq_f or pq_b:
            # Forward step
            if pq_f:
                d_f, u_f = heappop(pq_f)
                if u_f in finalized_f:
                    pass
                elif d_f <= best_dist:
                    finalized_f.add(u_f)
                    if u_f in dists_b:
                        total = d_f + dists_b[u_f]
                        if total < best_dist:
                            best_dist = total
                            best_node = u_f
                    for v, _ in adj.get(u_f, []):
                        if v in ignore_nodes:
                            continue
                        edge_key = (u_f, v)
                        if edge_key in ignore_edges:
                            continue
                        w = weight_func(u_f, v)
                        if w == float('inf'):
                            continue
                        new_dist = d_f + w
                        if v not in dists_f or new_dist < dists_f[v]:
                            dists_f[v] = new_dist
                            parents_f[v] = u_f
                            heappush(pq_f, (new_dist, v))

            # Backward step
            if pq_b:
                d_b, u_b = heappop(pq_b)
                if u_b in finalized_b:
                    pass
                elif d_b <= best_dist:
                    finalized_b.add(u_b)
                    if u_b in dists_f:
                        total = d_b + dists_f[u_b]
                        if total < best_dist:
                            best_dist = total
                            best_node = u_b
                    for v, _ in adj.get(u_b, []):
                        if v in ignore_nodes:
                            continue
                        edge_key = (v, u_b)
                        if edge_key in ignore_edges:
                            continue
                        w = weight_func(v, u_b)
                        if w == float('inf'):
                            continue
                        new_dist = d_b + w
                        if v not in dists_b or new_dist < dists_b[v]:
                            dists_b[v] = new_dist
                            parents_b[v] = u_b
                            heappush(pq_b, (new_dist, v))

            # Termination check
            min_f = pq_f[0][0] if pq_f else float('inf')
            min_b = pq_b[0][0] if pq_b else float('inf')
            if min_f + min_b >= best_dist:
                break

        if best_node is None:
            return (float('inf'), [])

        # Reconstruct path
        path_f = []
        node = best_node
        while node is not None:
            path_f.append(node)
            node = parents_f.get(node)
        path_f.reverse()

        path_b = []
        node = parents_b.get(best_node)
        while node is not None:
            path_b.append(node)
            node = parents_b.get(node)

        path = path_f + path_b
        return (best_dist, path)

    def _k_shortest_paths(
        self,
        adj: Dict,
        weight_func,
        source: int,
        target: int,
        k: int = 5,
        max_length: int = None
    ) -> List[List[int]]:
        """
        Yen's k-shortest simple paths algorithm.

        Returns:
            List of paths (each path is a list of node IDs).
        """
        # Find first shortest path
        dist, first_path = self._bidirectional_dijkstra(adj, weight_func, source, target)
        if not first_path:
            return []

        A = [first_path]  # confirmed shortest paths
        B = []  # candidate paths (heap of (cost, path))

        for i in range(1, k):
            prev_path = A[i - 1]
            for j in range(len(prev_path) - 1):
                spur_node = prev_path[j]
                root_path = prev_path[:j + 1]

                # Edges/nodes to ignore (from previously found paths sharing the root)
                ignore_edges = set()
                ignore_nodes = set()
                for p in A:
                    if len(p) > j and p[:j + 1] == root_path:
                        if j + 1 < len(p):
                            ignore_edges.add((p[j], p[j + 1]))

                for node in root_path[:-1]:
                    ignore_nodes.add(node)

                spur_dist, spur_path = self._bidirectional_dijkstra(
                    adj, weight_func, spur_node, target,
                    ignore_nodes=ignore_nodes,
                    ignore_edges=ignore_edges
                )

                if spur_path:
                    total_path = root_path[:-1] + spur_path
                    # Compute total cost
                    total_cost = 0
                    for idx in range(len(total_path) - 1):
                        total_cost += weight_func(total_path[idx], total_path[idx + 1])

                    # Check for duplicates
                    is_dup = any(p == total_path for _, p in B) or total_path in A
                    if not is_dup:
                        heappush(B, (total_cost, total_path))

            if not B:
                break

            _, next_path = heappop(B)

            # Apply max_length filter
            if max_length and len(next_path) > max_length + 1:
                # Skip this path and try next candidate
                found = False
                while B:
                    _, candidate = heappop(B)
                    if max_length is None or len(candidate) <= max_length + 1:
                        next_path = candidate
                        found = True
                        break
                if not found:
                    break

            A.append(next_path)

        # Filter by max_length
        if max_length:
            A = [p for p in A if len(p) <= max_length + 1]

        return A

    # ------------------------------------------------------------------
    # Path loss (list-based edge collection, matching original)
    # ------------------------------------------------------------------

    def _path_loss(
        self,
        edge_weights: torch.Tensor,
        paths: List[List[int]],
        sub_edge_index: torch.Tensor
    ) -> torch.Tensor:
        """
        Compute path loss matching the original PaGE-Link.

        Uses list-based edge collection (edges on multiple paths counted
        multiple times). This matches the original's get_eids_on_paths.
        """
        row = sub_edge_index[0]
        col = sub_edge_index[1]

        # Collect edge IDs on paths (LIST, not set)
        eids = []
        for path in paths:
            for i in range(len(path) - 1):
                u, v = path[i], path[i + 1]
                # Find edge ID for (u, v) or (v, u)
                match_fwd = ((row == u) & (col == v)).nonzero(as_tuple=True)[0]
                match_bwd = ((row == v) & (col == u)).nonzero(as_tuple=True)[0]
                if match_fwd.numel() > 0:
                    eids.append(match_fwd[0].item())
                elif match_bwd.numel() > 0:
                    eids.append(match_bwd[0].item())

        if not eids:
            return torch.tensor(0.0, device=self.device)

        eids_tensor = torch.tensor(eids, dtype=torch.long, device=self.device)

        # On-path loss: encourage high weights
        loss_on_path = -edge_weights[eids_tensor].mean()

        # Off-path loss: encourage low weights
        eids_off_path_mask = ~torch.isin(
            torch.arange(edge_weights.shape[0], device=self.device),
            eids_tensor
        )
        if eids_off_path_mask.any():
            loss_off_path = edge_weights[eids_off_path_mask].mean()
        else:
            loss_off_path = torch.tensor(0.0, device=self.device)

        return self.alpha * loss_on_path + self.beta * loss_off_path

    # ------------------------------------------------------------------
    # Main explain method
    # ------------------------------------------------------------------

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

        Matches the original PaGE-Link algorithm:
        1. Extract k-hop subgraph
        2. Prune (high-degree + k-core)
        3. Initialize per-type edge masks
        4. Optimize with prediction loss + path loss (re-running GNN each epoch)
        5. Extract paths via Yen's algorithm with degree penalty

        Args:
            head_idx: Head node index (global)
            tail_idx: Tail node index (global)
            rel_idx: Relation type index
            max_path_length: Maximum path length for path extraction
            top_k_edges: Number of top important edges to return
            verbose: Print progress

        Returns:
            Dictionary with paths, edge_mask, important_edges, subgraph_info
        """
        import time

        head_idx_t = torch.tensor([head_idx], device=self.device)
        tail_idx_t = torch.tensor([tail_idx], device=self.device)
        rel_idx_t = torch.tensor([rel_idx], device=self.device)

        # Step 1: Get initial prediction
        with torch.no_grad():
            node_emb, rel_emb = self.model.encode(self.edge_index_model, self.edge_type_model)
            original_score = self.model.decode(
                node_emb, rel_emb,
                head_idx_t, tail_idx_t, rel_idx_t
            ).item()
        pred = int(original_score > 0)

        if verbose:
            print(f"    Original score: {original_score:.4f} (pred={pred})", flush=True)

        # Step 2: Extract k-hop subgraph
        sub_edge_index, sub_edge_type, subset, mapping, full_edge_indices = \
            self._extract_khop_subgraph(head_idx, tail_idx, verbose=verbose)

        num_edges = sub_edge_index.size(1)
        if num_edges == 0:
            return {
                'paths': [], 'edge_mask': None, 'important_edges': None,
                'subgraph_info': {'num_nodes': len(subset), 'num_edges': 0, 'subset': subset},
                'error': 'No edges in subgraph'
            }

        head_local = mapping[head_idx]
        tail_local = mapping[tail_idx]
        num_sub_nodes = len(subset)

        # Step 3: Prune subgraph
        sub_edge_index, sub_edge_type, full_edge_indices = self._prune_graph(
            sub_edge_index, sub_edge_type, num_sub_nodes,
            head_local, tail_local, full_edge_indices,
            verbose=verbose
        )

        num_edges = sub_edge_index.size(1)
        if num_edges == 0:
            return {
                'paths': [], 'edge_mask': None, 'important_edges': None,
                'subgraph_info': {'num_nodes': num_sub_nodes, 'num_edges': 0, 'subset': subset},
                'error': 'No edges after pruning'
            }

        if verbose:
            print(f"    Pruned subgraph: {num_sub_nodes:,} nodes, {num_edges:,} edges", flush=True)

        # Step 4: Initialize per-type edge masks
        edge_masks = self._init_edge_masks(sub_edge_index, sub_edge_type)
        optimizer = torch.optim.Adam(edge_masks.values(), lr=self.lr)

        # Build adjacency for path finding (once, reused)
        adj = self._build_adjacency_list(sub_edge_index)

        if verbose:
            print(f"    Starting optimization ({self.num_epochs} epochs)...", flush=True)

        # Step 5: Optimization loop
        print_interval = max(1, self.num_epochs // 5)
        eweight_norm = 0.0
        EPS = 1e-3

        for epoch in range(self.num_epochs):
            optimizer.zero_grad()

            # Assemble edge weights from per-type masks
            edge_weights = self._assemble_edge_weights(edge_masks, sub_edge_type)

            # Build full-graph edge weight vector and re-run GNN
            full_edge_weight = self._build_full_edge_weight(edge_weights, full_edge_indices)
            node_emb_w, rel_emb_w = self.model.encode(
                self.edge_index_model, self.edge_type_model,
                edge_weight=full_edge_weight
            )

            # Prediction loss: (-1)^pred * sigmoid(score).log()
            score = self.model.decode(
                node_emb_w, rel_emb_w,
                head_idx_t, tail_idx_t, rel_idx_t
            )
            pred_loss = ((-1) ** pred) * score.sigmoid().log()

            # Path loss: find paths then penalize on/off path
            weight_func = self._get_neg_path_score_func(
                adj, edge_weights.detach(), sub_edge_index,
                num_sub_nodes, head_local, tail_local
            )
            paths = self._k_shortest_paths(
                adj, weight_func, head_local, tail_local,
                k=self.k_paths, max_length=max_path_length
            )
            path_loss = self._path_loss(edge_weights, paths, sub_edge_index)

            total_loss = pred_loss + path_loss

            total_loss.backward()
            optimizer.step()

            # Early stopping: check edge weight norm convergence
            curr_eweight_norm = edge_weights.detach().norm().item()
            if abs(eweight_norm - curr_eweight_norm) < EPS and epoch > 0:
                if verbose:
                    print(f"      Early stopping at epoch {epoch+1} (weight norm converged)", flush=True)
                break
            eweight_norm = curr_eweight_norm

            if verbose and ((epoch + 1) % print_interval == 0 or epoch == 0):
                print(f"      Epoch {epoch+1}/{self.num_epochs}: "
                      f"loss={total_loss.item():.4f} "
                      f"(pred={pred_loss.item():.4f}, path={path_loss.item():.4f})",
                      flush=True)

        # Step 6: Extract final paths from learned masks
        with torch.no_grad():
            final_weights = self._assemble_edge_weights(edge_masks, sub_edge_type)

        weight_func = self._get_neg_path_score_func(
            adj, final_weights, sub_edge_index,
            num_sub_nodes, head_local, tail_local
        )
        final_paths = self._k_shortest_paths(
            adj, weight_func, head_local, tail_local,
            k=self.k_paths, max_length=max_path_length
        )

        # Convert paths to global indices with relation info
        global_paths = []
        path_scores = []
        edge_lookup = {}
        for i in range(sub_edge_index.size(1)):
            s, d = sub_edge_index[0, i].item(), sub_edge_index[1, i].item()
            edge_lookup[(s, d)] = i
            edge_lookup[(d, s)] = i

        for path in final_paths:
            global_path = []
            score_sum = 0.0
            for j in range(len(path) - 1):
                u_local, v_local = path[j], path[j + 1]
                eidx = edge_lookup.get((u_local, v_local))
                if eidx is None:
                    eidx = edge_lookup.get((v_local, u_local))
                rel = sub_edge_type[eidx].item() if eidx is not None else -1
                u_global = subset[u_local].item()
                v_global = subset[v_local].item()
                global_path.append((u_global, rel, v_global))
                if eidx is not None:
                    score_sum += final_weights[eidx].item()
            global_paths.append(global_path)
            path_scores.append(score_sum / len(global_path) if global_path else 0.0)

        # Get top-k important edges
        top_k = min(top_k_edges, num_edges)
        top_k_values, top_k_indices = torch.topk(final_weights, top_k)

        important_edges = []
        for i in range(top_k):
            idx = top_k_indices[i].item()
            src_local = sub_edge_index[0, idx].item()
            dst_local = sub_edge_index[1, idx].item()
            rel = sub_edge_type[idx].item()
            important_edges.append({
                'src': subset[src_local].item(),
                'rel': rel,
                'dst': subset[dst_local].item(),
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
            'edge_mask': final_weights.cpu(),
            'important_edges': important_edges,
            'subgraph_info': {
                'num_nodes': num_sub_nodes,
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
    if 'edge_direction' in pyg_data:
        edge_direction = pyg_data['edge_direction']
    else:
        num_edges = edge_index.size(1)
        edge_direction = torch.cat([
            torch.zeros(num_edges // 2, dtype=torch.long),
            torch.ones(num_edges // 2, dtype=torch.long)
        ])
        print(f"  Reconstructed edge_direction: {num_edges // 2:,} forward, {num_edges // 2:,} inverse edges")

    # Get parameters
    pagelink_params = explainer_params.get('pagelink', {})
    exclude_inverse_edges = pagelink_params.get('exclude_inverse_edges', True)
    lr = pagelink_params.get('lr', 0.001)
    num_epochs = pagelink_params.get('num_epochs', 100)
    alpha = pagelink_params.get('alpha', 1.0)
    beta = pagelink_params.get('beta', 1.0)
    k_paths = pagelink_params.get('k_paths', 5)
    max_path_length = pagelink_params.get('max_path_length', 3)
    top_k_edges = pagelink_params.get('top_k_edges', 100)
    num_hops = pagelink_params.get('num_hops', 2)
    prune_max_degree = pagelink_params.get('prune_max_degree', -1)
    k_core = pagelink_params.get('k_core', 2)

    print(f"\nPaGE-Link Parameters:")
    print(f"  Learning rate: {lr}")
    print(f"  Epochs: {num_epochs}")
    print(f"  Alpha (on-path loss weight): {alpha}")
    print(f"  Beta (off-path loss weight): {beta}")
    print(f"  K paths to extract: {k_paths}")
    print(f"  Max path length: {max_path_length}")
    print(f"  Num hops (subgraph): {num_hops}")
    print(f"  Prune max degree: {prune_max_degree}")
    print(f"  K-core: {k_core}")
    print(f"  Exclude inverse edges: {exclude_inverse_edges}")

    # Create explainer
    print(f"\nInitializing PaGE-Link explainer...")
    print(f"  Graph size: {num_nodes:,} nodes, {edge_index.size(1):,} edges", flush=True)
    explainer = PaGELinkExplainer(
        model=model,
        edge_index=edge_index,
        edge_type=edge_type,
        edge_index_model=edge_index_model,
        edge_type_model=edge_type_model,
        num_nodes=num_nodes,
        num_relations=num_relations,
        lr=lr,
        num_epochs=num_epochs,
        alpha=alpha,
        beta=beta,
        k_paths=k_paths,
        num_hops=num_hops,
        prune_max_degree=prune_max_degree,
        k_core=k_core,
        device=device,
        exclude_inverse_edges=exclude_inverse_edges,
        edge_direction=edge_direction
    )
    print(f"  Explainer initialized successfully", flush=True)

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
            print(f"    Found {len(explanation['paths'])} paths in {triple_time:.1f}s")

            # Print top paths
            if explanation['paths']:
                print(f"    Top paths:")
                for j, path in enumerate(explanation['paths'][:3]):
                    path_str = " -> ".join([f"({p[0]})-[{p[1]}]->({p[2]})" for p in path])
                    print(f"      {j+1}. {path_str}")

        except Exception as e:
            print(f"    Error: {str(e)}")
            import traceback
            traceback.print_exc()
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
    print(f"\nPaGE-Link completed: {len(explanations)} explanations in {total_time:.1f}s")

    return {
        'explainer_type': 'PaGELink',
        'explanations': explanations,
        'num_explanations': len(explanations),
        'params': pagelink_params
    }
