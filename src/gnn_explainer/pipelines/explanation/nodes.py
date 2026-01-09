"""Nodes for explanation pipeline."""

import torch
import torch.nn as nn
from typing import Dict, List, Tuple, Optional, Set
from torch_geometric.explain import Explainer, GNNExplainer, PGExplainer
from torch_geometric.explain import Explanation
import numpy as np


class ModelWrapper(nn.Module):
    """
    Wrapper to make CompGCN/RGCN models compatible with PyG Explainer API.

    The PyG Explainer expects a model that takes (x, edge_index, ...) and returns
    predictions. Our CompGCN model has a different interface, so we wrap it.
    """

    def __init__(self, kg_model, edge_index, edge_type, mode='link_prediction'):
        """
        Args:
            kg_model: The trained KG embedding model (CompGCN or RGCN)
            edge_index: Full graph edge index
            edge_type: Full graph edge types
            mode: 'link_prediction' or 'node_classification'
        """
        super().__init__()
        self.kg_model = kg_model
        self.edge_index = edge_index
        self.edge_type = edge_type
        self.mode = mode
        # Store current subgraph node mapping (set before each explanation)
        self.current_subset = None

    def forward(self, x, edge_index, **kwargs):
        """
        Forward pass compatible with PyG Explainer.

        For link prediction, we use the edge_index to determine which
        links to score.

        Args:
            x: Node features (not used for KG models, but required by Explainer)
            edge_index: Edge index of the subgraph with RELABELED indices (shape: [2, num_edges])
                       Indices range from 0 to len(subset)-1

        Returns:
            Scores for the given edges
        """
        if self.mode == 'link_prediction':
            # Get edge types for the subgraph
            edge_type_for_encoding = kwargs.get('edge_type', None)
            if edge_type_for_encoding is None:
                # Fallback: assume relation 0
                edge_type_for_encoding = torch.zeros(edge_index.size(1), dtype=torch.long, device=edge_index.device)

            # OPTIMIZED VERSION: Direct subgraph encoding without embedding manipulation
            # The edge_index and edge_type passed here define the COMPLETE subgraph
            # CompGCN's encode() will only do message passing on these edges

            if self.current_subset is not None:
                # ALTERNATIVE APPROACH: Instead of swapping embeddings, just extract the subset
                # and manually run the encoding on the subgraph

                # Get the encoder reference
                if hasattr(self.kg_model, 'encoder'):
                    encoder = self.kg_model.encoder
                else:
                    encoder = self.kg_model

                # Get initial embeddings for the subset of nodes
                full_node_emb_param = encoder.node_emb
                subset_initial_emb = full_node_emb_param[self.current_subset]

                # Since the edge_index is already relabeled to 0..len(subset)-1,
                # we can't directly use encode() because it expects full graph indices.
                # Instead, we'll manually do what encode does but with subset embeddings.

                # For now, use a simpler approach: just use the full encode but only extract
                # the embeddings we need afterward
                node_emb_full, rel_emb = self.kg_model.encode(edge_index, edge_type_for_encoding)

                # The edge_index is relabeled, so node_emb_full won't work correctly.
                # We need to create embeddings for the subgraph specifically.
                # Use the subset initial embeddings and then run message passing.

                # Actually, the safest approach is to temporarily resize the parameter
                # Save the original state
                original_emb_data = full_node_emb_param.data
                original_requires_grad = full_node_emb_param.requires_grad

                try:
                    # Replace parameter data temporarily (this is in-place, no setattr needed)
                    with torch.no_grad():
                        # Store original size
                        original_size = full_node_emb_param.size()

                        # Create new parameter with subset size
                        subset_emb = subset_initial_emb.detach().clone()
                        subset_param = torch.nn.Parameter(subset_emb, requires_grad=False)

                        # Replace using module registration (safe way)
                        if hasattr(self.kg_model, 'encoder'):
                            self.kg_model.encoder.register_parameter('node_emb', subset_param)
                        else:
                            self.kg_model.register_parameter('node_emb', subset_param)

                    # Encode with the subset embeddings
                    node_emb_subset, rel_emb = self.kg_model.encode(edge_index, edge_type_for_encoding)

                finally:
                    # Restore original parameter
                    with torch.no_grad():
                        original_param = torch.nn.Parameter(original_emb_data, requires_grad=original_requires_grad)
                        if hasattr(self.kg_model, 'encoder'):
                            self.kg_model.encoder.register_parameter('node_emb', original_param)
                        else:
                            self.kg_model.register_parameter('node_emb', original_param)
            else:
                # No subset mapping, use original behavior
                node_emb_subset, rel_emb = self.kg_model.encode(edge_index, edge_type_for_encoding)

            # For decoding, we use the relabeled subgraph indices directly
            head_idx_subgraph = edge_index[0]
            tail_idx_subgraph = edge_index[1]

            # Decode using subset embeddings with relabeled indices
            scores = self.kg_model.decode(node_emb_subset, rel_emb, head_idx_subgraph, tail_idx_subgraph, edge_type_for_encoding)

            return scores
        else:
            raise NotImplementedError(f"Mode {self.mode} not implemented yet")


def extract_path_based_subgraph(
    head_node: int,
    tail_node: int,
    edge_index: torch.Tensor,
    edge_type: torch.Tensor,
    max_path_length: int = 3,
    device: torch.device = torch.device('cpu')
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    Extract subgraph based on paths between head and tail nodes using igraph.

    This follows the approach from filter_training_igraph.py:
    1. Build an igraph from the PyG edge_index
    2. Find all simple paths between head and tail up to max_path_length
    3. Extract edges from these paths
    4. Return subgraph with node relabeling

    Args:
        head_node: Source node index
        tail_node: Target node index
        edge_index: Full graph edge index [2, num_edges]
        edge_type: Full graph edge types [num_edges]
        max_path_length: Maximum path length to consider
        device: torch device

    Returns:
        Tuple of (subset, sub_edge_index, mapping, edge_mask)
        - subset: Node IDs in subgraph
        - sub_edge_index: Relabeled edge index
        - mapping: Mapping from [head, tail] to new indices
        - edge_mask: Boolean mask for edges in subgraph
    """
    try:
        import igraph as ig
    except ImportError:
        print("Warning: igraph not installed. Falling back to k-hop subgraph.")
        print("Install with: pip install igraph")
        from torch_geometric.utils import k_hop_subgraph
        return k_hop_subgraph(
            node_idx=[head_node, tail_node],
            num_hops=2,
            edge_index=edge_index,
            relabel_nodes=True,
            num_nodes=edge_index.max().item() + 1
        )

    # Convert PyG edge_index to numpy for igraph
    edge_list = edge_index.t().cpu().numpy()  # [num_edges, 2]
    num_nodes = edge_index.max().item() + 1

    # Build DIRECTED igraph to preserve edge directionality
    # Knowledge graphs are inherently directed (head -> relation -> tail)
    g = ig.Graph(n=num_nodes, edges=edge_list.tolist(), directed=True)

    # Find all simple paths between head and tail
    try:
        # Try with cutoff parameter (igraph >= 0.10.x)
        paths = g.get_all_simple_paths(head_node, to=tail_node, cutoff=max_path_length)
    except TypeError:
        try:
            # Try maxlen parameter (some igraph versions)
            paths = g.get_all_simple_paths(head_node, to=tail_node, maxlen=max_path_length)
        except TypeError:
            # Fallback: get all paths and filter by length
            all_paths = g.get_all_simple_paths(head_node, to=tail_node)
            paths = [p for p in all_paths if len(p) - 1 <= max_path_length]

    if not paths:
        # No paths found, fall back to k-hop
        print(f"  No paths found between nodes {head_node} and {tail_node}, using k-hop fallback")
        from torch_geometric.utils import k_hop_subgraph
        return k_hop_subgraph(
            node_idx=[head_node, tail_node],
            num_hops=2,
            edge_index=edge_index,
            relabel_nodes=True,
            num_nodes=num_nodes
        )

    # Extract all nodes and edges from paths
    nodes_in_paths = set()
    edges_in_paths = set()

    for path in paths:
        nodes_in_paths.update(path)
        # Extract edges from path (preserve direction for directed graph)
        for i in range(len(path) - 1):
            # Store directed edges as (src, dst) tuples
            edge = (path[i], path[i+1])
            edges_in_paths.add(edge)

    # Convert to sorted list for consistent ordering
    subset = torch.tensor(sorted(nodes_in_paths), dtype=torch.long)

    # Create node index mapping
    old_to_new = {old_idx.item(): new_idx for new_idx, old_idx in enumerate(subset)}

    # Find edges in original graph that are in our path-based subgraph
    edge_mask = torch.zeros(edge_index.size(1), dtype=torch.bool)

    for edge_idx in range(edge_index.size(1)):
        src = edge_index[0, edge_idx].item()
        dst = edge_index[1, edge_idx].item()
        # Use directed edge (src -> dst) for matching
        edge = (src, dst)

        if edge in edges_in_paths:
            edge_mask[edge_idx] = True

    # Extract subgraph edges and relabel
    sub_edge_index_list = []
    for edge_idx in range(edge_index.size(1)):
        if edge_mask[edge_idx]:
            src = edge_index[0, edge_idx].item()
            dst = edge_index[1, edge_idx].item()
            new_src = old_to_new[src]
            new_dst = old_to_new[dst]
            sub_edge_index_list.append([new_src, new_dst])

    if not sub_edge_index_list:
        # Empty subgraph, fall back to k-hop
        print(f"  Empty subgraph extracted, using k-hop fallback")
        from torch_geometric.utils import k_hop_subgraph
        return k_hop_subgraph(
            node_idx=[head_node, tail_node],
            num_hops=2,
            edge_index=edge_index,
            relabel_nodes=True,
            num_nodes=num_nodes
        )

    sub_edge_index = torch.tensor(sub_edge_index_list, dtype=torch.long).t()

    # Create mapping tensor for head and tail
    mapping = torch.tensor([old_to_new[head_node], old_to_new[tail_node]], dtype=torch.long)

    return subset.to(device), sub_edge_index.to(device), mapping.to(device), edge_mask


def prepare_model_for_explanation(
    trained_model_dict: Dict,
    dgl_data: Dict = None,
    pyg_data: Dict = None,
    device_str: str = "cpu"
) -> Dict:
    """
    Load trained model and prepare for explanation.

    Args:
        trained_model_dict: Dictionary with model state and metadata
        dgl_data: DGL format graph data (preferred)
        pyg_data: PyG format graph data (legacy)
        device_str: Device string ("cuda" or "cpu")

    Returns:
        Dictionary with wrapped model and graph data
    """
    # Determine which data format to use
    use_dgl = dgl_data is not None
    graph_data = dgl_data if use_dgl else pyg_data

    if graph_data is None:
        raise ValueError("Either dgl_data or pyg_data must be provided")
    print("\n" + "="*60)
    print("PREPARING MODEL FOR EXPLANATION")
    print("="*60)

    device = torch.device(device_str if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    print(f"Using {'DGL' if use_dgl else 'PyG'} graph format")

    # Recreate model from saved state
    from ..training.kg_models import CompGCNKGModel
    from ..training.kg_models_dgl import CompGCNKGModelDGL
    from ..training.model import RGCNDistMultModel

    # Extract model configuration (handles both old 'metadata' and new 'model_config' format)
    model_config = trained_model_dict.get('model_config', trained_model_dict.get('metadata', {}))
    model_type = model_config['model_type']

    print(f"\nRecreating {model_type.upper()} model...")
    print(f"  Decoder: {model_config['decoder_type']}")

    if model_type == 'compgcn':
        # Choose DGL or PyG model based on data format
        ModelClass = CompGCNKGModelDGL if use_dgl else CompGCNKGModel

        model = ModelClass(
            num_nodes=model_config['num_nodes'],
            num_relations=model_config['num_relations'],
            embedding_dim=model_config['embedding_dim'],
            decoder_type=model_config['decoder_type'],
            num_layers=model_config['num_layers'],
            comp_fn=model_config.get('comp_fn', 'sub'),
            dropout=model_config['dropout'],
            conve_kwargs=model_config.get('conve_kwargs')
        )
    elif model_type == 'rgcn':
        if use_dgl:
            raise NotImplementedError("RGCN with DGL is not yet implemented. Use CompGCN or PyG format.")

        model = RGCNDistMultModel(
            num_nodes=model_config['num_nodes'],
            num_relations=model_config['num_relations'],
            embedding_dim=model_config['embedding_dim'],
            num_layers=model_config['num_layers'],
            num_bases=model_config.get('num_bases', 30),
            dropout=model_config['dropout']
        )
    else:
        raise ValueError(f"Unknown model_type: {model_type}")

    # Load trained weights
    model.load_state_dict(trained_model_dict['model_state_dict'])
    model.to(device)
    model.eval()

    print(f"✓ Model loaded successfully")

    # Wrap model for Explainer API
    edge_index = graph_data['edge_index'].to(device)
    edge_type = graph_data['edge_type'].to(device)

    # Get DGL graph if available
    g = graph_data.get('graph') if use_dgl else None
    if g is not None:
        g = g.to(device)

    wrapped_model = ModelWrapper(
        kg_model=model,
        edge_index=edge_index,
        edge_type=edge_type,
        mode='link_prediction'
    )

    print(f"✓ Model wrapped for explanation")

    result = {
        'model': model,
        'wrapped_model': wrapped_model,
        'edge_index': edge_index,
        'edge_type': edge_type,
        'num_nodes': model_config['num_nodes'],
        'num_relations': model_config['num_relations'],
        'device': device,
        'use_dgl': use_dgl
    }

    if use_dgl and g is not None:
        result['graph'] = g

    return result


def select_triples_to_explain(
    dgl_data: Dict = None,
    pyg_data: Dict = None,
    knowledge_graph: Dict = None,
    selection_params: Dict = None,
    device_str: str = "cpu",
    triple_file_content: str = None
) -> Dict:
    """
    Select triples (edges) to explain.

    Args:
        dgl_data: DGL format graph data (preferred)
        pyg_data: PyG format graph data (legacy)
        knowledge_graph: Knowledge graph with dictionaries
        selection_params: Selection configuration
        knowledge_graph: Knowledge graph with dictionaries
        selection_params: Parameters for selecting triples
        device_str: Device string
        triple_file_content: Optional file content from Kedro catalog (for "from_file" strategy)

    Returns:
        Dictionary with selected triple indices and metadata
    """
    print("\n" + "="*60)
    print("SELECTING TRIPLES TO EXPLAIN")
    print("="*60)

    # Determine which data format to use
    use_dgl = dgl_data is not None
    graph_data = dgl_data if use_dgl else pyg_data

    if graph_data is None:
        raise ValueError("Either dgl_data or pyg_data must be provided")

    print(f"Using {'DGL' if use_dgl else 'PyG'} graph format")

    device = torch.device(device_str if torch.cuda.is_available() else "cpu")

    selection_strategy = selection_params.get('strategy', 'random')
    num_triples = selection_params.get('num_triples', 10)

    edge_index = graph_data['edge_index']
    edge_type = graph_data['edge_type']
    num_edges = edge_index.size(1)

    print(f"\nSelection strategy: {selection_strategy}")
    print(f"Number of triples to select: {num_triples}")
    print(f"Total edges in graph: {num_edges}")

    if selection_strategy == 'random':
        # Random selection from training edges
        indices = torch.randperm(num_edges)[:num_triples]

    elif selection_strategy == 'test_triples':
        # Select from test triples
        test_triples = graph_data.get('test_triples', None)

        if test_triples is None:
            print(f"Warning: No test triples found in {'dgl_data' if use_dgl else 'pyg_data'}, falling back to random")
            indices = torch.randperm(num_edges)[:num_triples]
        else:
            print(f"Total test triples available: {len(test_triples)}")

            # Randomly sample from test triples
            num_to_select = min(num_triples, len(test_triples))
            test_indices = torch.randperm(len(test_triples))[:num_to_select]

            # Extract the selected test triples
            selected_test_triples = test_triples[test_indices]

            # For test triples, we need to create edge_index and edge_type from triples
            # Test triples format: [head, relation, tail]
            selected_edge_index = torch.stack([
                selected_test_triples[:, 0],  # heads
                selected_test_triples[:, 2]   # tails
            ])
            selected_edge_type = selected_test_triples[:, 1]  # relations

            # Convert to readable format and return early
            triples_readable = []
            for i in range(len(selected_test_triples)):
                head = selected_test_triples[i, 0].item()
                tail = selected_test_triples[i, 2].item()
                rel = selected_test_triples[i, 1].item()

                # Get entity and relation names if available
                head_name = knowledge_graph.get('idx_to_entity', {}).get(head, f"node_{head}")
                tail_name = knowledge_graph.get('idx_to_entity', {}).get(tail, f"node_{tail}")
                rel_name = knowledge_graph.get('idx_to_relation', {}).get(rel, f"rel_{rel}")

                triples_readable.append({
                    'head_idx': head,
                    'tail_idx': tail,
                    'relation_idx': rel,
                    'head_name': head_name,
                    'tail_name': tail_name,
                    'relation_name': rel_name,
                    'triple': f"({head_name}, {rel_name}, {tail_name})"
                })

            # Print all selected triples
            print(f"\n✓ Selected {len(triples_readable)} test triples")
            print("\nSelected test triples:")
            for i, triple in enumerate(triples_readable):
                print(f"  {i+1}. {triple['triple']}")

            return {
                'selected_indices': test_indices,
                'selected_edge_index': selected_edge_index,
                'selected_edge_type': selected_edge_type,
                'triples_readable': triples_readable,
                'num_selected': len(triples_readable),
                'from_test_set': True
            }

    elif selection_strategy == 'from_file':
        # Load triples from Kedro catalog or file
        file_path = selection_params.get('file_path', None)

        # Use content from catalog if available, otherwise read from file
        if triple_file_content is not None:
            print(f"Loading triples from Kedro catalog dataset")
            file_lines = triple_file_content.strip().split('\n') if isinstance(triple_file_content, str) else triple_file_content.splitlines()
            print(f"Found {len(file_lines)} triples from catalog")
        elif file_path is not None:
            print(f"Loading triples from file: {file_path}")
            try:
                # Read triples from file (format: head\trelation\ttail)
                with open(file_path, 'r') as f:
                    file_lines = f.readlines()
                print(f"Found {len(file_lines)} triples in file")
            except Exception as e:
                print(f"Error loading file {file_path}: {e}")
                print("Falling back to random selection")
                indices = torch.randperm(num_edges)[:num_triples]
                file_lines = None
        else:
            print("Warning: No catalog content or file_path specified for 'from_file' strategy, falling back to random")
            indices = torch.randperm(num_edges)[:num_triples]
            file_lines = None

        if file_lines is not None:
            try:

                # Parse triples and convert to indices
                idx_to_entity = knowledge_graph.get('idx_to_entity', {})
                idx_to_relation = knowledge_graph.get('idx_to_relation', {})

                # Create reverse mappings
                entity_to_idx = {v: k for k, v in idx_to_entity.items()}
                relation_to_idx = {v: k for k, v in idx_to_relation.items()}

                selected_triples_list = []
                triples_readable = []

                for line in file_lines:
                    line = line.strip()
                    if not line:
                        continue

                    parts = line.split('\t')
                    if len(parts) != 3:
                        print(f"Warning: Skipping malformed line: {line}")
                        continue

                    head_name, rel_name, tail_name = parts

                    # Get indices
                    head_idx = entity_to_idx.get(head_name, None)
                    tail_idx = entity_to_idx.get(tail_name, None)
                    rel_idx = relation_to_idx.get(rel_name, None)

                    if head_idx is None or tail_idx is None or rel_idx is None:
                        print(f"Warning: Could not find indices for triple: {line}")
                        continue

                    selected_triples_list.append([head_idx, rel_idx, tail_idx])

                    triples_readable.append({
                        'head_idx': head_idx,
                        'tail_idx': tail_idx,
                        'relation_idx': rel_idx,
                        'head_name': head_name,
                        'tail_name': tail_name,
                        'relation_name': rel_name,
                        'triple': f"({head_name}, {rel_name}, {tail_name})"
                    })

                if not selected_triples_list:
                    print("Warning: No valid triples found in file, falling back to random")
                    indices = torch.randperm(num_edges)[:num_triples]
                else:
                    # Convert to tensors
                    selected_triples_tensor = torch.tensor(selected_triples_list, dtype=torch.long)

                    selected_edge_index = torch.stack([
                        selected_triples_tensor[:, 0],  # heads
                        selected_triples_tensor[:, 2]   # tails
                    ])
                    selected_edge_type = selected_triples_tensor[:, 1]  # relations

                    # Print all loaded triples
                    print(f"\n✓ Loaded {len(triples_readable)} triples from file")
                    print("\nLoaded triples:")
                    for i, triple in enumerate(triples_readable):
                        print(f"  {i+1}. {triple['triple']}")

                    return {
                        'selected_indices': torch.arange(len(triples_readable)),
                        'selected_edge_index': selected_edge_index,
                        'selected_edge_type': selected_edge_type,
                        'triples_readable': triples_readable,
                        'num_selected': len(triples_readable),
                        'from_file': True,
                        'file_path': file_path
                    }

            except Exception as e:
                print(f"Error loading file {file_path}: {e}")
                print("Falling back to random selection")
                indices = torch.randperm(num_edges)[:num_triples]

    elif selection_strategy == 'specific_relations':
        # Select triples with specific relation types
        target_relations = selection_params.get('target_relations', [0])
        mask = torch.zeros(num_edges, dtype=torch.bool)

        for rel in target_relations:
            mask |= (edge_type == rel)

        relation_edges = torch.where(mask)[0]

        if len(relation_edges) == 0:
            print(f"Warning: No edges found for relations {target_relations}")
            indices = torch.randperm(num_edges)[:num_triples]
        else:
            # Random sample from matching edges
            num_to_select = min(num_triples, len(relation_edges))
            perm = torch.randperm(len(relation_edges))[:num_to_select]
            indices = relation_edges[perm]

    elif selection_strategy == 'specific_nodes':
        # Select triples involving specific nodes
        target_nodes = selection_params.get('target_nodes', [])

        if not target_nodes:
            print("Warning: No target nodes specified, falling back to random")
            indices = torch.randperm(num_edges)[:num_triples]
        else:
            # Find edges involving target nodes
            mask = torch.zeros(num_edges, dtype=torch.bool)
            for node in target_nodes:
                mask |= (edge_index[0] == node) | (edge_index[1] == node)

            node_edges = torch.where(mask)[0]

            if len(node_edges) == 0:
                print(f"Warning: No edges found for nodes {target_nodes}")
                indices = torch.randperm(num_edges)[:num_triples]
            else:
                num_to_select = min(num_triples, len(node_edges))
                perm = torch.randperm(len(node_edges))[:num_to_select]
                indices = node_edges[perm]

    else:
        raise ValueError(f"Unknown selection strategy: {selection_strategy}")

    # Extract selected triples
    selected_edge_index = edge_index[:, indices]
    selected_edge_type = edge_type[indices]

    print(f"\n✓ Selected {len(indices)} triples")

    # Convert to readable format
    triples_readable = []
    for i in range(len(indices)):
        head = selected_edge_index[0, i].item()
        tail = selected_edge_index[1, i].item()
        rel = selected_edge_type[i].item()

        # Get entity and relation names if available
        head_name = knowledge_graph.get('idx_to_entity', {}).get(head, f"node_{head}")
        tail_name = knowledge_graph.get('idx_to_entity', {}).get(tail, f"node_{tail}")
        rel_name = knowledge_graph.get('idx_to_relation', {}).get(rel, f"rel_{rel}")

        triples_readable.append({
            'head_idx': head,
            'tail_idx': tail,
            'relation_idx': rel,
            'head_name': head_name,
            'tail_name': tail_name,
            'relation_name': rel_name,
            'triple': f"({head_name}, {rel_name}, {tail_name})"
        })

    # Print all selected triples
    print("\nSelected triples:")
    for i, triple in enumerate(triples_readable):
        print(f"  {i+1}. {triple['triple']}")

    return {
        'selected_indices': indices,
        'selected_edge_index': selected_edge_index,
        'selected_edge_type': selected_edge_type,
        'triples_readable': triples_readable,
        'num_selected': len(indices)
    }


def run_gnnexplainer(
    model_dict: Dict,
    selected_triples: Dict,
    explainer_params: Dict
) -> Dict:
    """
    Run GNNExplainer on selected triples.

    Args:
        model_dict: Dictionary with wrapped model and graph data
        selected_triples: Selected triples to explain
        explainer_params: GNNExplainer configuration

    Returns:
        Dictionary with explanations
    """
    print("\n" + "="*60)
    print("RUNNING GNNExplainer")
    print("="*60)

    import traceback as tb_module  # Import at top of function

    device = model_dict['device']
    wrapped_model = model_dict['wrapped_model']
    edge_index = model_dict['edge_index']
    edge_type = model_dict['edge_type']
    num_nodes = model_dict['num_nodes']

    # Extract GNNExplainer-specific configuration
    gnn_params = explainer_params.get('gnnexplainer', {})
    epochs = gnn_params.get('gnn_epochs', 200)
    lr = gnn_params.get('gnn_lr', 0.01)
    subgraph_method = gnn_params.get('subgraph_method', 'khop')

    print(f"\nGNNExplainer configuration:")
    print(f"  Epochs: {epochs}")
    print(f"  Learning rate: {lr}")
    print(f"  Subgraph method: {subgraph_method}")
    if subgraph_method == 'paths':
        print(f"  Max path length: {gnn_params.get('max_path_length', 3)}")
    else:
        print(f"  K-hop distance: {gnn_params.get('khop_distance', 2)}")

    # Create explainer
    explainer = Explainer(
        model=wrapped_model,
        algorithm=GNNExplainer(epochs=epochs, lr=lr),
        explanation_type='model',
        edge_mask_type='object',
        model_config=dict(
            mode='regression',  # For link prediction scores
            task_level='edge',
            return_type='raw'
        ),
    )

    print(f"\n✓ GNNExplainer initialized")

    # Create dummy node features (required by Explainer API but not used by KG models)
    # Since our KG model uses learned embeddings and doesn't use input features,
    # we create a minimal tensor instead of a massive identity matrix
    # The Explainer will only extract relevant nodes for subgraphs anyway
    print(f"\nNote: Using minimal node features (KG model uses learned embeddings)")
    x = torch.zeros((num_nodes, 1), device=device)  # Minimal features instead of identity matrix

    # Run explanation for each selected triple
    explanations = []

    selected_edge_index = selected_triples['selected_edge_index'].to(device)
    selected_edge_type = selected_triples['selected_edge_type'].to(device)
    triples_readable = selected_triples['triples_readable']

    print(f"\nGenerating explanations for {len(triples_readable)} triples...")
    print(f"Configuration: {epochs} optimization epochs per triple at lr={lr}")

    import time
    start_time = time.time()

    for i in range(len(triples_readable)):
        triple_start = time.time()
        print(f"\n  [{i+1}/{len(triples_readable)}] Explaining: {triples_readable[i]['triple']}")
        print(f"      Extracting 2-hop subgraph...", end='', flush=True)

        # Get the edge to explain
        edge_to_explain = selected_edge_index[:, i:i+1]
        edge_type_to_explain = selected_edge_type[i:i+1]

        try:
            # Extract subgraph to reduce memory usage
            # For explanation_type='model', we explain by providing the edge endpoints as index
            # The explainer will find important neighboring edges that lead to the prediction
            head_node = edge_to_explain[0, 0].item()
            tail_node = edge_to_explain[1, 0].item()

            # Use subgraph extraction method from config
            if subgraph_method == 'paths':
                # Use igraph-based path extraction
                max_path_length = gnn_params.get('max_path_length', 3)
                subset, sub_edge_index, mapping, edge_mask = extract_path_based_subgraph(
                    head_node, tail_node, edge_index, edge_type, max_path_length, device
                )
                print(f" Path-based subgraph (max_len={max_path_length}): ", end='', flush=True)
            else:
                # Use PyG k-hop subgraph extraction (default)
                from torch_geometric.utils import k_hop_subgraph
                khop_distance = gnn_params.get('khop_distance', 2)

                subset, sub_edge_index, mapping, edge_mask = k_hop_subgraph(
                    node_idx=[head_node, tail_node],
                    num_hops=khop_distance,
                    edge_index=edge_index,
                    relabel_nodes=True,
                    num_nodes=num_nodes
                )
                print(f" {khop_distance}-hop subgraph: ", end='', flush=True)

            # Extract edge types for subgraph
            sub_edge_type = edge_type[edge_mask]

            # Create subgraph node features
            # Ensure subset is on correct device and within bounds
            if subset.max().item() >= num_nodes:
                print(f"\n      Warning: subset contains indices >= num_nodes ({subset.max().item()} >= {num_nodes})")
                print(f"      Filtering out-of-bounds indices...")
                valid_mask = subset < num_nodes
                subset = subset[valid_mask]
                # Need to rebuild sub_edge_index with filtered nodes
                if len(subset) == 0:
                    raise ValueError("All nodes filtered out due to out-of-bounds indices")

            sub_x = x[subset]

            # Get remapped head node index
            head_node_remapped = mapping[0].item()

            # Validate indices
            if head_node_remapped >= len(subset):
                raise ValueError(f"head_node_remapped ({head_node_remapped}) >= len(subset) ({len(subset)})")
            if sub_edge_index.max().item() >= len(subset):
                raise ValueError(f"sub_edge_index contains indices >= len(subset) ({sub_edge_index.max().item()} >= {len(subset)})")

            print(f"{len(subset)} nodes, {sub_edge_index.size(1)} edges", end='', flush=True)

            # Diagnostic checks before explainer call
            print(f"\n      Diagnostic checks:")
            print(f"        - subset shape: {subset.shape}, min: {subset.min().item()}, max: {subset.max().item()}")
            print(f"        - sub_edge_index shape: {sub_edge_index.shape}, min: {sub_edge_index.min().item()}, max: {sub_edge_index.max().item()}")
            print(f"        - sub_edge_type shape: {sub_edge_type.shape}, min: {sub_edge_type.min().item()}, max: {sub_edge_type.max().item()}")
            print(f"        - sub_x shape: {sub_x.shape}")
            print(f"        - head_node_remapped: {head_node_remapped}")
            print(f"        - Expected: sub_edge_index.max() < len(subset) = {len(subset)}")
            print(f"        - Expected: sub_edge_type.max() < num_relations = {model_dict['num_relations']}")

            # Check if subset indices will be valid when accessing full graph embeddings
            print(f"        - Will access node_emb[{subset.min().item()}:{subset.max().item()}] from full graph embeddings")
            print(f"        - Full graph has {num_nodes} nodes")

            print(f"      Optimizing edge mask for {epochs} epochs...", end='', flush=True)

            # CRITICAL: Set the node subset mapping in the wrapped model
            # This allows the model to map relabeled subgraph indices back to global indices
            wrapped_model.current_subset = subset

            explanation = explainer(
                x=sub_x,
                edge_index=sub_edge_index,
                edge_type=sub_edge_type,  # Use subgraph edge types
                index=head_node_remapped  # Explain from the remapped head node
            )

            triple_time = time.time() - triple_start
            print(f" Done in {triple_time:.1f}s")

            # Extract explanation components
            edge_mask = explanation.edge_mask if hasattr(explanation, 'edge_mask') else None

            # Get top-k important edges
            top_k = gnn_params.get('top_k_edges', 10)

            if edge_mask is not None:
                top_k_indices = torch.topk(edge_mask, min(top_k, len(edge_mask))).indices

                # Map subgraph edges back to global graph
                # The important edges are in the subgraph space
                important_edges_subgraph = sub_edge_index[:, top_k_indices]
                important_edge_types_subgraph = sub_edge_type[top_k_indices]

                # Convert subgraph node indices back to global indices
                important_edges = torch.stack([
                    subset[important_edges_subgraph[0]],  # Map head nodes back
                    subset[important_edges_subgraph[1]]   # Map tail nodes back
                ])
                important_edge_types = important_edge_types_subgraph
                importance_scores = edge_mask[top_k_indices]
            else:
                important_edges = None
                important_edge_types = None
                importance_scores = None

            explanations.append({
                'triple': triples_readable[i],
                'explanation': explanation,
                'edge_mask': edge_mask,
                'important_edges': important_edges,
                'important_edge_types': important_edge_types,
                'importance_scores': importance_scores,
                'subgraph_size': len(subset)
            })

        except Exception as e:
            triple_time = time.time() - triple_start
            print(f" Failed in {triple_time:.1f}s")
            print(f"    ✗ Error type: {type(e).__name__}")
            print(f"    ✗ Error message: {str(e) if str(e) else '(empty error message)'}")
            print(f"    Full traceback:")
            tb_module.print_exc()
            explanations.append({
                'triple': triples_readable[i],
                'error': str(e) if str(e) else f"{type(e).__name__} (no message)",
                'error_type': type(e).__name__
            })

        # Show progress estimate
        if i < len(triples_readable) - 1:
            avg_time = (time.time() - start_time) / (i + 1)
            remaining = len(triples_readable) - (i + 1)
            eta_seconds = avg_time * remaining
            eta_mins = int(eta_seconds / 60)
            eta_secs = int(eta_seconds % 60)
            print(f"      Progress: {i+1}/{len(triples_readable)} done, ETA: {eta_mins}m {eta_secs}s", flush=True)

    total_time = time.time() - start_time
    print(f"\n✓ GNNExplainer completed: {len(explanations)} explanations generated in {total_time:.1f}s")

    return {
        'explainer_type': 'GNNExplainer',
        'explanations': explanations,
        'num_explanations': len(explanations),
        'params': gnn_params
    }


def run_pgexplainer(
    model_dict: Dict,
    selected_triples: Dict,
    pyg_data: Dict,
    explainer_params: Dict
) -> Dict:
    """
    Run PGExplainer (Parameterized Explainer) on selected triples.

    PGExplainer learns a parameterized explainer network that can generate
    explanations efficiently without optimization for each instance.

    Args:
        model_dict: Dictionary with wrapped model and graph data
        selected_triples: Selected triples to explain
        pyg_data: Full graph data for training PGExplainer
        explainer_params: PGExplainer configuration

    Returns:
        Dictionary with explanations
    """
    print("\n" + "="*60, flush=True)
    print("RUNNING PGExplainer", flush=True)
    print("="*60, flush=True)

    import traceback as tb_module  # Import at top of function
    import sys

    print("[PG] Step 1/6: Loading model and graph data...", flush=True)
    device = model_dict['device']
    wrapped_model = model_dict['wrapped_model']
    edge_index = model_dict['edge_index']
    edge_type = model_dict['edge_type']
    num_nodes = model_dict['num_nodes']
    print(f"[PG] ✓ Loaded: {num_nodes} nodes, {edge_index.size(1)} edges", flush=True)

    # Extract PGExplainer-specific configuration
    print("[PG] Step 2/6: Reading configuration...", flush=True)
    pg_params = explainer_params.get('pgexplainer', {})
    epochs = pg_params.get('pg_epochs', 30)
    lr = pg_params.get('pg_lr', 0.003)
    subgraph_method = pg_params.get('subgraph_method', 'khop')

    print(f"\n[PG] PGExplainer configuration:", flush=True)
    print(f"  Training epochs: {epochs}", flush=True)
    print(f"  Learning rate: {lr}", flush=True)
    print(f"  Subgraph method: {subgraph_method}", flush=True)
    if subgraph_method == 'paths':
        print(f"  Max path length: {pg_params.get('max_path_length', 3)}", flush=True)
    else:
        print(f"  K-hop distance: {pg_params.get('khop_distance', 2)}", flush=True)
    print(f"\n[PG] Note: PGExplainer trains an explainer network once,", flush=True)
    print(f"      then generates explanations efficiently for all instances.", flush=True)

    # Create explainer
    # Note: PGExplainer does NOT support task_level='edge', so we use 'node'
    # and explain link prediction as a node-level task on the head node
    # Using 'phenomenon' type to explain why specific links are predicted to exist
    print("[PG] Step 3/6: Initializing PGExplainer...", flush=True)
    explainer = Explainer(
        model=wrapped_model,
        algorithm=PGExplainer(epochs=epochs, lr=lr),
        explanation_type='phenomenon',  # Explain specific phenomenon (link existence)
        edge_mask_type='object',
        model_config=dict(
            mode='regression',
            task_level='node',  # PGExplainer only supports 'node' or 'graph' level
            return_type='raw'
        ),
    )

    print(f"[PG] ✓ PGExplainer initialized", flush=True)

    # Create dummy node features (required by Explainer API but not used by KG models)
    # Since our KG model uses learned embeddings and doesn't use input features,
    # we create a minimal tensor instead of a massive identity matrix
    print(f"[PG] Creating minimal node features tensor...", flush=True)
    x = torch.zeros((num_nodes, 1), device=device)  # Minimal features instead of identity matrix
    print(f"[PG] ✓ Created feature tensor: {x.shape}", flush=True)

    # Check if trained PGExplainer already exists
    import os
    from pathlib import Path

    explainer_cache_dir = Path("data/06_explainer_cache")
    explainer_cache_dir.mkdir(parents=True, exist_ok=True)
    explainer_cache_path = explainer_cache_dir / "pgexplainer_trained.pt"

    use_full_graph_training = pg_params.get('use_full_graph_training', False)
    force_retrain = pg_params.get('force_retrain', False)

    print(f"\n[PG] Step 4/6: Training/Loading PGExplainer network...", flush=True)

    if explainer_cache_path.exists() and not force_retrain:
        print(f"[PG] Found cached trained explainer at {explainer_cache_path}", flush=True)
        print(f"[PG] Loading trained PGExplainer network...", flush=True)
        try:
            checkpoint = torch.load(explainer_cache_path, map_location=device)
            explainer.algorithm.load_state_dict(checkpoint['explainer_state_dict'])
            print(f"[PG] ✓ Loaded trained explainer (trained for {checkpoint['epochs']} epochs)", flush=True)
            print(f"[PG]   Training config: {checkpoint['training_config']}", flush=True)
        except Exception as e:
            print(f"[PG] ⚠ Failed to load cached explainer: {e}", flush=True)
            print(f"[PG] Will retrain from scratch...", flush=True)
            force_retrain = True
    else:
        if force_retrain:
            print(f"[PG] Force retrain enabled, training from scratch...", flush=True)
        else:
            print(f"[PG] No cached explainer found, training from scratch...", flush=True)
        force_retrain = True

    if force_retrain or not explainer_cache_path.exists():
        print(f"[PG] This learns a parameterized explainer that works for all instances", flush=True)

        # Determine training data (full graph or subgraph)
        if use_full_graph_training:
            # Use full graph for training (slower but more comprehensive)
            print(f"[PG] Using FULL GRAPH for training (may be slow)...", flush=True)
            training_edge_index = edge_index
            training_edge_type = edge_type
            training_x = x
            training_subset = None  # No subset mapping needed
            training_subgraph_info = "full_graph"

            # Sample edges from full graph
            num_edges = edge_index.size(1)
            training_edges = pg_params.get('training_edges', 100)
            train_edge_indices = torch.randperm(num_edges, device=device)[:min(training_edges, num_edges)]
            train_node_pairs = edge_index[:, train_edge_indices]

            print(f"[PG] ✓ Sampled {train_node_pairs.size(1)} edges from full graph", flush=True)
        else:
            # OPTIMIZED APPROACH: Extract ONE representative subgraph, then train on edges within it
            print(f"[PG] Using SUBGRAPH-BASED training (faster)...", flush=True)
            print(f"[PG] Extracting representative training subgraph...", flush=True)

            # Select a random edge to center the training subgraph around
            num_edges = edge_index.size(1)
            center_edge_idx = torch.randint(0, num_edges, (1,), device=device).item()
            center_head = edge_index[0, center_edge_idx].item()
            center_tail = edge_index[1, center_edge_idx].item()

            print(f"[PG] Center edge: ({center_head}, {center_tail})", flush=True)

            # Extract a subgraph using the configured method
            if subgraph_method == 'paths':
                max_path_length = pg_params.get('max_path_length', 3)
                print(f"[PG] Using path-based extraction (max_path_length={max_path_length})...", flush=True)
                subset, sub_edge_index, mapping, edge_mask = extract_path_based_subgraph(
                    center_head, center_tail, edge_index, edge_type, max_path_length, device
                )
            else:
                khop_distance = pg_params.get('khop_distance', 2)
                print(f"[PG] Using k-hop extraction (khop={khop_distance})...", flush=True)
                from torch_geometric.utils import k_hop_subgraph
                subset, sub_edge_index, mapping, edge_mask = k_hop_subgraph(
                    node_idx=[center_head, center_tail],
                    num_hops=khop_distance,
                    edge_index=edge_index,
                    relabel_nodes=True,
                    num_nodes=num_nodes
                )

            # Extract subgraph data
            sub_edge_type = edge_type[edge_mask]
            sub_x = x[subset]

            # Ensure all tensors are on the correct device
            subset = subset.to(device)
            sub_edge_index = sub_edge_index.to(device)
            sub_edge_type = sub_edge_type.to(device)
            sub_x = sub_x.to(device)

            print(f"[PG] ✓ Training subgraph: {len(subset)} nodes, {sub_edge_index.size(1)} edges", flush=True)

            # Set training data to subgraph
            training_edge_index = sub_edge_index
            training_edge_type = sub_edge_type
            training_x = sub_x
            training_subset = subset
            training_subgraph_info = f"subgraph_{len(subset)}_nodes_{sub_edge_index.size(1)}_edges"

            # Now sample edges FROM WITHIN this subgraph for training
            training_edges = pg_params.get('training_edges', 100)
            num_subgraph_edges = sub_edge_index.size(1)
            actual_training_edges = min(training_edges, num_subgraph_edges)

            train_edge_indices = torch.randperm(num_subgraph_edges, device=device)[:actual_training_edges]
            train_node_pairs = training_edge_index[:, train_edge_indices]

            print(f"[PG] ✓ Sampled {train_node_pairs.size(1)} edges from training subgraph", flush=True)

        # Train the explainer
        import time
        train_start = time.time()
        successful_batches = 0
        failed_batches = 0

        # Set the subgraph/full graph context for the model
        wrapped_model.current_subset = training_subset

        for epoch in range(epochs):
            epoch_start = time.time()
            epoch_successful = 0
            epoch_failed = 0

            if epoch == 0:
                print(f"[PG-TRAIN] Starting epoch {epoch+1}/{epochs}...", flush=True)

            # Train on sampled edges
            for i in range(train_node_pairs.size(1)):
                try:
                    # Detailed logging for first 3 batches, then every 10th batch
                    if epoch == 0 and (i < 3 or (i + 1) % 10 == 0):
                        print(f"[PG-TRAIN] Processing batch {i+1}/{train_node_pairs.size(1)}...", flush=True)
                        if i < 3:
                            head_idx = train_node_pairs[0, i].item()
                            tail_idx = train_node_pairs[1, i].item()
                            print(f"[PG-TRAIN]   Edge: ({head_idx}, {tail_idx})", flush=True)
                            print(f"[PG-TRAIN]   Calling explainer.algorithm.train()...", flush=True)

                    # Get the head node index for this edge
                    head_node_idx = train_node_pairs[0, i].item()

                    # Train step - using the training data (subgraph or full graph)
                    target = torch.tensor([1.0], device=device)
                    explainer.algorithm.train(
                        epoch=epoch,
                        model=wrapped_model,
                        x=training_x,
                        edge_index=training_edge_index,
                        target=target,
                        index=head_node_idx,
                        edge_type=training_edge_type
                    )

                    if epoch == 0 and i < 3:
                        print(f"[PG-TRAIN]   ✓ Training step complete", flush=True)

                    epoch_successful += 1
                    successful_batches += 1

                except Exception as e:
                    epoch_failed += 1
                    failed_batches += 1
                    # Skip problematic training examples, but log for debugging
                    if epoch == 0 and (i < 3 or (i + 1) % 10 == 0):
                        print(f"[PG-TRAIN]   ✗ Error on batch {i+1}: {type(e).__name__}: {e}", flush=True)
                    if "CUDA" in str(e):
                        # Try to clear CUDA cache
                        if device.type == 'cuda':
                            torch.cuda.empty_cache()
                    continue

            # Print epoch progress
            epoch_time = time.time() - epoch_start
            if (epoch + 1) % 5 == 0 or epoch == 0:
                print(f"[PG-TRAIN] Epoch {epoch+1}/{epochs}: {epoch_successful} successful, {epoch_failed} failed ({epoch_time:.1f}s)", flush=True)

        # Training summary
        train_time = time.time() - train_start
        success_rate = (successful_batches / (successful_batches + failed_batches) * 100) if (successful_batches + failed_batches) > 0 else 0
        print(f"\n[PG] ✓ PGExplainer network trained ({epochs} epochs, {train_time:.1f}s)", flush=True)
        print(f"[PG]   Training summary: {successful_batches} successful, {failed_batches} failed ({success_rate:.1f}% success rate)", flush=True)
        print(f"[PG]   Training mode: {training_subgraph_info}", flush=True)

        # Save the trained explainer
        print(f"[PG] Saving trained explainer to {explainer_cache_path}...", flush=True)
        checkpoint = {
            'explainer_state_dict': explainer.algorithm.state_dict(),
            'epochs': epochs,
            'training_config': {
                'lr': lr,
                'subgraph_method': subgraph_method if not use_full_graph_training else 'full_graph',
                'training_edges': train_node_pairs.size(1),
                'training_mode': training_subgraph_info,
                'success_rate': success_rate
            }
        }
        torch.save(checkpoint, explainer_cache_path)
        print(f"[PG] ✓ Trained explainer saved successfully", flush=True)

    # Generate explanations for selected triples
    print(f"\n[PG] Step 5/6: Generating explanations for selected triples...", flush=True)
    explanations = []

    selected_edge_index = selected_triples['selected_edge_index'].to(device)
    selected_edge_type = selected_triples['selected_edge_type'].to(device)
    triples_readable = selected_triples['triples_readable']

    print(f"[PG] Generating explanations for {len(triples_readable)} triples...", flush=True)
    print(f"[PG] Configuration: Pre-trained explainer network (no per-triple optimization)", flush=True)

    import time
    start_time = time.time()

    for i in range(len(triples_readable)):
        triple_start = time.time()
        if i == 0 or (i + 1) % 5 == 0 or i == len(triples_readable) - 1:
            print(f"\n[PG-EXPLAIN] [{i+1}/{len(triples_readable)}] Explaining: {triples_readable[i]['triple']}", flush=True)
        if i < 3:
            print(f"[PG-EXPLAIN]   Extracting subgraph...", flush=True)

        edge_to_explain = selected_edge_index[:, i:i+1]
        edge_type_to_explain = selected_edge_type[i:i+1]

        try:
            # Extract subgraph to reduce memory usage
            # For explanation_type='model', we explain by providing the edge endpoints as index
            # The explainer will find important neighboring edges that lead to the prediction
            head_node = edge_to_explain[0, 0].item()
            tail_node = edge_to_explain[1, 0].item()

            # Use subgraph extraction method from config
            if subgraph_method == 'paths':
                # Use igraph-based path extraction
                max_path_length = pg_params.get('max_path_length', 3)
                subset, sub_edge_index, mapping, edge_mask = extract_path_based_subgraph(
                    head_node, tail_node, edge_index, edge_type, max_path_length, device
                )
                print(f" Path-based subgraph (max_len={max_path_length}): ", end='', flush=True)
            else:
                # Use PyG k-hop subgraph extraction (default)
                from torch_geometric.utils import k_hop_subgraph
                khop_distance = pg_params.get('khop_distance', 2)

                subset, sub_edge_index, mapping, edge_mask = k_hop_subgraph(
                    node_idx=[head_node, tail_node],
                    num_hops=khop_distance,
                    edge_index=edge_index,
                    relabel_nodes=True,
                    num_nodes=num_nodes
                )
                print(f" {khop_distance}-hop subgraph: ", end='', flush=True)

            # Extract edge types for subgraph
            sub_edge_type = edge_type[edge_mask]

            # Create subgraph node features
            # Ensure subset is on correct device and within bounds
            if subset.max().item() >= num_nodes:
                print(f"\n      Warning: subset contains indices >= num_nodes ({subset.max().item()} >= {num_nodes})")
                print(f"      Filtering out-of-bounds indices...")
                valid_mask = subset < num_nodes
                subset = subset[valid_mask]
                if len(subset) == 0:
                    raise ValueError("All nodes filtered out due to out-of-bounds indices")

            sub_x = x[subset]

            # Get remapped head node index
            head_node_remapped = mapping[0].item()

            # Validate indices
            if head_node_remapped >= len(subset):
                raise ValueError(f"head_node_remapped ({head_node_remapped}) >= len(subset) ({len(subset)})")
            if sub_edge_index.max().item() >= len(subset):
                raise ValueError(f"sub_edge_index contains indices >= len(subset) ({sub_edge_index.max().item()} >= {len(subset)})")

            if i < 3:
                print(f"[PG-EXPLAIN]   ✓ Subgraph: {len(subset)} nodes, {sub_edge_index.size(1)} edges", flush=True)
                print(f"[PG-EXPLAIN]   Setting model subset and generating explanation...", flush=True)

            # CRITICAL: Set the node subset mapping in the wrapped model
            # This allows the model to map relabeled subgraph indices back to global indices
            wrapped_model.current_subset = subset

            # For PGExplainer, get the target (model prediction for the link)
            # Since we're explaining test triples, the target is 1 (link exists)
            # For node-level task, the target is a scalar value for that node
            target = torch.tensor([1.0], device=device)

            explanation = explainer(
                x=sub_x,
                edge_index=sub_edge_index,
                edge_type=sub_edge_type,  # Use subgraph edge types
                index=head_node_remapped,  # Explain from the remapped head node
                target=target  # Target: 1.0 = link exists (test triple)
            )

            triple_time = time.time() - triple_start
            if i < 3:
                print(f"[PG-EXPLAIN]   ✓ Explanation generated in {triple_time:.1f}s", flush=True)

            # Extract explanation components
            edge_mask = explanation.edge_mask if hasattr(explanation, 'edge_mask') else None

            # Get top-k important edges
            top_k = pg_params.get('top_k_edges', 10)

            if edge_mask is not None:
                top_k_indices = torch.topk(edge_mask, min(top_k, len(edge_mask))).indices

                # Map subgraph edges back to global graph
                important_edges_subgraph = sub_edge_index[:, top_k_indices]
                important_edge_types_subgraph = sub_edge_type[top_k_indices]

                # Convert subgraph node indices back to global indices
                important_edges = torch.stack([
                    subset[important_edges_subgraph[0]],
                    subset[important_edges_subgraph[1]]
                ])
                important_edge_types = important_edge_types_subgraph
                importance_scores = edge_mask[top_k_indices]
            else:
                important_edges = None
                important_edge_types = None
                importance_scores = None

            explanations.append({
                'triple': triples_readable[i],
                'explanation': explanation,
                'edge_mask': edge_mask,
                'important_edges': important_edges,
                'important_edge_types': important_edge_types,
                'importance_scores': importance_scores,
                'subgraph_size': len(subset)
            })

        except Exception as e:
            triple_time = time.time() - triple_start
            print(f" Failed in {triple_time:.1f}s")
            print(f"    ✗ Error type: {type(e).__name__}")
            print(f"    ✗ Error message: {str(e) if str(e) else '(empty error message)'}")
            print(f"    Full traceback:")
            tb_module.print_exc()
            explanations.append({
                'triple': triples_readable[i],
                'error': str(e) if str(e) else f"{type(e).__name__} (no message)",
                'error_type': type(e).__name__
            })

        # Show progress estimate
        if (i + 1) % 5 == 0 and i < len(triples_readable) - 1:
            avg_time = (time.time() - start_time) / (i + 1)
            remaining = len(triples_readable) - (i + 1)
            eta_seconds = avg_time * remaining
            eta_mins = int(eta_seconds / 60)
            eta_secs = int(eta_seconds % 60)
            print(f"[PG-EXPLAIN] Progress: {i+1}/{len(triples_readable)} done, ETA: {eta_mins}m {eta_secs}s", flush=True)

    total_time = time.time() - start_time
    print(f"\n[PG] Step 6/6: Finalizing results...", flush=True)
    print(f"[PG] ✓ PGExplainer completed: {len(explanations)} explanations generated in {total_time:.1f}s", flush=True)

    return {
        'explainer_type': 'PGExplainer',
        'explanations': explanations,
        'num_explanations': len(explanations),
        'params': pg_params
    }


def run_page_explainer(
    model_dict: Dict,
    selected_triples: Dict,
    pyg_data: Dict,
    explainer_params: Dict
) -> Dict:
    """
    Run Improved PAGE (Parametric Generative Explainer) on selected triples.

    This improved version:
    1. Uses frozen CompGCN encoder embeddings (model-aware)
    2. Prediction-aware training (weighted by model scores)
    3. Explains: "Why did CompGCN predict this triple?"

    Args:
        model_dict: Dictionary with wrapped model and graph data
        selected_triples: Selected triples to explain
        pyg_data: Full graph data for training PAGE
        explainer_params: PAGE configuration

    Returns:
        Dictionary with explanations
    """
    print("\n" + "="*60)
    print("RUNNING Improved PAGE Explainer")
    print("="*60)
    print("Using: CompGCN features + Prediction-aware training")

    device = model_dict['device']
    edge_index = model_dict['edge_index']
    edge_type = model_dict['edge_type']
    num_nodes = model_dict['num_nodes']
    trained_model = model_dict['model']

    # Import improved PAGE components
    from .page_improved import ImprovedPAGEExplainer, extract_link_subgraph

    # Extract PAGE-specific configuration
    page_params = explainer_params.get('page', {})
    train_epochs = page_params.get('train_epochs', 100)
    lr = page_params.get('lr', 0.003)
    k_hops = page_params.get('k_hops', 2)
    hidden_dim = page_params.get('encoder_hidden1', 32)
    latent_dim = page_params.get('latent_dim', 16)
    prediction_weight = page_params.get('prediction_weight', 1.0)

    # Get subgraph extraction method
    subgraph_method = page_params.get('subgraph_method', 'khop')
    max_path_length = page_params.get('max_path_length', 3)

    print(f"\nImproved PAGE configuration:")
    print(f"  Training epochs: {train_epochs}")
    print(f"  Learning rate: {lr}")
    print(f"  Subgraph method: {subgraph_method}")
    if subgraph_method == 'paths':
        print(f"  Max path length: {max_path_length}")
    else:
        print(f"  K-hops: {k_hops}")
    print(f"  Latent dim: {latent_dim}")
    print(f"  Prediction weight: {prediction_weight}")
    print(f"  Using CompGCN embeddings: {trained_model.embedding_dim}D")

    # Initialize Improved PAGE explainer
    # Uses CompGCN embedding dimension as input
    page_explainer = ImprovedPAGEExplainer(
        compgcn_model=trained_model,
        edge_index=edge_index,
        edge_type=edge_type,
        embedding_dim=trained_model.embedding_dim,
        hidden_dim=hidden_dim,
        latent_dim=latent_dim,
        decoder_hidden_dim=page_params.get('decoder_hidden1', 16),
        dropout=page_params.get('dropout', 0.0),
        device=device
    )

    print(f"\n✓ Improved PAGE explainer initialized (with frozen CompGCN features)")

    # Extract subgraphs for training
    selected_edge_index = selected_triples['selected_edge_index']
    selected_edge_type = selected_triples['selected_edge_type']
    triples_readable = selected_triples['triples_readable']

    print(f"\nExtracting subgraphs with CompGCN features for {len(triples_readable)} triples...")

    subgraphs_data = []
    subgraph_info = []

    for i in range(len(triples_readable)):
        head_idx = selected_edge_index[0, i].item()
        tail_idx = selected_edge_index[1, i].item()
        rel_idx = selected_edge_type[i].item()

        try:
            # Extract subgraph around head and tail
            subgraph_nodes, subgraph_edges, adj_matrix = extract_link_subgraph(
                edge_index=edge_index.cpu(),
                head_idx=head_idx,
                tail_idx=tail_idx,
                num_hops=k_hops,
                num_nodes=num_nodes,
                method=subgraph_method,
                edge_type=edge_type.cpu() if subgraph_method == 'paths' else None,
                max_path_length=max_path_length
            )

            num_subgraph_nodes = len(subgraph_nodes)
            if num_subgraph_nodes > 0:
                # Get CompGCN features for subgraph nodes
                subgraph_features = page_explainer.get_subgraph_features(subgraph_nodes)

                # Add batch dimension
                x = subgraph_features.unsqueeze(0)  # (1, num_nodes, embedding_dim)
                adj = adj_matrix.unsqueeze(0)  # (1, num_nodes, num_nodes)

                # Get prediction score from CompGCN
                prediction_score = page_explainer.get_triple_score(head_idx, tail_idx, rel_idx)

                subgraphs_data.append({
                    'features': x,
                    'adj': adj,
                    'prediction_score': prediction_score
                })

                subgraph_info.append({
                    'triple_idx': i,
                    'num_nodes': num_subgraph_nodes,
                    'num_edges': subgraph_edges.size(1) if subgraph_edges.numel() > 0 else 0,
                    'subgraph_nodes': subgraph_nodes,
                    'subgraph_edges': subgraph_edges,
                    'prediction_score': prediction_score
                })

        except Exception as e:
            print(f"  Warning: Failed to extract subgraph for triple {i}: {e}")

    print(f"✓ Extracted {len(subgraphs_data)} subgraphs with CompGCN features")

    # Show prediction score statistics
    if subgraphs_data:
        scores = [d['prediction_score'] for d in subgraphs_data]
        print(f"\nPrediction scores: min={min(scores):.4f}, max={max(scores):.4f}, mean={sum(scores)/len(scores):.4f}")

    if len(subgraphs_data) == 0:
        print("✗ No valid subgraphs extracted, cannot train PAGE")
        return {
            'explainer_type': 'ImprovedPAGE',
            'explanations': [],
            'num_explanations': 0,
            'params': page_params,
            'error': 'No valid subgraphs extracted'
        }

    # Train PAGE with prediction-aware loss
    print(f"\nTraining Improved PAGE on {len(subgraphs_data)} subgraphs...")
    print(f"  Using prediction-aware training (weight={prediction_weight})")
    page_explainer.train_on_subgraphs(
        subgraphs_data=subgraphs_data,
        epochs=train_epochs,
        lr=lr,
        kl_weight=page_params.get('kl_weight', 0.2),
        prediction_weight=prediction_weight,
        verbose=True
    )

    print(f"✓ Improved PAGE training completed")

    # Generate explanations
    print(f"\nGenerating explanations for {len(triples_readable)} triples...")
    print(f"Configuration: Using trained VGAE to decode edge importance")

    import time
    start_time = time.time()

    explanations = []

    for i, data in enumerate(subgraphs_data):
        triple_start = time.time()
        info = subgraph_info[i]
        triple_idx = info['triple_idx']
        triple = triples_readable[triple_idx]
        pred_score = info['prediction_score']

        print(f"\n  [{i+1}/{len(subgraphs_data)}] Explaining: {triple['triple']} (score={pred_score:.4f})")
        print(f"      Generating explanation...", end='', flush=True)

        try:
            # Generate explanation using improved PAGE
            edge_importance, latent_z = page_explainer.explain(data['features'], data['adj'])

            triple_time = time.time() - triple_start
            print(f" Done in {triple_time:.1f}s")

            # Extract edge importance scores
            edge_importance_matrix = edge_importance.squeeze(0).cpu()  # (num_nodes, num_nodes)

            # Get top-k important edges
            top_k = page_params.get('top_k_edges', 10)

            # Flatten and get top-k
            num_subgraph_nodes = edge_importance_matrix.size(0)
            importance_flat = edge_importance_matrix.flatten()
            top_k_actual = min(top_k, importance_flat.numel())

            if top_k_actual > 0:
                top_k_values, top_k_indices = torch.topk(importance_flat, top_k_actual)

                # Convert flat indices to edge indices
                top_k_edges_local = torch.stack([
                    top_k_indices // num_subgraph_nodes,
                    top_k_indices % num_subgraph_nodes
                ])

                # Map back to original node indices
                subgraph_nodes = info['subgraph_nodes']
                top_k_edges_global = torch.tensor([
                    [subgraph_nodes[top_k_edges_local[0, j]].item(),
                     subgraph_nodes[top_k_edges_local[1, j]].item()]
                    for j in range(top_k_edges_local.size(1))
                ]).t()

                # Get edge types (approximate - use most common type in subgraph)
                # This is a simplification since we don't track edge types in subgraph extraction
                top_k_edge_types = torch.zeros(top_k_actual, dtype=torch.long)

                importance_scores = top_k_values

            else:
                top_k_edges_global = None
                top_k_edge_types = None
                importance_scores = None

            explanations.append({
                'triple': triple,
                'edge_importance_matrix': edge_importance_matrix,
                'important_edges': top_k_edges_global,
                'important_edge_types': top_k_edge_types,
                'importance_scores': importance_scores,
                'subgraph_info': info,
                'latent_representation': latent_z.squeeze(0).cpu(),
                'prediction_score': pred_score
            })

        except Exception as e:
            triple_time = time.time() - triple_start
            print(f" Failed in {triple_time:.1f}s")
            print(f"    ✗ Error: {str(e)}")
            explanations.append({
                'triple': triple,
                'error': str(e)
            })

        # Show progress estimate
        if i < len(subgraphs_data) - 1:
            avg_time = (time.time() - start_time) / (i + 1)
            remaining = len(subgraphs_data) - (i + 1)
            eta_seconds = avg_time * remaining
            eta_mins = int(eta_seconds / 60)
            eta_secs = int(eta_seconds % 60)
            print(f"      Progress: {i+1}/{len(subgraphs_data)} done, ETA: {eta_mins}m {eta_secs}s", flush=True)

    total_time = time.time() - start_time
    print(f"\n✓ Improved PAGE explainer completed: {len(explanations)} explanations generated in {total_time:.1f}s")
    print(f"   Uses: CompGCN embeddings + Prediction-aware training")

    return {
        'explainer_type': 'ImprovedPAGE',
        'explanations': explanations,
        'num_explanations': len(explanations),
        'params': page_params,
        'subgraph_info': subgraph_info,
        'model_aware': True,
        'uses_encoder': True,
        'uses_decoder': True
    }


def summarize_explanations(
    gnn_explanations: Optional[Dict] = None,
    pg_explanations: Optional[Dict] = None,
    knowledge_graph: Dict = None,
    page_explanations: Optional[Dict] = None
) -> Dict:
    """
    Summarize and compare explanations from different explainers.

    All explainer outputs are optional - only available explainers will be summarized.

    Args:
        gnn_explanations: GNNExplainer results (optional)
        pg_explanations: PGExplainer results (optional)
        knowledge_graph: Knowledge graph with dictionaries
        page_explanations: PAGE explainer results (optional)

    Returns:
        Summary dictionary with comparisons and insights
    """
    print("\n" + "="*60)
    print("SUMMARIZING EXPLANATIONS")
    print("="*60)

    summary = {
        'explainers_run': [],
        'comparisons': []
    }

    # Add GNNExplainer if available
    if gnn_explanations is not None:
        summary['gnn_explainer'] = {
            'num_explanations': gnn_explanations['num_explanations'],
            'successful': sum(1 for e in gnn_explanations['explanations'] if 'error' not in e),
            'failed': sum(1 for e in gnn_explanations['explanations'] if 'error' in e)
        }
        summary['explainers_run'].append('gnn_explainer')
        print(f"\nGNNExplainer: {summary['gnn_explainer']['successful']}/{summary['gnn_explainer']['num_explanations']} successful")

    # Add PGExplainer if available
    if pg_explanations is not None:
        summary['pg_explainer'] = {
            'num_explanations': pg_explanations['num_explanations'],
            'successful': sum(1 for e in pg_explanations['explanations'] if 'error' not in e),
            'failed': sum(1 for e in pg_explanations['explanations'] if 'error' in e)
        }
        summary['explainers_run'].append('pg_explainer')
        print(f"PGExplainer: {summary['pg_explainer']['successful']}/{summary['pg_explainer']['num_explanations']} successful")

    # Add PAGE if available
    if page_explanations is not None:
        summary['page_explainer'] = {
            'num_explanations': page_explanations['num_explanations'],
            'successful': sum(1 for e in page_explanations['explanations'] if 'error' not in e),
            'failed': sum(1 for e in page_explanations['explanations'] if 'error' in e)
        }
        summary['explainers_run'].append('page_explainer')
        print(f"PAGE Explainer: {summary['page_explainer']['successful']}/{summary['page_explainer']['num_explanations']} successful")

    # Skip comparison if less than 2 explainers were run
    if len(summary['explainers_run']) < 2:
        print(f"\nOnly {len(summary['explainers_run'])} explainer(s) run - skipping comparison")
        return summary

    # Compare explanations for each triple
    print(f"\nComparing explanations...")

    # Get any available explanations list to iterate over
    if gnn_explanations is not None and pg_explanations is not None:
        # Compare GNN vs PG
        for gnn_exp, pg_exp in zip(gnn_explanations['explanations'], pg_explanations['explanations']):
            if 'error' in gnn_exp or 'error' in pg_exp:
                continue

            triple = gnn_exp['triple']

            # Compare important edges
            comparison = {
                'triple': triple['triple'],
                'gnn_top_edges': [],
                'pg_top_edges': [],
                'overlap': 0
            }

            # Get top edges from each explainer
            if gnn_exp.get('important_edges') is not None:
                for j in range(min(5, len(gnn_exp['importance_scores']))):
                    head = gnn_exp['important_edges'][0, j].item()
                    tail = gnn_exp['important_edges'][1, j].item()
                    rel = gnn_exp['important_edge_types'][j].item()
                    score = gnn_exp['importance_scores'][j].item()

                    head_name = knowledge_graph.get('idx_to_entity', {}).get(head, f"node_{head}")
                    tail_name = knowledge_graph.get('idx_to_entity', {}).get(tail, f"node_{tail}")
                    rel_name = knowledge_graph.get('idx_to_relation', {}).get(rel, f"rel_{rel}")

                    comparison['gnn_top_edges'].append({
                        'edge': f"({head_name}, {rel_name}, {tail_name})",
                        'score': score
                    })

            if pg_exp.get('important_edges') is not None:
                for j in range(min(5, len(pg_exp['importance_scores']))):
                    head = pg_exp['important_edges'][0, j].item()
                    tail = pg_exp['important_edges'][1, j].item()
                    rel = pg_exp['important_edge_types'][j].item()
                    score = pg_exp['importance_scores'][j].item()

                    head_name = knowledge_graph.get('idx_to_entity', {}).get(head, f"node_{head}")
                    tail_name = knowledge_graph.get('idx_to_entity', {}).get(tail, f"node_{tail}")
                    rel_name = knowledge_graph.get('idx_to_relation', {}).get(rel, f"rel_{rel}")

                    comparison['pg_top_edges'].append({
                        'edge': f"({head_name}, {rel_name}, {tail_name})",
                        'score': score
                    })

            # Calculate overlap
            gnn_edges_set = set(e['edge'] for e in comparison['gnn_top_edges'])
            pg_edges_set = set(e['edge'] for e in comparison['pg_top_edges'])
            comparison['overlap'] = len(gnn_edges_set & pg_edges_set)

            summary['comparisons'].append(comparison)

    if summary['comparisons']:
        print(f"✓ Compared {len(summary['comparisons'])} explanations")
        avg_overlap = sum(c['overlap'] for c in summary['comparisons']) / len(summary['comparisons'])
        print(f"\nAverage overlap in top-5 important edges: {avg_overlap:.2f}")

    return summary
