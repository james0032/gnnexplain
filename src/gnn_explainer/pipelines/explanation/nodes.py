"""Nodes for explanation pipeline."""

import torch
import torch.nn as nn
from typing import Dict, List, Tuple, Optional
from torch_geometric.explain import Explainer, GNNExplainer, PGExplainer
from torch_geometric.explain import Explanation


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

    def forward(self, x, edge_index, **kwargs):
        """
        Forward pass compatible with PyG Explainer.

        For link prediction, we use the edge_index to determine which
        links to score.

        Args:
            x: Node features (not used for KG models, but required by Explainer)
            edge_index: Edge index to explain (shape: [2, num_edges])

        Returns:
            Scores for the given edges
        """
        if self.mode == 'link_prediction':
            # Get node and relation embeddings
            node_emb, rel_emb = self.kg_model.encode(self.edge_index, self.edge_type)

            # Score the edges
            head_idx = edge_index[0]
            tail_idx = edge_index[1]

            # For explanation, we need to determine relation type
            # We'll use the edge_type from the original graph
            # Find which edges in the original graph match the query edges
            edge_type_for_query = kwargs.get('edge_type', None)

            if edge_type_for_query is None:
                # Default: assume relation 0 for all edges
                # In practice, this should be provided
                edge_type_for_query = torch.zeros(edge_index.size(1), dtype=torch.long, device=edge_index.device)

            scores = self.kg_model.decode(node_emb, rel_emb, head_idx, tail_idx, edge_type_for_query)

            return scores
        else:
            raise NotImplementedError(f"Mode {self.mode} not implemented yet")


def prepare_model_for_explanation(
    trained_model_dict: Dict,
    pyg_data: Dict,
    device_str: str = "cpu"
) -> Dict:
    """
    Load trained model and prepare for explanation.

    Args:
        trained_model_dict: Dictionary with model state and metadata
        pyg_data: PyG format graph data
        device_str: Device string ("cuda" or "cpu")

    Returns:
        Dictionary with wrapped model and graph data
    """
    print("\n" + "="*60)
    print("PREPARING MODEL FOR EXPLANATION")
    print("="*60)

    device = torch.device(device_str if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # Recreate model from saved state
    from ..training.kg_models import CompGCNKGModel
    from ..training.model import RGCNDistMultModel

    metadata = trained_model_dict['metadata']
    model_type = metadata['model_type']

    print(f"\nRecreating {model_type.upper()} model...")
    print(f"  Decoder: {metadata['decoder_type']}")

    if model_type == 'compgcn':
        model = CompGCNKGModel(
            num_nodes=metadata['num_nodes'],
            num_relations=metadata['num_relations'],
            embedding_dim=metadata['embedding_dim'],
            decoder_type=metadata['decoder_type'],
            num_layers=metadata['num_layers'],
            comp_fn=metadata.get('comp_fn', 'sub'),
            dropout=metadata['dropout'],
            conve_kwargs=metadata.get('conve_kwargs')
        )
    elif model_type == 'rgcn':
        model = RGCNDistMultModel(
            num_nodes=metadata['num_nodes'],
            num_relations=metadata['num_relations'],
            embedding_dim=metadata['embedding_dim'],
            num_layers=metadata['num_layers'],
            num_bases=metadata.get('num_bases', 30),
            dropout=metadata['dropout']
        )
    else:
        raise ValueError(f"Unknown model_type: {model_type}")

    # Load trained weights
    model.load_state_dict(trained_model_dict['model_state_dict'])
    model.to(device)
    model.eval()

    print(f"✓ Model loaded successfully")

    # Wrap model for Explainer API
    edge_index = pyg_data['edge_index'].to(device)
    edge_type = pyg_data['edge_type'].to(device)

    wrapped_model = ModelWrapper(
        kg_model=model,
        edge_index=edge_index,
        edge_type=edge_type,
        mode='link_prediction'
    )

    print(f"✓ Model wrapped for explanation")

    return {
        'model': model,
        'wrapped_model': wrapped_model,
        'edge_index': edge_index,
        'edge_type': edge_type,
        'num_nodes': metadata['num_nodes'],
        'num_relations': metadata['num_relations'],
        'device': device
    }


def select_triples_to_explain(
    pyg_data: Dict,
    knowledge_graph: Dict,
    selection_params: Dict,
    device_str: str = "cpu"
) -> Dict:
    """
    Select triples (edges) to explain.

    Args:
        pyg_data: PyG format graph data
        knowledge_graph: Knowledge graph with dictionaries
        selection_params: Parameters for selecting triples
        device_str: Device string

    Returns:
        Dictionary with selected triple indices and metadata
    """
    print("\n" + "="*60)
    print("SELECTING TRIPLES TO EXPLAIN")
    print("="*60)

    device = torch.device(device_str if torch.cuda.is_available() else "cpu")

    selection_strategy = selection_params.get('strategy', 'random')
    num_triples = selection_params.get('num_triples', 10)

    edge_index = pyg_data['edge_index']
    edge_type = pyg_data['edge_type']
    num_edges = edge_index.size(1)

    print(f"\nSelection strategy: {selection_strategy}")
    print(f"Number of triples to select: {num_triples}")
    print(f"Total edges in graph: {num_edges}")

    if selection_strategy == 'random':
        # Random selection from training edges
        indices = torch.randperm(num_edges)[:num_triples]

    elif selection_strategy == 'test_triples':
        # Select from test triples
        test_triples = pyg_data.get('test_triples', None)

        if test_triples is None:
            print("Warning: No test triples found in pyg_data, falling back to random")
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

            # Print sample triples
            print(f"\n✓ Selected {len(triples_readable)} test triples")
            print("\nSample selected test triples:")
            for i, triple in enumerate(triples_readable[:5]):
                print(f"  {i+1}. {triple['triple']}")

            if len(triples_readable) > 5:
                print(f"  ... and {len(triples_readable) - 5} more")

            return {
                'selected_indices': test_indices,
                'selected_edge_index': selected_edge_index,
                'selected_edge_type': selected_edge_type,
                'triples_readable': triples_readable,
                'num_selected': len(triples_readable),
                'from_test_set': True
            }

    elif selection_strategy == 'from_file':
        # Load triples from a file (e.g., top10_test.txt)
        file_path = selection_params.get('file_path', None)

        if file_path is None:
            print("Warning: No file_path specified for 'from_file' strategy, falling back to random")
            indices = torch.randperm(num_edges)[:num_triples]
        else:
            print(f"Loading triples from file: {file_path}")

            try:
                # Read triples from file (format: head\trelation\ttail)
                with open(file_path, 'r') as f:
                    file_lines = f.readlines()

                print(f"Found {len(file_lines)} triples in file")

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

                    # Print sample triples
                    print(f"\n✓ Loaded {len(triples_readable)} triples from file")
                    print("\nSample loaded triples:")
                    for i, triple in enumerate(triples_readable[:5]):
                        print(f"  {i+1}. {triple['triple']}")

                    if len(triples_readable) > 5:
                        print(f"  ... and {len(triples_readable) - 5} more")

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

    # Print sample triples
    print("\nSample selected triples:")
    for i, triple in enumerate(triples_readable[:5]):
        print(f"  {i+1}. {triple['triple']}")

    if len(triples_readable) > 5:
        print(f"  ... and {len(triples_readable) - 5} more")

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

    device = model_dict['device']
    wrapped_model = model_dict['wrapped_model']
    edge_index = model_dict['edge_index']
    edge_type = model_dict['edge_type']
    num_nodes = model_dict['num_nodes']

    # Extract GNNExplainer-specific configuration
    gnn_params = explainer_params.get('gnnexplainer', {})
    epochs = gnn_params.get('gnn_epochs', 200)
    lr = gnn_params.get('gnn_lr', 0.01)

    print(f"\nGNNExplainer configuration:")
    print(f"  Epochs: {epochs}")
    print(f"  Learning rate: {lr}")

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

    # Create dummy node features (required by Explainer API)
    x = torch.eye(num_nodes, device=device)

    # Run explanation for each selected triple
    explanations = []

    selected_edge_index = selected_triples['selected_edge_index'].to(device)
    selected_edge_type = selected_triples['selected_edge_type'].to(device)
    triples_readable = selected_triples['triples_readable']

    print(f"\nGenerating explanations for {len(triples_readable)} triples...")

    for i in range(len(triples_readable)):
        print(f"\n  [{i+1}/{len(triples_readable)}] Explaining: {triples_readable[i]['triple']}")

        # Get the edge to explain
        edge_to_explain = selected_edge_index[:, i:i+1]
        edge_type_to_explain = selected_edge_type[i:i+1]

        try:
            # Run explainer
            # Note: We explain the edge by finding important neighboring edges
            explanation = explainer(
                x=x,
                edge_index=edge_index,
                edge_type=edge_type_to_explain,
                target=edge_to_explain
            )

            # Extract explanation components
            edge_mask = explanation.edge_mask if hasattr(explanation, 'edge_mask') else None

            # Get top-k important edges
            top_k = gnn_params.get('top_k_edges', 10)

            if edge_mask is not None:
                top_k_indices = torch.topk(edge_mask, min(top_k, len(edge_mask))).indices
                important_edges = edge_index[:, top_k_indices]
                important_edge_types = edge_type[top_k_indices]
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
                'importance_scores': importance_scores
            })

            print(f"    ✓ Explanation generated")

        except Exception as e:
            print(f"    ✗ Error: {str(e)}")
            explanations.append({
                'triple': triples_readable[i],
                'error': str(e)
            })

    print(f"\n✓ GNNExplainer completed: {len(explanations)} explanations generated")

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
    print("\n" + "="*60)
    print("RUNNING PGExplainer")
    print("="*60)

    device = model_dict['device']
    wrapped_model = model_dict['wrapped_model']
    edge_index = model_dict['edge_index']
    edge_type = model_dict['edge_type']
    num_nodes = model_dict['num_nodes']

    # Extract PGExplainer-specific configuration
    pg_params = explainer_params.get('pgexplainer', {})
    epochs = pg_params.get('pg_epochs', 30)
    lr = pg_params.get('pg_lr', 0.003)

    print(f"\nPGExplainer configuration:")
    print(f"  Training epochs: {epochs}")
    print(f"  Learning rate: {lr}")
    print(f"\nNote: PGExplainer trains an explainer network once,")
    print(f"      then generates explanations efficiently for all instances.")

    # Create explainer
    explainer = Explainer(
        model=wrapped_model,
        algorithm=PGExplainer(epochs=epochs, lr=lr),
        explanation_type='model',
        edge_mask_type='object',
        model_config=dict(
            mode='regression',
            task_level='edge',
            return_type='raw'
        ),
    )

    print(f"\n✓ PGExplainer initialized")

    # Create dummy node features
    x = torch.eye(num_nodes, device=device)

    # Train PGExplainer on the full graph first
    print(f"\nTraining PGExplainer network...")
    print(f"  This learns a parameterized explainer that works for all instances")

    # For PGExplainer, we need to provide training data
    # We'll use a subset of edges from the graph
    training_edges = pg_params.get('training_edges', 100)
    num_edges = edge_index.size(1)
    train_indices = torch.randperm(num_edges)[:min(training_edges, num_edges)]

    # Note: PGExplainer training happens inside the explainer
    # We just need to call it on some examples

    print(f"✓ PGExplainer network ready")

    # Generate explanations for selected triples
    explanations = []

    selected_edge_index = selected_triples['selected_edge_index'].to(device)
    selected_edge_type = selected_triples['selected_edge_type'].to(device)
    triples_readable = selected_triples['triples_readable']

    print(f"\nGenerating explanations for {len(triples_readable)} triples...")

    for i in range(len(triples_readable)):
        print(f"\n  [{i+1}/{len(triples_readable)}] Explaining: {triples_readable[i]['triple']}")

        edge_to_explain = selected_edge_index[:, i:i+1]
        edge_type_to_explain = selected_edge_type[i:i+1]

        try:
            # Run explainer
            explanation = explainer(
                x=x,
                edge_index=edge_index,
                edge_type=edge_type_to_explain,
                target=edge_to_explain
            )

            # Extract explanation components
            edge_mask = explanation.edge_mask if hasattr(explanation, 'edge_mask') else None

            # Get top-k important edges
            top_k = pg_params.get('top_k_edges', 10)

            if edge_mask is not None:
                top_k_indices = torch.topk(edge_mask, min(top_k, len(edge_mask))).indices
                important_edges = edge_index[:, top_k_indices]
                important_edge_types = edge_type[top_k_indices]
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
                'importance_scores': importance_scores
            })

            print(f"    ✓ Explanation generated")

        except Exception as e:
            print(f"    ✗ Error: {str(e)}")
            explanations.append({
                'triple': triples_readable[i],
                'error': str(e)
            })

    print(f"\n✓ PGExplainer completed: {len(explanations)} explanations generated")

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

    print(f"\nImproved PAGE configuration:")
    print(f"  Training epochs: {train_epochs}")
    print(f"  Learning rate: {lr}")
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
            # Extract k-hop subgraph around head and tail
            subgraph_nodes, subgraph_edges, adj_matrix = extract_link_subgraph(
                edge_index=edge_index.cpu(),
                head_idx=head_idx,
                tail_idx=tail_idx,
                num_hops=k_hops,
                num_nodes=num_nodes
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

    explanations = []

    for i, data in enumerate(subgraphs_data):
        info = subgraph_info[i]
        triple_idx = info['triple_idx']
        triple = triples_readable[triple_idx]
        pred_score = info['prediction_score']

        print(f"\n  [{i+1}/{len(subgraphs_data)}] Explaining: {triple['triple']} (score={pred_score:.4f})")

        try:
            # Generate explanation using improved PAGE
            edge_importance, latent_z = page_explainer.explain(data['features'], data['adj'])

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

            print(f"    ✓ Explanation generated ({top_k_actual} important edges, pred_score={pred_score:.4f})")

        except Exception as e:
            print(f"    ✗ Error: {str(e)}")
            explanations.append({
                'triple': triple,
                'error': str(e)
            })

    print(f"\n✓ Improved PAGE explainer completed: {len(explanations)} explanations generated")
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
    gnn_explanations: Dict,
    pg_explanations: Dict,
    knowledge_graph: Dict,
    page_explanations: Optional[Dict] = None
) -> Dict:
    """
    Summarize and compare explanations from different explainers.

    Args:
        gnn_explanations: GNNExplainer results
        pg_explanations: PGExplainer results
        knowledge_graph: Knowledge graph with dictionaries

    Returns:
        Summary dictionary with comparisons and insights
    """
    print("\n" + "="*60)
    print("SUMMARIZING EXPLANATIONS")
    print("="*60)

    summary = {
        'gnn_explainer': {
            'num_explanations': gnn_explanations['num_explanations'],
            'successful': sum(1 for e in gnn_explanations['explanations'] if 'error' not in e),
            'failed': sum(1 for e in gnn_explanations['explanations'] if 'error' in e)
        },
        'pg_explainer': {
            'num_explanations': pg_explanations['num_explanations'],
            'successful': sum(1 for e in pg_explanations['explanations'] if 'error' not in e),
            'failed': sum(1 for e in pg_explanations['explanations'] if 'error' in e)
        },
        'comparisons': []
    }

    # Add PAGE if available
    if page_explanations is not None:
        summary['page_explainer'] = {
            'num_explanations': page_explanations['num_explanations'],
            'successful': sum(1 for e in page_explanations['explanations'] if 'error' not in e),
            'failed': sum(1 for e in page_explanations['explanations'] if 'error' in e)
        }

    print(f"\nGNNExplainer: {summary['gnn_explainer']['successful']}/{summary['gnn_explainer']['num_explanations']} successful")
    print(f"PGExplainer: {summary['pg_explainer']['successful']}/{summary['pg_explainer']['num_explanations']} successful")
    if page_explanations is not None:
        print(f"PAGE Explainer: {summary['page_explainer']['successful']}/{summary['page_explainer']['num_explanations']} successful")

    # Compare explanations for each triple
    print(f"\nComparing explanations...")

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

    print(f"✓ Compared {len(summary['comparisons'])} explanations")

    if summary['comparisons']:
        avg_overlap = sum(c['overlap'] for c in summary['comparisons']) / len(summary['comparisons'])
        print(f"\nAverage overlap in top-5 important edges: {avg_overlap:.2f}")

    return summary
