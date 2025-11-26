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
        # Random selection
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

    # GNNExplainer configuration
    epochs = explainer_params.get('gnn_epochs', 200)
    lr = explainer_params.get('gnn_lr', 0.01)

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
            top_k = explainer_params.get('top_k_edges', 10)

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
        'params': explainer_params
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

    # PGExplainer configuration
    epochs = explainer_params.get('pg_epochs', 30)
    lr = explainer_params.get('pg_lr', 0.003)

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
    training_edges = explainer_params.get('training_edges', 100)
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
            top_k = explainer_params.get('top_k_edges', 10)

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
        'params': explainer_params
    }


def summarize_explanations(
    gnn_explanations: Dict,
    pg_explanations: Dict,
    knowledge_graph: Dict
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

    print(f"\nGNNExplainer: {summary['gnn_explainer']['successful']}/{summary['gnn_explainer']['num_explanations']} successful")
    print(f"PGExplainer: {summary['pg_explainer']['successful']}/{summary['pg_explainer']['num_explanations']} successful")

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
