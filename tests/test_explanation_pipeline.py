"""Tests for explanation pipeline components."""

import torch
import pytest
from gnn_explainer.pipelines.explanation.nodes import (
    ModelWrapper,
    select_triples_to_explain
)
from gnn_explainer.pipelines.training.kg_models import CompGCNKGModel


def create_simple_kg_model():
    """Create a simple CompGCN model for testing."""
    num_nodes = 10
    num_relations = 3
    embedding_dim = 16

    model = CompGCNKGModel(
        num_nodes=num_nodes,
        num_relations=num_relations,
        embedding_dim=embedding_dim,
        decoder_type='distmult',
        num_layers=2,
        comp_fn='sub',
        dropout=0.1
    )

    return model, num_nodes, num_relations


def create_simple_graph():
    """Create a simple test graph."""
    # 10 nodes, 3 relations, 20 edges
    edge_index = torch.tensor([
        [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 0, 2, 4, 6, 8, 1, 3, 5, 7, 9],
        [1, 2, 3, 4, 5, 6, 7, 8, 9, 0, 2, 4, 6, 8, 0, 3, 5, 7, 9, 1]
    ])

    edge_type = torch.tensor([0, 1, 2, 0, 1, 2, 0, 1, 2, 0, 1, 2, 0, 1, 2, 0, 1, 2, 0, 1])

    pyg_data = {
        'edge_index': edge_index,
        'edge_type': edge_type
    }

    knowledge_graph = {
        'num_nodes': 10,
        'num_relations': 3,
        'idx_to_entity': {i: f'node_{i}' for i in range(10)},
        'idx_to_relation': {i: f'rel_{i}' for i in range(3)}
    }

    return pyg_data, knowledge_graph


class TestModelWrapper:
    """Test ModelWrapper for PyG Explainer compatibility."""

    def test_model_wrapper_initialization(self):
        """Test that ModelWrapper initializes correctly."""
        model, num_nodes, num_relations = create_simple_kg_model()
        pyg_data, _ = create_simple_graph()

        wrapper = ModelWrapper(
            kg_model=model,
            edge_index=pyg_data['edge_index'],
            edge_type=pyg_data['edge_type'],
            mode='link_prediction'
        )

        assert wrapper.kg_model == model
        assert wrapper.mode == 'link_prediction'
        assert torch.equal(wrapper.edge_index, pyg_data['edge_index'])
        assert torch.equal(wrapper.edge_type, pyg_data['edge_type'])

    def test_model_wrapper_forward(self):
        """Test that ModelWrapper forward pass works."""
        model, num_nodes, num_relations = create_simple_kg_model()
        pyg_data, _ = create_simple_graph()

        wrapper = ModelWrapper(
            kg_model=model,
            edge_index=pyg_data['edge_index'],
            edge_type=pyg_data['edge_type'],
            mode='link_prediction'
        )

        # Create dummy node features
        x = torch.eye(num_nodes)

        # Test forward pass with a subset of edges
        test_edges = pyg_data['edge_index'][:, :5]
        test_edge_types = pyg_data['edge_type'][:5]

        with torch.no_grad():
            scores = wrapper(x, test_edges, edge_type=test_edge_types)

        assert scores.shape == (5,)
        assert not torch.isnan(scores).any()
        assert not torch.isinf(scores).any()


class TestTripleSelection:
    """Test triple selection strategies."""

    def test_random_selection(self):
        """Test random triple selection."""
        pyg_data, knowledge_graph = create_simple_graph()

        selection_params = {
            'strategy': 'random',
            'num_triples': 5
        }

        result = select_triples_to_explain(
            pyg_data=pyg_data,
            knowledge_graph=knowledge_graph,
            selection_params=selection_params,
            device_str='cpu'
        )

        assert result['num_selected'] == 5
        assert len(result['triples_readable']) == 5
        assert result['selected_edge_index'].shape == (2, 5)
        assert result['selected_edge_type'].shape == (5,)

    def test_specific_relations_selection(self):
        """Test selection by specific relations."""
        pyg_data, knowledge_graph = create_simple_graph()

        selection_params = {
            'strategy': 'specific_relations',
            'num_triples': 10,
            'target_relations': [0, 1]
        }

        result = select_triples_to_explain(
            pyg_data=pyg_data,
            knowledge_graph=knowledge_graph,
            selection_params=selection_params,
            device_str='cpu'
        )

        # Check that all selected triples have relation 0 or 1
        selected_relations = result['selected_edge_type']
        assert all(r in [0, 1] for r in selected_relations.tolist())

    def test_specific_nodes_selection(self):
        """Test selection by specific nodes."""
        pyg_data, knowledge_graph = create_simple_graph()

        selection_params = {
            'strategy': 'specific_nodes',
            'num_triples': 10,
            'target_nodes': [0, 1, 2]
        }

        result = select_triples_to_explain(
            pyg_data=pyg_data,
            knowledge_graph=knowledge_graph,
            selection_params=selection_params,
            device_str='cpu'
        )

        # Check that all selected triples involve nodes 0, 1, or 2
        selected_edges = result['selected_edge_index']
        heads = selected_edges[0]
        tails = selected_edges[1]

        for i in range(len(heads)):
            head = heads[i].item()
            tail = tails[i].item()
            assert head in [0, 1, 2] or tail in [0, 1, 2]

    def test_readable_triples_format(self):
        """Test that readable triples are formatted correctly."""
        pyg_data, knowledge_graph = create_simple_graph()

        selection_params = {
            'strategy': 'random',
            'num_triples': 3
        }

        result = select_triples_to_explain(
            pyg_data=pyg_data,
            knowledge_graph=knowledge_graph,
            selection_params=selection_params,
            device_str='cpu'
        )

        # Check format of readable triples
        for triple in result['triples_readable']:
            assert 'head_idx' in triple
            assert 'tail_idx' in triple
            assert 'relation_idx' in triple
            assert 'head_name' in triple
            assert 'tail_name' in triple
            assert 'relation_name' in triple
            assert 'triple' in triple

            # Check that triple string is formatted correctly
            assert triple['triple'].startswith('(')
            assert triple['triple'].endswith(')')
            assert ',' in triple['triple']


class TestExplanationPipelineIntegration:
    """Integration tests for explanation pipeline."""

    def test_full_pipeline_components(self):
        """Test that all pipeline components can work together."""
        # Create model and graph
        model, num_nodes, num_relations = create_simple_kg_model()
        pyg_data, knowledge_graph = create_simple_graph()

        # Step 1: Wrap model
        wrapper = ModelWrapper(
            kg_model=model,
            edge_index=pyg_data['edge_index'],
            edge_type=pyg_data['edge_type'],
            mode='link_prediction'
        )

        # Step 2: Select triples
        selection_params = {
            'strategy': 'random',
            'num_triples': 3
        }

        selected = select_triples_to_explain(
            pyg_data=pyg_data,
            knowledge_graph=knowledge_graph,
            selection_params=selection_params,
            device_str='cpu'
        )

        # Step 3: Verify model can score selected triples
        x = torch.eye(num_nodes)
        test_edges = selected['selected_edge_index']
        test_edge_types = selected['selected_edge_type']

        with torch.no_grad():
            scores = wrapper(x, test_edges, edge_type=test_edge_types)

        assert scores.shape == (3,)
        assert not torch.isnan(scores).any()

        print("\n✓ All pipeline components work together")


if __name__ == "__main__":
    # Run tests
    print("Testing ModelWrapper...")
    test_wrapper = TestModelWrapper()
    test_wrapper.test_model_wrapper_initialization()
    test_wrapper.test_model_wrapper_forward()
    print("✓ ModelWrapper tests passed")

    print("\nTesting Triple Selection...")
    test_selection = TestTripleSelection()
    test_selection.test_random_selection()
    test_selection.test_specific_relations_selection()
    test_selection.test_specific_nodes_selection()
    test_selection.test_readable_triples_format()
    print("✓ Triple Selection tests passed")

    print("\nTesting Integration...")
    test_integration = TestExplanationPipelineIntegration()
    test_integration.test_full_pipeline_components()
    print("✓ Integration tests passed")

    print("\n" + "="*60)
    print("ALL TESTS PASSED! ✓")
    print("="*60)
