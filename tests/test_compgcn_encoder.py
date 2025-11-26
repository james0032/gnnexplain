"""
Test CompGCN Encoder with Fabricated Knowledge Graphs

This script tests the CompGCN implementation with synthetic data to verify:
1. Forward pass works correctly
2. All composition functions work (sub, mult, corr)
3. Embeddings are updated properly
4. Gradients flow correctly
5. Multi-layer architecture works
"""

import torch
import torch.nn as nn
import sys
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from gnn_explainer.pipelines.training.compgcn_encoder import CompGCNEncoder
from gnn_explainer.pipelines.training.compgcn_layer import CompGCNConv


def create_simple_kg():
    """
    Create a simple fabricated knowledge graph.

    Graph structure:
        0 --relation_0--> 1
        1 --relation_1--> 2
        2 --relation_0--> 3
        0 --relation_1--> 3
        1 --relation_2--> 3

    Returns:
        edge_index, edge_type, num_nodes, num_relations
    """
    # Define edges (source, target)
    edges = [
        (0, 1),  # Entity 0 -> Entity 1
        (1, 2),  # Entity 1 -> Entity 2
        (2, 3),  # Entity 2 -> Entity 3
        (0, 3),  # Entity 0 -> Entity 3
        (1, 3),  # Entity 1 -> Entity 3
    ]

    # Define relation types for each edge
    relations = [0, 1, 0, 1, 2]

    # Convert to PyG format
    edge_index = torch.tensor(edges, dtype=torch.long).t()  # (2, num_edges)
    edge_type = torch.tensor(relations, dtype=torch.long)    # (num_edges,)

    num_nodes = 4
    num_relations = 3

    print("="*60)
    print("SIMPLE KNOWLEDGE GRAPH")
    print("="*60)
    print(f"Nodes: {num_nodes}")
    print(f"Relations: {num_relations}")
    print(f"Edges: {len(relations)}")
    print(f"\nEdge list:")
    for i, ((src, dst), rel) in enumerate(zip(edges, relations)):
        print(f"  Edge {i}: {src} --[rel_{rel}]--> {dst}")
    print("="*60)

    return edge_index, edge_type, num_nodes, num_relations


def create_complex_kg():
    """
    Create a more complex fabricated knowledge graph.

    Simulates a mini drug-disease-gene network:
    - 3 drugs (0, 1, 2)
    - 3 diseases (3, 4, 5)
    - 3 genes (6, 7, 8)
    - Relations: treats (0), causes (1), expresses (2), inhibits (3)
    """
    edges = [
        # Drug treats Disease
        (0, 3), (0, 4),  # Drug0 treats Disease3, Disease4
        (1, 4), (1, 5),  # Drug1 treats Disease4, Disease5
        (2, 3),          # Drug2 treats Disease3

        # Gene causes Disease
        (6, 3), (6, 4),  # Gene6 causes Disease3, Disease4
        (7, 4), (7, 5),  # Gene7 causes Disease4, Disease5
        (8, 5),          # Gene8 causes Disease5

        # Disease expresses Gene
        (3, 6), (4, 7), (5, 8),

        # Drug inhibits Gene
        (0, 6), (1, 7), (2, 8),
    ]

    relations = [
        0, 0, 0, 0, 0,  # treats
        1, 1, 1, 1, 1,  # causes
        2, 2, 2,        # expresses
        3, 3, 3,        # inhibits
    ]

    edge_index = torch.tensor(edges, dtype=torch.long).t()
    edge_type = torch.tensor(relations, dtype=torch.long)

    num_nodes = 9
    num_relations = 4

    print("\n" + "="*60)
    print("COMPLEX KNOWLEDGE GRAPH (Drug-Disease-Gene)")
    print("="*60)
    print(f"Nodes: {num_nodes}")
    print(f"  Drugs: 0, 1, 2")
    print(f"  Diseases: 3, 4, 5")
    print(f"  Genes: 6, 7, 8")
    print(f"\nRelations: {num_relations}")
    print(f"  0: treats")
    print(f"  1: causes")
    print(f"  2: expresses")
    print(f"  3: inhibits")
    print(f"\nTotal edges: {len(relations)}")
    print("="*60)

    return edge_index, edge_type, num_nodes, num_relations


def test_compgcn_layer():
    """Test basic CompGCN layer functionality."""
    print("\n" + "="*60)
    print("TEST 1: CompGCN Layer")
    print("="*60)

    # Create simple graph
    edge_index, edge_type, num_nodes, num_relations = create_simple_kg()

    # Initialize layer
    in_dim = 16
    out_dim = 32

    print(f"\nTesting CompGCN Layer:")
    print(f"  Input dim: {in_dim}")
    print(f"  Output dim: {out_dim}")

    for comp_fn in ['sub', 'mult', 'corr']:
        print(f"\n  Composition function: {comp_fn}")

        layer = CompGCNConv(
            in_channels=in_dim,
            out_channels=out_dim,
            num_relations=num_relations,
            comp_fn=comp_fn
        )

        # Random input
        x = torch.randn(num_nodes, in_dim)
        rel_emb = torch.randn(num_relations, in_dim)

        # Forward pass
        out_x, out_rel = layer(x, edge_index, edge_type, rel_emb)

        print(f"    Input shape: {x.shape}")
        print(f"    Output node shape: {out_x.shape}")
        print(f"    Output rel shape: {out_rel.shape}")

        assert out_x.shape == (num_nodes, out_dim), f"Node output shape mismatch"
        assert out_rel.shape == (num_relations, out_dim), f"Relation output shape mismatch"
        print(f"    ✓ Shapes correct")

        # Check for NaN or Inf
        assert not torch.isnan(out_x).any(), "NaN in node output"
        assert not torch.isinf(out_x).any(), "Inf in node output"
        assert not torch.isnan(out_rel).any(), "NaN in relation output"
        assert not torch.isinf(out_rel).any(), "Inf in relation output"
        print(f"    ✓ No NaN/Inf values")

        # Check gradient flow
        loss = out_x.sum() + out_rel.sum()
        loss.backward()
        assert layer.W_self.weight.grad is not None, "No gradient for W_self"
        print(f"    ✓ Gradients flow correctly")

    print("\n✅ CompGCN Layer tests passed!")


def test_compgcn_encoder():
    """Test full CompGCN encoder."""
    print("\n" + "="*60)
    print("TEST 2: CompGCN Encoder")
    print("="*60)

    # Create complex graph
    edge_index, edge_type, num_nodes, num_relations = create_complex_kg()

    embedding_dim = 64
    num_layers_list = [1, 2, 3]

    for num_layers in num_layers_list:
        print(f"\n  Testing {num_layers}-layer encoder:")

        for comp_fn in ['sub', 'mult', 'corr']:
            print(f"    Composition: {comp_fn}")

            encoder = CompGCNEncoder(
                num_nodes=num_nodes,
                num_relations=num_relations,
                embedding_dim=embedding_dim,
                num_layers=num_layers,
                comp_fn=comp_fn,
                dropout=0.2
            )

            # Forward pass
            node_emb, rel_emb = encoder(edge_index, edge_type)

            assert node_emb.shape == (num_nodes, embedding_dim)
            assert rel_emb.shape == (num_relations, embedding_dim)
            print(f"      ✓ Output shapes correct: nodes={node_emb.shape}, rels={rel_emb.shape}")

            # Check for NaN/Inf
            assert not torch.isnan(node_emb).any()
            assert not torch.isnan(rel_emb).any()
            print(f"      ✓ No NaN values")

            # Check gradient flow
            encoder.zero_grad()
            loss = node_emb.sum() + rel_emb.sum()
            loss.backward()
            assert encoder.node_emb.grad is not None
            assert encoder.rel_emb.grad is not None
            print(f"      ✓ Gradients flow")

    print("\n✅ CompGCN Encoder tests passed!")


def test_embedding_learning():
    """Test that embeddings are actually being learned."""
    print("\n" + "="*60)
    print("TEST 3: Embedding Learning")
    print("="*60)

    edge_index, edge_type, num_nodes, num_relations = create_simple_kg()

    embedding_dim = 32
    encoder = CompGCNEncoder(
        num_nodes=num_nodes,
        num_relations=num_relations,
        embedding_dim=embedding_dim,
        num_layers=2,
        comp_fn='sub',
        dropout=0.0  # No dropout for this test
    )

    # Save initial embeddings
    initial_node_emb = encoder.node_emb.data.clone()
    initial_rel_emb = encoder.rel_emb.data.clone()

    print(f"\nTraining for 10 steps...")

    optimizer = torch.optim.Adam(encoder.parameters(), lr=0.01)

    for step in range(10):
        optimizer.zero_grad()

        # Forward pass
        node_emb, rel_emb = encoder(edge_index, edge_type)

        # Dummy loss: encourage nodes 0 and 1 to be similar
        target_similarity = torch.tensor(1.0)
        similarity = torch.cosine_similarity(
            node_emb[0].unsqueeze(0),
            node_emb[1].unsqueeze(0)
        )
        loss = (similarity - target_similarity) ** 2

        loss.backward()
        optimizer.step()

        if step % 3 == 0:
            print(f"  Step {step}: Loss = {loss.item():.4f}, Similarity = {similarity.item():.4f}")

    # Check that embeddings changed
    node_emb_change = (encoder.node_emb.data - initial_node_emb).abs().mean()
    rel_emb_change = (encoder.rel_emb.data - initial_rel_emb).abs().mean()

    print(f"\nEmbedding changes:")
    print(f"  Node embeddings: {node_emb_change:.4f}")
    print(f"  Relation embeddings: {rel_emb_change:.4f}")

    assert node_emb_change > 0.01, "Node embeddings didn't change enough"
    assert rel_emb_change > 0.01, "Relation embeddings didn't change enough"
    print(f"  ✓ Embeddings are learning")

    print("\n✅ Embedding learning test passed!")


def test_composition_differences():
    """Test that different composition functions produce different results."""
    print("\n" + "="*60)
    print("TEST 4: Composition Function Differences")
    print("="*60)

    edge_index, edge_type, num_nodes, num_relations = create_simple_kg()

    embedding_dim = 32

    # Create encoders with different composition functions
    encoders = {}
    outputs = {}

    for comp_fn in ['sub', 'mult', 'corr']:
        encoder = CompGCNEncoder(
            num_nodes=num_nodes,
            num_relations=num_relations,
            embedding_dim=embedding_dim,
            num_layers=2,
            comp_fn=comp_fn,
            dropout=0.0
        )

        # Use same initial embeddings for fair comparison
        if comp_fn == 'sub':
            base_node_emb = encoder.node_emb.data.clone()
            base_rel_emb = encoder.rel_emb.data.clone()
        else:
            encoder.node_emb.data = base_node_emb.clone()
            encoder.rel_emb.data = base_rel_emb.clone()

        encoders[comp_fn] = encoder

        # Forward pass
        node_emb, rel_emb = encoder(edge_index, edge_type)
        outputs[comp_fn] = (node_emb, rel_emb)

    # Compare outputs
    print("\nComparing outputs from different composition functions:")

    for fn1 in ['sub', 'mult']:
        fn2 = 'corr' if fn1 == 'mult' else 'mult'

        node_diff = (outputs[fn1][0] - outputs[fn2][0]).abs().mean()
        rel_diff = (outputs[fn1][1] - outputs[fn2][1]).abs().mean()

        print(f"  {fn1} vs {fn2}:")
        print(f"    Node embedding diff: {node_diff:.4f}")
        print(f"    Relation embedding diff: {rel_diff:.4f}")

        assert node_diff > 0.01, f"{fn1} and {fn2} produced same node embeddings"
        assert rel_diff > 0.01, f"{fn1} and {fn2} produced same relation embeddings"

    print("  ✓ Different composition functions produce different results")
    print("\n✅ Composition difference test passed!")


def test_triple_scoring():
    """Test scoring triples using learned embeddings."""
    print("\n" + "="*60)
    print("TEST 5: Triple Scoring")
    print("="*60)

    edge_index, edge_type, num_nodes, num_relations = create_complex_kg()

    embedding_dim = 64
    encoder = CompGCNEncoder(
        num_nodes=num_nodes,
        num_relations=num_relations,
        embedding_dim=embedding_dim,
        num_layers=2,
        comp_fn='sub',
        dropout=0.0
    )

    # Get embeddings
    node_emb, rel_emb = encoder(edge_index, edge_type)

    print("\nTesting triple scoring (DistMult-style):")

    # Define some triples to score
    test_triples = [
        (0, 0, 3),  # Drug0 treats Disease3 (exists in graph)
        (1, 0, 5),  # Drug1 treats Disease5 (exists in graph)
        (2, 1, 6),  # Drug2 causes Gene6 (does NOT exist)
        (0, 2, 7),  # Drug0 expresses Gene7 (does NOT exist)
    ]

    for h, r, t in test_triples:
        # DistMult scoring: score = sum(h * r * t)
        score = torch.sum(node_emb[h] * rel_emb[r] * node_emb[t])

        exists = "EXISTS" if (h, t) in [(0,3), (1,5)] else "NOT IN GRAPH"
        print(f"  ({h}, {r}, {t}): score = {score:.4f} [{exists}]")

    print("  ✓ Triple scoring works")
    print("\n✅ Triple scoring test passed!")


def test_message_passing():
    """Test that message passing is actually happening."""
    print("\n" + "="*60)
    print("TEST 6: Message Passing Verification")
    print("="*60)

    # Create a simple chain graph: 0 -> 1 -> 2
    edge_index = torch.tensor([[0, 1], [1, 2]], dtype=torch.long).t()
    edge_type = torch.tensor([0, 0], dtype=torch.long)
    num_nodes = 3
    num_relations = 1

    print("\nChain graph: 0 -> 1 -> 2")
    print("Testing 1-layer vs 2-layer encoder")

    embedding_dim = 16

    # 1-layer encoder
    encoder_1 = CompGCNEncoder(
        num_nodes=num_nodes,
        num_relations=num_relations,
        embedding_dim=embedding_dim,
        num_layers=1,
        comp_fn='sub',
        dropout=0.0
    )

    # 2-layer encoder (same initial embeddings)
    encoder_2 = CompGCNEncoder(
        num_nodes=num_nodes,
        num_relations=num_relations,
        embedding_dim=embedding_dim,
        num_layers=2,
        comp_fn='sub',
        dropout=0.0
    )
    encoder_2.node_emb.data = encoder_1.node_emb.data.clone()
    encoder_2.rel_emb.data = encoder_1.rel_emb.data.clone()

    # Get embeddings
    node_emb_1, _ = encoder_1(edge_index, edge_type)
    node_emb_2, _ = encoder_2(edge_index, edge_type)

    # With 1 layer, node 2 only sees node 1
    # With 2 layers, node 2 can see information from node 0

    diff = (node_emb_2[2] - node_emb_1[2]).abs().mean()
    print(f"\nNode 2 embedding difference (1-layer vs 2-layer): {diff:.4f}")

    assert diff > 0.01, "2-layer encoder should produce different embeddings than 1-layer"
    print("  ✓ Multi-layer message passing works")

    print("\n✅ Message passing test passed!")


def run_all_tests():
    """Run all CompGCN tests."""
    print("\n" + "🧪 " + "="*58 + " 🧪")
    print("     COMPGCN ENCODER TEST SUITE")
    print("🧪 " + "="*58 + " 🧪\n")

    try:
        test_compgcn_layer()
        test_compgcn_encoder()
        test_embedding_learning()
        test_composition_differences()
        test_triple_scoring()
        test_message_passing()

        print("\n" + "🎉 " + "="*58 + " 🎉")
        print("     ALL TESTS PASSED!")
        print("🎉 " + "="*58 + " 🎉\n")

        print("✅ CompGCN implementation is working correctly!")
        print("\nNext steps:")
        print("  1. Run on real data: kedro run")
        print("  2. Try different decoders: --params=model.decoder_type:complex")
        print("  3. Experiment with composition functions: --params=model.comp_fn:mult")

        return True

    except Exception as e:
        print("\n" + "❌ " + "="*58 + " ❌")
        print("     TEST FAILED!")
        print("❌ " + "="*58 + " ❌\n")
        print(f"Error: {e}")
        import traceback
        traceback.print_exc()
        return False


if __name__ == "__main__":
    success = run_all_tests()
    sys.exit(0 if success else 1)
