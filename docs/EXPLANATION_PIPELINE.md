# GNN Explainer - Explanation Pipeline Guide

**Date**: 2025-11-25
**Status**: ✅ **COMPLETE** - GNNExplainer and PGExplainer Integrated

---

## 🎉 **What We Built**

A complete explanation pipeline that integrates PyTorch Geometric's explainability framework with the trained CompGCN models. The pipeline supports **two state-of-the-art explanation methods**:

1. **GNNExplainer** - Instance-level explanations via gradient-based optimization
2. **PGExplainer** - Parameterized explainer with efficient inference

---

## 📁 **Files Created**

### **Core Implementation**

1. **[nodes.py](../src/gnn_explainer/pipelines/explanation/nodes.py)** (~800 lines)
   - `ModelWrapper`: Wraps CompGCN/RGCN for PyG Explainer API
   - `prepare_model_for_explanation`: Loads trained model
   - `select_triples_to_explain`: Selects edges to explain
   - `run_gnnexplainer`: Runs GNNExplainer on selected triples
   - `run_pgexplainer`: Runs PGExplainer on selected triples
   - `summarize_explanations`: Compares and summarizes results

2. **[pipeline.py](../src/gnn_explainer/pipelines/explanation/pipeline.py)**
   - 5-node Kedro pipeline
   - Model preparation → Triple selection → GNN/PG Explainer → Summary

3. **[__init__.py](../src/gnn_explainer/pipelines/explanation/__init__.py)**
   - Lazy imports for standalone use

### **Configuration**

4. **[catalog.yml](../conf/base/catalog.yml)** (Updated)
   - `prepared_model`: Wrapped model for explanation
   - `selected_triples`: Triples to explain
   - `gnn_explanations`: GNNExplainer results
   - `pg_explanations`: PGExplainer results
   - `explanation_summary`: Comparison summary

5. **[parameters.yml](../conf/base/parameters.yml)** (Updated)
   - Triple selection strategies
   - GNNExplainer configuration
   - PGExplainer configuration

---

## 🏗️ **Architecture Overview**

```
┌─────────────────────────────────────────────┐
│          Trained CompGCN Model              │
│     (from training pipeline output)         │
└─────────────────────────────────────────────┘
                      ↓
┌─────────────────────────────────────────────┐
│       ModelWrapper (PyG Compatible)         │
│  - Wraps KG model for Explainer API         │
│  - Handles forward pass for link prediction │
└─────────────────────────────────────────────┘
                      ↓
┌─────────────────────────────────────────────┐
│          Select Triples to Explain          │
│  Strategies:                                │
│  - Random sampling                          │
│  - Specific relations (e.g., "treats")      │
│  - Specific nodes (e.g., drug X)            │
└─────────────────────────────────────────────┘
                      ↓
        ┌─────────────┴─────────────┐
        ↓                           ↓
┌──────────────────┐      ┌──────────────────┐
│  GNNExplainer    │      │  PGExplainer     │
│  (Instance-level)│      │  (Parameterized) │
│                  │      │                  │
│  - Per-instance  │      │  - Train once    │
│    optimization  │      │  - Fast inference│
│  - Edge masks    │      │  - Edge masks    │
│  - Slow but      │      │  - Fast but      │
│    accurate      │      │    approximate   │
└──────────────────┘      └──────────────────┘
        ↓                           ↓
        └─────────────┬─────────────┘
                      ↓
┌─────────────────────────────────────────────┐
│        Explanation Summary                  │
│  - Compare GNN vs PG explanations           │
│  - Extract top-k important edges            │
│  - Calculate overlap                        │
│  - Generate insights                        │
└─────────────────────────────────────────────┘
```

---

## 🚀 **Quick Start**

### **Prerequisites**

Train a CompGCN model first:

```bash
# Train CompGCN-ComplEx
kedro run --pipeline=training
```

This saves the trained model to `data/06_models/trained_model.pkl`.

### **Run Explanation Pipeline**

```bash
# Run full explanation pipeline
kedro run --pipeline=explanation

# Or run entire workflow (training + explanation)
kedro run
```

---

## 📊 **Explanation Methods**

### **1. GNNExplainer** 🔍

**Paper**: [GNNExplainer: Generating Explanations for Graph Neural Networks](https://arxiv.org/abs/1903.03894) (NeurIPS 2019)

**How it works**:
- Instance-level explanations
- Optimizes edge mask for each triple
- Gradient-based optimization (200 epochs default)
- Identifies which neighboring edges are most important

**Pros**:
- ✅ Accurate, instance-specific explanations
- ✅ Well-established method
- ✅ Works with any GNN

**Cons**:
- ❌ Slow (optimizes for each instance)
- ❌ Requires gradients

**Configuration**:
```yaml
explanation:
  gnnexplainer:
    gnn_epochs: 200        # Optimization epochs
    gnn_lr: 0.01           # Learning rate
    top_k_edges: 10        # Top important edges
```

**Example output**:
```
Explaining: (Aspirin, treats, Headache)
  Important edges:
    1. (Aspirin, inhibits, COX-2) - score: 0.89
    2. (COX-2, causes, Inflammation) - score: 0.76
    3. (Inflammation, leads_to, Headache) - score: 0.72
```

---

### **2. PGExplainer** ⚡

**Paper**: [Parameterized Explainer for Graph Neural Network](https://arxiv.org/abs/2011.04573) (NeurIPS 2020)

**How it works**:
- Trains a parameterized explainer network once
- Uses explainer to generate masks efficiently
- Fast inference after training (no per-instance optimization)
- Learns general explanation patterns

**Pros**:
- ✅ Fast inference (after training)
- ✅ Learns general patterns
- ✅ Amortized explanations

**Cons**:
- ❌ Requires training phase
- ❌ Less instance-specific than GNNExplainer
- ❌ May miss unique patterns

**Configuration**:
```yaml
explanation:
  pgexplainer:
    pg_epochs: 30          # Training epochs
    pg_lr: 0.003           # Learning rate
    training_edges: 100    # Edges for training
    top_k_edges: 10        # Top important edges
```

---

## 🎛️ **Configuration Guide**

### **Triple Selection Strategies**

#### **1. Random Sampling** (Default)

```yaml
explanation:
  triple_selection:
    strategy: "random"
    num_triples: 10
```

Randomly selects 10 triples from the knowledge graph.

#### **2. Specific Relations**

```yaml
explanation:
  triple_selection:
    strategy: "specific_relations"
    num_triples: 20
    target_relations: [0, 1, 5]  # Relation indices
```

Only explains triples with specific relation types (e.g., "treats", "causes").

#### **3. Specific Nodes**

```yaml
explanation:
  triple_selection:
    strategy: "specific_nodes"
    num_triples: 15
    target_nodes: [100, 200, 300]  # Node indices
```

Explains triples involving specific entities (e.g., Aspirin, Diabetes).

---

## 📈 **Usage Examples**

### **Example 1: Explain Random Drug-Disease Triples**

```bash
kedro run --pipeline=explanation \
  --params=explanation.triple_selection.strategy:random,\
explanation.triple_selection.num_triples:20
```

### **Example 2: Explain Specific "treats" Relations**

First, find the relation index for "treats":
```python
# In Python
import pickle
kg = pickle.load(open('data/02_intermediate/knowledge_graph.pkl', 'rb'))
treats_idx = kg['relation_to_idx']['treats']
print(f"treats relation index: {treats_idx}")
```

Then run:
```bash
kedro run --pipeline=explanation \
  --params=explanation.triple_selection.strategy:specific_relations,\
explanation.triple_selection.target_relations:[treats_idx]
```

### **Example 3: Compare GNN vs PG Explainer**

```bash
# Run both explainers and check summary
kedro run --pipeline=explanation

# Check the summary
python -c "
import pickle
summary = pickle.load(open('data/05_model_explanations/explanation_summary.pkl', 'rb'))
print(f'Average overlap: {summary[\"avg_overlap\"]:.2f} edges')
"
```

### **Example 4: Tune GNNExplainer for Better Quality**

```yaml
# parameters.yml
explanation:
  gnnexplainer:
    gnn_epochs: 500        # More epochs
    gnn_lr: 0.005          # Lower learning rate
    top_k_edges: 20        # More important edges
```

---

## 🔬 **Understanding Explanations**

### **What is an Edge Mask?**

An **edge mask** assigns importance scores to edges in the graph:

```
Edge: (Drug_A, treats, Disease_B)
Mask value: 0.85

Interpretation: This edge is 85% important for the prediction
```

### **Top-K Important Edges**

For a triple `(Aspirin, treats, Headache)`, the explainer finds:

```
Top-5 Important Edges (GNNExplainer):
  1. (Aspirin, inhibits, COX-2) - 0.89
  2. (COX-2, regulates, Prostaglandin) - 0.78
  3. (Prostaglandin, causes, Pain) - 0.75
  4. (Pain, symptom_of, Headache) - 0.72
  5. (Aspirin, metabolized_by, Liver) - 0.45
```

**Interpretation**: The model predicts `(Aspirin, treats, Headache)` primarily because:
- Aspirin inhibits COX-2
- COX-2 regulates prostaglandin production
- Prostaglandins cause pain
- Pain is a symptom of headache

This forms a **mechanistic pathway** explaining the prediction.

---

## 🐛 **Troubleshooting**

### **Issue: "No module named 'torch_geometric.explain'"**

**Solution**: Upgrade PyTorch Geometric
```bash
pip install torch-geometric>=2.3.0
```

### **Issue: GNNExplainer is very slow**

**Solutions**:
1. Reduce epochs:
```yaml
gnnexplainer:
  gnn_epochs: 100  # Instead of 200
```

2. Explain fewer triples:
```yaml
triple_selection:
  num_triples: 5  # Instead of 10
```

3. Use PGExplainer instead (faster after training)

### **Issue: Explanations don't make sense**

**Solutions**:
1. Check model quality first:
```bash
# Evaluate model
kedro run --pipeline=evaluation
```

2. Increase GNNExplainer epochs for better quality:
```yaml
gnnexplainer:
  gnn_epochs: 500
  gnn_lr: 0.005  # Lower LR
```

3. Try different explainer:
- GNNExplainer for instance-specific
- PGExplainer for general patterns

### **Issue: CUDA Out of Memory**

**Solution**: Reduce batch size or use CPU
```yaml
# parameters.yml
device: "cpu"  # Use CPU instead

# Or reduce number of triples
triple_selection:
  num_triples: 5
```

---

## 📚 **Output Files**

After running the explanation pipeline, you'll find:

```
data/05_model_explanations/
├── selected_triples.pkl        # Selected triples with metadata
├── gnn_explanations.pkl        # GNNExplainer results
├── pg_explanations.pkl         # PGExplainer results
└── explanation_summary.pkl     # Comparison summary
```

### **Reading Explanations**

```python
import pickle

# Load GNNExplainer results
gnn_exp = pickle.load(open('data/05_model_explanations/gnn_explanations.pkl', 'rb'))

for exp in gnn_exp['explanations']:
    triple = exp['triple']
    print(f"\nExplaining: {triple['triple']}")

    if 'important_edges' in exp:
        print("Top important edges:")
        for i in range(len(exp['importance_scores'])):
            head = exp['important_edges'][0, i].item()
            tail = exp['important_edges'][1, i].item()
            score = exp['importance_scores'][i].item()
            print(f"  {i+1}. ({head} -> {tail}): {score:.3f}")
```

---

## 🎯 **Advanced Usage**

### **Custom Explanation Strategy**

You can modify [nodes.py](../src/gnn_explainer/pipelines/explanation/nodes.py) to add custom triple selection:

```python
# In select_triples_to_explain()

elif selection_strategy == 'custom':
    # Your custom logic
    # Example: Select high-confidence predictions
    scores = model.predict_all()
    top_indices = torch.topk(scores, num_triples).indices
```

### **Visualizing Explanations**

Create a visualization node:

```python
import networkx as nx
import matplotlib.pyplot as plt

def visualize_explanation(explanation, knowledge_graph):
    G = nx.DiGraph()

    # Add important edges
    for i in range(len(explanation['importance_scores'])):
        head = explanation['important_edges'][0, i].item()
        tail = explanation['important_edges'][1, i].item()
        weight = explanation['importance_scores'][i].item()

        G.add_edge(head, tail, weight=weight)

    # Draw
    pos = nx.spring_layout(G)
    nx.draw(G, pos, with_labels=True)
    plt.savefig('explanation.png')
```

---

## 📖 **References**

### **Papers**

1. **GNNExplainer**: [Generating Explanations for Graph Neural Networks](https://arxiv.org/abs/1903.03894) (NeurIPS 2019)
2. **PGExplainer**: [Parameterized Explainer for Graph Neural Network](https://arxiv.org/abs/2011.04573) (NeurIPS 2020)
3. **CompGCN**: [Composition-based Multi-Relational GCN](https://arxiv.org/abs/1911.03082) (ICLR 2020)

### **Documentation**

- [PyG Explainability Tutorial](https://pytorch-geometric.readthedocs.io/en/latest/tutorial/explain.html)
- [PyG Explain API](https://pytorch-geometric.readthedocs.io/en/latest/modules/explain.html)
- [GNNExplainer API](https://pytorch-geometric.readthedocs.io/en/latest/generated/torch_geometric.explain.algorithm.GNNExplainer.html)
- [PGExplainer API](https://pytorch-geometric.readthedocs.io/en/latest/generated/torch_geometric.explain.algorithm.PGExplainer.html)

### **Implementations**

- [PyTorch Geometric](https://github.com/pyg-team/pytorch_geometric)
- [GNNExplainer Example](https://github.com/pyg-team/pytorch_geometric/blob/master/examples/explain/gnn_explainer.py)

---

## 💡 **Key Takeaways**

1. **GNNExplainer** is best for instance-specific, high-quality explanations
2. **PGExplainer** is best for fast inference after training
3. Both explainers work seamlessly with CompGCN models
4. Explanations reveal **mechanistic pathways** in the knowledge graph
5. Compare both methods for comprehensive understanding

---

## 🔮 **Future Enhancements**

### **Potential Additions**

1. **More Explainers**:
   - AttentionExplainer (attention-based)
   - CaptumExplainer (attribution methods)
   - GraphMaskExplainer (differentiable masks)

2. **Evaluation Metrics**:
   - Fidelity (how well explanation matches model)
   - Sparsity (how compact the explanation)
   - Stability (consistency across runs)

3. **Visualization**:
   - Interactive subgraph visualization
   - Pathway diagrams
   - Heatmaps of edge importance

4. **Comparison with Ground Truth**:
   - If you have known mechanisms, compare explanations
   - Metrics: precision, recall, F1 for retrieved pathways

---

**Implementation Complete!** 🎉

Both GNNExplainer and PGExplainer are now integrated into the Kedro pipeline and ready to explain your CompGCN model predictions.
