# PAGE Integration Plan

**Date**: 2025-11-26
**Status**: 🔧 **IN PROGRESS** - Planning PAGE Integration

---

## 📋 **Overview**

This document outlines the plan to integrate **PAGE (Parametric Generative Explainer)** into the GNN Explainer pipeline alongside GNNExplainer and PGExplainer.

---

## 🔍 **Research Findings**

### **What is PAGE?**

**Paper**: [PAGE: Parametric Generative Explainer for Graph Neural Network](https://arxiv.org/abs/2408.14042) (2024)

**Key Innovation**: PAGE is a parameterized generative framework that learns to generate explanations without requiring:
- Prior knowledge of the GNN internals
- Per-instance optimization (unlike GNNExplainer)
- Perturbation-based approaches

### **Architecture Components**

PAGE consists of three main components:

1. **Variational Graph Auto-Encoder (VGAE)**
   - **Encoder**: 3-layer Graph Convolutional Network
     - Produces latent representation (μ, σ) for variational inference
     - Dimensionality reduction to extract causal features
   - **Decoder**: MLP-based decoder
     - Maps latent features back to graph substructures
     - Reconstructs adjacency matrix from latent space

2. **Discriminator**
   - Identifies causal relationships between latent features and predictions
   - Guides the encoder to focus on causally relevant features

3. **Training Strategy**
   - **GAE Loss**: Reconstruction loss for graph structure
   - **KL Divergence**: Regularization for latent distribution
   - **Causal Loss**: Encourages causal feature learning
   - **Size Loss**: Promotes compact explanations

### **Key Differences from GNNExplainer/PGExplainer**

| Feature | GNNExplainer | PGExplainer | PAGE |
|---------|-------------|-------------|------|
| **Per-instance optimization** | ✅ Yes | ❌ No | ❌ No |
| **Generative model** | ❌ No | ❌ No | ✅ Yes (VAE) |
| **Sample-level operation** | ❌ Edge-level | ❌ Edge-level | ✅ Sample-level |
| **Causal reasoning** | ⚠️ Implicit | ⚠️ Implicit | ✅ Explicit (discriminator) |
| **Training required** | ❌ No | ✅ Yes | ✅ Yes |
| **Speed** | 🐢 Slow | ⚡ Fast | ⚡ Fast |

---

## 🚧 **Challenges for Knowledge Graph Link Prediction**

### **Problem 1: PAGE is designed for node/graph classification**

The original PAGE implementation (`PAGE_node.py`, `PAGE_graph.py`) is built for:
- Node classification (predict node labels)
- Graph classification (predict graph labels)

Our task is **link prediction** (predict edge existence/score).

**Impact**: PAGE's architecture assumes node-level or graph-level predictions, not edge-level scores.

### **Problem 2: Different computational graph**

- **PAGE**: Explains k-hop subgraph around a node
- **Our task**: Explain why a specific triple `(head, relation, tail)` is predicted

**Impact**: Need to adapt PAGE's subgraph extraction to focus on edge explanations.

### **Problem 3: KG-specific considerations**

- Knowledge graphs have **multiple relation types**
- Relations carry semantic meaning (not just connectivity)
- CompGCN learns both node AND relation embeddings

**Impact**: PAGE's VAE needs to handle edge types, not just adjacency.

### **Problem 4: Integration with CompGCN**

- PAGE was designed for simpler GNN models (GCN, GAT)
- CompGCN has a different architecture (composition-based message passing)
- Need to ensure PAGE can explain CompGCN's predictions

**Impact**: ModelWrapper may need modifications.

---

## 💡 **Proposed Integration Approach**

### **Option A: Adapt PAGE for Link Prediction** (Recommended)

**Strategy**: Modify PAGE to explain link predictions instead of node predictions.

#### **Key Modifications**:

1. **Redefine the "sample" for PAGE**
   - Instead of a k-hop subgraph around a node
   - Use a k-hop subgraph around both `head` and `tail` nodes
   - The "prediction" is the link score between head and tail

2. **Adapt the discriminator**
   - Train discriminator to distinguish between:
     - Latent features of positive triples (high scores)
     - Latent features of negative triples (low scores)
   - This identifies causal features for link existence

3. **Modify explanation generation**
   - Extract subgraph from latent space
   - Identify edges that contribute to the link prediction
   - Return top-k important edges (like GNNExplainer)

#### **Implementation Steps**:

1. **Create `PAGELinkPredictor` class**
   ```python
   class PAGELinkPredictor:
       def __init__(self, kg_model, vgae, discriminator):
           self.kg_model = kg_model  # Trained CompGCN
           self.vgae = vgae          # VGAE3MLP from PAGE
           self.discriminator = discriminator
   ```

2. **Implement training loop**
   ```python
   def train(self, edge_index, edge_type, positive_triples, negative_triples):
       # 1. Extract k-hop subgraphs for each triple
       # 2. Encode subgraphs with VGAE
       # 3. Compute causal loss with discriminator
       # 4. Optimize VGAE and discriminator jointly
   ```

3. **Implement explanation generation**
   ```python
   def explain(self, head_idx, tail_idx, relation_idx):
       # 1. Extract k-hop subgraph around head and tail
       # 2. Encode with trained VGAE
       # 3. Sample from latent space
       # 4. Decode to get explanation subgraph
       # 5. Return top-k important edges
   ```

#### **Pros**:
- ✅ Native PAGE approach
- ✅ Leverages PAGE's causal reasoning
- ✅ Sample-level explanations

#### **Cons**:
- ❌ Significant implementation effort
- ❌ Requires adapting PAGE's training loop
- ❌ May need extensive debugging

#### **Estimated Effort**: **High** (~2-3 days)

---

### **Option B: Use PAGE's Encoder as a Feature Extractor** (Simpler)

**Strategy**: Use PAGE's VGAE encoder to extract features, but use a simpler explanation method.

#### **Approach**:

1. **Train VGAE on the KG**
   - Reconstruct adjacency matrix
   - Learn latent representations of subgraphs

2. **For each triple to explain**:
   - Extract k-hop subgraph
   - Encode with VGAE to get latent representation
   - Use attention/importance scores to identify key edges

3. **Skip the discriminator**
   - Use reconstruction error or gradient-based importance instead

#### **Pros**:
- ✅ Simpler to implement
- ✅ Reuses PAGE's VAE component
- ✅ Faster development

#### **Cons**:
- ❌ Loses PAGE's causal reasoning
- ❌ Not a "true" PAGE implementation
- ❌ May not outperform GNNExplainer/PGExplainer

#### **Estimated Effort**: **Medium** (~1 day)

---

### **Option C: Port PAGE's Node Classification Approach** (Direct Port)

**Strategy**: Treat link prediction as node pair classification and port PAGE directly.

#### **Approach**:

1. **Create virtual "link nodes"**
   - For each triple `(h, r, t)`, create a virtual node representing the link
   - Connect to both `h` and `t` with the relation `r`

2. **Use PAGE's node explainer**
   - Explain the virtual link node's prediction
   - The explanation subgraph shows why the link exists

#### **Pros**:
- ✅ Minimal changes to PAGE code
- ✅ Proven architecture

#### **Cons**:
- ❌ Awkward representation (virtual nodes)
- ❌ May not capture KG semantics well
- ❌ Increased graph complexity

#### **Estimated Effort**: **Medium** (~1-2 days)

---

## 🎯 **Recommendation: Option A (Adapt for Link Prediction)**

Despite the higher effort, **Option A** is recommended because:

1. **Maintains PAGE's innovation**: Leverages the causal reasoning and generative approach
2. **Proper semantic fit**: Directly addresses link prediction instead of workarounds
3. **Research value**: Creates a novel application of PAGE to KG link prediction
4. **Completeness**: Provides a true third explainer method alongside GNN/PG

---

## 📝 **Implementation Plan**

### **Phase 1: Core PAGE Components** (Day 1)

1. **Port VGAE modules**
   - Copy `gae/` directory to `src/gnn_explainer/pipelines/explanation/page/`
   - Test VGAE encoder/decoder on simple graphs

2. **Implement subgraph extraction**
   - Create `extract_k_hop_subgraph(head, tail, k_hops)` function
   - Extract relevant edges and nodes around head/tail

3. **Create discriminator**
   - Implement MLP discriminator for causal feature identification
   - Input: latent representation, Output: causality score

### **Phase 2: Training Loop** (Day 2)

1. **Implement PAGE training**
   - Loss functions: GAE loss, KL divergence, causal loss, size loss
   - Optimization loop with positive/negative triple sampling

2. **Integration with CompGCN**
   - Ensure compatibility with CompGCN's predictions
   - Handle edge types and relation embeddings

3. **Checkpoint saving/loading**
   - Save trained PAGE model
   - Load for explanation generation

### **Phase 3: Explanation Generation** (Day 2-3)

1. **Implement explanation method**
   - Sample from latent space
   - Decode to explanation subgraph
   - Extract top-k important edges

2. **Create `run_page_explainer()` node**
   - Similar interface to `run_gnnexplainer()` and `run_pgexplainer()`
   - Returns explanations in same format

3. **Update pipeline**
   - Add PAGE to explanation pipeline
   - Update catalog and parameters

### **Phase 4: Testing and Validation** (Day 3)

1. **Unit tests**
   - Test VGAE forward/backward pass
   - Test subgraph extraction
   - Test discriminator

2. **Integration tests**
   - Test PAGE with trained CompGCN
   - Compare PAGE vs GNN vs PG explanations

3. **Documentation**
   - Usage guide
   - Configuration options
   - Troubleshooting

---

## 📊 **Expected Output Format**

PAGE explanations should match the format of GNNExplainer/PGExplainer:

```python
{
    'triple': {
        'head_name': 'Aspirin',
        'relation_name': 'treats',
        'tail_name': 'Headache',
        'triple': '(Aspirin, treats, Headache)'
    },
    'explanation': {
        'latent_representation': tensor(...),  # μ, σ from VGAE
        'explanation_subgraph': tensor(...),   # Decoded adjacency
        'edge_mask': tensor(...),              # Importance scores
    },
    'important_edges': tensor([[h1, h2, ...], [t1, t2, ...]]),
    'important_edge_types': tensor([r1, r2, ...]),
    'importance_scores': tensor([s1, s2, ...])
}
```

---

## ⚙️ **Configuration Parameters**

```yaml
explanation:
  page:
    # Training parameters
    train_epochs: 300
    batch_size: 32
    learning_rate: 0.003
    dropout: 0.0

    # Architecture parameters
    encoder_hidden1: 32
    encoder_hidden2: 16
    encoder_output: 16
    decoder_hidden1: 16
    decoder_hidden2: 16

    # Subgraph parameters
    k_hops: 3                    # Number of hops for subgraph

    # Latent space parameters
    K: 3                         # Dimensions for alpha
    NX: 2                        # Samples of X
    Nalpha: 25                   # Samples of alpha
    Nbeta: 100                   # Samples of beta

    # Loss coefficients
    coef_lambda: 0.1             # GAE loss coefficient
    coef_kl: 0.2                 # KL divergence coefficient
    coef_causal: 1.0             # Causal loss coefficient
    coef_size: 0.1               # Size loss coefficient

    # Explanation parameters
    top_k_edges: 10              # Top important edges
    num_samples: 100             # Samples from latent space
```

---

## 🐛 **Anticipated Challenges**

### **Challenge 1: Training instability**
- VGAE training can be unstable
- **Solution**: Careful hyperparameter tuning, gradient clipping

### **Challenge 2: Subgraph extraction complexity**
- Large KGs may have huge k-hop neighborhoods
- **Solution**: Limit subgraph size, use sampling

### **Challenge 3: Relation type handling**
- PAGE's original code doesn't handle edge types
- **Solution**: Modify VGAE to include edge features

### **Challenge 4: Computational cost**
- Training PAGE is expensive (300 epochs default)
- **Solution**: Use smaller batch sizes, cache trained models

---

## 📚 **Dependencies**

Additional dependencies for PAGE:

```bash
pip install tensorboardX  # For training visualization
pip install causaleffect  # For causal loss computation (if available)
pip install scipy         # For sparse matrix operations
```

---

## 🎯 **Success Criteria**

PAGE integration is successful if:

1. ✅ PAGE trains without errors on KG data
2. ✅ PAGE generates explanations for selected triples
3. ✅ Explanations are interpretable (top-k edges make sense)
4. ✅ PAGE outputs match GNN/PG explainer format
5. ✅ Summary node can compare all three explainers
6. ✅ Tests pass for PAGE components
7. ✅ Documentation is complete

---

## 🚦 **Decision Point**

**Before proceeding with full implementation, please confirm**:

1. **Which option to pursue?**
   - Option A: Adapt PAGE for link prediction (recommended, high effort)
   - Option B: Use VGAE as feature extractor (simpler, medium effort)
   - Option C: Port node classification approach (medium effort)

2. **Priority?**
   - High: Implement immediately
   - Medium: Implement after evaluating GNN/PG explainers
   - Low: Keep as future enhancement

3. **Expectations?**
   - Full PAGE implementation with all features
   - Minimal viable PAGE integration
   - Experimental/research prototype

---

## 📖 **References**

- **PAGE Paper**: https://arxiv.org/abs/2408.14042
- **PAGE Code**: https://github.com/anders1123/PAGE
- **VGAE Paper**: [Variational Graph Auto-Encoders](https://arxiv.org/abs/1611.07308)

---

**Next Steps**: Awaiting decision on integration approach before proceeding with implementation.
