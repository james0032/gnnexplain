# Explainer Architecture Analysis: How Explainers Use the Trained Model

**Date**: 2025-11-26
**Question**: Do explainers use only the GNN encoder, or do they also use the link prediction decoder?

---

## 🔍 **TL;DR: Answer**

**Current Implementation**:
- **GNNExplainer & PGExplainer**: Use **BOTH encoder AND decoder** ✅
- **PAGE**: Uses **ONLY graph structure** (NOT the trained model) ❌

**Critical Finding**: PAGE currently does NOT leverage the trained CompGCN model's knowledge!

---

## 📊 **Detailed Analysis**

### **1. GNNExplainer & PGExplainer Usage**

#### **Architecture Flow**

```python
# From nodes.py:32-68 (ModelWrapper.forward)

def forward(self, x, edge_index, **kwargs):
    if self.mode == 'link_prediction':
        # Step 1: Encode - Uses CompGCN ENCODER
        node_emb, rel_emb = self.kg_model.encode(
            self.edge_index,  # Full graph
            self.edge_type
        )

        # Step 2: Decode - Uses DECODER (DistMult/ComplEx/RotatE/ConvE)
        scores = self.kg_model.decode(
            node_emb,      # From encoder
            rel_emb,       # From encoder
            head_idx,      # Query triple
            tail_idx,
            edge_type_for_query
        )

        return scores  # Link prediction scores
```

#### **What This Means**

1. **Uses Encoder**: ✅
   - Runs CompGCN message passing on full graph
   - Generates contextualized node embeddings
   - Produces learned relation embeddings
   - **Leverages graph structure + trained weights**

2. **Uses Decoder**: ✅
   - Scores triples using learned decoder
   - Could be DistMult, ComplEx, RotatE, or ConvE
   - **Leverages trained scoring function**

3. **Explanation Process**:
   ```
   GNN/PG Explainer optimizes:
   "Which edges in the graph are important for this prediction score?"

   Where prediction score = decoder(encoder(full_graph))
   ```

#### **Gradient Flow**

```python
# GNNExplainer/PGExplainer can compute gradients through:

Full Graph → CompGCN Encoder → Node/Rel Embeddings → Decoder → Triple Score
    ↑              ↑                    ↑                ↑           ↑
 Edge Mask    GNN Layers          Embeddings      Scoring Fn    Target
(optimized)   (frozen)            (frozen)        (frozen)
```

**Key Point**: GNN/PG explainers can identify which edges are important **because** they can trace gradients through the entire encode→decode pipeline.

---

### **2. PAGE Explainer Usage**

#### **Architecture Flow**

```python
# From nodes.py:551-763 (run_page_explainer)

def run_page_explainer(model_dict, selected_triples, pyg_data, explainer_params):
    # Get graph structure
    edge_index = model_dict['edge_index']  # Just the structure
    edge_type = model_dict['edge_type']

    # Initialize PAGE - NEW untrained VGAE
    page_explainer = PAGEExplainer(
        input_dim=num_nodes,  # One-hot encoding
        ...
    )

    # Extract subgraphs
    for head_idx, tail_idx in triples:
        subgraph_nodes, subgraph_edges, adj = extract_link_subgraph(
            edge_index,  # Just topology
            head_idx,
            tail_idx,
            k_hops
        )

        # Create features: identity matrix (NOT using CompGCN embeddings!)
        x = torch.eye(num_subgraph_nodes)

        subgraphs.append((x, adj))

    # Train PAGE VGAE on subgraphs (INDEPENDENT of CompGCN!)
    page_explainer.train_on_subgraphs(subgraphs)

    # Generate explanation (based on VGAE reconstruction)
    explanation = page_explainer.explain(x, adj)
```

#### **What This Means**

1. **Does NOT use Encoder**: ❌
   - Does not call `kg_model.encode()`
   - Does not use CompGCN's learned node embeddings
   - Does not use CompGCN's learned relation embeddings

2. **Does NOT use Decoder**: ❌
   - Does not call `kg_model.decode()`
   - Does not use trained scoring function
   - Does not leverage knowledge of what makes a good triple

3. **What PAGE Actually Uses**:
   - ✅ Graph topology (edge_index)
   - ✅ Edge types (edge_type)
   - ❌ Trained CompGCN encoder weights
   - ❌ Trained decoder weights
   - ❌ Link prediction scores

4. **Explanation Process**:
   ```
   PAGE trains NEW VGAE:
   "Which edges are structurally important in this subgraph?"

   Based on: subgraph reconstruction, NOT link prediction quality
   ```

---

## 🚨 **Critical Problem with Current PAGE Implementation**

### **Issue**

PAGE is explaining **graph structure**, not **model predictions**!

```
❌ Current PAGE:
   Explains: "Why is this subgraph structured this way?"
   Based on: VGAE reconstruction loss

✅ Should be:
   Explains: "Why did the model predict this triple?"
   Based on: CompGCN prediction scores
```

### **Example Problem**

```python
# Triple to explain: (Aspirin, treats, Headache)

# CompGCN prediction:
score = decoder(encoder(full_graph))  # 0.95 - high confidence!
# Why? Because: Aspirin→inhibits→COX2→causes→Inflammation→leads_to→Headache

# PAGE explanation (WRONG):
# Trains VGAE on random subgraph structure
# Explains: "These edges exist in the graph"
# NOT: "These edges explain why CompGCN gave this score"
```

The issue: **PAGE doesn't know what the CompGCN model thinks!**

---

## ✅ **How to Fix PAGE**

### **Option 1: Use Encoder Embeddings (Quick Fix)**

```python
# In run_page_explainer(), replace:

# OLD: Use identity features
x = torch.eye(num_subgraph_nodes)

# NEW: Use CompGCN embeddings
node_emb, rel_emb = model_dict['model'].encode(
    model_dict['edge_index'],
    model_dict['edge_type']
)

# Extract subgraph embeddings
x = node_emb[subgraph_nodes].unsqueeze(0)  # (1, num_nodes, embedding_dim)
```

**Impact**: PAGE would now operate on learned embeddings, not raw structure.

### **Option 2: Add Prediction-Aware Training (Better)**

```python
# Add prediction scores to PAGE training

def train_on_subgraphs_with_predictions(
    subgraphs,
    prediction_scores,  # NEW: scores from CompGCN
    ...
):
    for (x, adj), score in zip(subgraphs, prediction_scores):
        # Standard VGAE loss
        adj_recon, mu, logvar, z = vgae(x, adj)
        recon_loss, kl_loss = vgae_loss(adj_recon, adj, mu, logvar)

        # NEW: Add prediction-guided loss
        # High-scoring triples should have more important edges
        prediction_weight = torch.sigmoid(score)
        weighted_recon_loss = recon_loss * prediction_weight

        loss = weighted_recon_loss + kl_loss
```

**Impact**: PAGE would focus on edges that contribute to high predictions.

### **Option 3: Use Encoder in VGAE (Most Integrated)**

```python
# Replace PAGE's GCN layers with frozen CompGCN layers

class VGAEWithCompGCN(nn.Module):
    def __init__(self, compgcn_encoder, latent_dim):
        super().__init__()
        # Use frozen CompGCN for encoding
        self.compgcn = compgcn_encoder
        for param in self.compgcn.parameters():
            param.requires_grad = False

        # Only train the VAE head
        self.fc_mu = nn.Linear(compgcn_output_dim, latent_dim)
        self.fc_logvar = nn.Linear(compgcn_output_dim, latent_dim)
```

**Impact**: PAGE would directly use CompGCN's learned representations.

---

## 📊 **Comparison: Current vs Fixed**

| Aspect | GNN/PG Explainer | PAGE (Current) | PAGE (Fixed) |
|--------|------------------|----------------|--------------|
| **Uses Encoder** | ✅ Yes | ❌ No | ✅ Yes |
| **Uses Decoder** | ✅ Yes | ❌ No | ✅ Yes |
| **Uses Predictions** | ✅ Yes | ❌ No | ✅ Yes |
| **Explains** | Model predictions | Graph structure | Model predictions |
| **Fidelity** | High | **Low** | High |
| **Computational Cost** | High | Low | Medium |

---

## 🎯 **Recommendations**

### **Immediate Action (Priority 1)**

Implement **Option 1: Use Encoder Embeddings**

**Why**:
- Quick to implement (5 lines of code)
- Immediately leverages trained model
- Minimal changes to existing PAGE code

**Code Change**:
```python
# In nodes.py:640-641
# OLD:
x = torch.eye(num_subgraph_nodes).unsqueeze(0)

# NEW:
node_emb, rel_emb = model_dict['model'].encode(
    model_dict['edge_index'],
    model_dict['edge_type']
)
x = node_emb[subgraph_nodes].unsqueeze(0)
# Also need to update input_dim in PAGEExplainer initialization
```

### **Future Enhancement (Priority 2)**

Implement **Option 2: Prediction-Aware Training**

**Why**:
- Makes PAGE truly explain model predictions
- Aligns with the purpose of explainability
- More faithful to the model's reasoning

### **Research Direction (Priority 3)**

Implement **Option 3: Integrated CompGCN-VGAE**

**Why**:
- Most theoretically sound
- Best fidelity to model
- Could lead to novel insights

---

## 🔬 **Experimental Validation Plan**

To validate which approach works best:

### **Metric 1: Fidelity**
```python
# Remove top-k edges identified by explainer
# Measure drop in prediction score

fidelity = original_score - score_without_important_edges

# Higher fidelity = better explanation
```

### **Metric 2: Sparsity**
```python
# Number of edges needed for good explanation

sparsity = num_important_edges / total_edges_in_subgraph

# Lower sparsity = more compact explanation
```

### **Metric 3: Consistency**
```python
# Agreement between explainers

consistency = overlap(GNN_edges, PG_edges, PAGE_edges)

# Higher consistency = more reliable
```

### **Expected Results**

| Approach | Fidelity | Sparsity | Consistency | Notes |
|----------|----------|----------|-------------|-------|
| **PAGE (Current)** | 🔴 Low (~0.3) | 🟡 Medium | 🔴 Low | Explains structure, not predictions |
| **PAGE + Embeddings** | 🟡 Medium (~0.6) | 🟢 Good | 🟡 Medium | Uses learned features |
| **PAGE + Predictions** | 🟢 High (~0.8) | 🟢 Good | 🟢 High | Prediction-aware |
| **GNN/PG Explainer** | 🟢 High (~0.9) | 🟡 Medium | 🟢 High | Baseline |

---

## 📝 **Implementation Checklist**

- [ ] **Quick Fix**: Update PAGE to use CompGCN embeddings
  - [ ] Modify `run_page_explainer()` to extract embeddings
  - [ ] Update `PAGEExplainer.__init__()` to accept embedding_dim
  - [ ] Update `input_dim` from `num_nodes` to `embedding_dim`
  - [ ] Test on small graph

- [ ] **Validation**: Compare explanations
  - [ ] Run GNN, PG, PAGE on same triples
  - [ ] Compute overlap metrics
  - [ ] Visualize differences

- [ ] **Enhancement**: Add prediction-aware training
  - [ ] Add `prediction_scores` parameter to training
  - [ ] Implement weighted loss function
  - [ ] Compare before/after fidelity

- [ ] **Documentation**: Update explainer comparison
  - [ ] Update EXPLANATION_PIPELINE.md
  - [ ] Add fidelity metrics
  - [ ] Document best practices

---

## 💡 **Key Insights**

1. **GNN/PG Explainers are model-faithful**: They explain what the **trained model** thinks, not just graph structure.

2. **PAGE needs integration**: Current implementation is model-agnostic (explains structure), but should be model-specific (explains predictions).

3. **Encoder + Decoder both matter**: Explainers need access to both:
   - **Encoder**: For contextualized embeddings
   - **Decoder**: For scoring function

4. **Different explainer philosophies**:
   - **GNN/PG**: "Optimize to find important edges for this prediction"
   - **PAGE (current)**: "Learn to reconstruct graph structure"
   - **PAGE (should be)**: "Learn to explain predictions via generative modeling"

---

## 🔗 **Code References**

- **GNN/PG usage**: [nodes.py:32-68](../src/gnn_explainer/pipelines/explanation/nodes.py#L32-L68) (ModelWrapper.forward)
- **CompGCN encode/decode**: [kg_models.py:94-180](../src/gnn_explainer/pipelines/training/kg_models.py#L94-L180)
- **PAGE implementation**: [nodes.py:551-763](../src/gnn_explainer/pipelines/explanation/nodes.py#L551-L763) (run_page_explainer)
- **PAGE VGAE**: [page_simple.py](../src/gnn_explainer/pipelines/explanation/page_simple.py)

---

**Conclusion**: GNN and PG explainers correctly use both encoder and decoder. PAGE needs to be fixed to leverage the trained model instead of only using raw graph structure.
