# GNN Explainer - Model Architectures Guide

This guide describes all available model architectures in the GNN Explainer Kedro pipeline.

## 🎯 **Overview**

The pipeline supports multiple encoder-decoder combinations for knowledge graph embedding:

### **Encoders**
1. **RGCN** - Relational Graph Convolutional Network
2. **CompGCN** - Composition-based Multi-Relational GCN ⭐ **NEW**

### **Decoders**
1. **DistMult** - Simple bilinear model
2. **ComplEx** - Complex-valued embeddings ⭐ **NEW**
3. **RotatE** - Rotations in complex space ⭐ **NEW**
4. **ConvE** - 2D Convolutional embeddings ⭐ **NEW**

---

## 📊 **Model Comparison**

| Model | Encoder | Decoder | Params | Performance | Best For |
|-------|---------|---------|--------|-------------|----------|
| **RGCN-DistMult** | RGCN | DistMult | Medium | Good | Baseline, Symmetric relations |
| **CompGCN-Complex** | CompGCN | ComplEx | Medium-High | **Very Good** | General purpose, Asymmetric relations |
| **CompGCN-RotatE** | CompGCN | RotatE | Medium-High | **Very Good** | Hierarchical, Composition patterns |
| **CompGCN-ConvE** | CompGCN | ConvE | **Lowest** | **Very Good** | Parameter efficiency |
| **CompGCN-DistMult** | CompGCN | DistMult | Medium | Good | Faster training |

---

## 🏗️ **Architecture Details**

### **1. RGCN + DistMult** (Original)

**Encoder**: RGCN
- Learns only **node embeddings**
- Relations are parameters (not learned embeddings)
- Uses basis decomposition for parameter efficiency

**Decoder**: DistMult
- Scoring: `score = Σ(h ⊙ r ⊙ t)`
- Simple bilinear model
- Good for symmetric relations

**Best For**: Baseline comparison, simple use cases

---

### **2. CompGCN + ComplEx** ⭐ **RECOMMENDED**

**Encoder**: CompGCN
- Learns **both node AND relation embeddings**
- Uses composition operations (sub, mult, corr)
- Jointly optimizes nodes and relations

**Decoder**: ComplEx
- Complex-valued embeddings (real + imaginary)
- Scoring: `score = Re(⟨h, r, conj(t)⟩)`
- Handles symmetric and asymmetric relations

**Best For**: General purpose KG embedding, drug-disease prediction

**Parameters**:
```yaml
model:
  model_type: "compgcn"
  decoder_type: "complex"
  embedding_dim: 200
  comp_fn: "sub"  # or "mult", "corr"
```

**Example**:
```bash
kedro run --params=model.model_type:compgcn,model.decoder_type:complex
```

---

### **3. CompGCN + RotatE**

**Encoder**: CompGCN

**Decoder**: RotatE
- Relations as **rotations in complex space**
- Scoring: `score = -‖h ∘ r - t‖`
- Excellent for modeling:
  - Hierarchies (is-a relations)
  - Symmetric relations
  - Composition patterns

**Best For**: Hierarchical knowledge graphs, chain-like reasoning

**Parameters**:
```yaml
model:
  model_type: "compgcn"
  decoder_type: "rotate"
  embedding_dim: 200
  comp_fn: "mult"  # multiplication works well with RotatE
```

---

### **4. CompGCN + ConvE** (Most Parameter-Efficient)

**Encoder**: CompGCN

**Decoder**: ConvE
- **2D convolutions** over reshaped embeddings
- 8-17x **fewer parameters** than ComplEx/RotatE
- Excellent performance despite parameter efficiency

**Best For**: Large-scale KGs, parameter-constrained environments

**Parameters**:
```yaml
model:
  model_type: "compgcn"
  decoder_type: "conve"
  embedding_dim: 200
  comp_fn: "sub"

  # ConvE-specific
  conve_input_drop: 0.2
  conve_hidden_drop: 0.3
  conve_feature_drop: 0.2
  conve_num_filters: 32
  conve_kernel_size: 3
```

---

### **5. CompGCN + DistMult**

**Encoder**: CompGCN

**Decoder**: DistMult
- Simpler than ComplEx/RotatE
- Faster training
- Still benefits from CompGCN's joint embedding

**Best For**: Quick experiments, symmetric relations

---

## 🎛️ **Configuration Guide**

### **Basic Configuration** ([conf/base/parameters.yml](../conf/base/parameters.yml))

```yaml
model:
  # Choose your architecture
  model_type: "compgcn"     # "rgcn" or "compgcn"
  decoder_type: "complex"    # "distmult", "complex", "rotate", "conve"

  # Common parameters
  embedding_dim: 200         # Recommended: 200 for ComplEx/RotatE/ConvE
  num_layers: 2             # 2-3 layers typical
  dropout: 0.2              # 0.1-0.3 recommended

  # CompGCN composition function
  comp_fn: "sub"            # "sub", "mult", or "corr"
```

### **Composition Functions**

CompGCN supports three composition operations:

1. **Subtraction** (`sub`) - Default
   - `φ(h, r) = h - r`
   - Works well for most cases
   - Fast computation

2. **Multiplication** (`mult`)
   - `φ(h, r) = h * r`
   - Good with RotatE
   - Element-wise product

3. **Circular Correlation** (`corr`)
   - `φ(h, r) = corr(h, r)`
   - Most expressive
   - Slowest computation

---

## 🚀 **Usage Examples**

### **Example 1: Train with CompGCN-ComplEx**

```bash
# Using defaults (ComplEx is default decoder)
kedro run

# Explicit configuration
kedro run --params=model.model_type:compgcn,model.decoder_type:complex
```

### **Example 2: Compare All Decoders**

```bash
# ComplEx
kedro run --params=model.decoder_type:complex

# RotatE
kedro run --params=model.decoder_type:rotate

# ConvE
kedro run --params=model.decoder_type:conve

# DistMult (baseline)
kedro run --params=model.decoder_type:distmult
```

### **Example 3: Experiment with Composition Functions**

```bash
# Subtraction
kedro run --params=model.comp_fn:sub

# Multiplication
kedro run --params=model.comp_fn:mult

# Circular correlation
kedro run --params=model.comp_fn:corr
```

### **Example 4: Use Original RGCN**

```bash
kedro run --params=model.model_type:rgcn
```

---

## 📈 **Performance Tuning**

### **For Best Performance**

```yaml
model:
  model_type: "compgcn"
  decoder_type: "complex"
  embedding_dim: 200
  num_layers: 3              # Try 3 layers
  dropout: 0.1               # Lower dropout
  comp_fn: "sub"

training:
  learning_rate: 0.0005      # Lower learning rate
  batch_size: 4096           # Larger batch
  num_epochs: 200            # More epochs
  patience: 15               # More patience
```

### **For Faster Training**

```yaml
model:
  model_type: "compgcn"
  decoder_type: "distmult"   # Simpler decoder
  embedding_dim: 128         # Smaller embeddings
  num_layers: 2
  dropout: 0.3               # More dropout (regularization)
  comp_fn: "sub"             # Fastest composition

training:
  batch_size: 2048
  num_epochs: 50
```

### **For Parameter Efficiency**

```yaml
model:
  model_type: "compgcn"
  decoder_type: "conve"      # Most parameter-efficient
  embedding_dim: 200
  num_layers: 2
  comp_fn: "sub"

  # ConvE tuning
  conve_num_filters: 32      # Fewer filters = fewer params
  conve_kernel_size: 3
```

---

## 🔬 **Technical Details**

### **CompGCN Layer**

```python
# Message passing with composition
message = φ(h_neighbor, r_edge)  # Compose node and relation

# Aggregation
h_new = σ(W_self * h + Σ W_rel * message)
r_new = σ(W_r * r)
```

Where:
- `φ`: Composition function (sub, mult, corr)
- `σ`: Activation function (ReLU)
- `W`: Learnable weight matrices

### **Decoder Scoring Functions**

**DistMult**:
```
score(h, r, t) = Σ(h_i * r_i * t_i)
```

**ComplEx**:
```
score(h, r, t) = Re(⟨h, r, conj(t)⟩)
               = Σ(Re(h) * Re(r) * Re(t) + Re(h) * Im(r) * Im(t) +
                   Im(h) * Re(r) * Im(t) - Im(h) * Im(r) * Re(t))
```

**RotatE**:
```
score(h, r, t) = -‖h ∘ r - t‖
where r is rotation in complex space
```

**ConvE**:
```
score(h, r, t) = f(conv2d([h; r]))^T * t + b
where conv2d is 2D convolution over reshaped embeddings
```

---

## 📖 **References**

### **Papers**

1. **CompGCN**: [Composition-based Multi-Relational Graph Convolutional Networks](https://arxiv.org/abs/1911.03082) (ICLR 2020)
2. **ComplEx**: [Complex Embeddings for Simple Link Prediction](https://arxiv.org/abs/1606.06357) (ICML 2016)
3. **RotatE**: [RotatE: Knowledge Graph Embedding by Relational Rotation in Complex Space](https://arxiv.org/abs/1902.10197) (ICLR 2019)
4. **ConvE**: [Convolutional 2D Knowledge Graph Embeddings](https://arxiv.org/abs/1707.01476) (AAAI 2018)
5. **RGCN**: [Modeling Relational Data with Graph Convolutional Networks](https://arxiv.org/abs/1703.06103) (ESWC 2018)

### **Implementations**

- [Official CompGCN](https://github.com/malllabiisc/CompGCN)
- [Official ConvE](https://github.com/TimDettmers/ConvE)
- [PyTorch Geometric](https://pytorch-geometric.readthedocs.io/)

---

## 🎯 **Quick Decision Guide**

**Choose CompGCN-ComplEx if:**
- ✅ You want the best general-purpose model
- ✅ You have asymmetric relations
- ✅ You want built-in PyG support

**Choose CompGCN-RotatE if:**
- ✅ Your KG has hierarchical structure
- ✅ You need composition pattern modeling
- ✅ You want built-in PyG support

**Choose CompGCN-ConvE if:**
- ✅ You need maximum parameter efficiency
- ✅ You have limited GPU memory
- ✅ You want 2D convolutional features

**Choose RGCN-DistMult if:**
- ✅ You want the baseline/comparison
- ✅ You need fast prototyping
- ✅ Relations are mostly symmetric

---

## ❓ **FAQ**

**Q: Which model should I start with?**
A: **CompGCN-ComplEx** - It's the best general-purpose choice with strong performance.

**Q: How do I compare models?**
A: Run the pipeline with different `decoder_type` values and compare the evaluation metrics.

**Q: Can I use my own composition function?**
A: Yes! Modify [compgcn_layer.py](../src/gnn_explainer/pipelines/training/compgcn_layer.py) and add your custom composition in the `message()` method.

**Q: Which is fastest to train?**
A: **CompGCN-DistMult** is fastest, followed by ComplEx, then RotatE, then ConvE.

**Q: Which uses least memory?**
A: **CompGCN-ConvE** has the fewest parameters, though ComplEx and RotatE are similar.

**Q: Can I mix encoders and decoders?**
A: Currently supports RGCN+DistMult and CompGCN+(all decoders). You can extend by modifying the training nodes.

---

For more information, see:
- [README_KEDRO.md](../README_KEDRO.md) - General usage guide
- [conf/base/parameters.yml](../conf/base/parameters.yml) - Configuration file
- [IMPLEMENTATION_STATUS.md](IMPLEMENTATION_STATUS.md) - Implementation status
