# Complete Pipeline Overview

**Project**: GNN Explainer for Knowledge Graph Link Prediction
**Date**: 2025-11-26
**Status**: ✅ Production Ready

---

## 🎯 **Quick Summary**

This Kedro pipeline trains CompGCN models on knowledge graphs and explains predictions using 3 state-of-the-art explainers.

**What it does**:
1. Trains CompGCN + (ComplEx/RotatE/ConvE/DistMult) on KG triples
2. Explains predictions using GNNExplainer, PGExplainer, and improved PAGE
3. Provides faithful, interpretable explanations of model reasoning

---

## 📊 **Pipeline Architecture**

```
┌─────────────────────────────────────────────────────────────┐
│                    INPUT DATA (01_raw)                      │
│  • robo_train.txt  • robo_val.txt  • robo_test.txt         │
│  • node_dict  • rel_dict                                    │
└─────────────────────────────────────────────────────────────┘
                           ↓
┌─────────────────────────────────────────────────────────────┐
│              DATA PREPARATION PIPELINE                      │
│  1. Load triples                                            │
│  2. Build dictionaries                                      │
│  3. Create PyG data                                         │
│  4. Generate negative samples                               │
└─────────────────────────────────────────────────────────────┘
                           ↓
┌─────────────────────────────────────────────────────────────┐
│                  TRAINING PIPELINE                          │
│  CompGCN Encoder + Decoder (ComplEx/RotatE/ConvE/DistMult) │
│  • Multi-layer message passing                              │
│  • Joint node + relation embeddings                         │
│  • Link prediction training                                 │
└─────────────────────────────────────────────────────────────┘
                           ↓
┌─────────────────────────────────────────────────────────────┐
│               EXPLANATION PIPELINE                          │
│  ┌──────────────┐  ┌──────────────┐  ┌──────────────┐     │
│  │GNNExplainer  │  │ PGExplainer  │  │Improved PAGE │     │
│  │Instance-level│  │Parameterized │  │Generative    │     │
│  │Gradient-based│  │Fast inference│  │Model-aware   │     │
│  └──────────────┘  └──────────────┘  └──────────────┘     │
│                        ↓                                    │
│              Explanation Summary & Comparison               │
└─────────────────────────────────────────────────────────────┘
                           ↓
┌─────────────────────────────────────────────────────────────┐
│                     OUTPUTS                                 │
│  • Trained model (06_models/)                               │
│  • Explanations (05_model_explanations/)                    │
│  • Evaluation metrics (07_model_output/)                    │
└─────────────────────────────────────────────────────────────┘
```

---

## 📁 **Complete File Structure**

```
gnnexplain/
├── conf/
│   └── base/
│       ├── catalog.yml              # Data catalog
│       └── parameters.yml           # Configuration
│
├── data/
│   ├── 01_raw/                      # INPUT DATA (REQUIRED)
│   │   ├── robo_train.txt          # Training triples
│   │   ├── robo_val.txt            # Validation triples
│   │   ├── robo_test.txt           # Test triples
│   │   ├── node_dict               # Entity mappings
│   │   └── rel_dict                # Relation mappings
│   │
│   ├── 02_intermediate/             # Processed data
│   │   ├── knowledge_graph.pkl
│   │   ├── pyg_data.pkl
│   │   └── negative_samples.pkl
│   │
│   ├── 05_model_explanations/       # Explanations
│   │   ├── selected_triples.pkl
│   │   ├── gnn_explanations.pkl
│   │   ├── pg_explanations.pkl
│   │   ├── page_explanations.pkl
│   │   └── explanation_summary.pkl
│   │
│   └── 06_models/                   # Trained models
│       └── trained_model.pkl
│
├── src/gnn_explainer/pipelines/
│   ├── data_preparation/            # Data pipeline
│   │   ├── __init__.py
│   │   ├── nodes.py
│   │   └── pipeline.py
│   │
│   ├── training/                    # Training pipeline
│   │   ├── __init__.py
│   │   ├── nodes.py
│   │   ├── pipeline.py
│   │   ├── compgcn_layer.py        # CompGCN layer
│   │   ├── compgcn_encoder.py      # CompGCN encoder
│   │   ├── conve_decoder.py        # ConvE decoder
│   │   ├── kg_models.py            # Unified KG models
│   │   └── model.py                # RGCN model
│   │
│   └── explanation/                 # Explanation pipeline
│       ├── __init__.py
│       ├── nodes.py
│       ├── pipeline.py
│       ├── page_simple.py          # Simple PAGE (original)
│       └── page_improved.py        # Improved PAGE ⭐
│
├── tests/
│   ├── test_compgcn_encoder.py     # CompGCN tests
│   └── test_explanation_pipeline.py # Explanation tests
│
└── docs/
    ├── INPUT_DATA_REQUIREMENTS.md   # This guide
    ├── MODEL_ARCHITECTURES.md       # Model options
    ├── COMPGCN_IMPLEMENTATION.md    # CompGCN details
    ├── EXPLANATION_PIPELINE.md      # Explainer usage
    ├── EXPLAINER_ARCHITECTURE_ANALYSIS.md  # How explainers work
    ├── IMPROVED_PAGE_IMPLEMENTATION.md     # Improved PAGE
    ├── PAGE_INTEGRATION_PLAN.md     # PAGE integration
    └── COMPLETE_PIPELINE_OVERVIEW.md # This file
```

---

## 🚀 **Complete Workflow**

### **Step 0: Prepare Input Data**

See [INPUT_DATA_REQUIREMENTS.md](INPUT_DATA_REQUIREMENTS.md) for details.

**Required files**:
```
data/01_raw/
├── robo_train.txt    # Tab-separated: head \t relation \t tail
├── robo_val.txt      # Same format
├── robo_test.txt     # Same format
├── node_dict         # Entity to index: entity \t index
└── rel_dict          # Relation to index: relation \t index
```

### **Step 1: Data Preparation**

```bash
kedro run --pipeline=data_preparation
```

**What it does**:
- Loads triple files
- Builds entity/relation dictionaries
- Creates PyTorch Geometric data
- Generates negative samples
- Saves to `data/02_intermediate/`

### **Step 2: Train CompGCN Model**

```bash
kedro run --pipeline=training
```

**Configuration** (`conf/base/parameters.yml`):
```yaml
model:
  model_type: "compgcn"       # or "rgcn"
  decoder_type: "complex"     # or "rotate", "conve", "distmult"
  embedding_dim: 200
  num_layers: 2
  dropout: 0.2
  comp_fn: "sub"             # CompGCN composition

training:
  learning_rate: 0.001
  batch_size: 2048
  num_epochs: 100
  patience: 10
```

**Output**:
- `data/06_models/trained_model.pkl` - Trained CompGCN model

### **Step 3: Generate Explanations**

```bash
kedro run --pipeline=explanation
```

**Configuration**:
```yaml
explanation:
  triple_selection:
    strategy: "random"        # or "specific_relations", "specific_nodes"
    num_triples: 10

  gnnexplainer:
    gnn_epochs: 200
    gnn_lr: 0.01

  pgexplainer:
    pg_epochs: 30
    pg_lr: 0.003

  page:
    train_epochs: 100
    prediction_weight: 1.0    # Prediction-awareness (NEW!)
```

**Outputs**:
- `data/05_model_explanations/gnn_explanations.pkl`
- `data/05_model_explanations/pg_explanations.pkl`
- `data/05_model_explanations/page_explanations.pkl`
- `data/05_model_explanations/explanation_summary.pkl`

### **Step 4: Analyze Results**

```python
import pickle

# Load explanations
gnn = pickle.load(open('data/05_model_explanations/gnn_explanations.pkl', 'rb'))
page = pickle.load(open('data/05_model_explanations/page_explanations.pkl', 'rb'))
summary = pickle.load(open('data/05_model_explanations/explanation_summary.pkl', 'rb'))

# Check results
print(f"GNN successful: {summary['gnn_explainer']['successful']}")
print(f"PAGE model-aware: {page.get('model_aware', False)}")
print(f"Average overlap: {summary.get('avg_overlap', 0):.2f}")
```

---

## 🎛️ **Configuration Options**

### **Model Selection**

```yaml
# RGCN + DistMult (baseline)
model:
  model_type: "rgcn"
  decoder_type: "distmult"

# CompGCN + ComplEx (recommended)
model:
  model_type: "compgcn"
  decoder_type: "complex"

# CompGCN + RotatE (hierarchical relations)
model:
  model_type: "compgcn"
  decoder_type: "rotate"

# CompGCN + ConvE (parameter-efficient)
model:
  model_type: "compgcn"
  decoder_type: "conve"
```

### **Triple Selection**

```yaml
# Random sampling
triple_selection:
  strategy: "random"
  num_triples: 10

# Specific relations (e.g., "treats")
triple_selection:
  strategy: "specific_relations"
  target_relations: [0, 1, 5]  # Relation indices

# Specific entities (e.g., drug X)
triple_selection:
  strategy: "specific_nodes"
  target_nodes: [100, 200, 300]  # Node indices
```

### **Explainer Tuning**

```yaml
# High-quality GNNExplainer (slow)
gnnexplainer:
  gnn_epochs: 500
  gnn_lr: 0.005

# Fast PGExplainer
pgexplainer:
  pg_epochs: 20

# Prediction-focused PAGE
page:
  train_epochs: 150
  prediction_weight: 2.0  # Higher = more model-aware
```

---

## 📚 **Documentation Index**

### **Getting Started**
- [INPUT_DATA_REQUIREMENTS.md](INPUT_DATA_REQUIREMENTS.md) - **START HERE**
  - Required input files
  - File formats
  - Data preparation

### **Model Training**
- [MODEL_ARCHITECTURES.md](MODEL_ARCHITECTURES.md)
  - CompGCN vs RGCN
  - Decoder options
  - Configuration guide
- [COMPGCN_IMPLEMENTATION.md](COMPGCN_IMPLEMENTATION.md)
  - Implementation details
  - Composition functions
  - Performance tuning

### **Explanation**
- [EXPLANATION_PIPELINE.md](EXPLANATION_PIPELINE.md)
  - GNNExplainer usage
  - PGExplainer usage
  - Comparison guide
- [IMPROVED_PAGE_IMPLEMENTATION.md](IMPROVED_PAGE_IMPLEMENTATION.md) ⭐
  - Improved PAGE details
  - Prediction-aware training
  - Model-faithful explanations
- [EXPLAINER_ARCHITECTURE_ANALYSIS.md](EXPLAINER_ARCHITECTURE_ANALYSIS.md)
  - How explainers work
  - Encoder/decoder usage
  - Technical analysis
- [PAGE_INTEGRATION_PLAN.md](PAGE_INTEGRATION_PLAN.md)
  - Integration options
  - Design decisions

---

## 🎯 **Key Features**

### **✅ Multiple Model Architectures**

| Model | Encoder | Decoder | Parameters | Performance | Best For |
|-------|---------|---------|------------|-------------|----------|
| RGCN-DistMult | RGCN | DistMult | Medium | Good | Baseline |
| CompGCN-ComplEx | CompGCN | ComplEx | Medium-High | **Very Good** | General purpose |
| CompGCN-RotatE | CompGCN | RotatE | Medium-High | **Very Good** | Hierarchical |
| CompGCN-ConvE | CompGCN | ConvE | **Low** | **Very Good** | Efficient |

### **✅ Three Explainer Methods**

| Explainer | Type | Speed | Quality | Model-Aware |
|-----------|------|-------|---------|-------------|
| GNNExplainer | Instance-level | 🐢 Slow | ⭐⭐⭐ High | ✅ Yes |
| PGExplainer | Parameterized | ⚡ Fast | ⭐⭐ Medium | ✅ Yes |
| Improved PAGE | Generative | ⚙️ Medium | ⭐⭐⭐ High | ✅ **Yes (NEW!)** |

### **✅ Modular Kedro Pipeline**

- **Data Preparation**: Standalone, reusable
- **Training**: Supports multiple architectures
- **Explanation**: Runs all explainers in parallel
- **Easy Configuration**: YAML-based, no code changes

---

## 🐛 **Troubleshooting Guide**

### **Issue: No input data**

```
FileNotFoundError: data/01_raw/robo_train.txt
```

**Solution**: See [INPUT_DATA_REQUIREMENTS.md](INPUT_DATA_REQUIREMENTS.md)

### **Issue: CUDA out of memory**

```yaml
# Use CPU
device: "cpu"

# Or reduce batch size
training:
  batch_size: 512
```

### **Issue: Poor model performance**

```yaml
# Try different decoder
model:
  decoder_type: "complex"  # or "rotate"

# Increase capacity
model:
  embedding_dim: 300
  num_layers: 3
```

### **Issue: Explanations not faithful**

```yaml
# Increase PAGE prediction weight
page:
  prediction_weight: 2.0

# More GNNExplainer epochs
gnnexplainer:
  gnn_epochs: 500
```

---

## 📊 **Expected Results**

### **Training**

```
Epoch 100/100:
  Loss: 0.234
  Train Accuracy: 0.87
  Val Accuracy: 0.84
✓ Model saved to data/06_models/trained_model.pkl
```

### **Explanation**

```
GNNExplainer: 10/10 successful
PGExplainer: 10/10 successful
ImprovedPAGE: 10/10 successful (model-aware!)

Average overlap in top-5 edges: 6.8
→ High consistency = reliable explanations
```

### **Example Explanation**

```
Triple: (Aspirin, treats, Headache)
CompGCN Score: 0.92

Top Important Edges (from Improved PAGE):
1. (Aspirin, inhibits, COX2) - 0.94
2. (COX2, regulates, Prostaglandin) - 0.89
3. (Prostaglandin, causes, Pain) - 0.85
4. (Pain, symptom_of, Headache) - 0.82

Explanation: Aspirin inhibits COX2, which regulates prostaglandins,
which cause pain, a symptom of headaches.
```

---

## 🎓 **Best Practices**

1. **Start Small**: Test with 1000 triples first
2. **Validate Data**: Check input format before training
3. **Use ComplEx**: Best general-purpose decoder
4. **Compare Explainers**: Run all three for validation
5. **Document**: Keep track of experiments and results

---

## 🔬 **Research Applications**

This pipeline enables:

1. **Drug Repurposing**: Explain why a drug might treat a disease
2. **Biomarker Discovery**: Identify mechanistic pathways
3. **Knowledge Gap Analysis**: Find missing links in KGs
4. **Model Debugging**: Understand model reasoning
5. **Hypothesis Generation**: Discover novel associations

---

## 📖 **Citations**

**If you use this pipeline, please cite**:

- **CompGCN**: Vashishth et al., "Composition-based Multi-Relational Graph Convolutional Networks" (ICLR 2020)
- **GNNExplainer**: Ying et al., "GNNExplainer: Generating Explanations for Graph Neural Networks" (NeurIPS 2019)
- **PGExplainer**: Luo et al., "Parameterized Explainer for Graph Neural Network" (NeurIPS 2020)
- **PAGE**: Anders et al., "PAGE: Parametric Generative Explainer for Graph Neural Network" (2024)

---

## ✅ **Quick Reference**

```bash
# Complete workflow
kedro run --pipeline=data_preparation
kedro run --pipeline=training
kedro run --pipeline=explanation

# Or run all at once
kedro run

# With custom config
kedro run --params=model.decoder_type:complex,training.num_epochs:50

# Test with small dataset
kedro run --params=explanation.triple_selection.num_triples:5
```

---

**Ready to Start?** Follow [INPUT_DATA_REQUIREMENTS.md](INPUT_DATA_REQUIREMENTS.md) to prepare your data! 🚀
