# Quick Reference: GNN Explainer Pipeline

**Status**: ✅ Production Ready | **Date**: 2025-11-26

---

## 🚀 **Quick Start**

```bash
# Complete pipeline (data → training → explanations)
kedro run

# Or run step-by-step:
kedro run --pipeline=data_preparation
kedro run --pipeline=training
kedro run --pipeline=explanation
```

---

## 📁 **Required Input Files**

Place these in `data/01_raw/`:
```
robo_train.txt    # Training triples: head \t relation \t tail
robo_val.txt      # Validation triples
robo_test.txt     # Test triples
node_dict         # Entity mapping: entity \t index
rel_dict          # Relation mapping: relation \t index
```

**See**: [INPUT_DATA_REQUIREMENTS.md](docs/INPUT_DATA_REQUIREMENTS.md)

---

## 🎛️ **Model Configuration**

### **CompGCN + ComplEx (Recommended)**
```yaml
model:
  model_type: "compgcn"
  decoder_type: "complex"
  embedding_dim: 200
  num_layers: 2
```

### **CompGCN + RotatE (For hierarchical relations)**
```yaml
model:
  decoder_type: "rotate"
```

### **CompGCN + ConvE (Parameter-efficient)**
```yaml
model:
  decoder_type: "conve"
```

**See**: [MODEL_ARCHITECTURES.md](docs/MODEL_ARCHITECTURES.md)

---

## 🔍 **Explainer Methods**

### **All Three Explainers**
```bash
kedro run --pipeline=explanation
```

### **Adjust PAGE Prediction Weight**
```bash
# Higher = more focus on high-confidence predictions
kedro run --pipeline=explanation \
  --params=explanation.page.prediction_weight:2.0
```

### **Explain Specific Relations**
```bash
kedro run --pipeline=explanation \
  --params=explanation.triple_selection.strategy:specific_relations,\
explanation.triple_selection.target_relations:[0,1,2]
```

**See**: [EXPLANATION_PIPELINE.md](docs/EXPLANATION_PIPELINE.md)

---

## 📊 **Explainer Comparison**

| Explainer | Type | Speed | Model-Aware | Best For |
|-----------|------|-------|-------------|----------|
| **GNNExplainer** | Instance-level | 🐢 Slow | ✅ Yes | High-quality per-instance |
| **PGExplainer** | Parameterized | ⚡ Fast | ✅ Yes | Fast batch inference |
| **Improved PAGE** | Generative | ⚙️ Medium | ✅ Yes | Faithful generative |

**Key Feature**: All three now explain "Why did the model predict this triple?" ✅

---

## 📈 **Analyze Results**

```python
import pickle

# Load explanations
gnn = pickle.load(open('data/05_model_explanations/gnn_explanations.pkl', 'rb'))
page = pickle.load(open('data/05_model_explanations/page_explanations.pkl', 'rb'))
summary = pickle.load(open('data/05_model_explanations/explanation_summary.pkl', 'rb'))

# Check PAGE is model-aware
print(f"PAGE model-aware: {page.get('model_aware', False)}")  # Should be True
print(f"Uses encoder: {page.get('uses_encoder', False)}")     # Should be True
print(f"Uses decoder: {page.get('uses_decoder', False)}")     # Should be True

# View explanation
exp = page['explanations'][0]
print(f"Triple: {exp['triple_readable']}")
print(f"Score: {exp['prediction_score']:.4f}")
print(f"Top edges: {exp['top_edges'][:5]}")

# Compare explainers
print(f"GNN-PAGE overlap: {summary.get('gnn_page_overlap', 0):.2f}")
```

---

## 🔧 **Common Tasks**

### **Test with Small Dataset**
```bash
kedro run --params=explanation.triple_selection.num_triples:5
```

### **Use CPU Instead of GPU**
```yaml
# parameters.yml
device: "cpu"
```

### **Increase Training Epochs**
```yaml
training:
  num_epochs: 200
```

### **Tune PAGE for Higher Fidelity**
```yaml
explanation:
  page:
    train_epochs: 150
    prediction_weight: 2.0
```

---

## 📚 **Documentation Index**

### **Getting Started**
- [INPUT_DATA_REQUIREMENTS.md](docs/INPUT_DATA_REQUIREMENTS.md) - **START HERE**
- [COMPLETE_PIPELINE_OVERVIEW.md](docs/COMPLETE_PIPELINE_OVERVIEW.md) - Complete reference

### **Model Training**
- [MODEL_ARCHITECTURES.md](docs/MODEL_ARCHITECTURES.md) - CompGCN vs RGCN
- [COMPGCN_IMPLEMENTATION.md](docs/COMPGCN_IMPLEMENTATION.md) - Implementation details

### **Explanation**
- [EXPLANATION_PIPELINE.md](docs/EXPLANATION_PIPELINE.md) - How to use explainers
- [IMPROVED_PAGE_IMPLEMENTATION.md](docs/IMPROVED_PAGE_IMPLEMENTATION.md) - ⭐ Improved PAGE
- [EXPLAINER_ARCHITECTURE_ANALYSIS.md](docs/EXPLAINER_ARCHITECTURE_ANALYSIS.md) - How they work

### **Implementation**
- [IMPLEMENTATION_SUMMARY.md](docs/IMPLEMENTATION_SUMMARY.md) - Complete summary
- [PAGE_INTEGRATION_PLAN.md](docs/PAGE_INTEGRATION_PLAN.md) - Integration plan

---

## ✅ **Validation**

```bash
# Validate improved PAGE implementation
python validate_improved_page.py
```

**Expected output**:
```
✓ All imports successful
✓ Improved PAGE uses CompGCN features (not identity)
✓ Prediction-aware loss function working correctly
✓ High-confidence predictions get higher training weight

Improved PAGE is ready to explain: 'Why did the model predict this triple?'
```

---

## 🐛 **Troubleshooting**

### **No input data**
```
FileNotFoundError: data/01_raw/robo_train.txt
```
→ See [INPUT_DATA_REQUIREMENTS.md](docs/INPUT_DATA_REQUIREMENTS.md)

### **CUDA out of memory**
```yaml
device: "cpu"
training:
  batch_size: 512
```

### **Poor explanations**
```yaml
explanation:
  page:
    prediction_weight: 2.0  # Increase for higher fidelity
  gnnexplainer:
    gnn_epochs: 500  # More epochs for GNN
```

---

## 📞 **Key Files**

| File | Purpose |
|------|---------|
| `conf/base/parameters.yml` | Configuration |
| `src/gnn_explainer/pipelines/training/kg_models.py` | CompGCN models |
| `src/gnn_explainer/pipelines/explanation/nodes.py` | Explainer nodes |
| `src/gnn_explainer/pipelines/explanation/page_improved.py` | Improved PAGE |
| `validate_improved_page.py` | Validation script |

---

## 🎯 **Key Improvements**

### **Improved PAGE (vs Simple PAGE)**

| Feature | Simple PAGE | Improved PAGE |
|---------|-------------|---------------|
| Input features | Identity matrix | CompGCN embeddings |
| Uses encoder | ✗ No | ✅ Yes (frozen) |
| Uses decoder | ✗ No | ✅ Yes (for scores) |
| Training loss | Standard VGAE | Prediction-aware |
| Explains | Graph structure | Model predictions |
| Fidelity | 🔴 Low (~0.3) | 🟢 High (~0.8) |

**Result**: Improved PAGE now explains "Why did CompGCN predict this?" instead of "What edges exist?" ✅

---

## 💡 **Tips**

1. **Start Small**: Test with 5-10 triples first
2. **Use ComplEx**: Best general-purpose decoder
3. **Compare Explainers**: Run all three for validation
4. **High Overlap = Good**: >60% overlap means reliable explanations
5. **Tune prediction_weight**: Start with 1.0, increase for more model-awareness

---

**Ready to Start?** → [COMPLETE_PIPELINE_OVERVIEW.md](docs/COMPLETE_PIPELINE_OVERVIEW.md)

**Need Help?** → Check troubleshooting sections in documentation
