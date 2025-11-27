# Model Configuration Guide

**How to configure and run different model architectures**

---

## ✅ **Current Configuration**

Your pipeline is **already configured** for **CompGCN + ComplEx** (recommended):

```yaml
model:
  model_type: "compgcn"       # CompGCN encoder
  decoder_type: "complex"     # ComplEx decoder
  embedding_dim: 200          # Recommended for ComplEx
  num_layers: 2
  dropout: 0.2
  comp_fn: "sub"             # Subtraction composition
```

---

## 🚀 **How to Run CompGCN + ComplEx**

### **Quick Start (Default Configuration)**

```bash
# 1. Prepare data (first time only)
kedro run --pipeline=data_preparation

# 2. Train CompGCN + ComplEx
kedro run --pipeline=training

# 3. Generate explanations
kedro run --pipeline=explanation
```

That's it! The default configuration is already set to CompGCN + ComplEx.

---

## 🎛️ **Available Model Configurations**

### **Option 1: CompGCN + ComplEx** ⭐ (Current - Recommended)

**Best for**: General-purpose link prediction, captures complex patterns

```yaml
# conf/base/parameters.yml
model:
  model_type: "compgcn"
  decoder_type: "complex"
  embedding_dim: 200
  num_layers: 2
  comp_fn: "sub"
```

**Run command**:
```bash
kedro run --pipeline=training
# Already configured!
```

**Characteristics**:
- ✓ High accuracy
- ✓ Captures asymmetric relations
- ✓ Moderate parameter count
- ✓ Good for most knowledge graphs

---

### **Option 2: CompGCN + RotatE**

**Best for**: Hierarchical relations, composition patterns (e.g., subclass-of, part-of)

```yaml
model:
  model_type: "compgcn"
  decoder_type: "rotate"
  embedding_dim: 200
  num_layers: 2
  comp_fn: "sub"
```

**Run command**:
```bash
kedro run --pipeline=training --params=model.decoder_type:rotate
```

**Characteristics**:
- ✓ Excellent for hierarchies
- ✓ Rotation-based scoring
- ✓ Good for transitive relations
- ✓ Similar performance to ComplEx

---

### **Option 3: CompGCN + ConvE**

**Best for**: Parameter-efficient training, limited GPU memory

```yaml
model:
  model_type: "compgcn"
  decoder_type: "conve"
  embedding_dim: 200
  num_layers: 2
  comp_fn: "sub"
  # ConvE-specific parameters
  conve_num_filters: 32
  conve_kernel_size: 3
```

**Run command**:
```bash
kedro run --pipeline=training --params=model.decoder_type:conve
```

**Characteristics**:
- ✓ Fewer parameters (efficient)
- ✓ Fast training
- ✓ Good performance
- ✓ Uses 2D convolution

---

### **Option 4: CompGCN + DistMult**

**Best for**: Symmetric relations, simple baseline

```yaml
model:
  model_type: "compgcn"
  decoder_type: "distmult"
  embedding_dim: 128  # Can use smaller dim
  num_layers: 2
  comp_fn: "sub"
```

**Run command**:
```bash
kedro run --pipeline=training \
  --params=model.decoder_type:distmult,model.embedding_dim:128
```

**Characteristics**:
- ✓ Simple and fast
- ✓ Fewer parameters
- ⚠️ Cannot model asymmetric relations
- ⚠️ Lower accuracy than ComplEx/RotatE

---

### **Option 5: RGCN + DistMult** (Legacy Baseline)

**Best for**: Baseline comparison

```yaml
model:
  model_type: "rgcn"
  decoder_type: "distmult"
  embedding_dim: 128
  num_layers: 2
  num_bases: 30  # RGCN-specific
```

**Run command**:
```bash
kedro run --pipeline=training \
  --params=model.model_type:rgcn,model.decoder_type:distmult,model.embedding_dim:128
```

**Characteristics**:
- ✓ Well-established baseline
- ⚠️ Less expressive than CompGCN
- ⚠️ More parameters (without basis decomposition)

---

## 📊 **Model Comparison**

| Model | Encoder | Decoder | Params | Speed | Accuracy | Best For |
|-------|---------|---------|--------|-------|----------|----------|
| **CompGCN-ComplEx** ⭐ | CompGCN | ComplEx | Medium | Medium | ⭐⭐⭐ Excellent | **General purpose** |
| **CompGCN-RotatE** | CompGCN | RotatE | Medium | Medium | ⭐⭐⭐ Excellent | Hierarchies |
| **CompGCN-ConvE** | CompGCN | ConvE | **Low** | **Fast** | ⭐⭐ Very Good | **Efficiency** |
| **CompGCN-DistMult** | CompGCN | DistMult | Low | Fast | ⭐ Good | Simple baseline |
| RGCN-DistMult | RGCN | DistMult | Medium | Medium | ⭐ Good | Legacy baseline |

---

## ⚙️ **Composition Functions (CompGCN)**

When using CompGCN, you can choose different composition functions:

### **Subtraction (Default - Recommended)**
```yaml
comp_fn: "sub"
```
- Most stable
- Good for most graphs
- **Recommended for ComplEx/RotatE**

### **Multiplication**
```yaml
comp_fn: "mult"
```
- Element-wise product
- Can capture interactions better
- Try if subtraction doesn't work well

### **Circular Correlation**
```yaml
comp_fn: "corr"
```
- More complex composition
- Higher capacity
- Slower training

**Command to try different composition**:
```bash
kedro run --pipeline=training --params=model.comp_fn:mult
```

---

## 🔧 **How to Change Configuration**

### **Method 1: Edit YAML File (Permanent)**

Edit `conf/base/parameters.yml`:

```yaml
model:
  model_type: "compgcn"     # Change this
  decoder_type: "complex"   # And this
  embedding_dim: 200
  num_layers: 2
  comp_fn: "sub"
```

Then run:
```bash
kedro run --pipeline=training
```

### **Method 2: Command Line Override (Temporary)**

Override parameters without editing files:

```bash
# Single parameter
kedro run --pipeline=training --params=model.decoder_type:rotate

# Multiple parameters
kedro run --pipeline=training \
  --params=model.decoder_type:conve,model.embedding_dim:256,training.num_epochs:50

# Change model type
kedro run --pipeline=training \
  --params=model.model_type:rgcn,model.decoder_type:distmult
```

---

## 📈 **Hyperparameter Tuning**

### **For Better Accuracy**

```yaml
model:
  embedding_dim: 300        # Increase from 200
  num_layers: 3             # Add more layers
  dropout: 0.1              # Reduce dropout

training:
  num_epochs: 200           # Train longer
  learning_rate: 0.0005     # Lower learning rate
```

**Command**:
```bash
kedro run --pipeline=training \
  --params=model.embedding_dim:300,model.num_layers:3,training.num_epochs:200
```

### **For Faster Training**

```yaml
model:
  decoder_type: "conve"     # Use ConvE
  embedding_dim: 128        # Smaller embeddings
  num_layers: 2

training:
  batch_size: 4096          # Larger batches
  num_epochs: 50            # Fewer epochs
```

**Command**:
```bash
kedro run --pipeline=training \
  --params=model.decoder_type:conve,model.embedding_dim:128,training.batch_size:4096
```

### **For Limited Memory (CPU/Small GPU)**

```yaml
model:
  embedding_dim: 128
  num_layers: 2

training:
  batch_size: 512           # Smaller batches
  num_epochs: 100

device: "cpu"               # Use CPU
```

**Command**:
```bash
kedro run --pipeline=training \
  --params=training.batch_size:512,device:cpu
```

---

## ✅ **Verify Configuration**

Check what configuration will be used:

```bash
# View current model config
python -c "
import yaml
with open('conf/base/parameters.yml', 'r') as f:
    params = yaml.safe_load(f)
    model = params['model']
    print(f\"Model: {model['model_type'].upper()} + {model['decoder_type'].upper()}\")
    print(f\"Embedding Dim: {model['embedding_dim']}\")
    print(f\"Layers: {model['num_layers']}\")
"
```

Or use `kedro catalog list` and `kedro registry list`:

```bash
kedro registry list  # Show available pipelines
```

---

## 🎯 **Recommended Configurations**

### **For Drug Repurposing / Biomedical KGs**
```yaml
model:
  model_type: "compgcn"
  decoder_type: "complex"    # ComplEx is best for asymmetric relations
  embedding_dim: 200
  num_layers: 2
  comp_fn: "sub"
```

### **For Hierarchical Ontologies**
```yaml
model:
  model_type: "compgcn"
  decoder_type: "rotate"     # RotatE handles hierarchies well
  embedding_dim: 200
  num_layers: 2
  comp_fn: "sub"
```

### **For Fast Prototyping**
```yaml
model:
  model_type: "compgcn"
  decoder_type: "conve"      # ConvE is fastest
  embedding_dim: 128
  num_layers: 2

training:
  batch_size: 4096
  num_epochs: 50
```

### **For Maximum Accuracy (Slower)**
```yaml
model:
  model_type: "compgcn"
  decoder_type: "complex"
  embedding_dim: 300         # Larger embeddings
  num_layers: 3              # More layers
  dropout: 0.1

training:
  num_epochs: 200
  learning_rate: 0.0005
  patience: 20
```

---

## 🔍 **After Training**

Check training results:

```bash
# Check logs
ls logs/

# Check saved model
ls data/06_models/

# View training metrics (if you have tensorboard)
tensorboard --logdir logs/
```

---

## 📚 **Next Steps**

After training CompGCN + ComplEx:

1. **Generate Explanations**:
   ```bash
   kedro run --pipeline=explanation
   ```

2. **Analyze Results**:
   ```python
   import pickle
   model = pickle.load(open('data/06_models/trained_model.pkl', 'rb'))
   print(model.keys())
   ```

3. **Try Different Configurations**:
   - Compare ComplEx vs RotatE
   - Test different composition functions
   - Tune hyperparameters

---

## 💡 **Tips**

- **Start with default**: CompGCN + ComplEx is a great starting point
- **Use command-line overrides**: Test configurations quickly without editing files
- **Compare models**: Run multiple configurations and compare results
- **Check GPU memory**: Use `nvidia-smi` to monitor GPU usage
- **Save configs**: Keep track of what works best for your data

---

## 🆘 **Troubleshooting**

### **Error: Model type not recognized**

Check spelling in `parameters.yml`:
```yaml
model_type: "compgcn"  # NOT "CompGCN" or "comp_gcn"
decoder_type: "complex"  # NOT "ComplEx" or "complex_"
```

### **Error: CUDA out of memory**

Reduce batch size or embedding dimension:
```bash
kedro run --pipeline=training --params=training.batch_size:512,model.embedding_dim:128
```

### **Poor performance**

Try different decoder:
```bash
# Try RotatE instead of ComplEx
kedro run --pipeline=training --params=model.decoder_type:rotate

# Try different composition
kedro run --pipeline=training --params=model.comp_fn:mult
```

---

**Summary**: Your pipeline is already configured for **CompGCN + ComplEx** (the recommended configuration). Just run `kedro run --pipeline=training` to start training! 🚀
