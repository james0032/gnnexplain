# Migration Guide: Converting GNN Explainer to Kedro

This guide explains how the original scripts have been converted into a Kedro pipeline structure.

## 📋 Summary of Changes

### Original Structure
```
gnnexplain/
├── src/
│   ├── cl_model.py              # Training script
│   ├── cl_eval.py               # Evaluation & explanation
│   ├── explainers.py            # Explainer implementations
│   ├── visualize_explanation.py # Visualization
│   ├── utils.py                 # Utilities
│   └── triple_filter_prefix.py  # Prefix filtering
└── data/                        # Unstructured data
```

### New Kedro Structure
```
gnnexplain/
├── conf/base/
│   ├── catalog.yml              # Data catalog
│   └── parameters.yml           # All parameters
├── data/                        # Kedro data layers
│   ├── 01_raw/
│   ├── 02_intermediate/
│   ├── 06_models/
│   ├── 07_model_output/
│   └── 08_reporting/
└── src/gnn_explainer/
    ├── pipeline_registry.py
    └── pipelines/
        ├── data_preparation/
        ├── training/
        ├── evaluation/          # TODO
        ├── explanation/         # TODO
        ├── metrics/             # TODO
        └── utils/
```

## 🔄 Code Migration Mapping

### 1. Data Loading (cl_model.py → data_preparation/)

| Original | Kedro Pipeline |
|----------|----------------|
| `KGDataLoader` class | `data_preparation/nodes.py` |
| `load_dict()` | `load_dictionaries()` node |
| `load_triples()` | `load_triples_from_files()` node |
| `create_pyg_data()` | `convert_to_pyg_format()` node |

**Benefits**:
- ✅ Modular, testable functions
- ✅ Automatic caching of intermediate results
- ✅ Configuration via YAML instead of CLI args

### 2. Model Definition (cl_model.py → training/)

| Original | Kedro Pipeline |
|----------|----------------|
| `DistMult` class | `training/model.py` |
| `RGCNDistMultModel` class | `training/model.py` |
| `train_epoch()` | Integrated into `train_model()` node |
| `evaluate()` | → `evaluation/nodes.py` (TODO) |

**Benefits**:
- ✅ Clean separation of model architecture from training logic
- ✅ Model artifacts stored with metadata
- ✅ Easy to swap models or architectures

### 3. Utilities (utils.py → pipelines/utils/)

| Original | Kedro Pipeline |
|----------|----------------|
| `utils.py` | `pipelines/utils/data_utils.py` |
| `triple_filter_prefix.py` | `pipelines/utils/prefix_filter.py` |

**Benefits**:
- ✅ Organized into logical modules
- ✅ Reusable across pipelines
- ✅ Proper Python package structure

### 4. Explainers (explainers.py → explanation/)

| Original | Kedro Status |
|----------|--------------|
| `link_prediction_explainer()` | → `explanation/perturbation_explainer.py` (TODO) |
| `simple_path_explanation()` | → `explanation/path_explainer.py` (TODO) |

**Planned Structure**:
```python
# explanation/nodes.py
def generate_path_based_explanations(...) -> List[Dict]:
    # Uses path_explainer.py

def generate_perturbation_based_explanations(...) -> List[Dict]:
    # Uses perturbation_explainer.py
```

### 5. Visualization (visualize_explanation.py → metrics/)

| Original | Kedro Status |
|----------|--------------|
| `visualize_explanation()` | → `metrics/visualization.py` (TODO) |
| `visualize_simple_explanation()` | → `metrics/visualization.py` (TODO) |

## 🎯 Parameter Configuration Migration

### Original CLI Arguments
```bash
python src/cl_model.py \
    --train_file data/robo_train.txt \
    --embedding_dim 128 \
    --num_layers 2 \
    --num_epochs 100 \
    --learning_rate 0.001
```

### New Kedro Configuration
```yaml
# conf/base/parameters.yml
data:
  train_file: "data/01_raw/robo_train.txt"

model:
  embedding_dim: 128
  num_layers: 2

training:
  num_epochs: 100
  learning_rate: 0.001
```

```bash
# Run with default parameters
kedro run

# Override parameters
kedro run --params=model.embedding_dim:256,training.num_epochs:200
```

## 🔀 Workflow Comparison

### Original Workflow
```bash
# Step 1: Train
python src/cl_model.py --train_file ... --num_epochs 100

# Step 2: Evaluate
python src/cl_eval.py --model_path best_model.pt

# Step 3: Generate explanations
python src/cl_eval.py --model_path best_model.pt --num_explain 20
```

### New Kedro Workflow
```bash
# Run full pipeline (when complete)
kedro run

# Or run specific stages
kedro run --pipeline=data_prep
kedro run --pipeline=training
kedro run --pipeline=evaluation      # TODO
kedro run --pipeline=explanation     # TODO
kedro run --pipeline=metrics         # TODO

# Combined workflows
kedro run --pipeline=data_and_train
kedro run --pipeline=train_eval      # TODO
kedro run --pipeline=explain_viz     # TODO
```

## 📊 Data Flow Improvements

### Original: Manual Data Passing
```python
# Load data
loader = KGDataLoader(...)
train_triples = loader.load_triples('train.txt')

# Train model
model = RGCNDistMultModel(...)
train_epoch(model, train_triples, ...)

# Save manually
torch.save(model.state_dict(), 'best_model.pt')

# Load manually for evaluation
model.load_state_dict(torch.load('best_model.pt'))
```

### New Kedro: Automatic Data Management
```python
# Kedro automatically:
# 1. Loads inputs from catalog
# 2. Passes between nodes
# 3. Caches intermediate results
# 4. Saves outputs to catalog

# No manual file I/O needed!
# Just define node functions and pipeline
```

## ✅ Migration Checklist

### Completed ✅
- [x] Project structure and configuration
- [x] Data preparation pipeline
- [x] Training pipeline
- [x] Utility modules migrated
- [x] Parameters externalized to YAML
- [x] Data catalog configuration

### In Progress 🔄
- [ ] Evaluation pipeline nodes
- [ ] Explanation pipeline nodes
- [ ] Metrics/visualization pipeline

### TODO ⏳
- [ ] Unit tests for each pipeline
- [ ] Integration tests
- [ ] Documentation for each node
- [ ] Example notebooks
- [ ] CI/CD pipeline

## 🚀 Running the Migration

### 1. Install Kedro Project
```bash
cd /Users/jchung/Documents/RENCI/everycure/experiments/Influence_estimate/gnnexplain
pip install -e .
```

### 2. Move Data to Kedro Structure
```bash
# If you have existing data files, move them:
mv data/* data/01_raw/  # Or symlink if data is large

# Expected structure:
# data/01_raw/
# ├── robo_train.txt
# ├── robo_val.txt
# ├── robo_test.txt
# ├── node_dict
# ├── rel_dict
# ├── edge_map.json
# └── id_to_name.map
```

### 3. Test Data Preparation
```bash
kedro run --pipeline=data_prep
```

### 4. Test Training
```bash
kedro run --pipeline=training

# Or run both:
kedro run --pipeline=data_and_train
```

### 5. Monitor Progress
```bash
# Launch Kedro-Viz
kedro viz

# This opens a web interface showing:
# - Pipeline DAG
# - Node execution status
# - Data lineage
# - Performance metrics
```

## 🔧 Extending the Pipeline

### Adding a New Pipeline

1. Create pipeline directory:
```bash
mkdir -p src/gnn_explainer/pipelines/my_pipeline
touch src/gnn_explainer/pipelines/my_pipeline/__init__.py
touch src/gnn_explainer/pipelines/my_pipeline/nodes.py
touch src/gnn_explainer/pipelines/my_pipeline/pipeline.py
```

2. Define nodes in `nodes.py`:
```python
def my_node(input_data: Dict, params: Dict) -> Dict:
    # Process data
    return output_data
```

3. Create pipeline in `pipeline.py`:
```python
from kedro.pipeline import Pipeline, pipeline, node
from .nodes import my_node

def create_pipeline(**kwargs) -> Pipeline:
    return pipeline([
        node(
            func=my_node,
            inputs=["input_dataset", "params:my_params"],
            outputs="output_dataset",
            name="my_node_name",
        ),
    ])
```

4. Register in `pipeline_registry.py`:
```python
from gnn_explainer.pipelines import my_pipeline

def register_pipelines() -> Dict[str, Pipeline]:
    return {
        "my_pipeline": my_pipeline.create_pipeline(),
        ...
    }
```

## 📖 Additional Resources

- **Kedro Documentation**: https://docs.kedro.org/
- **Original README**: [README.md](README.md)
- **Kedro README**: [README_KEDRO.md](README_KEDRO.md)
- **Configuration Guide**: See `conf/base/parameters.yml`
- **Data Catalog Guide**: See `conf/base/catalog.yml`

## ❓ FAQ

**Q: Can I still use the original scripts?**
A: Yes! The original scripts in `src/` are preserved. The Kedro implementation is in `src/gnn_explainer/`.

**Q: How do I debug a failed node?**
A: Use `kedro run --pipeline=my_pipeline --to-nodes=failing_node` to run up to the failing node, then inspect the data in `data/` folders.

**Q: How do I change parameters without editing YAML?**
A: Use `--params` flag: `kedro run --params=model.embedding_dim:256`

**Q: How do I add logging?**
A: Use Python's built-in logging in your nodes. Kedro will capture and display it.

**Q: Can I run nodes in parallel?**
A: Yes! Kedro can run independent nodes in parallel. Use `kedro run --runner=ParallelRunner`.

## 🎉 Benefits of Migration

1. **Reproducibility**: All parameters tracked in version control
2. **Modularity**: Each stage is independent and testable
3. **Caching**: Kedro caches intermediate results automatically
4. **Visualization**: Pipeline DAG visualization with Kedro-Viz
5. **Scalability**: Easy to parallelize and distribute
6. **Maintainability**: Clear structure and separation of concerns
7. **Collaboration**: Easy for team members to understand and contribute
8. **Production-Ready**: Built on industry-standard framework
