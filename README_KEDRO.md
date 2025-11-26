# GNN Explainer - Kedro Pipeline Implementation

This directory contains a Kedro-based implementation of the GNN Explainer pipeline for knowledge graph link prediction and explanation.

## 🏗️ Project Structure

```
gnnexplain/
├── conf/
│   └── base/
│       ├── catalog.yml          # Data catalog configuration
│       └── parameters.yml       # Pipeline parameters
├── data/                        # Data storage (Kedro convention)
│   ├── 01_raw/                 # Raw input data (triples, dicts, etc.)
│   ├── 02_intermediate/        # Intermediate processed data
│   ├── 06_models/              # Trained models
│   ├── 07_model_output/        # Model predictions and metrics
│   └── 08_reporting/           # Visualizations and reports
├── src/
│   └── gnn_explainer/
│       ├── __init__.py
│       ├── __main__.py
│       ├── settings.py
│       ├── pipeline_registry.py  # Register all pipelines
│       └── pipelines/
│           ├── data_preparation/  # Load and prepare KG data
│           │   ├── nodes.py
│           │   └── pipeline.py
│           ├── training/          # Train RGCN-DistMult model
│           │   ├── model.py
│           │   ├── nodes.py
│           │   └── pipeline.py
│           ├── evaluation/        # TODO: Evaluate model (MRR, Hit@K)
│           ├── explanation/       # TODO: Generate explanations
│           ├── metrics/           # TODO: Visualizations
│           └── utils/             # Shared utilities
│               ├── data_utils.py
│               └── prefix_filter.py
├── pyproject.toml              # Project dependencies
└── README_KEDRO.md            # This file
```

## 🚀 Quick Start

### 1. Install Dependencies

```bash
cd /Users/jchung/Documents/RENCI/everycure/experiments/Influence_estimate/gnnexplain

# Install in development mode
pip install -e .
```

### 2. Prepare Data

Place your data files in `data/01_raw/`:
- `robo_train.txt` - Training triples
- `robo_val.txt` - Validation triples
- `robo_test.txt` - Test triples
- `node_dict` - Node ID mappings
- `rel_dict` - Relation ID mappings
- `edge_map.json` - Edge/predicate names
- `id_to_name.map` - Human-readable node names

### 3. Run the Pipeline

```bash
# Run the full pipeline (currently: data prep + training)
kedro run

# Run specific pipelines
kedro run --pipeline=data_prep
kedro run --pipeline=training
kedro run --pipeline=data_and_train

# Override parameters
kedro run --params=model.embedding_dim:256,training.num_epochs:200
```

### 4. Visualize the Pipeline

```bash
# Launch Kedro-Viz for interactive visualization
kedro viz
```

This will open a web interface showing the pipeline DAG, data catalog, and execution timeline.

## 📊 Pipeline Overview

### Implemented Pipelines

#### 1. Data Preparation (`data_prep`)

**Purpose**: Load and prepare knowledge graph data for training

**Nodes**:
- `load_triple_files_node` - Load train/val/test file paths
- `load_dictionaries_node` - Load node and relation dictionaries
- `load_triples_node` - Parse triple files into tensors
- `create_kg_node` - Combine into knowledge graph structure
- `convert_to_pyg_node` - Convert to PyTorch Geometric format
- `generate_neg_samples_node` - Generate negative samples for evaluation

**Outputs**:
- `knowledge_graph` - Complete KG with dictionaries
- `pyg_data` - PyG-formatted graph data
- `negative_samples` - Negative samples for evaluation

#### 2. Training (`training`)

**Purpose**: Train RGCN-DistMult model for link prediction

**Nodes**:
- `train_rgcn_distmult` - Full training loop with early stopping

**Model Architecture**:
- **Encoder**: RGCN (Relational Graph Convolutional Network)
  - Multi-layer message passing
  - Basis decomposition for parameter efficiency
- **Decoder**: DistMult bilinear scoring

**Outputs**:
- `trained_model_artifact` - Model state dict + metadata

### Pipelines To Be Implemented

#### 3. Evaluation (`evaluation`) - TODO

**Purpose**: Compute evaluation metrics

**Planned Nodes**:
- `load_trained_model` - Load model from artifact
- `compute_binary_accuracy` - Binary classification accuracy
- `compute_filtered_mrr` - Mean Reciprocal Rank
- `compute_hits_at_k` - Hit@1, Hit@3, Hit@10
- `aggregate_evaluation_metrics` - Combine metrics into report

#### 4. Explanation (`explanation`) - TODO

**Purpose**: Generate explanations for predictions

**Planned Nodes**:
- `filter_test_triples_by_prefix` - Filter to drug-disease triples
- `generate_path_based_explanations` - BFS path finding
- `generate_perturbation_based_explanations` - Edge importance scoring

#### 5. Metrics (`metrics`) - TODO

**Purpose**: Visualizations and analysis

**Planned Nodes**:
- `create_all_visualizations` - Generate explanation graphs
- `generate_summary_report` - Create final report
- `analyze_metapaths` - Metapath pattern analysis (future)

## ⚙️ Configuration

### Parameters (`conf/base/parameters.yml`)

Key parameters you can adjust:

```yaml
# Model architecture
model:
  embedding_dim: 128      # Embedding dimensionality
  num_layers: 2           # Number of RGCN layers
  num_bases: 30           # Basis decomposition size
  dropout: 0.2            # Dropout rate

# Training
training:
  learning_rate: 0.001
  batch_size: 2048
  num_epochs: 100
  patience: 10            # Early stopping patience

# Device
device: "cuda"            # or "cpu" or "mps"
```

### Data Catalog (`conf/base/catalog.yml`)

Defines all data inputs/outputs. Key datasets:

- **Raw Data**: Triple files, dictionaries, mappings
- **Intermediate**: Processed KG data, PyG format
- **Models**: Trained model checkpoints
- **Outputs**: Metrics, explanations, visualizations

## 🔧 Development Workflow

### Adding New Nodes

1. Create node function in appropriate `nodes.py`:
```python
def my_new_node(input_data: Dict, params: Dict) -> Dict:
    # Process data
    return output_data
```

2. Add to `pipeline.py`:
```python
node(
    func=my_new_node,
    inputs=["input_dataset", "params:my_params"],
    outputs="output_dataset",
    name="my_node_name",
)
```

3. Define datasets in `catalog.yml`:
```yaml
output_dataset:
  type: pickle.PickleDataset
  filepath: data/02_intermediate/output.pkl
```

### Running Tests

```bash
# Run all tests
pytest

# Run specific test file
pytest tests/pipelines/test_data_preparation.py
```

### Code Quality

```bash
# Format code
ruff format src/

# Lint code
ruff check src/
```

## 📈 Expected Performance

**Training**:
- ~5-10 minutes per epoch (depends on graph size and GPU)
- Typical convergence: 50-100 epochs

**Metrics** (typical results):
- Accuracy: 0.75 - 0.90
- MRR: 0.30 - 0.60
- Hit@10: 0.50 - 0.80

**Explanation Speed** (GPU):
- Path-based: ~0.1 sec/triple
- Perturbation (fast): ~5-10 sec/triple

## 🐛 Troubleshooting

### Common Issues

**Issue**: CUDA out of memory
```bash
# Solution: Reduce batch size
kedro run --params=training.batch_size:1024
```

**Issue**: Module not found
```bash
# Solution: Reinstall in dev mode
pip install -e .
```

**Issue**: Data not found
```bash
# Solution: Check data paths in parameters.yml
# Ensure data files exist in data/01_raw/
```

## 📚 Next Steps

1. ✅ **Completed**: Data preparation and training pipelines
2. 🔄 **In Progress**: Evaluation pipeline
3. ⏳ **TODO**: Explanation pipeline
4. ⏳ **TODO**: Metrics and visualization pipeline
5. ⏳ **TODO**: End-to-end testing
6. ⏳ **TODO**: Documentation and examples

## 🔗 Useful Links

- [Kedro Documentation](https://docs.kedro.org/)
- [PyTorch Geometric](https://pytorch-geometric.readthedocs.io/)
- [Original GNN Explainer README](README.md)

## 💡 Tips

- Use `kedro viz` to visualize the pipeline structure
- Use `kedro catalog list` to see all available datasets
- Use `kedro pipeline list` to see all registered pipelines
- Use `--params` flag to override parameters without editing YAML files
- Check `data/` folder for intermediate outputs for debugging
