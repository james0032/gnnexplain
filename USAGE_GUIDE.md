# CL_EVAL.PY Usage Guide

## Overview
`cl_eval.py` is a comprehensive evaluation script that:
1. Loads a trained RGCN-DistMult model
2. Evaluates on test triples with multiple metrics
3. Generates explanations using all available explainers

## Quick Start

### Basic Evaluation
```bash
python src/cl_eval.py \
    --model_path best_model.pt \
    --test_file robo_test.txt
```

### Complete Workflow
```bash
python src/cl_eval.py \
    --model_path best_model.pt \
    --node_dict node_dict \
    --rel_dict rel_dict \
    --train_file robo_train.txt \
    --val_file robo_val.txt \
    --test_file robo_test.txt \
    --edge_map edge_map.json \
    --id_to_name_map id_to_name.map \
    --num_explain 20
```

## Evaluation Metrics

### Accuracy
Binary classification accuracy with negative sampling:
```bash
python src/cl_eval.py --model_path best_model.pt
# Output: accuracy: 0.8542
```

### MRR (Mean Reciprocal Rank)
Filtered ranking metric:
```bash
python src/cl_eval.py --model_path best_model.pt --compute_mrr
# Output: mrr: 0.4521
```

### Hit@K
Hit rate at different K values:
```bash
python src/cl_eval.py \
    --model_path best_model.pt \
    --compute_hits \
    --hit_k_values 1 3 10 20
# Output:
#   hit@1: 0.3214
#   hit@3: 0.5123
#   hit@10: 0.7234
#   hit@20: 0.8456
```

### Metrics Only (No Explanations)
```bash
python src/cl_eval.py \
    --model_path best_model.pt \
    --skip_explanation
```

## Explanation Generation

### Both Explainers (Default)
```bash
python src/cl_eval.py \
    --model_path best_model.pt \
    --num_explain 10
```

This generates:
- Path-based explanations (BFS path finding)
- Perturbation-based explanations (edge importance)

### Fast GPU Mode (Default)
```bash
python src/cl_eval.py \
    --model_path best_model.pt \
    --num_explain 20
```

### Standard Mode (Slower but Accurate)
```bash
python src/cl_eval.py \
    --model_path best_model.pt \
    --num_explain 20 \
    --use_slow_explainer
```

### Explanation Parameters
```bash
python src/cl_eval.py \
    --model_path best_model.pt \
    --num_explain 15 \
    --explanation_khops 3 \
    --top_k_edges 30 \
    --max_edges 3000
```

Parameters:
- `--explanation_khops`: Neighborhood hops for subgraph (default: 2)
- `--top_k_edges`: Max edges to show in visualization (default: 20)
- `--max_edges`: Skip if subgraph > N edges (default: 2000)

## Prefix Filtering

### Drug-Disease Triples Only
```bash
python src/cl_eval.py \
    --model_path best_model.pt \
    --subject_prefixes CHEBI UNII PUBCHEM.COMPOUND \
    --object_prefixes MONDO
```

### Show Available Prefixes
```bash
python src/cl_eval.py \
    --model_path best_model.pt \
    --show_prefix_inventory
```

### Disable Filtering
```bash
python src/cl_eval.py \
    --model_path best_model.pt \
    --no_prefix_filter
```

## Model Configuration

### Match Training Hyperparameters
```bash
python src/cl_eval.py \
    --model_path best_model.pt \
    --embedding_dim 256 \
    --num_layers 3 \
    --num_bases 40 \
    --dropout 0.3
```

**Important:** These must match your training configuration!

### Default Values
- `--embedding_dim 128`
- `--num_layers 2`
- `--num_bases 30`
- `--dropout 0.2`

## Output Files

### Evaluation Metrics
```bash
python src/cl_eval.py \
    --model_path best_model.pt \
    --metrics_save_path my_metrics.pkl
```

Output: `my_metrics.pkl` containing:
```python
{
    'accuracy': 0.8542,
    'mrr': 0.4521,
    'hit@1': 0.3214,
    'hit@3': 0.5123,
    'hit@10': 0.7234
}
```

### Explanations
```bash
python src/cl_eval.py \
    --model_path best_model.pt \
    --explanation_save_path my_explanations.pkl \
    --explanation_dir my_visualizations/
```

Output structure:
```
my_explanations.pkl          # Pickled explanation data
my_visualizations/
├── path_explanation_1.png   # Path-based viz
├── path_explanation_2.png
├── perturbation_explanation_1.png  # Perturbation viz
└── perturbation_explanation_2.png
```

## Common Workflows

### 1. Quick Model Evaluation
```bash
# Just get the metrics
python src/cl_eval.py \
    --model_path best_model.pt \
    --skip_explanation
```

### 2. Generate Explanations for Paper
```bash
# High-quality visualizations
python src/cl_eval.py \
    --model_path best_model.pt \
    --num_explain 50 \
    --explanation_khops 2 \
    --top_k_edges 15 \
    --subject_prefixes CHEBI UNII \
    --object_prefixes MONDO \
    --explanation_dir paper_figures/
```

### 3. Full Evaluation Report
```bash
# All metrics + explanations
python src/cl_eval.py \
    --model_path best_model.pt \
    --compute_mrr \
    --compute_hits \
    --hit_k_values 1 5 10 20 50 \
    --num_explain 25 \
    --show_prefix_inventory
```

### 4. Debug Specific Triples
```bash
# Use slow mode for accuracy
python src/cl_eval.py \
    --model_path best_model.pt \
    --num_explain 5 \
    --use_slow_explainer \
    --explanation_khops 3
```

## Performance Tips

### Speed Up Evaluation
1. Use fast explainer mode (default)
2. Reduce `--num_explain`
3. Lower `--max_edges` threshold
4. Skip MRR/Hit@K for large graphs: `--no_mrr --no_hits`

### Memory Management
1. Reduce `--batch_size` if OOM
2. Lower `--explanation_khops`
3. Use `--max_edges` to skip large subgraphs

### GPU Optimization
```bash
# Ensure model and data are on GPU
CUDA_VISIBLE_DEVICES=0 python src/cl_eval.py \
    --model_path best_model.pt \
    --batch_size 2048
```

## Troubleshooting

### Model Loading Fails
```bash
# Check hyperparameters match training
python src/cl_eval.py \
    --model_path best_model.pt \
    --embedding_dim 128 \  # Must match training!
    --num_layers 2
```

### Out of Memory
```bash
# Reduce batch size and explanation parameters
python src/cl_eval.py \
    --model_path best_model.pt \
    --batch_size 512 \
    --max_edges 1000 \
    --explanation_khops 1
```

### Slow Explanations
```bash
# Use fast mode (default) or reduce samples
python src/cl_eval.py \
    --model_path best_model.pt \
    --num_explain 5 \
    --max_edges 1500
```

### No Filtered Triples Found
```bash
# Check your prefixes or disable filtering
python src/cl_eval.py \
    --model_path best_model.pt \
    --show_prefix_inventory \  # See what's available
    --no_prefix_filter         # Or disable filtering
```

## Advanced Usage

### Custom Batch Evaluation
```python
# In Python script
from cl_eval import evaluate, RGCNDistMultModel, KGDataLoader

# Load model and data
model = RGCNDistMultModel(...)
model.load_state_dict(torch.load('best_model.pt'))

# Evaluate
metrics = evaluate(
    model, edge_index, edge_type, test_triples,
    all_triples, device,
    compute_mrr=True,
    compute_hits=True,
    hit_k_values=[1, 3, 10]
)

print(metrics)
```

### Programmatic Explanation Generation
```python
from cl_eval import explain_triples_all_methods

explanations = explain_triples_all_methods(
    model, edge_index, edge_type, test_triples,
    node_dict, rel_dict, device,
    num_samples=10,
    use_fast_mode=True
)

# Access results
path_explanations = explanations['path_based']
pert_explanations = explanations['perturbation_based']
```

## Example Output

### Console Output
```
============================================================
COMPREHENSIVE MODEL EVALUATION
============================================================
Device: cuda
Model: best_model.pt

============================================================
LOADING DATA
============================================================
✓ Loaded triples:
  Train: 25000
  Val:   5000
  Test:  5000
✓ Graph statistics:
  Nodes:     15234
  Relations: 45

============================================================
EVALUATION METRICS
============================================================

[1/3] Computing Accuracy...
  ✓ Accuracy: 0.8542

[2/3] Computing Ranking Metrics...
  Building filtered candidate sets...
  Computing rankings...
    Processed 5000/5000 test triples
  ✓ MRR: 0.4521
  ✓ Hit@1: 0.3214
  ✓ Hit@3: 0.5123
  ✓ Hit@10: 0.7234

============================================================
GENERATING EXPLANATIONS FOR 10 TEST TRIPLES
============================================================

[1/10] Explaining triple:
  CHEBI:123 -> MONDO:456

[Method 1] Path-based explanation...
  ✓ Found 3 connecting paths
  ✓ Saved visualization to explanations/path_explanation_1.png

[Method 2] Perturbation-based explanation...
  Testing 234 edges in 2-hop subgraph...
  Using GPU-accelerated FAST mode (batched graph encoding)...
    Processed 200/234 edges...
  ✓ Original score: 2.3456
  ✓ Top 5 important edges:
    1. CHEBI:123 -[treats]-> MONDO:789 (importance: 1.0000)
    2. MONDO:789 -[related_to]-> MONDO:456 (importance: 0.8234)
    ...
  ✓ Saved visualization to explanations/perturbation_explanation_1.png

...

============================================================
FINAL RESULTS SUMMARY
============================================================

Test Metrics:
  accuracy    : 0.8542
  mrr         : 0.4521
  hit@1       : 0.3214
  hit@3       : 0.5123
  hit@10      : 0.7234

Explanations Generated:
  Path-based:         10
  Perturbation-based: 9

============================================================
EVALUATION COMPLETE!
============================================================
```
