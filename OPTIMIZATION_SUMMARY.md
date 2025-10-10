# GPU Optimization Summary for link_prediction_explainer

## Problem
The original `link_prediction_explainer` function was running very slowly due to sequential edge perturbation analysis. For each edge in the k-hop subgraph, it:
1. Created a mask to remove that edge
2. Performed a full forward pass through the model
3. Computed importance as the score change

This required **N forward passes** for N edges, processed sequentially on CPU/GPU.

## Solution
Implemented GPU-accelerated batch processing with two modes:

### **Fast Mode (Default)** - GPU Batched Processing
- **Batching**: Process 50 edges at a time instead of one-by-one
- **GPU Mask Creation**: Create all masks on GPU using vectorized operations
- **Parallel Execution**: All mask operations done in parallel on GPU
- **Progress Tracking**: Shows progress every 100 edges

**Key Improvements:**
```python
# OLD: Sequential processing
for edge_idx in original_edge_indices:
    mask = torch.ones(...)  # CPU operation
    mask[edge_idx] = False
    # ... forward pass ...

# NEW: Batched GPU processing
for batch_start in range(0, num_edges, batch_size):
    batch_masks = torch.ones((batch_size, num_edges), device=device)  # GPU
    batch_masks[row_idx, batch_indices] = False  # Vectorized on GPU
    # ... batched forward passes ...
```

**Expected Speedup:** 3-5x faster than original implementation

### **Standard Mode** - Original Implementation
- Available via `--use_slow_explainer` flag
- Processes each edge individually
- More accurate but significantly slower

## Usage

### In cl_eval.py:
```bash
# Use fast mode (default)
python cl_eval.py --model_path best_model.pt

# Use standard mode
python cl_eval.py --model_path best_model.pt --use_slow_explainer
```

### In cl_model.py:
The `explain_triples` function in cl_model.py also supports the optimization via:
```python
explanations = explain_triples(
    model, edge_index, edge_type, test_triples,
    node_dict, rel_dict, device,
    use_perturbation=True,  # Enable perturbation-based explainer
    # Fast mode is default in explainers.py
)
```

## Technical Details

### Optimizations Applied:
1. **Vectorized Mask Creation**: Uses `torch.arange` and advanced indexing
2. **GPU Memory Management**: Batch size of 50 to prevent OOM
3. **Reduced CPU-GPU Transfers**: All operations stay on GPU
4. **Progress Monitoring**: Clear feedback every 100 edges

### Memory Considerations:
- Batch size set to 50 edges (configurable in code)
- For very large graphs, may need to reduce batch size
- Mask tensor size: `(batch_size, num_total_edges)`

### Code Location:
- **Explainer Function**: [explainers.py:8](src/explainers.py#L8-L140)
- **Integration**: [cl_eval.py:440](src/cl_eval.py#L440-L453)
- **Model Integration**: [cl_model.py:386](src/cl_model.py#L386-L557)

## Benchmark Results (Expected)

For a typical k=2 hop subgraph with ~500 edges:

| Mode | Time | Speedup |
|------|------|---------|
| Original (Sequential) | ~25 seconds | 1x |
| Fast (GPU Batched) | ~5-8 seconds | 3-5x |

*Note: Actual speedup depends on GPU model, graph size, and batch size.*

## Future Improvements

Potential further optimizations:
1. **Fully Vectorized Encoding**: If model supports batch graph encoding
2. **Sparse Matrix Operations**: Use sparse tensors for very large graphs
3. **Multi-GPU Support**: Distribute batches across multiple GPUs
4. **Gradient-based Importance**: Use gradient computation instead of perturbation
5. **Cached Embeddings**: Cache and reuse node embeddings where possible

## Related Files Modified

1. `explainers.py` - Added fast mode with GPU batching
2. `cl_eval.py` - Added `--use_fast_explainer` argument
3. `cl_model.py` - Updated to use optimized explainer (already compatible)
