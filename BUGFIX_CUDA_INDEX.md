# CUDA Index Out of Bounds Bug Fix

## Problem Description

### Error Message
```
/opt/pytorch/pytorch/aten/src/ATen/native/cuda/IndexKernel.cu:92: operator(): block: [0,0,0], thread: [0,0,0]
Assertion `-sizes[i] <= index && index < sizes[i] && "index out of bounds"` failed.
✗ Error: CUDA error: device-side assert triggered
```

### When It Occurred
- **First explanation**: Works fine
- **Second explanation onwards**: CUDA assertion failure
- **Trigger**: After `visualize_explanation` runs, subsequent explanations with <2000 edges fail

### Root Cause
Multiple tensor device mismatch and indexing issues in `link_prediction_explainer`:

1. **Improper Tensor Indexing** (Line 89-90)
   - Using Python slice notation on tensor values: `triple[0]:triple[0]+1`
   - `triple[0]` is a tensor, not an int
   - Results in invalid slice indices

2. **CPU/GPU Tensor Mismatch** (Line 81)
   - `original_edge_indices` from `k_hop_subgraph` is on CPU
   - `batch_masks` is on GPU
   - Mixing them causes CUDA indexing errors

3. **NumPy Array Indexing with Tensors** (Line 136)
   - Using tensor as numpy array index: `full_edge_mask[edge_idx]`
   - `edge_idx` is a CUDA tensor, numpy expects int

## Solutions Applied

### Fix 1: Proper Tensor Indexing
**Before:**
```python
masked_head_emb = masked_node_emb[triple[0]:triple[0]+1]
masked_tail_emb = masked_node_emb[triple[2]:triple[2]+1]
score = model.decoder(masked_head_emb, masked_tail_emb, triple[1:2].to(device))
```

**After:**
```python
# Use already extracted integer values
head_idx_val = head_idx  # Already .item() extracted above
tail_idx_val = tail_idx  # Already .item() extracted above
masked_head_emb = masked_node_emb[head_idx_val:head_idx_val+1]
masked_tail_emb = masked_node_emb[tail_idx_val:tail_idx_val+1]

# Proper tensor creation for relation index
rel_idx_tensor = torch.tensor([rel_idx], device=device)
score = model.decoder(masked_head_emb, masked_tail_emb, rel_idx_tensor)
```

### Fix 2: Device Consistency for k_hop_subgraph
**Before:**
```python
nodes_of_interest = torch.tensor([head_idx, tail_idx])
subset, sub_edge_index, mapping, edge_mask_sub = k_hop_subgraph(
    nodes_of_interest, k_hops, edge_index, ...
)
original_edge_indices = torch.where(edge_mask_sub)[0]
```

**After:**
```python
# Ensure proper CPU usage for k_hop_subgraph
nodes_of_interest = torch.tensor([head_idx, tail_idx], dtype=torch.long)

# Move to CPU for k_hop_subgraph (it works on CPU)
edge_index_cpu = edge_index.cpu()
edge_type_cpu = edge_type.cpu()

subset, sub_edge_index, mapping, edge_mask_sub = k_hop_subgraph(
    nodes_of_interest, k_hops, edge_index_cpu, ...
)
original_edge_indices = torch.where(edge_mask_sub)[0]  # CPU tensor
```

### Fix 3: GPU Batch Index Handling
**Before:**
```python
batch_indices = original_edge_indices[batch_start:batch_end]
batch_masks = torch.ones((...), device=device)  # GPU
row_idx = torch.arange(current_batch_size, device=device)  # GPU
batch_masks[row_idx, batch_indices] = False  # ERROR: batch_indices is CPU!
```

**After:**
```python
batch_indices = original_edge_indices[batch_start:batch_end]  # CPU tensor
batch_masks = torch.ones((...), device=device)  # GPU

# FIX: Move indices to GPU before indexing
batch_indices_gpu = batch_indices.to(device)
row_idx = torch.arange(current_batch_size, device=device)
batch_masks[row_idx, batch_indices_gpu] = False  # Both on GPU now!
```

### Fix 4: NumPy Indexing with Tensors
**Before:**
```python
full_edge_mask = np.zeros(edge_index.shape[1])
for i, edge_idx in enumerate(original_edge_indices):
    full_edge_mask[edge_idx] = importance_scores[i]  # ERROR: edge_idx is tensor!
```

**After:**
```python
full_edge_mask = np.zeros(edge_index.shape[1])
for i, edge_idx in enumerate(original_edge_indices):
    # Convert tensor to int to avoid index errors
    idx = edge_idx.item() if torch.is_tensor(edge_idx) else int(edge_idx)
    full_edge_mask[idx] = importance_scores[i]
```

## Testing

### How to Verify the Fix
```bash
# Run with multiple explanations to trigger the bug
python src/cl_eval.py \
    --model_path best_model.pt \
    --num_explain 5 \
    --use_fast_explainer
```

**Expected Behavior:**
- All 5 explanations complete successfully
- No CUDA assertion errors
- Visualizations are generated correctly

### Debug Mode (if issues persist)
```bash
# Enable CUDA launch blocking for detailed error tracking
CUDA_LAUNCH_BLOCKING=1 python src/cl_eval.py \
    --model_path best_model.pt \
    --num_explain 5
```

## Files Modified

1. **[explainers.py](src/explainers.py)** - Lines 35-142
   - Fixed tensor indexing (lines 89-95)
   - Fixed CPU/GPU device handling (lines 36-52, 80-82)
   - Fixed numpy array indexing (lines 139-142)

## Prevention Guidelines

### Best Practices for CUDA Tensors

1. **Always Extract Scalar Values Early**
   ```python
   # Good
   idx = tensor_value.item()
   result = array[idx:idx+1]

   # Bad
   result = array[tensor_value:tensor_value+1]
   ```

2. **Maintain Device Consistency**
   ```python
   # Good
   tensor_gpu = tensor_cpu.to(device)
   result = gpu_tensor[tensor_gpu]

   # Bad
   result = gpu_tensor[tensor_cpu]  # Device mismatch!
   ```

3. **Check Tensor Types Before Indexing**
   ```python
   # Good
   idx = val.item() if torch.is_tensor(val) else int(val)
   numpy_array[idx] = value

   # Bad
   numpy_array[tensor_val] = value  # Type mismatch!
   ```

4. **Know Your Library's Device Requirements**
   - `k_hop_subgraph`: Works on CPU
   - `model.encode()`: Works on GPU
   - Explicitly move tensors when crossing boundaries

## Related Issues

- **PyTorch Issue**: CUDA tensors cannot be used directly as numpy indices
- **torch_geometric**: `k_hop_subgraph` operates on CPU tensors
- **Assertion Failures**: Often occur after first successful run due to corrupted state

## Performance Impact

**No performance degradation** - The fixes only:
- Add proper device transfers (minimal overhead)
- Use correct indexing methods
- Maintain same computational complexity

The GPU acceleration optimizations remain fully effective.

## Verification Checklist

- [x] Fixed tensor slicing with scalar values
- [x] Ensured CPU/GPU device consistency
- [x] Added tensor-to-int conversion for numpy indexing
- [x] Tested with multiple consecutive explanations
- [x] Verified GPU batch processing still works
- [x] No performance regression

## Additional Notes

### Why This Wasn't Caught Earlier
- First explanation works because tensors haven't been reused yet
- Error only appears when graph operations are repeated
- CUDA assertions are asynchronous - error shows up later in code

### Debug Commands
```bash
# Minimal reproduction
python src/cl_eval.py --model_path best_model.pt --num_explain 2

# Verbose CUDA debugging
CUDA_LAUNCH_BLOCKING=1 TORCH_USE_CUDA_DSA=1 python src/cl_eval.py \
    --model_path best_model.pt --num_explain 3
```

---

**Bug Status**: ✅ FIXED
**Tested**: Multiple consecutive explanations work correctly
**Performance**: No degradation, GPU optimization intact
