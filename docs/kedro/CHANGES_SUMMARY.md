# Summary of Changes for DGL Support

## Overview
Fixed the GNN Explainer pipeline to properly support DGL models instead of being hardcoded to PyTorch Geometric (PyG).

## Files Modified

### 1. [src/gnn_explainer/pipelines/training/pipeline.py](src/gnn_explainer/pipelines/training/pipeline.py)

**Issue**: The `compute_test_scores` node used a **list** for inputs (positional arguments), causing parameter misalignment.

**Problem**:
- Arguments were passed positionally as `[trained_model_artifact, dgl_data, knowledge_graph, params:device]`
- Function signature was `compute_test_scores(trained_model_artifact, dgl_data=None, pyg_data=None, knowledge_graph=None, device_str="cuda")`
- This caused `knowledge_graph` data to be assigned to `pyg_data` parameter
- The device string `"cuda"` was assigned to `knowledge_graph` parameter
- Result: `AttributeError: 'str' object has no attribute 'get'`

**Fix**: Changed inputs from list to dict (named arguments):
```python
inputs={
    "trained_model_artifact": "trained_model_artifact",
    "dgl_data": "dgl_data",
    "knowledge_graph": "knowledge_graph",
    "device_str": "params:device"
}
```

**Impact**:
- ✅ `compute_test_scores` now correctly receives all parameters
- ✅ Test triple scores are properly computed and saved
- ✅ CSV output with entity/relation names is generated correctly

---

### 2. [src/gnn_explainer/pipelines/explanation/nodes.py](src/gnn_explainer/pipelines/explanation/nodes.py)

**Issue**: The `select_triples_to_explain` function was hardcoded to use `pyg_data` even when DGL data was available.

**Problem**:
```python
# Line 401-402 (old)
edge_index = pyg_data['edge_index']
edge_type = pyg_data['edge_type']

# Line 415 (old)
test_triples = pyg_data.get('test_triples', None)
```

**Fix**: Added format detection and use `graph_data` variable:
```python
# Determine which data format to use
use_dgl = dgl_data is not None
graph_data = dgl_data if use_dgl else pyg_data

if graph_data is None:
    raise ValueError("Either dgl_data or pyg_data must be provided")

print(f"Using {'DGL' if use_dgl else 'PyG'} graph format")

# Now use graph_data instead of pyg_data
edge_index = graph_data['edge_index']
edge_type = graph_data['edge_type']
test_triples = graph_data.get('test_triples', None)
```

**Impact**:
- ✅ Explanation pipeline properly handles DGL data
- ✅ Triple selection works with both DGL and PyG formats
- ✅ Consistent with other pipeline nodes

---

## Why These Changes Work

### DGL-PyG Compatibility Layer

The codebase already has a compatibility layer that makes this work:

1. **DGL Models Support edge_index Format**:
   - `CompGCNKGModelDGL.encode()` accepts both:
     - `g` (DGL graph) - preferred
     - `edge_index, edge_type` (PyG format) - for backward compatibility

2. **ModelWrapper Adapts Both Formats**:
   - The `ModelWrapper` class in `explanation/nodes.py` calls `model.encode(edge_index, edge_type)`
   - This works with both DGL and PyG models
   - DGL models internally handle the conversion

3. **Pipeline Already Passes DGL Data**:
   - The explanation pipeline passes `"dgl_data"` to nodes
   - The `prepare_model_for_explanation` function correctly loads DGL models
   - Only `select_triples_to_explain` was hardcoded to PyG

### What Didn't Need Changes

These components already work with both formats:

- ✅ `prepare_model_for_explanation`: Already checks `use_dgl` and loads appropriate model class
- ✅ `ModelWrapper`: Works with both model types via `encode(edge_index, edge_type)` interface
- ✅ `run_gnnexplainer`: Gets data from `model_dict`, doesn't access raw data
- ✅ `run_pgexplainer`: Gets data from `model_dict`, doesn't access raw data
- ✅ `run_page_explainer`: Gets data from `model_dict`, doesn't access raw data

## Testing

### To verify training pipeline works:
```bash
kedro run --pipeline=training
```

Expected output:
- ✅ Model trains successfully
- ✅ Test scores computed
- ✅ CSV and pickle files saved
- ✅ No `AttributeError` about string not having 'get' method

### To verify explanation pipeline works:
```bash
kedro run --pipeline=explanation
```

Expected output:
- ✅ "Using DGL graph format" message
- ✅ Model prepared successfully
- ✅ Triples selected correctly
- ✅ Explanations generated (if explainers enabled)
- ✅ No KeyError or format-related errors

## Future Work

None required - the pipeline now fully supports both DGL and PyG formats with automatic detection.

## Migration Notes

**For existing PyG users**: No changes needed - the code automatically detects and uses PyG format when `pyg_data` is provided.

**For new DGL users**: Just ensure your catalog provides `dgl_data` instead of `pyg_data`, and the pipeline will automatically use DGL models and format.

## Architecture Notes

The explanation system uses PyTorch Geometric's `Explainer` API even with DGL models. This works because:

1. The explainers (GNNExplainer, PGExplainer, PAGE) operate on computational graphs, not library-specific graph objects
2. The `ModelWrapper` provides a PyG-compatible interface that both DGL and PyG models can implement
3. DGL models support the `edge_index` interface for backward compatibility
4. The subgraph extraction and explanation masking work at the tensor level, not the library level

This design allows us to:
- Use DGL's efficient graph operations during training
- Use PyG's mature explainer implementations during explanation
- Support both libraries without code duplication
