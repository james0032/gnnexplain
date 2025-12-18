# Migration from PyTorch Geometric to DGL

This document tracks the migration of the GNN Explainer project from PyTorch Geometric (PyG) to Deep Graph Library (DGL).

## Motivation

- **Better Batching**: DGL provides native batching support for graphs
- **Better Performance**: More efficient message passing implementation
- **PGExplainer Batching**: DGL's PGExplainer supports batch training
- **Production Ready**: Better suited for production deployments

## Migration Status

### ✅ Completed
- [x] Migration plan documented
- [x] Requirements updated (pyproject.toml)
  - Added `dgl>=1.1.0` and `dgllife>=0.3.2`
  - Moved PyG to optional dependencies
- [x] Data preparation pipeline converted to DGL
  - `convert_to_dgl_format()` creates DGL graphs with edge types and direction
  - Backward compatible with PyG format (marked as deprecated)
- [x] CompGCN layer ported to DGL ([compgcn_layer_dgl.py](src/gnn_explainer/pipelines/training/compgcn_layer_dgl.py))
  - Message passing using DGL's `update_all()` paradigm
  - Supports all composition functions: sub, mult, corr
- [x] CompGCN encoder ported to DGL ([compgcn_encoder_dgl.py](src/gnn_explainer/pipelines/training/compgcn_encoder_dgl.py))
  - Multi-layer CompGCN with DGL graphs
  - `forward_with_edge_index()` for backward compatibility
- [x] CompGCN model wrapper ported to DGL ([kg_models_dgl.py](src/gnn_explainer/pipelines/training/kg_models_dgl.py))
  - Supports all decoders: DistMult, ComplEx, RotatE, ConvE
  - No PyG dependencies - all decoders implemented directly
- [x] Training pipeline updated for DGL support
  - `train_model()` accepts both `dgl_data` and `pyg_data`
  - Automatically selects backend based on input
  - Pipeline configured to use `dgl_data` by default
- [x] PGExplainer with native DGL batching ([pgexplainer_dgl.py](src/gnn_explainer/pipelines/explanation/pgexplainer_dgl.py))
  - Uses `dgl.nn.pytorch.explain.PGExplainer`
  - Temperature annealing for better training
  - Batch explanation generation (expected 5-10x faster)
- [x] Explanation pipeline nodes updated for DGL
  - `prepare_model_for_explanation()` supports both formats
  - `select_triples_to_explain()` supports both formats

### ⏳ Pending
- [ ] Integrate DGL PGExplainer into main explanation pipeline
- [ ] Testing DGL CompGCN training end-to-end
- [ ] Testing DGL PGExplainer with batching
- [ ] Performance benchmarking vs PyG
- [ ] Update remaining explainers (GNNExplainer, PAGE) for DGL

## Key Differences: PyG vs DGL

### Graph Representation

**PyG:**
```python
data = Data(edge_index=edge_index, edge_type=edge_type, num_nodes=num_nodes)
```

**DGL:**
```python
g = dgl.graph((src_nodes, dst_nodes), num_nodes=num_nodes)
g.edata['etype'] = edge_type
```

### Message Passing

**PyG:**
```python
class CompGCNLayer(MessagePassing):
    def message(self, x_j, edge_type):
        ...
```

**DGL:**
```python
class CompGCNLayer(nn.Module):
    def forward(self, g, node_feat, edge_feat):
        with g.local_scope():
            g.ndata['h'] = node_feat
            g.edata['e'] = edge_feat
            g.update_all(message_func, reduce_func)
            return g.ndata['h_new']
```

### Batching

**PyG:**
```python
# Manual batching with Batch.from_data_list()
batch = Batch.from_data_list([data1, data2, data3])
```

**DGL:**
```python
# Native batching
batched_graph = dgl.batch([g1, g2, g3])
```

## File Changes

### Data Preparation
- `data_preparation/nodes.py` - Convert to DGL graph format

### Training
- `training/compgcn_layer.py` - Port to DGL message passing
- `training/compgcn_encoder.py` - Update for DGL graphs
- `training/kg_models.py` - Update model wrapper
- `training/nodes.py` - Update training loop

### Explanation
- `explanation/nodes.py` - Port PGExplainer to DGL
- Add batch training support

## Backward Compatibility

The PyG version is preserved in the `pyg_version` branch for reference and rollback if needed.

## Dependencies

### Added
- `dgl>=1.1.0`
- `dgllife` (for explainability tools)

### Removed
- `torch-geometric` (kept as optional for comparison)
- `torch-scatter`
- `torch-sparse`

## Performance Expectations

- **Training Speed**: 2-3x faster with DGL batching
- **Memory Usage**: 20-30% reduction with DGL's efficient storage
- **PGExplainer Training**: 5-10x faster with batch training

## Testing Plan

1. **Unit Tests**: Verify each component independently
2. **Integration Tests**: End-to-end pipeline validation
3. **Performance Tests**: Benchmark against PyG version
4. **Correctness Tests**: Verify model outputs match PyG version

## Rollback Plan

If issues arise:
1. Switch back to `pyg_version` branch
2. Document issues encountered
3. Address issues before re-attempting migration

## Known Issues and Fixes

### Graphbolt Compatibility Error

**Issue**: On some systems (especially macOS ARM and certain Linux configurations), DGL's graphbolt C++ library may not be available:

```
FileNotFoundError: Cannot find DGL C++ graphbolt library at
.../site-packages/dgl/graphbolt/libgraphbolt_pytorch_*.so
```

**Cause**: Graphbolt is DGL's distributed graph library. The C++ binaries may not be available for all platform/Python/PyTorch combinations.

**Impact**: This only affects distributed graph operations. Standard DGL usage (including our knowledge graph pipelines) does not require graphbolt.

**Fix**: Run the provided `fix_graphbolt.py` script in your environment:

```bash
python fix_graphbolt.py
```

This script:
1. Locates the DGL graphbolt module in your environment
2. Creates a backup of the original `__init__.py`
3. Patches it to disable graphbolt with a warning
4. Verifies DGL still works correctly

**Manual Fix** (if script doesn't work):

1. Find your DGL installation: `python -c "import dgl; print(dgl.__file__)"`
2. Navigate to the graphbolt directory: `cd <dgl_path>/graphbolt/`
3. Backup the original: `cp __init__.py __init__.py.backup`
4. Replace `__init__.py` with:

```python
"""GraphBolt disabled for compatibility"""
import warnings
warnings.warn("Graphbolt is disabled (not needed for standard DGL usage)", RuntimeWarning)
```

### TorchData Version Compatibility

**Issue**: `ModuleNotFoundError: No module named 'torchdata.datapipes'`

**Cause**: TorchData 0.8.0+ removed the `datapipes` module that Kedro depends on.

**Fix**: The `pyproject.toml` now pins `torchdata>=0.7.0,<0.8.0` to ensure a compatible version is installed.

If you encounter this error:
```bash
pip install "torchdata>=0.7.0,<0.8.0"
```
