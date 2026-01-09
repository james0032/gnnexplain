# Troubleshooting: knowledge_graph String vs Dictionary Issue

## Problem Summary

When running `kedro run --nodes=compute_test_scores`, you're getting this error:

```
AttributeError: 'str' object has no attribute 'get'
```

at [nodes.py:474](src/gnn_explainer/pipelines/training/nodes.py#L474)

This happens because `knowledge_graph` is being passed as a **string** (filepath) instead of as a **dictionary** (loaded data).

## Root Cause Analysis

The `compute_test_scores` function expects `knowledge_graph` to be a dictionary containing:
- `train_triples`, `val_triples`, `test_triples` (tensors)
- `node_dict`, `rel_dict` (entity/relation mappings)
- `idx_to_entity`, `idx_to_relation` (reverse mappings)
- `num_nodes`, `num_relations` (integers)

However, Kedro is passing it as a string path instead of loading the pickle file.

## Most Likely Causes

### 1. **File Doesn't Exist on Server** (Most Likely)

The `knowledge_graph.pkl` file hasn't been generated yet on the server. This can happen if:
- You only ran the training pipeline without running data_prep first
- You're running from a different directory than where data was prepared
- The data prep pipeline failed partway through

**Solution**: Run the data preparation pipeline first:

```bash
# From the Kedro project root
cd /projects/aixb/jchung/everycure/git/gnnexplain
kedro run --pipeline=data_prep
```

This will generate:
- `knowledge_graph.pkl`
- `dgl_data.pkl`
- `negative_samples.pkl`

Then run the compute_test_scores node:

```bash
kedro run --nodes=compute_test_scores
```

### 2. **Kedro Version Mismatch**

Different versions of Kedro might handle pickle datasets differently.

**Check your Kedro version**:

```bash
kedro --version
```

Should be: `0.19.10` (from pyproject.toml)

If different, reinstall:

```bash
pip install kedro~=0.19.10
```

### 3. **Catalog Configuration Issue**

The catalog might not be loading correctly in your environment.

**Check which catalog is being used**:

```bash
kedro catalog list | grep knowledge_graph
```

Should show:
```
knowledge_graph (PickleDataset)
```

### 4. **Wrong Working Directory**

You might be running from the wrong directory.

**Verify you're in the Kedro project root**:

```bash
pwd
ls -la conf/  # Should show base/ and local/ directories
```

The project root should be: `/projects/aixb/jchung/everycure/git/gnnexplain`

## Diagnostic Steps

### Step 1: Run the Debug Script

I've created a comprehensive debug script that will help identify the issue:

```bash
# From the Kedro project root
cd /projects/aixb/jchung/everycure/git/gnnexplain
python debug_kg_load.py
```

This will:
1. Check if the pickle file exists
2. Try to load it directly
3. Verify the contents are correct
4. Test Kedro's catalog loading
5. Provide specific recommendations

### Step 2: Verify File Paths

Check if the expected files exist:

```bash
ls -lh /projects/aixb/jchung/everycure/influence_estimate/robokop/gnn_ROBOKOP_clean_baseline/data/02_intermediate/
```

You should see:
- `knowledge_graph.pkl` (should be ~100-500 MB depending on data size)
- `dgl_data.pkl`

### Step 3: Check Catalog Configuration

View your catalog configuration:

```bash
cat conf/local/catalog.yml | grep -A 3 knowledge_graph
```

Should show:
```yaml
knowledge_graph:
  type: pickle.PickleDataset
  filepath: /projects/aixb/jchung/everycure/influence_estimate/robokop/gnn_ROBOKOP_clean_baseline/data/02_intermediate/knowledge_graph.pkl
```

### Step 4: Test Direct Loading

Try loading the file directly in Python:

```python
import pickle
from pathlib import Path

kg_path = Path("/projects/aixb/jchung/everycure/influence_estimate/robokop/gnn_ROBOKOP_clean_baseline/data/02_intermediate/knowledge_graph.pkl")

if kg_path.exists():
    with open(kg_path, 'rb') as f:
        kg = pickle.load(f)
    print(f"Type: {type(kg)}")
    print(f"Keys: {list(kg.keys())}")
else:
    print("File does not exist!")
```

## Expected Debug Output

When you run the debug script, you should see:

**If file exists and is correct:**
```
✓ Successfully loaded!
  Type: <class 'dict'>
  Keys: ['train_triples', 'val_triples', 'test_triples', 'node_dict', 'rel_dict', ...]
```

**If file doesn't exist:**
```
✗ File does not exist!
⚠ The knowledge_graph.pkl file does not exist.

To fix this, run the data preparation pipeline first:
  kedro run --pipeline=data_prep
```

## Complete Workflow

Here's the correct order of operations:

```bash
# 1. Navigate to project root
cd /projects/aixb/jchung/everycure/git/gnnexplain

# 2. (Optional) Run debug script to verify current state
python debug_kg_load.py

# 3. Run data preparation pipeline (if needed)
kedro run --pipeline=data_prep

# 4. Run training pipeline
kedro run --pipeline=training

# 5. Or run just compute_test_scores if training already completed
kedro run --nodes=compute_test_scores
```

## Additional Notes

### Why the Local File Works

The local file at `/Users/jchung/Documents/RENCI/everycure/experiments/Influence_estimate/gnnexplain/data/02_intermediate/knowledge_graph.pkl` loads correctly because:
1. You downloaded it from the server
2. You placed it in the correct relative path
3. It was already generated by a previous successful data_prep run

### Server vs Local Paths

- **Local (Mac)**: `/Users/jchung/Documents/RENCI/everycure/experiments/Influence_estimate/gnnexplain/`
- **Server (Git repo)**: `/projects/aixb/jchung/everycure/git/gnnexplain/`
- **Server (Data directory)**: `/projects/aixb/jchung/everycure/influence_estimate/robokop/gnn_ROBOKOP_clean_baseline/data/`

The catalog in `conf/local/catalog.yml` points to the data directory, not the git repo.

## Contact for Help

If the debug script doesn't reveal the issue, please share:
1. The complete output of `python debug_kg_load.py`
2. Your current working directory: `pwd`
3. Kedro version: `kedro --version`
4. Contents of catalog: `cat conf/local/catalog.yml | grep -A 3 knowledge_graph`
