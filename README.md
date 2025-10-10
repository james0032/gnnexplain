# gnnexplain

## How to use it

### Default: Only drug->disease triples (CHEBI, UNII, PUBCHEM.COMPOUND -> MONDO)
```bash
python kg_model.py --use_perturbation --num_explain 10
```

### Show what prefixes exist in the test set first
```bash
python kg_model.py --show_prefix_inventory --skip_explanation
```

### Custom prefixes
```bash
python kg_model.py --use_perturbation --num_explain 10 \
  --subject_prefixes CHEBI DRUGBANK \
  --object_prefixes MONDO HP
```

### Disable filtering (use all test triples randomly)
```bash
python kg_model.py --use_perturbation --num_explain 10 --no_prefix_filter
```

### Drug->disease with top 15 edges displayed
```bash
python kg_model.py --use_perturbation --num_explain 20 \
  --subject_prefixes CHEBI UNII PUBCHEM.COMPOUND \
  --object_prefixes MONDO \
  --top_k_edges 15
```