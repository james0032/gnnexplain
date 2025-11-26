# Input Data Requirements for Training Pipeline

**Date**: 2025-11-26
**Purpose**: Document all required input files for the GNN Explainer training pipeline

---

## 📋 **Overview**

The training pipeline requires **5 input files** in the `data/01_raw/` directory:

1. ✅ **Training triples** (`robo_train.txt`)
2. ✅ **Validation triples** (`robo_val.txt`)
3. ✅ **Test triples** (`robo_test.txt`)
4. ✅ **Node dictionary** (`node_dict`)
5. ✅ **Relation dictionary** (`rel_dict`)

---

## 📁 **Directory Structure**

```
experiments/Influence_estimate/gnnexplain/
├── data/
│   └── 01_raw/                    # Raw input data directory
│       ├── robo_train.txt         # Training triples (REQUIRED)
│       ├── robo_val.txt           # Validation triples (REQUIRED)
│       ├── robo_test.txt          # Test triples (REQUIRED)
│       ├── node_dict              # Entity ID mapping (REQUIRED)
│       ├── rel_dict               # Relation ID mapping (REQUIRED)
│       ├── edge_map.json          # Edge metadata (OPTIONAL)
│       └── id_to_name.map         # ID to name mapping (OPTIONAL)
```

---

## 📝 **File Format Specifications**

### **1. Triple Files** (train/val/test)

**Format**: Tab-separated values (TSV)
**Extension**: `.txt`
**Encoding**: UTF-8

#### **Structure**:
```
<head_entity>\t<relation>\t<tail_entity>
```

#### **Example** (`robo_train.txt`):
```
CHEBI:15365	biolink:treats	MONDO:0005015
UNII:R16CO5Y76E	biolink:treats	MONDO:0005148
PUBCHEM.COMPOUND:2244	biolink:interacts_with	HGNC:11998
CHEBI:3002	biolink:causes	HP:0001824
MONDO:0005015	biolink:related_to	HP:0002315
```

#### **Format Details**:
- **Separator**: Tab character (`\t`)
- **No header row**
- **Entity format**: `<namespace>:<identifier>`
  - Examples: `CHEBI:15365`, `MONDO:0005015`, `HGNC:11998`
- **Relation format**: `<namespace>:<relation_type>`
  - Examples: `biolink:treats`, `biolink:interacts_with`

#### **Typical Sizes**:
```
robo_train.txt:  ~80% of total triples
robo_val.txt:    ~10% of total triples
robo_test.txt:   ~10% of total triples
```

---

### **2. Node Dictionary** (`node_dict`)

**Format**: Tab-separated values (TSV)
**Extension**: None (no extension)
**Encoding**: UTF-8

#### **Structure**:
```
<entity_id>\t<integer_index>
```

#### **Example** (`node_dict`):
```
CHEBI:15365	0
MONDO:0005015	1
UNII:R16CO5Y76E	2
PUBCHEM.COMPOUND:2244	3
HGNC:11998	4
CHEBI:3002	5
HP:0001824	6
HP:0002315	7
```

#### **Format Details**:
- **Separator**: Tab character (`\t`)
- **No header row**
- **Column 1**: Entity ID (string)
- **Column 2**: Integer index (0-based)
- **All entities from triples must be included**
- **Indices must be continuous: 0, 1, 2, ..., N-1**

#### **Purpose**:
Maps entity IDs (strings) to integer indices for tensor operations.

---

### **3. Relation Dictionary** (`rel_dict`)

**Format**: Tab-separated values (TSV)
**Extension**: None (no extension)
**Encoding**: UTF-8

#### **Structure**:
```
<relation_id>\t<integer_index>
```

#### **Example** (`rel_dict`):
```
biolink:treats	0
biolink:interacts_with	1
biolink:causes	2
biolink:related_to	3
biolink:affects	4
biolink:prevents	5
```

#### **Format Details**:
- **Separator**: Tab character (`\t`)
- **No header row**
- **Column 1**: Relation ID (string)
- **Column 2**: Integer index (0-based)
- **All relations from triples must be included**
- **Indices must be continuous: 0, 1, 2, ..., R-1**

#### **Purpose**:
Maps relation IDs (strings) to integer indices for tensor operations.

---

### **4. Edge Map** (`edge_map.json`) - OPTIONAL

**Format**: JSON
**Extension**: `.json`
**Encoding**: UTF-8

#### **Structure**:
```json
{
  "edge_index": {
    "source_nodes": [0, 1, 2, ...],
    "target_nodes": [1, 2, 3, ...]
  },
  "edge_type": [0, 1, 0, ...],
  "num_nodes": 1000,
  "num_edges": 5000
}
```

#### **Purpose**:
Optional pre-computed edge mappings for faster loading.

---

### **5. ID to Name Map** (`id_to_name.map`) - OPTIONAL

**Format**: Pickle
**Extension**: `.map` (pickle format)

#### **Structure** (Python dict):
```python
{
    'CHEBI:15365': 'Aspirin',
    'MONDO:0005015': 'Diabetes Mellitus',
    'HGNC:11998': 'TP53',
    ...
}
```

#### **Purpose**:
Maps entity IDs to human-readable names for visualization.

---

## 🔧 **How Data is Processed**

### **Step 1: Data Preparation Pipeline**

```python
# 1. Load triple files
triple_files = load_triple_files(
    train_file='data/01_raw/robo_train.txt',
    val_file='data/01_raw/robo_val.txt',
    test_file='data/01_raw/robo_test.txt'
)

# 2. Load dictionaries
dictionaries = load_dictionaries(
    node_dict='data/01_raw/node_dict',
    rel_dict='data/01_raw/rel_dict'
)

# 3. Convert triples to tensors
# Reads triple files and converts entity/relation IDs to indices
triple_tensors = load_triples_from_files(triple_files, dictionaries)

# 4. Build knowledge graph
knowledge_graph = build_knowledge_graph(triple_tensors, dictionaries)

# 5. Create PyG data
pyg_data = create_pyg_data(knowledge_graph)
```

### **Output**:
- `knowledge_graph.pkl`: Complete KG with mappings
- `pyg_data.pkl`: PyTorch Geometric Data object
- `negative_samples.pkl`: Negative triples for training

---

## ✅ **Validation Checklist**

Before running the training pipeline, verify:

- [ ] **File existence**:
  - [ ] `data/01_raw/robo_train.txt` exists
  - [ ] `data/01_raw/robo_val.txt` exists
  - [ ] `data/01_raw/robo_test.txt` exists
  - [ ] `data/01_raw/node_dict` exists
  - [ ] `data/01_raw/rel_dict` exists

- [ ] **File format**:
  - [ ] Triple files are tab-separated
  - [ ] No header rows in triple files
  - [ ] Dictionary files are tab-separated
  - [ ] Entity/relation IDs are consistent across files

- [ ] **Data consistency**:
  - [ ] All entities in triples exist in `node_dict`
  - [ ] All relations in triples exist in `rel_dict`
  - [ ] Indices in dictionaries are continuous (0, 1, 2, ...)
  - [ ] No duplicate entries in dictionaries

- [ ] **Data quality**:
  - [ ] No empty lines in files
  - [ ] No malformed triples
  - [ ] UTF-8 encoding
  - [ ] Reasonable train/val/test split (~80/10/10)

---

## 🛠️ **Creating Input Files from Raw Data**

### **Option 1: From Existing Knowledge Graph**

If you have a knowledge graph in another format:

```python
import pandas as pd

# Example: Load from CSV
kg = pd.read_csv('knowledge_graph.csv')
# Columns: head, relation, tail

# Split into train/val/test
from sklearn.model_selection import train_test_split

train, temp = train_test_split(kg, test_size=0.2, random_state=42)
val, test = train_test_split(temp, test_size=0.5, random_state=42)

# Save as TSV
train.to_csv('data/01_raw/robo_train.txt', sep='\t', index=False, header=False)
val.to_csv('data/01_raw/robo_val.txt', sep='\t', index=False, header=False)
test.to_csv('data/01_raw/robo_test.txt', sep='\t', index=False, header=False)

# Create dictionaries
entities = set(kg['head'].unique()) | set(kg['tail'].unique())
relations = set(kg['relation'].unique())

# Save node_dict
with open('data/01_raw/node_dict', 'w') as f:
    for i, entity in enumerate(sorted(entities)):
        f.write(f"{entity}\t{i}\n")

# Save rel_dict
with open('data/01_raw/rel_dict', 'w') as f:
    for i, relation in enumerate(sorted(relations)):
        f.write(f"{relation}\t{i}\n")
```

### **Option 2: From Biomedical Databases**

```python
# Example: Query from database
import requests

# Fetch drug-disease associations
# ... query code ...

# Format as triples
triples = []
for drug, disease, relation in associations:
    triples.append(f"{drug}\t{relation}\t{disease}")

# Save to files
# ... save code ...
```

---

## 📊 **Example Dataset Statistics**

Typical sizes for a drug-disease knowledge graph:

```
Dataset Statistics:
├── Entities: 10,000 - 50,000 nodes
│   ├── Drugs: 2,000 - 5,000
│   ├── Diseases: 1,000 - 3,000
│   ├── Genes: 5,000 - 20,000
│   └── Phenotypes: 2,000 - 10,000
│
├── Relations: 10 - 50 types
│   ├── treats
│   ├── causes
│   ├── interacts_with
│   ├── prevents
│   └── ...
│
└── Triples: 100,000 - 1,000,000
    ├── Train: 80,000 - 800,000
    ├── Val: 10,000 - 100,000
    └── Test: 10,000 - 100,000
```

---

## 🚀 **Quick Start**

### **1. Check if data exists**:
```bash
cd /Users/jchung/Documents/RENCI/everycure/experiments/Influence_estimate/gnnexplain

ls -lh data/01_raw/
```

### **2. Validate data format**:
```bash
# Check triple file format
head -5 data/01_raw/robo_train.txt

# Check dictionary format
head -5 data/01_raw/node_dict
head -5 data/01_raw/rel_dict

# Count lines
wc -l data/01_raw/robo_train.txt
wc -l data/01_raw/node_dict
wc -l data/01_raw/rel_dict
```

### **3. Run data preparation**:
```bash
# Run only data preparation pipeline
kedro run --pipeline=data_preparation
```

### **4. Verify output**:
```bash
# Check intermediate data
ls -lh data/02_intermediate/

# Should see:
# - knowledge_graph.pkl
# - pyg_data.pkl
# - negative_samples.pkl
```

---

## 🐛 **Common Issues**

### **Issue 1: File not found**

```
FileNotFoundError: data/01_raw/robo_train.txt
```

**Solution**: Create the `data/01_raw/` directory and add files:
```bash
mkdir -p data/01_raw
# Copy your data files to data/01_raw/
```

### **Issue 2: Entity not in dictionary**

```
KeyError: 'CHEBI:12345'
```

**Solution**: Ensure all entities in triples exist in `node_dict`:
```python
# Validate
entities_in_triples = set()
with open('data/01_raw/robo_train.txt') as f:
    for line in f:
        head, rel, tail = line.strip().split('\t')
        entities_in_triples.add(head)
        entities_in_triples.add(tail)

entities_in_dict = set()
with open('data/01_raw/node_dict') as f:
    for line in f:
        entity, idx = line.strip().split('\t')
        entities_in_dict.add(entity)

missing = entities_in_triples - entities_in_dict
if missing:
    print(f"Missing entities: {missing}")
```

### **Issue 3: Malformed file**

```
ValueError: not enough values to unpack
```

**Solution**: Check file format (must be tab-separated, no extra whitespace):
```bash
# Check for proper tabs
cat -A data/01_raw/robo_train.txt | head

# Should see: entity^Irelation^Ientity$
# ^I = tab character
```

---

## 📖 **References**

- **PyTorch Geometric Data Format**: https://pytorch-geometric.readthedocs.io/en/latest/modules/data.html
- **Knowledge Graph Embeddings**: https://github.com/pykeen/pykeen
- **Biolink Model**: https://biolink.github.io/biolink-model/

---

## 💡 **Tips**

1. **Start Small**: Test with a subset (1000 triples) first
2. **Validate Format**: Use the validation script before training
3. **Backup Data**: Keep original data files safe
4. **Document**: Record data provenance and preprocessing steps
5. **Version Control**: Track data versions if iterating

---

**Ready to Train?** Once you have all 5 required files in `data/01_raw/`, run:

```bash
kedro run --pipeline=data_preparation
kedro run --pipeline=training
```
