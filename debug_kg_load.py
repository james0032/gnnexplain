#!/usr/bin/env python3
"""
Debug script to check knowledge_graph loading.

Run this on the server to verify the pickle file loads correctly and
identify any catalog configuration issues.
"""

import pickle
from pathlib import Path
import sys

# Path from your local catalog config
kg_path = Path("/projects/aixb/jchung/everycure/influence_estimate/robokop/gnn_ROBOKOP_clean_baseline/data/02_intermediate/knowledge_graph.pkl")

print("="*80)
print("KNOWLEDGE GRAPH DEBUG SCRIPT")
print("="*80)
print(f"\nCurrent working directory: {Path.cwd()}")
print(f"Python executable: {sys.executable}")

print("\n" + "="*80)
print("STEP 1: CHECK FILE PATH")
print("="*80)

print(f"\nTarget path: {kg_path}")
print(f"  Exists: {kg_path.exists()}")
if kg_path.exists():
    print(f"  Is file: {kg_path.is_file()}")
    print(f"  Is directory: {kg_path.is_dir()}")

if kg_path.exists():
    print(f"  Size: {kg_path.stat().st_size:,} bytes")

    print("\n" + "="*80)
    print("STEP 2: LOAD PICKLE FILE")
    print("="*80)

    try:
        with open(kg_path, 'rb') as f:
            kg = pickle.load(f)

        print(f"\n✓ Successfully loaded!")
        print(f"  Type: {type(kg)}")

        if isinstance(kg, dict):
            print(f"  Keys: {list(kg.keys())}")
            print(f"\n  Key details:")
            for key in kg.keys():
                val = kg[key]
                print(f"    {key}: {type(val).__name__}", end='')
                if hasattr(val, 'shape'):
                    print(f" - shape: {val.shape}")
                elif isinstance(val, (list, dict)):
                    print(f" - len: {len(val)}")
                else:
                    print(f" - value: {val}")
        else:
            print(f"  Content: {kg}")
            print(f"\n⚠ WARNING: Expected a dictionary, but got {type(kg).__name__}")

    except Exception as e:
        print(f"\n✗ Error loading pickle: {e}")
        import traceback
        traceback.print_exc()
else:
    print("\n✗ File does not exist!")

    print("\n" + "="*80)
    print("STEP 3: INVESTIGATE DIRECTORY STRUCTURE")
    print("="*80)

    # Check if directory exists
    parent_dir = kg_path.parent
    print(f"\nParent directory: {parent_dir}")
    print(f"  Exists: {parent_dir.exists()}")

    if parent_dir.exists():
        print(f"\n  Contents:")
        for item in sorted(parent_dir.iterdir())[:20]:
            print(f"    {item.name}")
    else:
        # Check grandparent
        grandparent = parent_dir.parent
        print(f"\nGrandparent directory: {grandparent}")
        print(f"  Exists: {grandparent.exists()}")

        if grandparent.exists():
            print(f"\n  Contents:")
            for item in sorted(grandparent.iterdir())[:20]:
                print(f"    {item.name}")

# Test Kedro catalog loading
print("\n" + "="*80)
print("STEP 4: TEST KEDRO CATALOG LOADING")
print("="*80)

try:
    from kedro.io import DataCatalog
    from kedro.config import OmegaConfigLoader
    from pathlib import Path

    # Try to load the catalog
    conf_path = Path.cwd() / "conf"

    if not conf_path.exists():
        print(f"\n⚠ WARNING: Config directory not found at {conf_path}")
        print("  Make sure you're running this from the Kedro project root")
    else:
        print(f"\nConfig path: {conf_path}")
        print(f"  Exists: {conf_path.exists()}")

        config_loader = OmegaConfigLoader(conf_source=str(conf_path))
        catalog_config = config_loader["catalog"]

        print(f"\n✓ Loaded catalog configuration")

        if 'knowledge_graph' in catalog_config:
            kg_config = catalog_config['knowledge_graph']
            print(f"\nknowledge_graph catalog entry:")
            for key, val in kg_config.items():
                print(f"  {key}: {val}")
        else:
            print("\n⚠ WARNING: 'knowledge_graph' not found in catalog")

except Exception as e:
    print(f"\n⚠ Could not test Kedro catalog: {e}")
    print("  This is normal if running outside the Kedro project directory")

print("\n" + "="*80)
print("RECOMMENDATIONS")
print("="*80)

if not kg_path.exists():
    print("\n⚠ The knowledge_graph.pkl file does not exist.")
    print("\nTo fix this, run the data preparation pipeline first:")
    print("  kedro run --pipeline=data_prep")
    print("\nThis will generate:")
    print("  - knowledge_graph.pkl")
    print("  - dgl_data.pkl (or pyg_data.pkl)")
    print("  - negative_samples.pkl")

print("\n" + "="*80)
