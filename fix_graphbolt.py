#!/usr/bin/env python3
"""
Fix DGL graphbolt compatibility issue.

This script patches the DGL graphbolt module to disable it,
which resolves FileNotFoundError on systems where the graphbolt
C++ library is not available or incompatible.

Run this in your target environment:
    python fix_graphbolt.py
"""

import os
import sys
from pathlib import Path


def find_dgl_graphbolt():
    """Find the DGL graphbolt __init__.py file."""
    try:
        import dgl
        dgl_path = Path(dgl.__file__).parent
        graphbolt_init = dgl_path / "graphbolt" / "__init__.py"

        if graphbolt_init.exists():
            return graphbolt_init
        else:
            print(f"❌ Could not find graphbolt/__init__.py at {graphbolt_init}")
            return None
    except ImportError:
        print("❌ DGL is not installed")
        return None


def patch_graphbolt(graphbolt_init_path):
    """Patch the graphbolt __init__.py to disable it."""

    # Backup original file
    backup_path = graphbolt_init_path.with_suffix('.py.backup')
    if not backup_path.exists():
        print(f"📦 Creating backup at {backup_path}")
        graphbolt_init_path.rename(backup_path)
    else:
        print(f"✓ Backup already exists at {backup_path}")

    # Write patched version
    patched_content = '''"""
GraphBolt module disabled for compatibility.

This module has been patched to resolve compatibility issues
with the graphbolt C++ library on some systems.

Graphbolt is only needed for distributed graph operations,
which are not required for standard DGL usage.
"""

import warnings

warnings.warn(
    "Graphbolt is disabled (not needed for standard DGL usage)",
    RuntimeWarning
)
'''

    print(f"✏️  Writing patched graphbolt/__init__.py")
    with open(graphbolt_init_path, 'w') as f:
        f.write(patched_content)

    print(f"✅ Graphbolt successfully patched!")


def verify_fix():
    """Verify the fix works."""
    print("\n🔍 Verifying fix...")

    try:
        import dgl
        import torch

        print(f"✓ DGL version: {dgl.__version__}")
        print(f"✓ PyTorch version: {torch.__version__}")

        # Test creating a DGL graph
        src = torch.tensor([0, 1, 2])
        dst = torch.tensor([1, 2, 3])
        g = dgl.graph((src, dst), num_nodes=4)
        g.edata['etype'] = torch.tensor([0, 1, 0])

        print(f"✓ DGL graph creation test passed")
        print(f"  - Nodes: {g.num_nodes()}, Edges: {g.num_edges()}")
        print(f"\n🎉 Fix verified! DGL is working correctly.")
        return True

    except Exception as e:
        print(f"❌ Verification failed: {e}")
        return False


def main():
    """Main function."""
    print("=" * 60)
    print("DGL Graphbolt Compatibility Fix")
    print("=" * 60)
    print()

    # Find graphbolt
    graphbolt_init = find_dgl_graphbolt()
    if graphbolt_init is None:
        sys.exit(1)

    print(f"📍 Found graphbolt at: {graphbolt_init}")
    print()

    # Patch it
    try:
        patch_graphbolt(graphbolt_init)
    except Exception as e:
        print(f"❌ Failed to patch: {e}")
        sys.exit(1)

    # Verify
    if verify_fix():
        print()
        print("=" * 60)
        print("You can now use DGL without graphbolt errors!")
        print("=" * 60)
        sys.exit(0)
    else:
        sys.exit(1)


if __name__ == "__main__":
    main()
