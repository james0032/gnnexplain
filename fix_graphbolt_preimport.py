#!/usr/bin/env python3
"""
Fix DGL graphbolt compatibility issue (pre-import version).

This script patches the DGL graphbolt module BEFORE importing DGL,
which resolves the FileNotFoundError that prevents DGL from loading.

Run this in your target environment:
    python fix_graphbolt_preimport.py
"""

import os
import sys
import site
from pathlib import Path


def find_site_packages():
    """Find site-packages directories."""
    site_packages = site.getsitepackages()

    # Also check user site-packages
    user_site = site.getusersitepackages()
    if user_site:
        site_packages.append(user_site)

    # Filter to only existing directories
    existing = [Path(p) for p in site_packages if Path(p).exists()]

    if not existing:
        print("❌ Could not find any site-packages directories")
        return []

    return existing


def find_dgl_graphbolt_in_path(site_pkg_path):
    """Find DGL graphbolt in a specific site-packages path."""
    dgl_path = site_pkg_path / "dgl"

    if not dgl_path.exists():
        return None

    graphbolt_init = dgl_path / "graphbolt" / "__init__.py"

    if graphbolt_init.exists():
        return graphbolt_init

    return None


def patch_graphbolt(graphbolt_init_path):
    """Patch the graphbolt __init__.py to disable it."""

    # Backup original file
    backup_path = graphbolt_init_path.with_suffix('.py.backup')
    if not backup_path.exists():
        print(f"📦 Creating backup at {backup_path}")
        import shutil
        shutil.copy2(graphbolt_init_path, backup_path)
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
    RuntimeWarning,
    stacklevel=2
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
        import traceback
        traceback.print_exc()
        return False


def main():
    """Main function."""
    print("=" * 60)
    print("DGL Graphbolt Compatibility Fix (Pre-Import)")
    print("=" * 60)
    print()

    # Find site-packages
    print("🔍 Searching for DGL installation...")
    site_packages = find_site_packages()

    if not site_packages:
        print("❌ No site-packages directories found")
        sys.exit(1)

    print(f"   Found {len(site_packages)} site-packages directories")

    # Find graphbolt in any of them
    graphbolt_init = None
    for site_pkg in site_packages:
        print(f"   Checking: {site_pkg}")
        found = find_dgl_graphbolt_in_path(site_pkg)
        if found:
            graphbolt_init = found
            print(f"   ✓ Found DGL graphbolt")
            break

    if graphbolt_init is None:
        print("\n❌ Could not find DGL graphbolt module in any site-packages")
        print("\nSearched in:")
        for sp in site_packages:
            print(f"  - {sp}")
        print("\nIs DGL installed? Try: pip install dgl")
        sys.exit(1)

    print(f"\n📍 Found graphbolt at: {graphbolt_init}")
    print()

    # Patch it
    try:
        patch_graphbolt(graphbolt_init)
    except Exception as e:
        print(f"❌ Failed to patch: {e}")
        import traceback
        traceback.print_exc()
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
