# DGL Graphbolt Compatibility Fix

If you encounter this error when using DGL:

```
FileNotFoundError: Cannot find DGL C++ graphbolt library at
.../site-packages/dgl/graphbolt/libgraphbolt_pytorch_*.so
```

This is a known compatibility issue on some systems (macOS ARM, certain Linux configurations) where the graphbolt C++ library is not available.

## Quick Fix

**Use this fix script** (works even if DGL fails to import):

```bash
python fix_graphbolt_preimport.py
```

This will:
1. ✓ Find your DGL installation
2. ✓ Create a backup of the graphbolt module
3. ✓ Patch it to disable graphbolt (which is not needed for standard DGL operations)
4. ✓ Verify DGL works correctly

## Why This Works

- **Graphbolt** is only needed for distributed graph operations
- Our knowledge graph pipelines use standard DGL operations
- Disabling graphbolt has **no impact** on functionality
- The fix adds a harmless warning but DGL works normally

## Manual Fix

If the script doesn't work, you can manually patch it:

1. Find your DGL installation:
   ```bash
   python -c "import site; print([p for p in site.getsitepackages() if 'site-packages' in p][0])"
   ```

2. Navigate to graphbolt:
   ```bash
   cd <site-packages>/dgl/graphbolt/
   ```

3. Backup and replace `__init__.py`:
   ```bash
   cp __init__.py __init__.py.backup
   cat > __init__.py << 'EOF'
"""GraphBolt disabled for compatibility"""
import warnings
warnings.warn("Graphbolt is disabled (not needed for standard DGL usage)", RuntimeWarning)
EOF
   ```

4. Verify:
   ```bash
   python -c "import dgl; print(f'DGL {dgl.__version__} works!')"
   ```

## Alternative Script

If DGL imports successfully but you still want to fix it, use:

```bash
python fix_graphbolt.py
```

This version imports DGL first to locate the module.

## Need Help?

See [MIGRATION_DGL.md](MIGRATION_DGL.md#known-issues-and-fixes) for more details about the DGL migration.
