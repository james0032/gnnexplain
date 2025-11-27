# Installation Guide

**How to set up the GNN Explainer environment**

---

## 📋 **Prerequisites**

- Python 3.9 or higher
- pip or conda package manager
- Git (for cloning the repository)

---

## 🚀 **Quick Installation**

### **Option 1: Using pip (Recommended)**

```bash
# 1. Navigate to project directory
cd /path/to/gnnexplain

# 2. Create virtual environment
python -m venv venv

# 3. Activate virtual environment
# On macOS/Linux:
source venv/bin/activate
# On Windows:
venv\Scripts\activate

# 4. Install the package in editable mode
pip install -e .

# 5. Verify installation
kedro --version
python -c "import gnn_explainer; print('✓ Installation successful!')"
```

### **Option 2: Using conda**

```bash
# 1. Create conda environment
conda create -n gnnexplain python=3.9 -y

# 2. Activate environment
conda activate gnnexplain

# 3. Install PyTorch (choose your platform)
# For CPU:
conda install pytorch cpuonly -c pytorch

# For CUDA 11.8:
conda install pytorch pytorch-cuda=11.8 -c pytorch -c nvidia

# For CUDA 12.1:
conda install pytorch pytorch-cuda=12.1 -c pytorch -c nvidia

# For Mac M1/M2:
conda install pytorch -c pytorch

# 4. Install the package
cd /path/to/gnnexplain
pip install -e .
```

---

## 🔧 **Detailed Installation Steps**

### **Step 1: Set Up Python Environment**

**Check Python version:**
```bash
python --version
# Should be Python 3.9 or higher
```

**Create and activate virtual environment:**
```bash
# Create venv
python -m venv venv

# Activate (macOS/Linux)
source venv/bin/activate

# Activate (Windows)
venv\Scripts\activate

# Your prompt should now show (venv)
```

### **Step 2: Install Dependencies**

The project uses `pyproject.toml` for dependency management. When you run `pip install -e .`, it will install:

**Core Dependencies:**
- `kedro~=0.19.10` - Pipeline framework
- `torch>=2.0.0` - PyTorch
- `torch-geometric>=2.3.0` - Graph neural networks
- `networkx>=3.0` - Graph data structures
- `matplotlib>=3.5.0` - Visualization
- `numpy>=1.24.0` - Numerical computing
- `pandas>=2.0.0` - Data manipulation
- `scikit-learn>=1.3.0` - Machine learning utilities

**Install the package:**
```bash
pip install -e .
```

**Expected output:**
```
Successfully installed gnn_explainer-0.1
```

### **Step 3: Install Development Dependencies (Optional)**

For development, testing, and jupyter notebooks:

```bash
pip install -e ".[dev]"
```

This installs additional packages:
- `pytest>=7.2` - Testing framework
- `pytest-cov>=3.0` - Coverage reports
- `ruff>=0.1.8` - Linting
- `ipython>=8.10` - Interactive Python
- `jupyterlab>=3.0` - Jupyter notebooks

### **Step 4: Verify Installation**

```bash
# Check Kedro is installed
kedro --version
# Output: kedro, version 0.19.10

# Check package is importable
python -c "import gnn_explainer; print('✓ Package imported successfully')"

# Check PyTorch Geometric
python -c "import torch_geometric; print('✓ PyG installed')"

# List installed pipelines
kedro registry list
```

**Expected output:**
```
✓ Package imported successfully
✓ PyG installed

Pipelines:
- data_preparation
- training
- explanation
```

---

## 🐛 **Troubleshooting**

### **Issue 1: `pip install -e .` fails with "No module named 'setuptools'"`

**Solution:**
```bash
pip install --upgrade pip setuptools wheel
pip install -e .
```

### **Issue 2: PyTorch installation fails**

**Solution:** Install PyTorch separately first:
```bash
# For CPU only:
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cpu

# For CUDA 11.8:
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118

# For CUDA 12.1:
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121

# Then install the package:
pip install -e .
```

### **Issue 3: PyTorch Geometric installation fails**

**Solution:** Install PyG with specific versions:
```bash
pip install torch-scatter torch-sparse torch-cluster torch-spline-conv -f https://data.pyg.org/whl/torch-2.0.0+cpu.html
pip install torch-geometric
pip install -e .
```

### **Issue 4: `kedro viz` fails with "MissingConfigException: conf/local"`

**Solution:** The `conf/local/` directory is required by Kedro:
```bash
mkdir -p conf/local
touch conf/local/.gitkeep
```

This has been fixed in the repository.

### **Issue 5: Import errors after installation**

**Solution:** Make sure you're in the virtual environment:
```bash
# Check which Python is being used
which python
# Should point to venv/bin/python

# If not, activate venv:
source venv/bin/activate  # macOS/Linux
venv\Scripts\activate     # Windows
```

### **Issue 6: CUDA out of memory**

**Solution:** Use CPU mode in `conf/base/parameters.yml`:
```yaml
device: "cpu"
```

Or install CPU-only PyTorch:
```bash
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cpu
```

---

## ✅ **Verification Checklist**

After installation, verify everything works:

- [ ] Python 3.9+ installed (`python --version`)
- [ ] Virtual environment created and activated
- [ ] Package installed (`pip list | grep gnn-explainer`)
- [ ] Kedro working (`kedro --version`)
- [ ] PyTorch installed (`python -c "import torch; print(torch.__version__)"`)
- [ ] PyTorch Geometric installed (`python -c "import torch_geometric"`)
- [ ] conf/local/ directory exists (`ls conf/local/`)
- [ ] Can import package (`python -c "import gnn_explainer"`)
- [ ] Pipelines registered (`kedro registry list`)
- [ ] Validation script works (`python validate_improved_page.py`)

---

## 📦 **Alternative: Install from requirements.txt**

If you prefer `requirements.txt`:

```bash
# Create requirements.txt from pyproject.toml
pip install pip-tools
pip-compile pyproject.toml

# Install from requirements.txt
pip install -r requirements.txt
```

---

## 🔄 **Updating the Environment**

If dependencies change:

```bash
# Reinstall in editable mode
pip install -e . --upgrade

# Or force reinstall all dependencies
pip install -e . --force-reinstall
```

---

## 🌍 **Platform-Specific Notes**

### **macOS (M1/M2 Apple Silicon)**

PyTorch has native support for Apple Silicon:

```bash
# Install PyTorch with MPS support
pip install torch torchvision torchaudio

# Use MPS device in parameters.yml
device: "mps"
```

### **Linux with CUDA**

```bash
# Check CUDA version
nvidia-smi

# Install matching PyTorch version
# For CUDA 11.8:
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118

# For CUDA 12.1:
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121
```

### **Windows**

```bash
# Activate venv (PowerShell)
venv\Scripts\Activate.ps1

# Or (Command Prompt)
venv\Scripts\activate.bat

# Install as normal
pip install -e .
```

---

## 📊 **Testing the Installation**

### **Quick Test**

```bash
# Run validation script
python validate_improved_page.py
```

**Expected output:**
```
================================================================================
Improved PAGE Validation
================================================================================

1. Checking imports...
   ✓ SimplePAGE imported successfully
   ✓ ImprovedPAGE imported successfully

2. Architecture Comparison:
   [...]

✓ All imports successful
✓ Improved PAGE uses CompGCN features (not identity)
✓ Prediction-aware loss function working correctly
✓ High-confidence predictions get higher training weight

Improved PAGE is ready to explain: 'Why did the model predict this triple?'
================================================================================
```

### **Run Unit Tests**

```bash
# Install dev dependencies
pip install -e ".[dev]"

# Run tests
pytest tests/

# With coverage
pytest tests/ --cov=gnn_explainer --cov-report=html
```

---

## 🚀 **Next Steps**

After successful installation:

1. **Prepare input data**: See [INPUT_DATA_REQUIREMENTS.md](docs/INPUT_DATA_REQUIREMENTS.md)
2. **Configure pipeline**: Edit `conf/base/parameters.yml`
3. **Run pipeline**: See [QUICK_REFERENCE.md](QUICK_REFERENCE.md)

---

## 📞 **Need Help?**

If you encounter issues not covered here:

1. Check [QUICK_REFERENCE.md](QUICK_REFERENCE.md) for common tasks
2. Check [COMPLETE_PIPELINE_OVERVIEW.md](docs/COMPLETE_PIPELINE_OVERVIEW.md) for troubleshooting
3. Verify Python version: `python --version` (must be 3.9+)
4. Check if in virtual environment: `which python`
5. Try fresh install:
   ```bash
   deactivate  # Exit venv
   rm -rf venv  # Remove old venv
   python -m venv venv  # Create new venv
   source venv/bin/activate
   pip install --upgrade pip setuptools wheel
   pip install -e .
   ```

---

## 📝 **Summary**

```bash
# Complete installation in 4 steps:
python -m venv venv
source venv/bin/activate  # or venv\Scripts\activate on Windows
pip install -e .
python validate_improved_page.py
```

**That's it!** You're ready to use the GNN Explainer pipeline. 🎉

---

**Last Updated**: 2025-11-26
