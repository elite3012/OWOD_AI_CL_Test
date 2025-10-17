# Installation Guide for Enhanced OWOD

This guide will walk you through the installation process for the Enhanced Open World Object Detection (OWOD) system with Continual Learning capabilities.

## Table of Contents
- [System Requirements](#system-requirements)
- [Step-by-Step Installation](#step-by-step-installation)
- [Verification](#verification)
- [Common Issues](#common-issues)

## System Requirements

### Hardware Requirements
- **GPU**: NVIDIA GPU with at least 8GB VRAM (16GB+ recommended)
- **RAM**: 16GB minimum (32GB+ recommended)
- **Storage**: At least 50GB free space

### Software Requirements
- **OS**: Windows 10/11, Linux (Ubuntu 20.04+), or macOS
- **Python**: 3.8, 3.9, or 3.10
- **CUDA**: 11.8 or higher (for GPU support)
- **cuDNN**: Compatible version with CUDA

## Step-by-Step Installation

### 1. Clone the Repository

```bash
git clone https://github.com/yourusername/enhanced-owod.git
cd enhanced-owod
```

### 2. Create Virtual Environment

**Windows (PowerShell):**
```powershell
python -m venv venv
.\venv\Scripts\Activate.ps1
```

**Linux/Mac:**
```bash
python3 -m venv venv
source venv/bin/activate
```

### 3. Upgrade pip and Install Build Tools

```bash
python -m pip install --upgrade pip setuptools wheel
```

### 4. Install PyTorch

Visit [PyTorch's official website](https://pytorch.org/get-started/locally/) and select your configuration. For example:

**Windows/Linux with CUDA 11.8:**
```bash
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
```

**CPU only:**
```bash
pip install torch torchvision torchaudio
```

### 5. Install Detectron2

**Windows:**
```powershell
pip install 'git+https://github.com/facebookresearch/detectron2.git'
```

**Linux:**
```bash
pip install 'git+https://github.com/facebookresearch/detectron2.git'
```

If you encounter issues, see [Detectron2 Installation Guide](https://detectron2.readthedocs.io/en/latest/tutorials/install.html).

**Colab:**
```python
!pip install 'git+https://github.com/facebookresearch/detectron2.git'
```

### 6. Install Main Dependencies

```bash
pip install -r requirements.txt
```

### 7. Install Foundation Model Dependencies

```bash
# OpenCLIP for CLIP models
pip install open-clip-torch

# Transformers for various models
pip install transformers

# Segment Anything (SAM)
pip install git+https://github.com/facebookresearch/segment-anything.git
```

### 8. Install Development Tools (Optional)

```bash
pip install black flake8 isort pytest
```

### 9. Download Pre-trained Models

Run the download script to get foundation model weights:

```bash
python scripts/download_models.py
```

This will download:
- CLIP ViT-B/16 weights
- DINOv2 weights
- SAM weights (optional)

### 10. Setup Datasets

Follow the dataset setup instructions:

```bash
python scripts/prepare_datasets.py --dataset pascal_voc --data-dir ./datasets
```

For detailed dataset setup, see [DATASET.md](docs/DATASET.md).

## Verification

### Quick Verification

Run the verification script to ensure everything is installed correctly:

```bash
python scripts/verify_installation.py
```

Expected output:
```
✓ Python version: 3.10.x
✓ PyTorch: 2.x.x (CUDA available)
✓ Detectron2: Successfully imported
✓ OpenCLIP: Successfully imported
✓ Transformers: Successfully imported
✓ GPU: NVIDIA GeForce RTX 3090 (24GB)
✓ All dependencies installed correctly!
```

### Run a Quick Test

Test the model on a sample image:

```bash
python demo/demo.py --config configs/owod_cl/task1_clip_lora.yaml \
    --input demo/images/sample.jpg \
    --output demo/output/
```

## Common Issues

### Issue 1: Detectron2 Installation Fails on Windows

**Solution:**
1. Make sure you have Visual Studio Build Tools installed
2. Use pre-built wheels instead of building from source
3. Try a different CUDA version if the wheel is not available

```powershell
# Download from
https://dl.fbaipublicfiles.com/detectron2/wheels/index.html
```

### Issue 2: CUDA Out of Memory

**Solution:**
1. Reduce batch size in config: `SOLVER.IMS_PER_BATCH: 4`
2. Enable gradient checkpointing: `OPTIMIZATION.GRADIENT_CHECKPOINTING.ENABLED: True`
3. Use mixed precision training: `OPTIMIZATION.MIXED_PRECISION.ENABLED: True`

### Issue 3: Import Errors

**Solution:**
```bash
# Ensure project root is in Python path
export PYTHONPATH="${PYTHONPATH}:/path/to/enhanced-owod"

# Windows
set PYTHONPATH=%PYTHONPATH%;C:\path\to\enhanced-owod
```

### Issue 4: OpenCLIP Model Download Fails

**Solution:**
```bash
# Manually download and specify local path
# Edit configs/defaults.py and set:
# FOUNDATION_MODEL.CLIP.PRETRAINED: "/path/to/local/clip/weights.pth"
```

### Issue 5: Permission Denied (Windows Defender)

**Solution:**
1. Add project folder to Windows Defender exclusions
2. Or run PowerShell as Administrator

### Issue 6: Slow Data Loading

**Solution:**
1. Increase number of workers: `INPUT.NUM_WORKERS: 4`
2. Use SSD for dataset storage
3. Preprocess and cache datasets

## Environment Variables

Create a `.env` file in the project root:

```bash
# CUDA settings
CUDA_VISIBLE_DEVICES=0
TORCH_HOME=./pretrained_models

# Data paths
DATA_ROOT=./datasets
OUTPUT_ROOT=./output

# Logging
WANDB_API_KEY=your_key_here
```

## Docker Installation (Alternative)

For a containerized environment:

```bash
# Build Docker image
docker build -t enhanced-owod:latest .

# Run container
docker run --gpus all -v $(pwd):/workspace enhanced-owod:latest
```

See [DOCKER.md](docs/DOCKER.md) for detailed Docker instructions.

## Updating the Installation

To update to the latest version:

```bash
git pull origin main
pip install -r requirements.txt --upgrade
```

## Uninstalling

To completely remove the installation:

```bash
# Deactivate virtual environment
deactivate

# Remove virtual environment
rm -rf venv

# Remove downloaded models and outputs (optional)
rm -rf pretrained_models output
```

## Getting Help

If you encounter issues not covered here:

1. Check [FAQ](docs/FAQ.md)
2. Search [Issues](https://github.com/yourusername/enhanced-owod/issues)
3. Ask on [Discussions](https://github.com/yourusername/enhanced-owod/discussions)
4. Contact maintainers

## Next Steps

After successful installation:

1. Read the [Getting Started Guide](docs/GETTING_STARTED.md)
2. Try the [Tutorial](docs/TUTORIAL.md)
3. Explore [Example Notebooks](notebooks/)
4. Run experiments with provided configs

---

**Congratulations!** 🎉 You're now ready to use Enhanced OWOD for continual learning in open world object detection.
