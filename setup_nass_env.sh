#!/bin/bash
# NASS Remote GPU Environment Setup
# Run this on the remote machine after cloning the repo.
#
# Prerequisites: conda (or mamba) installed, NVIDIA driver with CUDA 12.x
# Check CUDA version: nvidia-smi (top right shows CUDA version)

set -e

ENV_NAME="nass"

# Step 1: Create conda env with Python 3.11
conda create -n $ENV_NAME python=3.11.15 -y
conda activate $ENV_NAME || source activate $ENV_NAME

# Step 2: JAX with CUDA 12 (adjust to cuda11 if nvidia-smi shows CUDA 11.x)
# This replaces the CPU-only jaxlib from your Mac
pip install "jax[cuda12]==0.4.35"

# Step 3: Core NASS dependencies (pinned to your local versions)
pip install \
    "jaxley==0.13.0" \
    "optax==0.2.7" \
    "numpy==1.26.4" \
    "allensdk==2.16.2" \
    "h5py==3.16.0" \
    "anthropic==0.85.0" \
    "python-dotenv==1.2.2" \
    "scipy==1.17.1" \
    "matplotlib==3.10.8" \
    "tridiax==0.2.1" \
    "pandas==2.3.3"

# Step 4: Verify GPU is visible
python -c "
import jax
print('JAX devices:', jax.devices())
assert any('gpu' in str(d).lower() or 'cuda' in str(d).lower() for d in jax.devices()), \
    'ERROR: No GPU found. Check CUDA drivers and jax[cuda] install.'
print('GPU OK')

import jaxley; print(f'jaxley {jaxley.__version__}')
import optax;  print(f'optax  {optax.__version__}')
import numpy;  print(f'numpy  {numpy.__version__}')
print('All imports OK')
"

echo ""
echo "Environment '$ENV_NAME' ready. Activate with:"
echo "  conda activate $ENV_NAME"
echo ""
echo "IMPORTANT: Remove JAX_PLATFORMS=cpu from .env if present."
echo "  sed -i '/JAX_PLATFORMS/d' .env 2>/dev/null || true"
