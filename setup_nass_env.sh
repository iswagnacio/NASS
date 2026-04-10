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

# Activate — handles both init'd and non-init'd shells
conda activate $ENV_NAME 2>/dev/null || source "$(conda info --base)/etc/profile.d/conda.sh" && conda activate $ENV_NAME

# Step 2: JAX with CUDA 12 (adjust to cuda11 if nvidia-smi shows CUDA 11.x)
# This replaces the CPU-only jaxlib from your Mac
pip install "jax[cuda12]==0.4.35"

# Step 3: Core NASS dependencies (pinned to your local versions)
# Install everything EXCEPT allensdk normally
pip install \
    "jaxley==0.13.0" \
    "optax==0.2.7" \
    "numpy==1.26.4" \
    "h5py==3.16.0" \
    "anthropic==0.85.0" \
    "python-dotenv==1.2.2" \
    "scipy==1.17.1" \
    "matplotlib==3.10.8" \
    "tridiax==0.2.1" \
    "pandas==2.3.3" \
    "pynwb==3.1.3" \
    "hdmf==4.3.1" \
    "tqdm==4.67.3" \
    "PyYAML==6.0.3" \
    "requests==2.32.5"

# Step 4: Install allensdk WITHOUT its deps.
#
# allensdk==2.16.2 declares numpy<1.24 as a dependency, but this constraint
# is overly conservative — it was added to block numpy 2.x, not 1.26.x.
# allensdk 2.16.2 works fine at runtime with numpy 1.26.4.
#
# Installing with --no-deps bypasses the resolution conflict. The actual
# runtime deps allensdk needs (requests, tqdm, h5py, pynwb, etc.) are already
# installed above from requirements_nass.txt.
pip install "allensdk==2.16.2" --no-deps

# Step 5: Verify GPU is visible
python -c "
import jax
print('JAX devices:', jax.devices())
assert any('gpu' in str(d).lower() or 'cuda' in str(d).lower() for d in jax.devices()), \
    'ERROR: No GPU found. Check CUDA drivers and jax[cuda] install.'
print('GPU OK')

import jaxley; print(f'jaxley {jaxley.__version__}')
import optax;  print(f'optax  {optax.__version__}')
import numpy;  print(f'numpy  {numpy.__version__}')
import allensdk; print(f'allensdk {allensdk.__version__}')
print('All imports OK')
"

echo ""
echo "Environment '$ENV_NAME' ready. Activate with:"
echo "  conda activate $ENV_NAME"
echo ""
echo "IMPORTANT: Remove JAX_PLATFORMS=cpu from .env if present."
echo "  sed -i '/JAX_PLATFORMS/d' .env 2>/dev/null || true"