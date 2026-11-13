#!/bin/bash
# setup_env.sh -- build the Python environment for the Python parts of the workshop
# (PyTorch and CuPy). The CUDA C++ programs need NO Python: skip this if you only want those.
#
# Run it INSIDE a GPU session on Roar (after salloc), once:
#       bash setup_env.sh
# It takes about 10 minutes and about 6 GB in your work folder (not your home folder).
# Afterwards, in every new session:   source env.sh
#
# Versions are fixed on purpose: PyTorch 2.6.0 built for CUDA 12.4 still supports the P100 and
# V100 cards of the open queue.
set -e
ENV_DIR=${1:-/storage/work/$USER/gpu_workshop_env}
module load python/3.12.12
python3 -m venv "$ENV_DIR"
"$ENV_DIR/bin/pip" install --no-cache-dir --upgrade pip
"$ENV_DIR/bin/pip" install --no-cache-dir torch==2.6.0 --index-url https://download.pytorch.org/whl/cu124
"$ENV_DIR/bin/pip" install --no-cache-dir cupy-cuda12x==14.2.0 numpy pillow
"$ENV_DIR/bin/python" -c "import torch, cupy, PIL; print('PyTorch', torch.__version__, '| CuPy', cupy.__version__, '| GPU visible:', torch.cuda.is_available())"
touch "$ENV_DIR/.complete"        # env.sh only uses an environment that finished building
echo "Environment ready in $ENV_DIR"
