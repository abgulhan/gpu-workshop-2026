# env.sh -- run this in every new GPU session on Roar:      source env.sh
# It loads the CUDA compiler and tools, and finds a Python that has PyTorch and CuPy.
unset PYTHONPATH PYTHONHOME      # a Python path left over from your own setup would hide the workshop packages
module load cuda/12.9.1 python/3.12.12

_own=/storage/work/$USER/gpu_workshop_env          # built by:  bash setup_env.sh
_shared=/scratch/abg6029/gpu_workshop_env          # ready-made by the instructors
_env=""
if   [ -f "$_own/.complete" ];      then _env=$_own      # only if setup_env.sh ran to the end
elif [ -x "$_shared/bin/python3" ]; then _env=$_shared
fi

if [ -n "$_env" ]; then
    export PATH=$_env/bin:$PATH
    echo "CUDA:   $(nvcc --version | grep -o 'release [0-9.]*')"
    echo "Python: $_env"
else
    echo "CUDA:   $(nvcc --version | grep -o 'release [0-9.]*')"
    echo "Python: no workshop environment found. The C++ parts work as they are."
    echo "        For the Python parts run once:   bash setup_env.sh   (about 10 minutes)"
fi
unset _own _shared _env
