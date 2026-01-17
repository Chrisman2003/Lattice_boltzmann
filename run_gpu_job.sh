#!/usr/bin/env bash
#SBATCH --job-name=lbm_gpu
#SBATCH --account=C3SE2025-2-17   # <— your project
#SBATCH --partition=vera          # use the Vera partition
#SBATCH --gpus-per-node=A40:1     # request one A40 GPU
#SBATCH --nodes=1
#SBATCH --time=04:00:00           # wall-time (4 hours)
#SBATCH --output=lbm_%j.out       # job output log

module purge
module load CUDA/12.9.1
source venv/bin/activate

#python3 samples/isentropic_vortex.py
python3 samples/taylor_green_vortex.py
