#!/usr/bin/env bash
#SBATCH --job-name=lbm_gpu
#SBATCH --account=C3SE2025-2-17   # <— your project
#SBATCH --partition=vera          # use the Vera partition
#SBATCH --gpus-per-node=H100:1     # specify GPU model
#SBATCH --nodes=1
#SBATCH --time=04:00:00           # wall-time (4 hours)
#SBATCH --output=output/lbm_%j.out       # job output log

module purge
module load foss/2025b
module load CUDA/12.9.1
module load numba-cuda/0.20.0-foss-2025b-CUDA-12.9.1

#python3 samples/isentropic_vortex.py
python3 samples/taylor_green_vortex.py
