#!/usr/bin/env bash
#SBATCH --job-name=lbm_mpi_gpu
#SBATCH --account=C3SE2025-2-17
#SBATCH --partition=vera
#SBATCH --nodes=2
#SBATCH --ntasks-per-node=1
#SBATCH --gpus-per-node=A40:1
#SBATCH --time=04:00:00
#SBATCH --output=lbm_mpi_gpu_%j.out

module purge
module load CUDA/12.9.1 
module load OpenMPI/4.1.6
source venv/bin/activate

srun python3 samples/taylor_green_vortex.py
