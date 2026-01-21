#!/usr/bin/env bash
#SBATCH --job-name=lbm_mpi_gpu
#SBATCH --account=C3SE2025-2-17
#SBATCH --partition=vera
#SBATCH --nodes=2
#SBATCH --ntasks-per-node=1
#SBATCH --gpus-per-node=H100:1
#SBATCH --time=00:02:00
#SBATCH --output=output/lbm_mpi_%j.out

module purge
module load gompi/2023a
module load CUDA/12.9.1
module load mpi4py/3.1.4-gompi-2023a
source venv/bin/activate

# Run MPI with GPU-aware CuPy
srun --mpi=pmix python3 samples/taylor_green_vortex.py
