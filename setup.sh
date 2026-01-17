#!/bin/bash
#SBATCH --gres=gpu:1

module purge
module load foss/2025b
module load CUDA/12.9.1
module load numba-cuda/0.20.0-foss-2025b-CUDA-12.9.1