#!/bin/bash

#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=1
#SBATCH --time=1:00:00
#SBATCH --mem=2GB
#SBATCH --gres=gpu
#SBATCH --job-name=short_time

module purge

singularity exec --nv \
	    --overlay /scratch/vv2102/povm/my_jax.ext3:ro \
	    /scratch/work/public/singularity/cuda12.1.1-cudnn8.9.0-devel-ubuntu22.04.2.sif\
	    /bin/bash -c "source /ext3/env.sh; conda activate venv; cd ..; cd tests; python short_time.py $SLURM_ARRAY_TASK_ID"
