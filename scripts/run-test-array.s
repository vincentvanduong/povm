#!/bin/bash
#
#SBATCH --job-name=myJobarrayTest
#SBATCH --nodes=1
#SBATCH --tasks-per-node=1
#SBATCH --time=5:00
#SBATCH --mem=2GB
#SBATCH --gres=gpu
#SBATCH --output=testarray_%A_%a.out
#SBATCH --error=testarray_%A_%a.err

singularity exec --nv \
	    --overlay /scratch/vv2102/povm/my_jax.ext3:ro \
	    /scratch/work/public/singularity/cuda12.1.1-cudnn8.9.0-devel-ubuntu22.04.2.sif\
	    /bin/bash -c "source /ext3/env.sh; conda activate venv; cd ..; cd tests; python two-body-array.py ${SLURM_ARRAY_TASK_ID}"