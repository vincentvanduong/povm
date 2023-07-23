#!/bin/bash
#
#SBATCH --job-name=myJobarrayTest
#SBATCH --nodes=1
#SBATCH --tasks-per-node=1
#SBATCH --time=1:00
#SBATCH --mem=1GB
#SBATCH --gres=gpu

singularity exec --nv \
	    --overlay /scratch/vv2102/povm/my_jax.ext3:ro \
	    /scratch/work/public/singularity/cuda12.1.1-cudnn8.9.0-devel-ubuntu22.04.2.sif\
	    /bin/bash -c "source /ext3/env.sh; conda activate venv; cd ..; cd tests; python two-body-array.py ${SLURM_ARRAY_TASK_ID}"