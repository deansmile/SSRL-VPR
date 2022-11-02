#!/bin/bash

#SBATCH --nodes=1
#SBATCH --ntasks-per-node=4
#SBATCH --cpus-per-task=1
#SBATCH --mem-per-cpu=16GB
#SBATCH --time=24:00:00
#SBATCH --gres=gpu
#SBATCH --job-name=test

module purge

singularity exec --nv \
	    --overlay /scratch/ds5725/environments/habitat1.ext3:rw \
	    /scratch/work/public/singularity/cuda11.1-cudnn8-devel-ubuntu18.04.sif \
	    /bin/bash -c "source /ext3/env.sh; python main.py --mode=train --pooling=netvlad --num_clusters=64"