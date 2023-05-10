#!/bin/bash

#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=2
#SBATCH --time=2:00:00
#SBATCH --mem=250GB
#SBATCH --gres=gpu:1
#SBATCH --job-name=stellar


module purge

singularity exec --nv \
	    --overlay /scratch/yj2369/senv/overlay-15GB-500K.ext3:ro \
	    /scratch/work/public/singularity/cuda11.7.99-cudnn8.5-devel-ubuntu22.04.2.sif \
	    /bin/bash -c "source /ext3/env.sh; cd /scratch/yj2369/stellar; python STELLAR_run.py --dataset TonsilBE --num-heads 13 --num-seed-class 3"
