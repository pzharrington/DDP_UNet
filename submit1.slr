#!/bin/bash -l
#SBATCH --nodes=1  --time=04:00:00  
#SBATCH -C gpu 
#SBATCH --account m1759
#SBATCH --gres=gpu:8
#SBATCH --exclusive
#SBATCH -c 80
#SBATCH -J single
#SBATCH -o %x-%j.out

module load pytorch/v1.4.0-gpu
module list

# Start training
export HDF5_USE_FILE_LOCKING=FALSE
srun python -m torch.distributed.launch --nproc_per_node=1 train.py --run_num=09

date

