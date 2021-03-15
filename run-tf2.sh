#!/bin/bash
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=10
#SBATCH --partition=gputest
#SBATCH --gres=gpu:v100:1
#SBATCH --time=0:15:00
#SBATCH --mem=64G
#SBATCH --account=project_2000859

module purge
module load python-data/3.7.6-1
module list

export DATADIR=/scratch/project_2000859/BDA2021/DatPhuongAnh

set -xv
srun python3 $*
