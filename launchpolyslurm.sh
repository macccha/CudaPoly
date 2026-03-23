#!/bin/bash

#SBATCH --ntasks=1  # number of processor cores (i.e. tasks)
#SBATCH --partition=graphic
#SBATCH --gres=gpu:1
#SBATCH --gpu-bind=single:1

##medium 2-00:00:00
##short 02:00:00
##long 14-00:00:00

#SBATCH --cpus-per-task=1
#SBATCH --mem-per-cpu=4G

# If you so choose Slurm will notify you when certain events happen for your job 
# for a full list of possible options look furhter down
#SBATCH --mail-type END

#SBATCH --job-name="Polycuda"

#SBATCH --output=/home/ciarchi/Documents/Slurm/R-%j-%x.out
#SBATCH --error=/home/ciarchi/Documents/Slurm/R-%j-%x.err

#
# Prepare your environment
#

# causes jobs to fail if one command fails - makes failed jobs easier to find with tools like sacct

set -e

# Load modules
# On the PKS cluster, the code works only with cuda/12.1
module load cuda/12.1
export CUDA_VISIBLE_DEVICES=$SLURM_JOB_GPUS

# Set variables you need
project="/data/others/ciarchi/PolymerDyn/CudaPoly"
results="/data/others/ciarchi/PolymerDyn/$SLURM_JOB_ID"
scratch="/scratch/$USER/$SLURM_JOB_ID"
export OMP_NUM_THREADS=$SLURM_CPUS_PER_TASK


TIMETOT=$(squeue -j $SLURM_JOB_ID -h --Format TimeLimit)

mkdir -p $scratch
#cd $scratch
cd $project

#Make executable
make polydyn

#Run without output buffering for real time output
echo "Running $SLURM_ARRAY_TASK_ID job."
srun --unbuffered ./polydyn run $SLURM_ARRAY_TASK_ID

# Clean up after yourself
cd
rm -rf $scratch

# exit gracefully
exit 0

EOT