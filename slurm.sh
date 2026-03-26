#!/bin/bash

  

#SBATCH --job-name=training # create a short name for your job

#SBATCH --nodes=1 # node count

#SBATCH --ntasks=1 # total number of tasks across all nodes

#SBATCH --cpus-per-task=4 # cpu-cores per task (>1 if multi-threaded tasks)

#SBATCH --mem-per-cpu=16G # memory per cpu-core (4G per cpu-core is default)

#SBATCH --time=12:00:00 # total run time limit (HH:MM:SS)

#SBATCH --gres=gpu:1 # number of gpus per node

#SBATCH --mail-type=begin # send email when job begins

#SBATCH --mail-type=end # send email when job ends

#SBATCH --mail-user=martin.barry@hevs.ch

#SBATCH --output=logs/job_%j.out # standard output and error log

#SBATCH --error=logs/job_%j.err

ulimit -n 65535

  

apptainer exec --nv --bind /home/martin.barry/datasets/:/home/martin.barry/projects/VisionBridge/largefiles/ /home/martin.barry/datasets/visionbridge.sif python -m src.scripts.DoorDetectionTrainerv7