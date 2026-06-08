#!/bin/bash
#SBATCH --account=gratis
#SBATCH --job-name="Train MATGCN model"
#SBATCH --time=08:00:00
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem-per-cpu=1G
#SBATCH --partition=epyc2

# Load Anaconda and activate environment
module load Anaconda3
eval "$(conda shell.bash hook)"
conda env update -f environment.yaml
conda activate dl_a2


# Run training
python graph_models/station_graph/training.py