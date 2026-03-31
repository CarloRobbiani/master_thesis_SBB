#!/bin/bash
#------------------------
#SBATCH --account=gratis
#------------------------
#SBATCH --job-name="Train MATGCN model"
#SBATCH --time=02:00:00
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem-per-cpu=1G
#SBATCH --partition=epyc2

# Your code below this line
module load Anaconda3
eval "$(conda shell.bash hook)"

conda activate dl_a2
conda config --add channels defaults
pip install pyarrow==15.0.2 fastparquet
conda env update -f environment.yaml
python3 graph_models/station_graph/training.py