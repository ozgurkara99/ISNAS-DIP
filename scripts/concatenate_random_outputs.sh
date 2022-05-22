#!/bin/bash
#SBATCH  --output=sbatch_log/%j.out
#SBATCH  --gres=gpu:1
#SBATCH  --mem=30G



eval "$(/itet-stor/aricanm/net_scratch/conda/bin/conda shell.bash hook)"

echo 'activating conda environment'
conda activate pytcu10

echo "args: $@"
python -u '/srv/beegfs02/scratch/biwismrschool21/data/NAS-DIP Summer Research/concatenate_random_outputs.py' "$@"

