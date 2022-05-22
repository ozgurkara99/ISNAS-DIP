#!/bin/bash
#SBATCH  --output=sbatch_log/%j.out
#SBATCH  --gres=gpu:1
#SBATCH  --mem=30G



eval "$(/itet-stor/aricanm/net_scratch/conda/bin/conda shell.bash hook)"

echo 'activating conda environment'
conda activate ersin

echo "args: $@"
python -u '/usr/bmicnas02/data-biwi-01/eisnas_dip/ISNAS-DIP/dip.py' "$@"
