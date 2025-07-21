#!/bin/bash
#SBATCH --account mutationalscanning
#SBATCH --time 6:00:00
#SBATCH -c 32
#SBATCH --mem 700g

source $(conda info --base)/etc/profile.d/conda.sh
conda activate methyl-jepa

python make_contexts_df_argparse.py -n 600000 -c 32 -o standard_600k_32
