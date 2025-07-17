#!/bin/bash
#SBATCH --account mutationalscanning
#SBATCH --time 10:00:00
#SBATCH -c 32
#SBATCH --mem 128g

source $(conda info --base)/etc/profile.d/conda.sh
conda activate methyl-jepa

python make_contexts_df_argparse.py -n 10_000 -c 32 -o standard_10k_32_test