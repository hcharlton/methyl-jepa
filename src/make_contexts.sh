#!/bin/bash
#SBATCH --account mutationalscanning
#SBATCH --time 10:00:00
#SBATCH -c 32
#SBATCH --mem 512g

source $(conda info --base)/etc/profile.d/conda.sh
conda activate methyl-jepa

python make_contexts_df_argparse.py --output-name standard_all_32 --context 32 --n_reads 0