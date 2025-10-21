#!/bin/bash
#SBATCH --account mutationalscanning
#SBATCH --time 240
#SBATCH -c 32
#SBATCH --mem 700g

source $(conda info --base)/etc/profile.d/conda.sh
conda activate methyl-jepa

python make_null_dataset.py -i "~/mutationalscanning/DerivedData/bam/HiFi/chimp/martin/kinetics/martin_kinetics_diploid.bam" -n 1_000_000 -p 0.24 -c 32 -o null/martin_null_p0.24_n1m


# BAM_PATH = "~/mutationalscanning/DerivedData/bam/HiFi/chimp/martin/kinetics/martin_kinetics_diploid.bam" 
# BAM_PATH = "~/mutationalscanning/Workspaces/chcharlton/methyl-jepa/data/raw/methylated_subset.bam"