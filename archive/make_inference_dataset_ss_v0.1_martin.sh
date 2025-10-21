#!/bin/bash
#SBATCH --account mutationalscanning
#SBATCH --time 4:00:00
#SBATCH -c 16
#SBATCH --mem 512g

source $(conda info --base)/etc/profile.d/conda.sh
conda activate methyl-jepa

python make_inference_dataset.py -i "~/mutationalscanning/DerivedData/bam/HiFi/chimp/martin/kinetics/martin_kinetics_diploid.bam" -n 1_000_000 -c 32 -o martin_ss_v0.1


# BAM_PATH = "~/mutationalscanning/DerivedData/bam/HiFi/chimp/martin/kinetics/martin_kinetics_diploid.bam" 
# BAM_PATH = "~/mutationalscanning/Workspaces/chcharlton/methyl-jepa/data/raw/methylated_subset.bam"