#!/bin/bash
#SBATCH --account mutationalscanning
#SBATCH --time 2:00:00
#SBATCH -c 16
#SBATCH --mem 128g

source $(conda info --base)/etc/profile.d/conda.sh
conda activate methyl-jepa

python make_inference_dataset.py -i "~/mutationalscanning/DerivedData/bam/HiFi/chimp/martin/kinetics/martin_kinetics_diploid.bam" -n 10000 -c 32 -o martin_ss_v0.1


# BAM_PATH = "~/mutationalscanning/DerivedData/bam/HiFi/chimp/martin/kinetics/martin_kinetics_diploid.bam" 
# BAM_PATH = "~/mutationalscanning/Workspaces/chcharlton/methyl-jepa/data/raw/methylated_subset.bam"