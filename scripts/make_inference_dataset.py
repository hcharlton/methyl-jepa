import pysam
import polars as pl
import re
import numpy as np
import argparse
import sys
import gc

### Purpose ###
# converts a BAM file into a parquet file of CpG instances. 


#### example usage ####
## short format
# python make_inference_dataset.py -n 1000 -c 32 -o output_file_name_str -r
## long format
# python make_inference_dataset.py --output-name output_file_name_str --context 32 --n_reads 1000 --restrict-instances


# GLOBAL
BAM_PATH = "~/mutationalscanning/DerivedData/bam/HiFi/chimp/martin/kinetics/martin_kinetics_diploid.bam" 

# Note for the reverse strand indexing: 
# The reverse strand is stored in the opposite direction of the 
# forward strand. So it is from the last base to the first. 
#
#   SEQ:  A   A   C   C   G   T   T   A   G   C
# fi/fp: f0, f1, f2, f3, f4, f5, f6, f7, f8, f9
# ri/rp: r9, r8, r7, r6, r5, r4, r3, r2, r1, r0
#
# So if we want to get the kinetic values associated with bases CCG,
# we should index using [2:5] on the forward strand, and [6:8] on 
# the reverse strand. So the calculation for the reverse indexing is:
# [L-forward_end: L-forward_start]
# info here https://pacbiofileformats.readthedocs.io/en/13.1/BAM.html

def bam_to_df(bam_path: str, n_reads: int, context: int, label: int, singletons: bool):
    # tags in the BAM file that we need to pull out each sample
    required_tags = {"fi", "ri", "fp", "rp", "fn", "rn", "sm", "sx"}
    col_data = {
        "read_name": [], 
        "cg_pos": [],
        "fn": [],
        "rn": [],
        "sm": [],
        "sx": [],
        "seq": [], 
        "fi": [], 
        "fp": [], 
        "ri": [], 
        "rp": []
        }
    
    # in other words, how far on each side of the CG should we gather context?
    # for context=32, this is 15 so that the total sample is 32 units long
    flank_size = (context - 2) // 2

    with pysam.AlignmentFile(bam_path, "rb", check_sq=False) as bam:
        # Iterate through a limited number of reads for a quick test
        for i, read in enumerate(bam):
            if i >= n_reads and n_reads !=0:
                break
            if not all(read.has_tag(tag) for tag in required_tags):
                continue
            seq = read.query_sequence
            # meta values
            fn_values = read.get_tag('fn')
            rn_values = read.get_tag('rn')
            sm_values = read.get_tag('sm')
            sx_values = read.get_tag('sx')
            # forward values
            fi_values = read.get_tag("fi")
            fp_values = read.get_tag("fp")
            # reverse values
            ri_values = read.get_tag("ri")
            rp_values = read.get_tag("rp")
            # check that none of the 
            if any([len(fi_values)==0,
                    len(fp_values)==0,
                    len(ri_values)==0,
                    len(rp_values)==0,
                    len(fn_values)==0,
                    len(rn_values)==0,
                    len(sm_values)==0,
                    len(sx_values)==0]):
                   continue
            # Find all non-overlapping "CG" sites in the sequence
            for match in re.finditer("CG", seq):
                L = len(fi_values)
                cg_pos = match.start()
                win_start = cg_pos - flank_size
                win_end = cg_pos + 2 + flank_size
                # perform reverse strand indexing calculation
                rev_win_start = L - win_end
                rev_win_end = L - win_start
                context_seq = seq[win_start:win_end]
                # logic for only including one CG per sample (singleton)
                if (singletons == True) and (context_seq.count("CG") != 1):
                    continue
                context_fi = fi_values[win_start:win_end]
                context_fp = fp_values[win_start:win_end]
                context_ri = ri_values[rev_win_start:rev_win_end]
                context_rp = rp_values[rev_win_start:rev_win_end]
                # these do not have funny indexing since they are not strand specific
                context_fn = fn_values[win_start:win_end]
                context_rn = rn_values[win_start:win_end]
                context_sm = sm_values[win_start:win_end]
                context_sx = sx_values[win_start:win_end]

                # make sure they all have the same length
                if set(map(len, [context_seq, context_fi, context_fp, context_ri, context_rp, context_fn, context_rn, context_sm, context_sx])) != {context}:
                    continue
                # Ensure the window is fully contained within the read
                if all([win_start >= 0, win_end <= L, rev_win_end >=0, rev_win_start <= L]):
                    col_data["read_name"].append(read.query_name)
                    col_data["cg_pos"].append(cg_pos)
                    col_data["seq"].append(context_seq)
                    col_data["fi"].append(context_fi)
                    col_data["fp"].append(context_fp)
                    col_data["ri"].append(context_ri)
                    col_data["rp"].append(context_rp)
                    col_data["fn"].append(context_fn)
                    col_data["rn"].append(context_rn)
                    col_data["sm"].append(context_sm)
                    col_data["sx"].append(context_sx)
    # use List(UInt16) for memory saving since polars defaults to UInt64
    df = pl.DataFrame({
        "read_name": col_data["read_name"],
        "cg_pos": col_data["cg_pos"],
        "seq": col_data["seq"],
        "fn": pl.Series(col_data["fn"], dtype = pl.List(pl.UInt8)),
        "rn": pl.Series(col_data["rn"], dtype = pl.List(pl.UInt8)),
        "sm": pl.Series(col_data["sm"], dtype = pl.List(pl.UInt8)),
        "sx": pl.Series(col_data["sx"], dtype = pl.List(pl.UInt8)),
        "fi": pl.Series(col_data["fi"], dtype = pl.List(pl.UInt16)),
        "fp": pl.Series(col_data["fp"], dtype = pl.List(pl.UInt16)),
        "ri": pl.Series(col_data["ri"], dtype = pl.List(pl.UInt16)),
        "rp": pl.Series(col_data["rp"], dtype = pl.List(pl.UInt16)),
    })

    return df


def main():
    parser = argparse.ArgumentParser(
        description="Processes the first N reads or 0 for all of the methylation dataset." \
        "Takes context size as a parameter. Outputs a train and test dataset."
    )
    parser.add_argument('-n', '--n_reads',
                        type=int,
                        required=True,
                        help="Number of reads to process. 0 for all reads.")
    parser.add_argument('-c', '--context',
                        type=int,
                        required=True,
                        help='How many base pairs to extract for each sample, including the CG pair.')
    parser.add_argument('-o', '--output_name',
                        type=str,
                        required=True,
                        help="The prefix for the two output parquet files. Train and test will be appended")
    parser.add_argument('--singletons',
                        action='store_true',
                        help="If specified, restrict samples to contain only one CG instance. Defaults to False if not specified.")
    args = parser.parse_args()
    if args.n_reads < 0:
        print("Error: n_reads should be positive or 0 (to indicate all reads).")
        sys.exit(1)

    df = bam_to_df(bam_path=BAM_PATH, 
                            n_reads=args.n_reads, 
                            context=args.context, 
                            label=1,
                            singletons=args.singletons)



    df.write_parquet(f'../data/processed/{args.output_name}.parquet')

if __name__ == "__main__":
    main()