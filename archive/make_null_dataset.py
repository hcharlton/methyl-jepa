import pysam
import polars as pl
import argparse
import sys
import os
import random

### Purpose ###
# converts a BAM file into a parquet file of null windows (windows which may or may not contain a CpG site)


#### example usage ####
## short format
# python make_null_dataset.py -n 1000 -c 32 -o output_file_name_str -r
## long format
# python make_null_dataset.py --output-name output_file_name_str --context 32 --n_reads 1000 


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
# our strategy is to align everything such that it is in accord with the forward strand. 
# This means that we reverse index the reverse kinetic tags, but keep everything else the same
# this also has the effect of preserving the seq->kinetics causal structure
# info here https://pacbiofileformats.readthedocs.io/en/13.1/BAM.html

def bam_to_df(bam_path: str, n_reads: int, p_sample: float, context: int):
    # tags in the BAM file that we need to pull out each sample
    required_tags = {"fi", "ri", "fp", "rp", "np", "sm", "sx"}
    col_data = {
        "read_name": [], 
        "seq": [], 
        "qual": [],
        "np": [],
        "sm": [],
        "sx": [],
        "fi": [], 
        "fp": [], 
        "ri": [], 
        "rp": []
        }


    counters = {
            "reads_processed": 0,
            "reads_missing_tags": 0,
            "reads_with_empty_tags": 0,
            "windows_processed": 0,
            "windows_filtered_by_length": 0,
            "windows_appended": 0
            }


    with pysam.AlignmentFile(bam_path, "rb", check_sq=False) as bam:
        for i, read in enumerate(bam):
            if i >= n_reads and n_reads !=0:
                break
            if random.random() > p_sample:
                continue
            counters["reads_processed"] += 1 # counter append
            if not all(read.has_tag(tag) for tag in required_tags):
                counters["reads_missing_tags"] += 1 # counter append
                continue
            seq = read.query_sequence
            # meta values
            np_value = read.get_tag('np')
            sm_values = read.get_tag('sm')
            sx_values = read.get_tag('sx')
            qual_values = list(read.query_qualities)
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
                    len(sm_values)==0,
                    len(sx_values)==0,
                    len(qual_values)==0]):
                   counters["reads_with_empty_tags"] += 1 # counter append
                   continue
            # Find all non-overlapping "CG" sites in the sequence
            L = len(fi_values)
            for win_start in range(0, L, context):
                counters["windows_processed"] += 1 # counter append
                win_end = win_start + context

                # perform reverse strand indexing calculation
                rev_win_start = L - win_end
                rev_win_end = L - win_start
                # forward slicing 
                context_seq = seq[win_start:win_end]
                context_qual = qual_values[win_start:win_end]
                context_fi = fi_values[win_start:win_end]
                context_fp = fp_values[win_start:win_end]
                context_sm = sm_values[win_start:win_end]
                context_sx = sx_values[win_start:win_end]

                # reverse slicing
                context_ri = ri_values[rev_win_start:rev_win_end]
                context_rp = rp_values[rev_win_start:rev_win_end]

                # make sure they all have the same length
                if set(map(len, [context_seq, context_qual, context_fi, context_fp, context_ri, context_rp, context_sm, context_sx])) != {context}:
                    counters["windows_filtered_by_length"] += 1 # counter append
                    continue
                counters["windows_appended"] += 1 # counter append
                col_data["read_name"].append(read.query_name)
                col_data["seq"].append(context_seq)
                col_data['qual'].append(context_qual)
                col_data["fi"].append(context_fi)
                col_data["fp"].append(context_fp)
                col_data["ri"].append(context_ri)
                col_data["rp"].append(context_rp)
                col_data["np"].append(np_value)
                col_data["sm"].append(context_sm)
                col_data["sx"].append(context_sx)
    # use List(UInt16) for memory saving since polars defaults to UInt64

    print("--- Debugging Counters ---")
    for key, value in counters.items():
        print(f"{key:<30}: {value}")
    print("--------------------------")
    df = pl.DataFrame({
        "read_name": pl.Series(col_data["read_name"], dtype = pl.String),
        "seq": pl.Series(col_data["seq"], dtype = pl.String),
        "qual": pl.Series(col_data['qual'], dtype = pl.List(pl.UInt8)),
        "np": pl.Series(col_data["np"], dtype = pl.UInt8),
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
        "Takes context size as a parameter. Outputs a parquet file containing 1 sample per row."
    )
    parser.add_argument('-i', '--input_path',
                    type=str,
                    required=True,
                    help="Path to the input BAM file.")
    
    parser.add_argument('-n', '--n_reads',
                        type=int,
                        required=True,
                        help="Number of reads to process. 0 for all reads.")
    parser.add_argument('-p', '--p_sample',
                        type=float,
                        required=True,
                        help="Probability to collect data from a read. Set to 1 for all reads."
                        )
    parser.add_argument('-c', '--context',
                        type=int,
                        required=True,
                        help='How many base pairs to extract for each sample, including the CG pair.')
    parser.add_argument('-o', '--output_name',
                        type=str,
                        required=True,
                        help="The output file name")
    args = parser.parse_args()
    if args.n_reads < 0:
        print("Error: n_reads should be positive or 0 (to indicate all reads).")
        sys.exit(1)

    df = bam_to_df(bam_path=os.path.expanduser(args.input_path), 
                            n_reads=args.n_reads, 
                            p_sample=args.p_sample,
                            context=args.context,
                            )



    df.write_parquet(f'../data/processed/{args.output_name}.parquet')

if __name__ == "__main__":
    main()