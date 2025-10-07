import pysam
import polars as pl
import re
import argparse
import sys
import os

### Purpose ###
# converts a BAM file into a parquet file of CpG instances. 


#### example usage ####
## short format
# python make_inference_dataset.py -n 1000 -c 32 -o output_file_name_str -r
## long format
# python make_inference_dataset.py --output-name output_file_name_str --context 32 --n_reads 1000 --restrict-instances


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

def bam_to_df(bam_path: str, n_reads: int, context: int, singletons: bool, required_tags = {"fi", "ri", "fp", "rp", "np", "sm", "sx"}):
    # tags in the BAM file that we need to pull out each sample
    col_data = {
        "read_name": [], 
        "seq": [], 
        "qual": [],
        "cg_pos": [],
        "np": [],
        "sm": [],
        "sx": [],
        "fi": [], 
        "fp": [], 
        "ri": [], 
        "rp": []
        }
    
    # in other words, how far on each side of the CG should we gather context?
    # for context=32, this is 15 so that the total sample is 32 units long
    flank_size = (context - 2) // 2


    counters = {
        "reads_processed": 0,
        "reads_missing_tags": 0,
        "reads_with_empty_tags": 0,
        "cpg_sites_found": 0,
        "cpg_filtered_out_by_window": 0,
        "cpg_appended": 0
        }


    with pysam.AlignmentFile(bam_path, "rb", check_sq=False) as bam:
        for i, read in enumerate(bam):
            if i >= n_reads and n_reads !=0:
                break
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
            for match in re.finditer("CG", seq):
                counters["cpg_sites_found"] += 1 # counter append
                L = len(fi_values)
                cg_pos = match.start()
                win_start = cg_pos - flank_size
                win_end = cg_pos + 2 + flank_size
                # perform reverse strand indexing calculation
                rev_win_start = L - win_end
                rev_win_end = L - win_start
                context_seq = seq[win_start:win_end]
                context_qual = qual_values[win_start:win_end]
                # logic for only including one CG per sample (singleton)
                if (singletons == True) and (context_seq.count("CG") != 1):
                    continue
                context_fi = fi_values[win_start:win_end]
                context_fp = fp_values[win_start:win_end]
                context_ri = ri_values[rev_win_start:rev_win_end]
                context_rp = rp_values[rev_win_start:rev_win_end]
                # these do not have funny indexing since they are not strand specific
                context_sm = sm_values[win_start:win_end]
                context_sx = sx_values[win_start:win_end]

                # make sure they all have the same length
                if set(map(len, [context_seq, context_qual, context_fi, context_fp, context_ri, context_rp, context_sm, context_sx])) != {context}:
                    counters["cpg_filtered_out_by_window"] += 1 # counter append
                    continue
                # Ensure the window is fully contained within the read
                if all([win_start >= 0, win_end <= L, rev_win_end >=0, rev_win_start <= L]):
                    counters["cpg_appended"] += 1 # counter append
                    col_data["read_name"].append(read.query_name)
                    col_data["cg_pos"].append(cg_pos)
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
        print(f"{key:<25}: {value}")
    print("--------------------------")
    df = pl.DataFrame({
        "read_name": pl.Series(col_data["read_name"], dtype = pl.String),
        "cg_pos": pl.Series(col_data["cg_pos"], dtype = pl.UInt16),
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
    parser.add_argument('-c', '--context',
                        type=int,
                        required=True,
                        help='How many base pairs to extract for each sample, including the CG pair.')
    parser.add_argument('-o', '--output_path',
                        type=str,
                        required=True,
                        help="path for output file")
    parser.add_argument('--singletons',
                        action='store_true',
                        help="If specified, restrict samples to contain only one CG instance. Defaults to False if not specified.")
    args = parser.parse_args()
    if args.n_reads < 0:
        print("Error: n_reads should be positive or 0 (to indicate all reads).")
        sys.exit(1)

    df = bam_to_df(bam_path=os.path.expanduser(args.input_path), 
                            n_reads=args.n_reads, 
                            context=args.context, 
                            singletons=args.singletons)



    df.write_parquet(args.output_path)

if __name__ == "__main__":
    main()