import pysam
import polars as pl
import re
import numpy as np
import argparse
import sys

### Purpose ###
# Converts the two BAM files containing wholey methylated and unmethylated PacBio SMRT 
# reads into tabular data format read for use by the DataSet class which for now 
# is still in the notebook. The source files are hardcoded for now, but the 
# script takes a few command line parameters:
# 1. how many reads to process
# 2. how large of a context (in total) for each CG sample
# 3. the name for the output files (gets some suffixes automatically)
# 4. whether to restrict the processing s.t. each sample has one CG


#### example usage ####
## short format
# python make_contexts_df_argparse.py -n 1000 -c 32 -o output_file_name_str -r
## long format
# python make_contexts_df_argparse.py --output-name output_file_name_str --context 32 --n_reads 1000 --restrict-instances


# GLOBAL
POS_BAM_PATH = "../data/raw/methylated_hifi_reads.bam" 
NEG_BAM_PATH = "../data/raw/unmethylated_hifi_reads.bam"       
TRAIN_PROP = 0.8     


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
    required_tags = {"fi", "ri", "fp", "rp"}
    records = []
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
            # forward values
            fi_values = read.get_tag("fi")
            fp_values = read.get_tag("fp")
            # reverse values
            ri_values = read.get_tag("ri")
            rp_values = read.get_tag("rp")
            # check that none of the 
            if any([len(fi_values)==0,len(fp_values)==0,len(ri_values)==0,len(rp_values)==0]):
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
                context_fi = list(fi_values[win_start:win_end])
                context_fp = list(fp_values[win_start:win_end])
                context_ri = list(ri_values[rev_win_start:rev_win_end])
                context_rp = list(rp_values[rev_win_start:rev_win_end])
                # make sure they all have the same length
                if set(map(len, [context_seq, context_fi, context_fp, context_ri, context_rp])) != {context}:
                    continue
                # Ensure the window is fully contained within the read
                if all([win_start >= 0, win_end <= L, rev_win_end >=0, rev_win_start <= L]):
                    records.append({
                        "read_name": read.query_name,
                        "cg_pos": cg_pos,
                        "seq": context_seq,
                        "fi": context_fi,
                        "fp": context_fp,
                        "ri": context_ri,
                        "rp": context_rp
                    })

    df = pl.from_dicts(records).with_columns(pl.lit(label).alias('label'))

    return df


def train_test_split(
    df: pl.DataFrame, train_prop: float = 0.8
) -> tuple[pl.DataFrame, pl.DataFrame]:
    """
    Args:
        df (pl.DataFrame): DataFrame to split.
        train_prop (float, optional): The proportion of the dataset to
                                      allocate to the training split.
                                      Defaults to 0.8.
    Returns:
        tuple[pl.DataFrame, pl.DataFrame]: A tuple containing the train
                                           and test DataFrames.
    """

    # shuffle the df
    shuffled_df = df.sample(fraction=1, shuffle=True, seed=1337)
    # get the unique reads as a series
    unique_readnames = shuffled_df['read_name'].unique()
    # calculate the index where training data ends
    split_idx = int(len(unique_readnames) * train_prop)
    # split the read_names into test/train partitions
    train_readnames = unique_readnames.slice(0,split_idx)
    test_readnames = unique_readnames.slice(split_idx, None)
    # check that we used all of them
    assert len(train_readnames) + len(test_readnames) == len(unique_readnames), "Partitioning did not capture all reads"
    
    # define the train/test sets by readname
    df_train = shuffled_df.filter(pl.col('read_name').is_in(train_readnames.implode()))
    df_test = shuffled_df.filter(pl.col('read_name').is_in(test_readnames.implode()))
    
    return df_train, df_test


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

# make the dataframe for the methylated data
    pos_df = bam_to_df(bam_path=POS_BAM_PATH, 
                            n_reads=args.n_reads, 
                            context=args.context, 
                            label=1,
                            singletons=args.singletons)

# and for the unmethylated data
    neg_df = bam_to_df(bam_path=NEG_BAM_PATH, 
                            n_reads=args.n_reads, 
                            context=args.context, 
                            label=0,
                            singletons=args.singletons)
# split them independently in to train/test
    pos_train_df, pos_test_df = train_test_split(pos_df, train_prop=TRAIN_PROP)
    neg_train_df, neg_test_df = train_test_split(neg_df, train_prop=TRAIN_PROP)
# concatenated the training and testing datasets together
# keeping them seperate and shuffling
    train_df = pl.concat([pos_train_df, neg_train_df]).sample(fraction=1, seed=1337, shuffle=True)
    test_df =  pl.concat([pos_test_df, neg_test_df]).sample(fraction=1, seed=1337, shuffle=True)
# check that the read names in the training and test sets have no overlaps
    assert set(train_df['read_name']).isdisjoint(set(test_df['read_name'])), "Train/test set readnames are not disjoint."

# write out 
    train_df.write_parquet(f'../data/processed/{args.output_name}_train.parquet')
    test_df.write_parquet(f'../data/processed/{args.output_name}_test.parquet')


if __name__ == "__main__":
    main()