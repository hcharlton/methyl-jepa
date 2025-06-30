import pysam
import polars as pl
import re
import numpy as np

# --- Configuration ---
POS_BAM_PATH = "../data/raw/methylated_hifi_reads.bam" 
NEG_BAM_PATH = "../data/raw/unmethylated_hifi_reads.bam" 
NUM_READS_TO_PROCESS = 200   
WINDOW_SIZE = 32         
TRAIN_PROP = 0.8     


required_tags = {"fi", "ri", "fp", "rp"}


def bam_to_parquet(bam_path: str, num_reads: int, window_size: int, label: int):
    records = []
    flank_size = (window_size - 2) // 2

    with pysam.AlignmentFile(bam_path, "rb", check_sq=False) as bam:
        # Iterate through a limited number of reads for a quick test
        for i, read in enumerate(bam):
            if i >= num_reads:
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
            if any([len(fi_values)==0,len(fp_values)==0,len(ri_values)==0,len(rp_values)==0]):
                   continue
            # Find all non-overlapping "CG" sites in the sequence
            for match in re.finditer("CG", seq):
                L = len(fi_values)
                cg_pos = match.start()
                win_start = cg_pos - flank_size
                win_end = cg_pos + 2 + flank_size
                rev_win_start = L - win_end
                rev_win_end = L - win_start
                window_seq = seq[win_start:win_end]
                window_fi = list(fi_values[win_start:win_end])
                window_fp = list(fp_values[win_start:win_end])
                window_ri = list(ri_values[rev_win_start:rev_win_end])
                window_rp = list(rp_values[rev_win_start:rev_win_end])
                # make sure they all have the same length
                if set(map(len, [window_seq, window_fi, window_fp, window_ri, window_rp])) != {WINDOW_SIZE}:
                    continue

                # Ensure the window is fully contained within the read
                if all([win_start >= 0, win_end <= L, rev_win_end >=0, rev_win_start <= L]):
                    records.append({
                        "read_name": read.query_name,
                        "cg_pos": cg_pos,
                        "seq": window_seq,
                        "fi": window_fi,
                        "fp": window_fp,
                        "ri": window_ri,
                        "rp": window_rp
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
    
    # calculate the index where training data ends
    split_idx = int(shuffled_df.height * train_prop)
    
    # split the shuffled datafame at the idx
    df_train = shuffled_df.slice(offset=0, length=split_idx)
    df_test = shuffled_df.slice(offset=split_idx)
    
    return df_train, df_test


pos_df = bam_to_parquet(bam_path=POS_BAM_PATH, num_reads=NUM_READS_TO_PROCESS, window_size=WINDOW_SIZE, label=1)
neg_df = bam_to_parquet(bam_path=NEG_BAM_PATH, num_reads=NUM_READS_TO_PROCESS, window_size=WINDOW_SIZE, label=0)
pos_train_df, pos_test_df = train_test_split(pos_df, train_prop=TRAIN_PROP)
neg_train_df, neg_test_df = train_test_split(neg_df, train_prop=TRAIN_PROP)

train_df = pl.concat([pos_train_df, neg_train_df]).sample(fraction=1, seed=1337, shuffle=True)
test_df =  pl.concat([pos_test_df, neg_test_df]).sample(fraction=1, seed=1337, shuffle=True)


train_df.write_parquet(f'../data/processed/train_cg_{WINDOW_SIZE}_{NUM_READS_TO_PROCESS}.parquet')
test_df.write_parquet(f'../data/processed/test_cg_{WINDOW_SIZE}_{NUM_READS_TO_PROCESS}.parquet')