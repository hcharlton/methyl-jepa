import pysam
import polars as pl
import re

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


            # Find all non-overlapping "CG" sites in the sequence
            for match in re.finditer("CG", seq):
                L = len(seq)
                cg_pos = match.start()
                win_start = cg_pos - flank_size
                win_end = cg_pos + 2 + flank_size
                rev_win_start = L - win_end
                rev_win_end = L - win_start

                # Ensure the window is fully contained within the read
                if win_start >= 0 and win_end <= len(seq):
                    records.append({
                        "read_name": read.query_name,
                        "cg_pos": cg_pos,
                        "window_seq": seq[win_start:win_end],
                        "window_fi": list(fi_values[win_start:win_end]),
                        "window_fp": list(fp_values[win_start:win_end]),
                        "window_ri": list(ri_values[rev_win_start:rev_win_end]),
                        "window_rp": list(rp_values[rev_win_start:rev_win_end])
                    })

    df = pl.from_dicts(records).with_columns(pl.lit(label).alias('label'))

    return df


def train_test_split_lazy(
    df: pl.DataFrame, train_prop: float = 0.8
    ) -> tuple[pl.DataFrame, pl.DataFrame]:
    """Split polars dataframe into two sets.
    Args:
        df (pl.DataFrame): Dataframe to split
        train_fraction (float, optional): Fraction that goes to train. Defaults to 0.8
    Returns:
        Tuple[pl.DataFrame, pl.DataFrame]: Tuple of train and test dataframes
    """
    df = df.with_row_index().sample(fraction=1, seed=1337, shuffle=True)
    df_train = df.filter(pl.col("index") < pl.col("index").max() * train_prop)
    df_test = df.filter(pl.col("index") >= pl.col("index").max() * train_prop)
    return df_train.drop('index'), df_test.drop('index')


pos_df = bam_to_parquet(bam_path=POS_BAM_PATH, num_reads=NUM_READS_TO_PROCESS, window_size=WINDOW_SIZE, label=1).sample(fraction=1, seed=1337, shuffle=True)
neg_df = bam_to_parquet(bam_path=NEG_BAM_PATH, num_reads=NUM_READS_TO_PROCESS, window_size=WINDOW_SIZE, label=0).sample(fraction=1, seed=1337, shuffle=True)

pos_train_df, pos_test_df = train_test_split_lazy(pos_df, train_prop=TRAIN_PROP)
neg_train_df, neg_test_df = train_test_split_lazy(neg_df, train_prop=TRAIN_PROP)

train_df = pl.concat([pos_train_df, neg_train_df]).sample(fraction=1, seed=1337, shuffle=True)
test_df =  pl.concat([pos_test_df, neg_test_df]).sample(fraction=1, seed=1337, shuffle=True)


train_df.write_parquet(f'../data/processed/train_cg_{WINDOW_SIZE}_{NUM_READS_TO_PROCESS}.parquet')
test_df.write_parquet(f'../data/processed/test_cg_{WINDOW_SIZE}_{NUM_READS_TO_PROCESS}.parquet')