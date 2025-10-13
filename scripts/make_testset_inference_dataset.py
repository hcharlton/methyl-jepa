import polars as pl
import argparse
import os
from methyl_jepa.paths import INFERENCE_DATA_DIR, TRAIN_DATA_DIR

def make_dataset(pos_ds_path, neg_ds_path, test_ds_path):
    test_q = (
        pl.scan_parquet(test_ds_path)
        .select('read_name')
        )

    test_df = test_q.collect()

    pos_q = (
        pl.scan_parquet(pos_ds_path)
        .filter(pl.col("read_name").is_in(test_df['read_name']))
        )

    neg_q = (
        pl.scan_parquet(neg_ds_path)
        .filter(pl.col("read_name").is_in(test_df['read_name']))
        )

    combined_q = (
        pl.concat([pos_q, neg_q], how='vertical')
    )

    combined_df = combined_q.collect()

    return combined_df 



def main():
    parser = argparse.ArgumentParser(
        description="Creates an inference dataset using reads from the test set. Does not include labels."
    )
    parser.add_argument('-p', '--positive-path',
                        type = str,
                        required=True,
                        help="Input filepath to the positive data")
    parser.add_argument('-n', '--negative-path',
                        type = str,
                        required=True,
                        help="Input filepath to the negative data")
    parser.add_argument('-t', '--test-path',
                        type = str,
                        required=True,
                        help="Input filepath to the test data")
    parser.add_argument('-o', '--output-path',
                        type = str,
                        required=True,
                        help="Output filepath")
    args = parser.parse_args()
    out_df = make_dataset(pos_ds_path=args.positive_path, 
                          neg_ds_path=args.negative_path, 
                          test_ds_path=args.test_path)
    out_df.write_parquet(args.output_path)

if __name__ == "__main__":
    main()