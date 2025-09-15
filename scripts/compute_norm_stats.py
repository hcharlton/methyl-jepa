import polars as pl
import os
from src.config import KINETICS_FEATURES
import yaml
import argparse

def compute_log_normalization_stats(df, features, epsilon=1):
    means = {col: (df[col].explode() + epsilon).log().mean() for col in features}
    stds = {col: (df[col].explode() + epsilon).log().explode().std() for col in features}
    output_dict = {'log_norm':{
        'means': means,
        'stds': stds
    }}
    return output_dict


def main():
    parser = argparse.ArgumentParser(
        description="Generates normalization statistics based on the train partition of methylation data"
    )
    parser.add_argument('-i', '--input_path',
                        type = str,
                        required=True,
                        help="Input filepath to the training data")
    parser.add_argument('-o', '--output_path',
                        type=str,
                        required=True,
                        help="Path to output file (including filename)")
    parser.add_argument('-t', '--truncate',
                        action='store_true',
                        help='If specified, only uses first 10_000 samples')
    args = parser.parse_args()
    print('began running compute stats')
    q = (
        pl.scan_parquet(os.path.expanduser(args.input_path),
                        schema = {'read_name': pl.String,
                                  'cg_pos': pl.Int64,
                                  'seq': pl.String,
                                  'fi': pl.List(pl.UInt16),
                                  'fp': pl.List(pl.UInt16),
                                  'ri': pl.List(pl.UInt16),
                                  'rp': pl.List(pl.UInt16),
                                  'label': pl.Int32,
                                  'qual': pl.List(pl.UInt8),
                                  'np': pl.UInt8,
                                  })
                                  )

    df = q.collect()
    print('collected df')
    exp_outpath = args.output_path
    stats_dict = compute_log_normalization_stats(df, KINETICS_FEATURES)
    print('computed stats')
    os.makedirs(os.path.dirname(exp_outpath), exist_ok=True)

    with open(exp_outpath, 'w') as f:
        yaml.dump(stats_dict, f, indent=4)

    print(f"Normalization stats saved to {exp_outpath}")
    print(yaml.dump(stats_dict, indent=4))


if __name__ == "__main__":
    main()