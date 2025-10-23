import pysam
import polars as pl
import re
import argparse
import sys
import os
from array import array

### Purpose ###
# converts a BAM file into a parquet file of CpG instances. 

### example usage ###
## short format
# python make_inference_dataset.py -n 1000 -c 32 -o output_file_name_str -r
## long format
# python make_inference_dataset.py --output-name output_file_name_str --context 32 --n_reads 1000 --restrict-instances

### Note for the reverse strand indexing ###
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

REQUIRED_TAGS = {"fi", "ri", "fp", "rp",}

DTYPE_MAP = {
    "read_name": pl.String,
    "cg_pos": pl.UInt16,
    "seq": pl.String,
    "qual": pl.List(pl.UInt8),
    "np": pl.UInt8,
    "sm": pl.List(pl.UInt8),
    "sx": pl.List(pl.UInt8),
    "fi": pl.List(pl.UInt8),
    "fp": pl.List(pl.UInt8),
    "ri": pl.List(pl.UInt8),
    "rp": pl.List(pl.UInt8),
}
# optional tags {"np", "sm", "sx"}

def _process_read(read, all_tags, required_tags):
    """
    Processes a single pysam.AlignmentRead.
    Checks for required tags and extracts all full-length tag data.
    Returns:
        A dictionary with full-length data ('seq', 'qual', 'tag_data'),
        or None if the read is invalid (missing/inconsistent tags).
    """
    # check required tags existence
    if not all(read.has_tag(tag) for tag in required_tags):
        return None
    read_tag_data = {}

    # iterate through all tags
    for tag in all_tags:
        value = read.get_tag(tag) if read.has_tag(tag) else None
        # if isinstance(value, array):
        #     value = value.tolist()
        read_tag_data[tag] = value

    # check validity of tag values
    if any(read_tag_data[tag] is None or len(read_tag_data[tag]) == 0 for tag in required_tags):
        return None
    # return the dict (the same keys regardless of optional tags)
    return {
        "name": read.query_name,
        "seq": read.query_sequence,
        "qual": list(read.query_qualities),
        "tag_data": read_tag_data
    }

def bam_to_df(bam_path: str, n_reads: int, context: int, singletons: bool, optional_tags: list):
    PER_BASE_TAGS = {"fi", "ri", "fp", "rp", "sm", "sx"}
    vocab = {'N': 1, 'A':2, 'T':3, 'C':4, 'G':5}
    required_tags = REQUIRED_TAGS
    optional_tags_set = set(optional_tags)
    all_tags = required_tags.union(optional_tags_set)

    final_cols = ["read_name", "cg_pos", "seq", "qual"] + list(all_tags)
    col_data = {key: [] for key in final_cols}
    
    counters = { "reads_processed": 0, "reads_skipped": 0, "cpg_windows_found": 0 }
    flank_size = (context - 2) // 2

    with pysam.AlignmentFile(bam_path, "rb", check_sq=False) as bam:
        for i, read in enumerate(bam):
            if i >= n_reads and n_reads != 0:
                break
            
            processed_read = _process_read(read, all_tags, required_tags)
            
            if processed_read is None:
                counters["reads_skipped"] += 1
                continue
            
            seq = processed_read["seq"]
            qual_values = processed_read["qual"]
            read_tag_data = processed_read["tag_data"]

            for match in re.finditer("CG", seq):
                L = len(read_tag_data["fi"])
                cg_pos = match.start()
                win_start = cg_pos - flank_size
                win_end = cg_pos + 2 + flank_size
                rev_win_start = L - win_end
                rev_win_end = L - win_start

                if not all([win_start >= 0, win_end <= L, rev_win_end >= 0, rev_win_start <= L]):
                    continue

                context_seq = seq[win_start:win_end]
                if singletons and context_seq.count("CG") != 1:
                    continue
                
                # Check that all sliced data will have the correct context length
                if len(context_seq) != context:
                    continue

                # --- Append data directly to the column lists ---
                col_data["read_name"].append(processed_read["name"])
                col_data["cg_pos"].append(cg_pos)
                col_data["seq"].append(context_seq)
                col_data["qual"].append(qual_values[win_start:win_end])

                for tag, values in read_tag_data.items():
                    if values is None:
                        # This should only happen for optional tags not present
                        col_data[tag].append(None) 
                        continue
                    
                    if tag in PER_BASE_TAGS:
                        sliced_values = values[rev_win_start:rev_win_end] if tag in {"ri", "rp"} else values[win_start:win_end]
                        col_data[tag].append(sliced_values)
                    else: # Non-per-base tags like 'np'
                        col_data[tag].append(values)

                counters["cpg_windows_found"] += 1
            counters["reads_processed"] += 1

    print("--- Debugging Counters ---")
    for key, value in counters.items():
        print(f"{key:<25}: {value}")
    print("--------------------------")

    if counters["cpg_windows_found"] == 0:
        print("Warning: No valid CpG windows were found. Returning an empty DataFrame.")
        return pl.DataFrame()
    
    final_schema = {k: v for k, v in DTYPE_MAP.items() if k in col_data and col_data[k]}
    df = pl.DataFrame(col_data, schema=final_schema)
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
    parser.add_argument('-t', '--optional_tags',
                        type=str,
                        nargs='*', 
                        default=[],
                        help='Space-separated list of optional tags to extract (e.g., np sm sx).')
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
                            singletons=args.singletons,
                            optional_tags=args.optional_tags)

    df.write_parquet(args.output_path)

if __name__ == "__main__":
    main()