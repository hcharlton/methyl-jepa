import pysam
import polars as pl
import re

# --- Configuration ---
BAM_PATH = "../data/raw/unmethylated_subset.bam" 
NUM_READS_TO_PROCESS = 10000            
WINDOW_SIZE = 64                

required_tags = {"fi", "ri", "fp", "rp"}

# --- Script ---
records = []
flank_size = (WINDOW_SIZE - 2) // 2



with pysam.AlignmentFile(BAM_PATH, "rb", check_sq=False) as bam:
    # Iterate through a limited number of reads for a quick test
    for i, read in enumerate(bam):
        if i >= NUM_READS_TO_PROCESS:
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

# Create a polars DataFrame from the collected records
# The schema will be automatically inferred.
df = pl.from_dicts(records)

df.write_parquet('../data/processed/unmethylated_cg_context_small.parquet')