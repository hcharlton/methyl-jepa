import polars as pl
import altair as alt
import pathlib
from methyl_jepa.paths import INFERENCE_DATA_DIR

alt.data_transformers.enable("vegafusion")

import os

def human_format(num):
    num = float('{:.3g}'.format(num))
    magnitude = 0
    while abs(num) >= 1000:
        magnitude += 1
        num /= 1000.0
    return '{}{}'.format('{:f}'.format(num).rstrip('0').rstrip('.'), ['', 'K', 'M', 'B', 'T'][magnitude])

q = (pl.scan_parquet(os.path.expanduser(INFERENCE_DATA_DIR/'da1.parquet'))
     .select(['seq'])
)
df = q.collect()



def plot_cg_distribution(df, scale='linear'):
  df_counts= df.with_columns(
      pl.col("seq").str.count_matches("CG").alias("per_sample_cg_count")
  )
  chart = alt.Chart(df_counts).mark_bar().encode(
      alt.X('per_sample_cg_count:O').bin(),
      alt.Y('count():Q').scale(type=scale)
  ).properties(
      width=700,
      height=500,
      title=f'da1-1m CG Distribution for {human_format(len(df))} Samples'
  )
  return chart

chart = plot_cg_distribution(df, scale='log')

chart.save(pathlib.Path(__file__).with_name("da1_1m_CG-DIST.svg"))