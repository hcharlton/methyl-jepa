import polars as pl
import altair as alt
alt.data_transformers.enable("vegafusion")
import os

def human_format(num):
    num = float('{:.3g}'.format(num))
    magnitude = 0
    while abs(num) >= 1000:
        magnitude += 1
        num /= 1000.0
    return '{}{}'.format('{:f}'.format(num).rstrip('0').rstrip('.'), ['', 'K', 'M', 'B', 'T'][magnitude])

q = (pl.scan_parquet(os.path.expanduser('~/mutationalscanning/Workspaces/chcharlton/methyl-jepa/data/processed/martin_1m.parquet'))
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
      title=f'Martin-1m CG Distribution for {human_format(len(df))} Samples'
  )
  return chart

chart = plot_cg_distribution(df, scale='log')

chart.save('./martin_1m_CG-DIST.svg')