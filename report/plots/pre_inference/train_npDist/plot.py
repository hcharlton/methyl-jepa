import polars as pl
import altair as alt
import os
alt.data_transformers.enable("vegafusion")


q = (
  pl.scan_parquet(os.path.expanduser('~/mutationalscanning/Workspaces/chcharlton/methyl-jepa/data/01_processed/train_sets/pacbio_standard_train.parquet'))
  .select('np', 'read_name')
  .unique(subset=["read_name"], maintain_order=True)
  # .head(10_000)
  )
df = q.collect()
print(df.head())


chart = alt.Chart(df).mark_bar().encode(
      alt.X('np:Q').title('Passes'),
      alt.Y('count():Q').scale(type='log').title('count'),
  ).properties(
      width=500,
      height=500,
  )
chart.save('./trainset_npDist.svg')