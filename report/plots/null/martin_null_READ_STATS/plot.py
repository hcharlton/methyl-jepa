import altair as alt
import polars as pl
import os
alt.data_transformers.enable('vegafusion')

q = (
    pl.scan_parquet(os.path.expanduser('~/mutationalscanning/Workspaces/chcharlton/methyl-jepa/data/processed/null/martin_null_p0.24_n1m.parquet'))
    .select(['read_name', 'np'])
    .unique(subset=['read_name'], keep='first')
    )
df = q.collect()

chart = alt.Chart(df).mark_bar().encode(
    alt.X('np').bin(maxbins=100).title('CCS Number of Passes (np)'),
    alt.Y('count()').title('Count')
).properties(
    width=500,
    height=500,
    title='martin-null Number of Passes Distribution'
)

chart.save('./martin_null_READ_STATS.svg')