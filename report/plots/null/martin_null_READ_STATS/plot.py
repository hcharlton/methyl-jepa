import altair as alt
import polars as pl
import os
alt.data_transformers.enable('vegafusion')

q = (
    pl.scan_parquet(os.path.expanduser('~/mutationalscanning/Workspaces/chcharlton/methyl-jepa/data/processed/null/martin_null_p0.24_n1m.parquet'))
    .select(['read_name']
)