import polars as pl
import altair as alt
import pathlib
from methyl_jepa.paths import RESULTS_DIR, TRAIN_DATA_DIR

alt.data_transformers.enable("vegafusion")

q_x = (
    pl.scan_parquet(RESULTS_DIR/'testset_inference.parquet')
    .head(10_000_000)
    # .head(100_000)
    .with_columns((pl.col('read_name')+pl.col('pos').cast(pl.String)).alias('site_id'))
    .select(['site_id','strand', 'prob'])
    )

q_join = (
    pl.scan_parquet(TRAIN_DATA_DIR/'pacbio_standard_test.parquet')
    .head(10_000_000)
    # .head(100_000)
    .with_columns((pl.col('read_name')+pl.col('cg_pos').cast(pl.String)).alias('site_id'))
    .join(other=q_x, on='site_id', how='inner')
    .with_columns((pl.col('fi').list[18]).alias('fi_18'))
    .select(['prob','fi_18', 'strand'])
    .filter(pl.col('strand')=='fwd')
    )

df_join = q_join.collect()
print(df_join.head())

chart = alt.Chart(df_join).mark_boxplot().encode(
    alt.X("fi_18:Q", bin=alt.Bin(maxbins=50)),
    alt.Y("prob:Q").scale(zero=True),
).properties(
    height=500,
    width=800
)
chart.save(pathlib.Path(__file__).with_name("ss_v01_testset_ipdProb.svg"))