import polars as pl
import altair as alt
import pathlib
from methyl_jepa.paths import RESULTS_DIR, INFERENCE_DATA_DIR

alt.data_transformers.enable("vegafusion")

q_x = (
    pl.scan_parquet(RESULTS_DIR/'martin_inference.parquet')
    .head(100_000_000)
    .with_columns((pl.col('read_name')+pl.col('pos').cast(pl.String)).alias('site_id'))
    .select(['site_id','strand', 'prob'])
    # .unpivot(index = 'siteid')
    )

q_join = (
    pl.scan_parquet(INFERENCE_DATA_DIR/'martin.parquet')
    .head(100_000_000)
    .with_columns((pl.col('read_name')+pl.col('cg_pos').cast(pl.String)).alias('site_id'))
    .join(other=q_x, on='site_id', how='inner')
    .with_columns((pl.col('fi').list[18]).alias('fi_18'))
    .select(['prob', 'fi_18', 'strand'])
    .filter(pl.col('strand')=='fwd')
    # .with_columns((pl.col('fi').list[14]).alias('fi_14'))
    # .with_columns((pl.col('ri').list[18]).alias('ri_18'))
    # .with_columns((pl.col('ri').list[14]).alias('ri_14'))
    )

df_join = q_join.collect().sample(n=2_000_000, seed=0)
print(len(df_join))

chart = alt.Chart(df_join).mark_bar().encode(
    alt.X("fi_18:Q"),
    alt.Y("count():Q"),
).properties(
    height=500,
    width=500
)
chart.save(pathlib.Path(__file__).with_name("ss_v01_martin_ipd18Dist.svg"))