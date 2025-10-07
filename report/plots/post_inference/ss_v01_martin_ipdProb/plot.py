import polars as pl
import altair as alt
from src.paths import RESULTS_DIR, INFERENCE_DATA_DIR

alt.data_transformers.enable("vegafusion")

q_x = (
    pl.scan_parquet(RESULTS_DIR/'martin_inference.parquet')
    .head(1000)
    .with_columns((pl.col('read_name')+pl.col('pos').cast(pl.String)).alias('site_id'))
    .select(['site_id','strand', 'prob'])
    # .unpivot(index = 'siteid')
    )

q_join = (
    pl.scan_parquet(INFERENCE_DATA_DIR/'martin.parquet')
    .head(1000)
    .with_columns((pl.col('read_name')+pl.col('cg_pos').cast(pl.String)).alias('site_id'))
    .join(other=q_x, on='site_id', how='inner')
    )

df_join = q_join.collect()
print(df_join.head())

# chart = alt.Chart(df_join).mark_