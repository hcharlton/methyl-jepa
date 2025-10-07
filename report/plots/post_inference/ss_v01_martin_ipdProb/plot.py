import polars as pl
import altair as alt
alt.data_transformers.enable("vegafusion")

q_x = (
    pl.scan_parquet('../../../../results/martin_inference.parquet')
    .head(1000)
    .with_columns((pl.col('read_name')+pl.col('pos').cast(pl.String)).alias('site_id'))
    .select(['site_id','strand', 'prob'])
    # .unpivot(index = 'siteid')
    )

q_join = (
    pl.scan_parquet('../../../../data/01_processed/inference_sets/martin.parquet')
    .head(1000)
    .with_columns((pl.col('read_name')+pl.col('cg_pos').cast(pl.String)).alias('site_id'))
    .join(other=q_x, on='site_id', how='inner')
    )

df_join = q_join.collect()
print(df_join.head())

# chart = alt.Chart(df_join).mark_