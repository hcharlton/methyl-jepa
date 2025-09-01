import polars as pl
import altair as alt



data_path='../../../output/v0.4.7-train_df.parquet'

output_path='v4.7_strandProbScatter.svg'

q = (
    pl.scan_parquet(data_path)
    .head(1_000_000)
    .with_columns((pl.col('read_name') + pl.col('pos').cast(pl.String)).alias('id'))
)

df = (
    q.collect()
    .pivot(
    on="strand",          
    index=["id", 'label'],         
    values="prob",        
    aggregate_function="first", 
    )
)

chart = alt.Chart(df).mark_circle(opacity=0.02).encode(
    alt.X('fwd'),
    alt.Y('rev'),
    alt.Color('label:N')
).properties(
    width=600,
    height=600
)


chart.save(output_path)