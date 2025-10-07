import polars as pl
import altair as alt

data_path='../../../../results/testset_inference.parquet'

output_path='ss_v01_testset_probDist.svg'

q = (
    pl.scan_parquet(data_path)
    .head(1_000_000)
    .with_columns((pl.col('read_name') + pl.col('pos').cast(pl.String)).alias('id'))
)

df = (
    q.collect()
    # .pivot(
    # on="strand",          
    # index=["id"],         
    # values="prob",        
    # aggregate_function="first", 
    # )
    # .with_columns(
    #     (pl.col('fwd')-pl.col('rev')).alias('deltap')
    # )
)

print(df.head())

chart = alt.Chart(df).mark_bar().encode(
    alt.X('prob').bin(maxbins=200).title('P(methylation)'),
    alt.Y('count()'),
).properties(
    width=600,
    height=600,
    title='Test Set Inferred SS P(methylation) Distribution'
)


chart.save(output_path)