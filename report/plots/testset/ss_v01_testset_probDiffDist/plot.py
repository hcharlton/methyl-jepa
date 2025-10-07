import polars as pl
import altair as alt
alt.data_transformers.enable("vegafusion")



data_path='../../../../results/testset_inference.parquet'

output_path='ss_v01_testset_strandProbDiffDist.svg'

LOW_DROP = 0.1
HIGH_DROP = 0.9

q = (
    pl.scan_parquet(data_path)
    # .head(1_000_000)
    .with_columns((pl.col('read_name') + pl.col('pos').cast(pl.String)).alias('id'))
)

df = (
    q.collect()
    .filter(
        (pl.col('prob') >= HIGH_DROP) | (pl.col('prob') <= LOW_DROP)
    )
    .drop_nulls()
    .pivot(
    on="strand",          
    index=["id"],         
    values="prob",        
    aggregate_function="first", 
    )
    .with_columns(
        abs((pl.col('fwd')-pl.col('rev')).alias('deltap'))
    )
    .drop_nulls()
)

print(df.head())

chart = alt.Chart(df).mark_bar().encode(
    alt.X('deltap').bin(maxbins=200).title('abs(P(fwd) - P(rev))'),
    alt.Y('count()').scale(type="log"),
).properties(
    width=600,
    height=600,
    title=f'Apply "{HIGH_DROP} < p(meth) < {LOW_DROP}" -> Per-Site P(fwd_meth) - P(rev_meth)'
)


chart.save(output_path)