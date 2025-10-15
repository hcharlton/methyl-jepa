import polars as pl
import altair as alt
import pathlib
alt.data_transformers.enable("vegafusion")

from methyl_jepa.paths import RESULTS_DIR, INFERENCE_DATA_DIR
da1_results_path=  RESULTS_DIR/'da1_inference.parquet'
da1_source_path =  INFERENCE_DATA_DIR/'da1.parquet'

output_path='ss_v01_da1_strandProbDist.svg'

da1_source_q = (
    pl.scan_parquet(da1_source_path)
    .select(['np','read_name', 'cg_pos'])
    .with_columns((pl.col('read_name') + pl.col('cg_pos').cast(pl.String)).alias('id'))
)
join_q = (
    pl.scan_parquet(da1_results_path)
    .with_columns((pl.col('read_name') + pl.col('pos').cast(pl.String)).alias('id'))
    .join(how='inner', other=da1_source_q, on='id')
    .select(['read_name','prob','np'])
    .filter(pl.col('np')>30)
    .head(1_000_000)
)

df = (
    join_q.collect()
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
    alt.Y('count()').title('count'),
).properties(
    width=600,
    height=600,
)


chart.save(pathlib.Path(__file__).with_name("ss_v01_da1_npG13ProbDist.svg"))