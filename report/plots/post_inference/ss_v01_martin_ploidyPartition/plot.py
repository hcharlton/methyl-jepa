import polars as pl
import altair as alt
import pathlib
alt.data_transformers.enable("vegafusion")
from methyl_jepa.paths import RESULTS_DIR, INFERENCE_DATA_DIR, ANALYSIS_DIR


martin_results_path=  RESULTS_DIR/'martin_inference.parquet'
martin_source_path =  INFERENCE_DATA_DIR/'martin.parquet'
martin_ploidy_path = ANALYSIS_DIR/'martin_read_labels.txt'

ploidy_q = (
    pl.scan_csv(
    martin_ploidy_path,
    separator=" ",
    has_header=False,
    new_columns=["read_name", "ploidy"]
    )
)

LOW_DROP = 0.1
HIGH_DROP = 0.9

source_q = (
    pl.scan_parquet(martin_source_path)
    .select(['np', 'read_name', 'cg_pos', 'qual'])
    .join(other=ploidy_q, on='read_name', how='left')
    .with_columns((pl.col('read_name') + pl.col('cg_pos').cast(pl.String)).alias('id'))
)

results_q = (
    pl.scan_parquet(martin_results_path)
    .with_columns((pl.col('read_name') + pl.col('pos').cast(pl.String)).alias('id'))
    .join(other=source_q, on='id', how='inner')
    .filter(pl.col('np')>20)
)

# df = (
#     results_q.collect()
#     .filter(
#         (pl.col('prob') >= HIGH_DROP) | (pl.col('prob') <= LOW_DROP)
#     )
#     .drop_nulls()
#     .pivot(
#     on="strand",          
#     index=["id"],         
#     values="prob",        
#     aggregate_function="first", 
#     )
#     .with_columns(
#         abs((pl.col('fwd')-pl.col('rev')).alias('deltap'))
#     )
#     .drop_nulls()
# )

df = results_q.collect()


print(df.shape)
print(df.head())

chart = alt.Chart(df).mark_bar().encode(
    alt.X('prob').bin(maxbins=200).title('P(methylation)'),
    alt.Y('count()').title('count'),
).properties(
    width=600,
    height=600,
).facet(
    column='ploidy:N'
).resolve_scale(
    y='independent'
)





chart.save(pathlib.Path(__file__).with_name("ss_v01_martin_ploidyPartitionNpG20.svg"))