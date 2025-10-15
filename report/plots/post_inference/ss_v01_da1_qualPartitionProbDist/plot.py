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
    .select(['np', 'read_name', 'cg_pos', 'qual'])
    .with_columns(
        pl.col('qual').list.get(15).alias('qual_16'),
        (pl.col('read_name') + pl.col('cg_pos').cast(pl.String)).alias('id'))
    .drop('qual')
)

aggs = [
    (
        ((pl.col('prob') < 0.1) | (pl.col('prob') > 0.9))
        .filter(pl.col("qual_16") > i)
        .sum()
        / pl.col("qual_16").filter(pl.col("qual_16") > i).len()
    ).alias(f"outer_sum_qual_gt_{i}")
    for i in range(20, 92)
]

join_q = (
    pl.scan_parquet(da1_results_path)
    .with_columns((pl.col('read_name') + pl.col('pos').cast(pl.String)).alias('id'))
    .join(how='inner', other=da1_source_q, on='id')
    .select(aggs)
    .unpivot(variable_name='partition', value_name='high_confidence_prop')
    .with_columns(
        pl.col("partition").str.extract(r"_(\d+)$", 1).cast(pl.Int64).alias("qual_cutoff")
        )
    )
df = join_q.collect()

print(df.shape)
print(df.head)



chart = alt.Chart(df).mark_line().encode(
    alt.Y('high_confidence_prop').title('Pr(0.1>methProb>0.9)'),
    alt.X('qual_cutoff').title('lower bound for qual_16'),
).properties(
    width=600,
    height=600,
)


chart.save(pathlib.Path(__file__).with_name("ss_v01_da1_qualPartitionProbDist.svg"))