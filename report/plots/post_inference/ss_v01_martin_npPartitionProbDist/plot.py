import polars as pl
import altair as alt
import pathlib
alt.data_transformers.enable("vegafusion")

from methyl_jepa.paths import RESULTS_DIR, INFERENCE_DATA_DIR
da1_results_path=  RESULTS_DIR/'martin_inference.parquet'
da1_source_path =  INFERENCE_DATA_DIR/'martin.parquet'



da1_source_q = (
    pl.scan_parquet(da1_source_path)
    .select(['np', 'read_name', 'cg_pos', 'qual'])
    .with_columns((pl.col('read_name') + pl.col('cg_pos').cast(pl.String)).alias('id'))
)

aggs = [
    (
        ((pl.col('prob') < 0.1) | (pl.col('prob') > 0.9))
        .filter(pl.col("np") > i)
        .sum()
        / pl.col("np").filter(pl.col("np") > i).len()
    ).alias(f"outer_sum_np_gt_{i}")
    for i in range(1, 40)
]

join_q = (
    pl.scan_parquet(da1_results_path)
    .with_columns((pl.col('read_name') + pl.col('pos').cast(pl.String)).alias('id'))
    .join(how='inner', other=da1_source_q, on='id')
    .select(aggs)
    .unpivot(variable_name='partition', value_name='high_confidence_prop')
    .with_columns(
        pl.col("partition").str.extract(r"_(\d+)$", 1).cast(pl.Int64).alias("np_cutoff")
        )
    )
df = join_q.collect()

print(df.shape)
print(df.head)



chart = alt.Chart(df).mark_line().encode(
    alt.Y('high_confidence_prop').title('Pr(0.1>methProb>0.9)'),
    alt.X('np_cutoff').title('lower bound for np'),
).properties(
    width=600,
    height=600,
)


chart.save(pathlib.Path(__file__).with_name("ss_v01_martin_npPartitionProbDist.svg"))