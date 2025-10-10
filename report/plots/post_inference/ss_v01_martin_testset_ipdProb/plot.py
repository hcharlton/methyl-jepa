import polars as pl
import altair as alt
import pathlib
from methyl_jepa.paths import RESULTS_DIR, INFERENCE_DATA_DIR, TRAIN_DATA_DIR

alt.data_transformers.enable("vegafusion")

q_martin_x = (
    pl.scan_parquet(RESULTS_DIR/'martin_inference.parquet')
    # .head(100_000)
    .with_columns((pl.col('read_name')+pl.col('pos').cast(pl.String)).alias('site_id'))
    .select(['site_id','strand', 'prob'])
    # .unpivot(index = 'siteid')
    )

q_martin_join = (
    pl.scan_parquet(INFERENCE_DATA_DIR/'martin.parquet')
    # .head(100_000)
    .with_columns((pl.col('read_name')+pl.col('cg_pos').cast(pl.String)).alias('site_id'))
    .join(other=q_martin_x, on='site_id', how='inner')
    .with_columns((pl.col('fi').list[18]).alias('fi_18'))
    .select(['prob','fi_18', 'strand'])
    .filter(pl.col('strand')=='fwd')
    .group_by(pl.col('fi_18'))
    .agg(pl.col("prob").mean())
    .with_columns((pl.lit('martin')).alias('sample'))
    )


q_testset_x = (
    pl.scan_parquet(RESULTS_DIR/'testset_inference.parquet')
    # .head(100_000)
    .with_columns((pl.col('read_name')+pl.col('pos').cast(pl.String)).alias('site_id'))
    .select(['site_id','strand', 'prob'])
    )

q_testset_join = (
    pl.scan_parquet(TRAIN_DATA_DIR/'pacbio_standard_test.parquet')
    # .head(100_000)
    .with_columns((pl.col('read_name')+pl.col('cg_pos').cast(pl.String)).alias('site_id'))
    .join(other=q_testset_x, on='site_id', how='inner')
    .with_columns((pl.col('fi').list[18]).alias('fi_18'))
    .select(['prob','fi_18', 'strand', 'label'])
    .filter(pl.col('strand')=='fwd')
    .group_by([pl.col('fi_18'), pl.col('label')])
    .agg(pl.col("prob").mean())
    .with_columns(pl.concat_str([pl.lit('testset_'), pl.col('label')]).alias('sample'))
    .drop(pl.col('label'))
    )

print(q_testset_join.collect().head())

martin_join = q_martin_join.collect()
test_join = q_testset_join.collect()
df = martin_join.extend(test_join)
print(df.head())
df.write_csv(pathlib.Path(__file__).with_name("martin_testset_fi_18_prob_mean.csv"))

chart = alt.Chart(df).mark_line().encode(
    alt.X("fi_18:Q").title('Forward IPD at Index 18'),
    alt.Y("prob:Q").scale(zero=True).title('Inferred P(Methylation)'),
    alt.Color('sample')
).properties(
    height=500,
    width=800
)

chart.save(pathlib.Path(__file__).with_name("ss_v01_martin_testset_ipdProb.svg"))