import polars as pl
import altair as alt
import pathlib
from methyl_jepa.paths import RESULTS_DIR, TRAIN_DATA_DIR

alt.data_transformers.enable("vegafusion")

q_x = (
    pl.scan_parquet(RESULTS_DIR/'testset_inference.parquet')
    .with_columns((pl.col('read_name')+pl.col('pos').cast(pl.String)).alias('site_id'))
    .select(['site_id','strand', 'prob'])
    )

q_join = (
    pl.scan_parquet(TRAIN_DATA_DIR/'pacbio_standard_test.parquet')
    .with_columns((pl.col('read_name')+pl.col('cg_pos').cast(pl.String)).alias('site_id'))
    .join(other=q_x, on='site_id', how='inner')
    .with_columns((pl.col('fi').list[18]).alias('fi_18'))
    # .with_columns((pl.col('fi').list[14]).alias('fi_14'))
    # .with_columns((pl.col('ri').list[18]).alias('ri_18'))
    # .with_columns((pl.col('ri').list[14]).alias('ri_14'))
    .select(['prob','fi_18', 'strand'])
    .filter(pl.col('strand')=='fwd')
    .group_by(pl.col('fi_18'))
    .agg(pl.col("prob").mean())  
    )

df_join = q_join.collect()
print(df_join.head())
df_join.write_csv(pathlib.Path(__file__).with_name("testset_fi_18_prob_mean.csv"))

chart = alt.Chart(df_join).mark_line().encode(
    alt.X("fi_18:Q").title('Forward IPD at Index 18'),
    alt.Y("prob:Q").scale(zero=True).title('Inferred P(Methylation)'),
).properties(
    height=500,
    width=800
)
chart.save(pathlib.Path(__file__).with_name("ss_v01_testset_ipdProb.svg"))