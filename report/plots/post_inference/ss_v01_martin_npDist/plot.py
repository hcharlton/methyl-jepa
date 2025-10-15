import polars as pl
import altair as alt
import pathlib
alt.data_transformers.enable("vegafusion")

from methyl_jepa.paths import RESULTS_DIR, INFERENCE_DATA_DIR
results_path=  RESULTS_DIR/'martin_inference.parquet'
source_path =  INFERENCE_DATA_DIR/'martin.parquet'

da1_source_q = (
    pl.scan_parquet(source_path)
    # .head(10_000)
    .select(['np', 'read_name', 'cg_pos', 'qual'])
    .with_columns(
        pl.col('qual').list.slice(14, 4).alias('qual_slice'))
    .with_columns(
        pl.col('qual_slice').list.min().alias('qual_slice_min'),
        (pl.col('read_name') + pl.col('cg_pos').cast(pl.String)).alias('id'))
    .drop('qual')
)

df = da1_source_q.collect()

chart = alt.Chart(df).mark_bar().encode(
    alt.X('np:Q').title('number of passes'),
    alt.Y('count():Q').scale(type='log').title('count')
).properties(
    width=600,
    height=600
)


chart.save(pathlib.Path(__file__).with_name("ss_v01_martin_npDist.svg"))