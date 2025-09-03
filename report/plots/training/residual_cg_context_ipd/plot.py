import polars as pl
import altair as alt


pos_path='../../data/processed/methylated_cg_context_small.parquet'
neg_path='../../data/processed/unmethylated_cg_context_small.parquet'

output_path='residual_cg_context_ipd.svg'

pos_means = (
    pl.read_parquet(pos_path)
    .select(pl.col("window_fi").alias("fwd"), pl.col("window_ri").alias("rev"))
    .unpivot(on=["fwd", "rev"], variable_name="strand", value_name="ipd_list")
    .with_columns(index=pl.int_ranges(start=0, end=pl.col("ipd_list").list.len()))
    .explode("index", "ipd_list")
    .rename({"ipd_list": "ipd"})
    .group_by("index", "strand").agg(pl.col("ipd").mean())
    # .with_columns(pl.lit('True').alias('methylated'))
)

neg_means = (
    pl.read_parquet(neg_path)
    .select(pl.col("window_fi").alias("fwd"), pl.col("window_ri").alias("rev"))
    .unpivot(on=["fwd", "rev"], variable_name="strand", value_name="ipd_list")
    .with_columns(index=pl.int_ranges(start=0, end=pl.col("ipd_list").list.len()))
    .explode("index", "ipd_list")
    .rename({"ipd_list": "ipd"})
    .group_by("index", "strand").agg(pl.col("ipd").mean())
    # .with_columns(pl.lit('False').alias('methylated'))
)

means = pos_means.join(
    neg_means, on=['index', 'strand'], suffix='_neg'
    ).with_columns((pl.col('ipd')-pl.col('ipd_neg')).alias('residual'))
# means = pl.concat([pos_means,neg_means])
# means.unpivot(on)

chart = alt.Chart(means).mark_line().encode(
    alt.X("index:Q", title="Position", axis=alt.Axis(tickCount=16)),
    alt.Y("residual:Q", title="IPD Mean Difference", scale=alt.Scale(domain=(-3, 15), clamp=True)),
    alt.Color("strand:N", title="Strand"),
    ).properties(
    title = "Mean IPD Residual (Meth-Unmeth) Across CG Context", 
    width=800, 
    height=600
    ).configure_axis(
    labelFontSize=12, 
    titleFontSize=14, 
    grid=True
    ).configure_title(
    fontSize=16, 
    anchor='middle'
    ).configure_legend(
    titleFontSize=12, 
    labelFontSize=11
    )

chart.save(output_path)
# print(means.head())

