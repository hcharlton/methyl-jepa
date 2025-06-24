import polars as pl
import altair as alt

def create_kinetics_plot(input_path: str, output_path: str) -> None:
    df_means = (
        pl.read_parquet(input_path)
        .select(pl.col("window_fi").alias("fwd"), pl.col("window_ri").alias("rev"))
        .unpivot(on=["fwd", "rev"], variable_name="strand", value_name="ipd_list")
        .with_columns(index=pl.int_ranges(start=0, end=pl.col("ipd_list").list.len()))
        .explode("index", "ipd_list")
        .rename({"ipd_list": "ipd"})
        .group_by("index", "strand").agg(pl.col("ipd").mean())
    )

    base = alt.Chart(df_means).mark_line().encode(
        alt.X("index:Q", title="Position", axis=alt.Axis(tickCount=16)),
        alt.Y("ipd:Q", title="Mean IPD", scale=alt.Scale(domain=(24, 44), clamp=True)),
        alt.Color("strand:N", title="Strand"),
        ).properties(
        title = "Unmethylated CG Context index-IPD Means", 
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
    base.save(output_path)


if __name__ == "__main__":
    create_kinetics_plot(
        input_path='../../data/processed/unmethylated_cg_context_small.parquet',
        output_path='neg_cg_context_ipd.svg'
    )














# import polars as pl
# import altair as alt
# import numpy as np

# df = pl.read_parquet('../../data/processed/unmethylated_cg_context_small.parquet')

# ipd_fwd_array_all = np.stack([s.to_numpy() for s in df['window_fi']])
# ipd_rev_array_all = np.stack([s.to_numpy() for s in df['window_ri']])

# df_long_fwd=pl.DataFrame(ipd_fwd_array_all, schema = [f'{i}' for i in range(64)]).unpivot(variable_name='index', value_name='ipd').cast({"index": pl.Int64})
# df_long_fwd = df_long_fwd.with_columns(strand = pl.lit('fwd'))
# df_long_rev=pl.DataFrame(ipd_rev_array_all, schema = [f'{i}' for i in range(64)]).unpivot(variable_name='index', value_name='ipd').cast({"index": pl.Int64})
# df_long_rev = df_long_rev.with_columns(strand = pl.lit('rev'))

# df_long = pl.concat([df_long_fwd, df_long_rev], how='vertical_relaxed')

# df_means = df_long.group_by('index', 'strand').agg(pl.col('ipd').mean())


# index_kinetics = alt.Chart(df_means).mark_line(point=True,).encode(
#     alt.X('index:O'),
#     alt.Y('ipd:Q', title = 'index-Wise IPD Mean').scale(domain=(24,44), clamp=True),
#     alt.Color('strand:N', legend=alt.Legend(title="fwd-rev strand"))
# ).properties(
#     title = 'Unmethylated CG Context index-IPD Means',
#     width = 800,
#     height = 600
# )

# index_kinetics.save('index_kinetics.svg')

