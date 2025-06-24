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
        title = "Methylated CG Context index-IPD Means", 
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
        input_path='../../data/processed/methylated_cg_context_small.parquet',
        output_path='pos_cg_context.svg'
    )