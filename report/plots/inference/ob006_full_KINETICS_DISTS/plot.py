import polars as pl
import altair as alt
import os
alt.data_transformers.enable("vegafusion")



KINETICS_FEATURES = ['fi', 'ri', 'fp', 'rp']

q = (
  pl.scan_parquet(os.path.expanduser('~/mutationalscanning/Workspaces/chcharlton/kinetics_modelling/data/processed/ob006-run2_full'))
#   .select(KINETICS_FEATURES)
  .head(1_000)
  )
df = q.collect()
print(df.schema)


# def plot_feature_dists(df, features, log_scale=True):
#   if log_scale:
#     scale_type = 'log'
#   else:
#     scale_type = 'linear'

#   plotting_df = df.explode(features).unpivot(on=features, variable_name = 'kinetics_feature')

#   chart = alt.Chart(plotting_df).mark_bar().encode(
#       alt.X('value:Q').title('zmw frames'),
#       alt.Y('count():Q').scale(type=scale_type).title('count'),
#   ).properties(
#       width=400,
#       height=400,
#   ).facet(
#       column='kinetics_feature:N'
#   ).properties(
#       title="ob006-full Kinetics Features Distributions"
#   )

#   return chart

# chart = plot_feature_dists(df, KINETICS_FEATURES, log_scale=False)
# chart.save('./ob006_full_KINETICS_DISTS.svg')