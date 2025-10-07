import polars as pl
import altair as alt
import os
alt.data_transformers.enable("vegafusion")



KINETICS_FEATURES = ['fi', 'ri', 'fp', 'rp']

q = (
  pl.scan_parquet(os.path.expanduser('~/mutationalscanning/Workspaces/chcharlton/methyl-jepa/data/01_processed/inference_sets/martin.parquet'))
  .select('np', 'read_name')
  .unique(subset=["read_name"], maintain_order=True)
  # .head(10_000)
  )
df = q.collect()
print(df.head())


chart = alt.Chart(df).mark_bar().encode(
      alt.X('np:Q').title('Passes'),
      alt.Y('count():Q').scale(type='log').title('count'),
  ).properties(
      width=500,
      height=500,
  )
chart.save('./martin_npDist.svg')

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
#       column='kinetics_feature:N',
#       columns=2
#   ).properties(
#       title="Martin-p0.024-n1m Kinetics Features Distributions"
#   )

#   return chart

# chart = plot_feature_dists(df, KINETICS_FEATURES, log_scale=False)
# chart.save('./martin_null_KINETICS_DISTS.svg')




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
#       column='kinetics_feature:N',
#       columns=2
#   ).properties(
#       title="Martin-p0.024-n1m Kinetics Features Distributions"
#   )

#   return chart

# chart = plot_feature_dists(df, KINETICS_FEATURES, log_scale=False)
# chart.save('./martin_null_KINETICS_DISTS.svg')