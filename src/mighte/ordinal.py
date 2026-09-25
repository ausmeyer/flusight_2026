"""Integration point for the forthcoming ordinal hospitalization model.

Implement predict(context) below and set ordinal_plugin to "mighte.ordinal:predict"
in config/settings.json. Return the eight hub columns, only for
"wk flu hosp rate change", output_type "pmf", with all five categories for
every location/horizon you forecast. The runner appends these rows only to Base.

context contains reference_date, snapshot_path, hospitalization_history,
base_hospitalization_quantiles, and settings. All input data have been cut off
at reference_date minus seven days. Do not fetch new data from inside the plugin.
"""


def predict(context):
    raise NotImplementedError("Integrate the validated ordinal model here before enabling ordinal_plugin")
