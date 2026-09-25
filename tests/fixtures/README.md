These forecasts were generated with the original `joint_twostage_distribution_study` implementation on a deterministic synthetic three-location history, using the settings in `settings.json`. The data generator lives in `tests/conftest.py`; synthetic covariates are defined in `test_golden_fits.py`.

Both feature tables and the original/new output arrays were compared directly on September 25, 2026. Maximum absolute difference was **0.0** for the distributional LightGBM forecasts and **0.0** for the partially pooled linear forecasts, in the pinned Python environment. CI allows small cross-platform numerical differences. These are unit-test reference values, not retrospective forecasts or prospective performance evidence.

Original source hashes are in `docs/port-provenance.json`. No tests require access to that source directory.
