Register MIGHTE-Base and MIGHTE-Linear and update MIGHTE-Nsemble for 2026–27. Contributors: Austin Meyer and Mauricio Santillana.

- **MIGHTE-Base:** pooled two-stage LightGBM for hospitalization and ED forecasts, with a separate multiclass LightGBM component for hospitalization trends.
- **MIGHTE-Linear:** partially pooled autoregression for hospitalizations.
- **MIGHTE-Nsemble:** fixed equal-weight quantile ensemble of two-stage LightGBM bases and MIGHTE-Linear.

Base and Nsemble are designated; Linear is submitted for evaluation. MIGHTE-Joint's designation is retired; its forecast archive is retained.

Metadata passed validation against the current hub schema. No forecast files are included.
