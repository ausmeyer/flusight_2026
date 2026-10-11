Update MIGHTE-Base, MIGHTE-Linear, and MIGHTE-Nsemble metadata for 2026–27. Contributors: Austin Meyer and Mauricio Santillana.

- **MIGHTE-Base:** pooled LightGBM for hospitalizations, ED visits, and seasonal hospitalization peaks; a separate multiclass fit predicts hospitalization trends.
- **MIGHTE-Linear:** partially pooled autoregression for hospitalizations.
- **MIGHTE-Nsemble:** fixed equal-weight quantile ensemble of two-stage LightGBM bases and MIGHTE-Linear.

Base and Nsemble are designated; Linear is submitted for evaluation.

Metadata passed validation against the current hub schema. No forecast files are included.
