# Reproducibility Package

This package records the locked protocol, data audit, environment, best hyperparameters, run commands, and reporting checklists for the OpenSees surrogate-model publication evaluation.

## Minimal Rebuild

Run the commands in `run_commands.md` in order. The scripts are designed to avoid changing the locked test split and to write auditable CSV/JSON/Markdown artifacts under `publication_eval_20260614/`.

## Important Boundaries

- High-drift independent stress labels are not yet present; use the TODO10 OpenSees generation plan before making high-drift safety claims.
- Controlled inference latency and OpenSees speedup are not measured in the current package.
- TODO12 SHAP/permutation was downgraded to model-free feature association because h5py was blocked by local application-control policy.
