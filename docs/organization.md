# Repository organization

The established src/core, cost, mcmc, io, metrics, visualization and experiments API is retained. This housekeeping does not change equations, adaptation modes, kernels or cost components. data holds inputs; kd_transit remains at its original path to preserve existing consumers; configs holds experiment settings; results holds retained research outputs.

Six identical synthetic-instance files also exist under src/experiments/benchmark/data. Both locations remain until their consumers are audited. Tracked pyc/checkpoints and machine-local IDE/assistant settings are removed from Git; original local project folders are untouched.

The formal source of truth remains docs/MATH_SPEC.md. TEST.md is a historical implementation plan; it is not the current test report. CLAUDE.md contains illustrative/planned paths that are not all implemented. Use the tracked tree and module --help for available entry points.
