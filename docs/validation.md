# Organization validation — 2026-11-16

Editable installation and structural/import checks were performed on Python 3.12. Six projects have three migration checks each; the existing MCMC-PESP suite has 423 passing tests. All input, report and retained output contents are preserved. Function bodies are preserved after normalization of import and input/output path changes. MCMC source, configs and mathematical specification are unchanged; Mancala rule and agent bodies are unchanged.

Tests run in an isolated environment (NumPy 1.26.4, SciPy 1.17.1). This is not a lockfile or a reconstruction of the old environment. Full financial experiments, GUI interaction and agent tournaments were not run. Known Mancala issues and Multiple-Decrement experiment parameter issues remain documented, not repaired by this file-classification change.
