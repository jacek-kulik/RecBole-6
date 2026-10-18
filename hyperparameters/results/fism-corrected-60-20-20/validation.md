# Verification

- Base: remote master `94d3610e`; source and runner commit: `71156ec7`.
- `.venv/Scripts/python.exe -m unittest discover -s project/tests -p test_fism.py -v`: all 3 regression tests pass. They check raw training logits, BCE loss and gradients, one sigmoid for pointwise prediction, full-sort score consistency, and equivalence of history aggregation scores and gradients for all three searched alpha values.
- `.venv/Scripts/python.exe -m unittest discover -s project/tests -q`: 61 of 62 tests pass. The unrelated `test_validate.ValidateTests.test_cli_routes_options_and_reports_errors` assertion expects the literal English text `File exists`; Windows instead reports `[WinError 183] Cannot create a file when that file already exists`. The CLI still reports the error and exits with code 2 as expected. No validation CLI code was changed.
- `git diff --check`: passes.
- Post-run artifact checks: 48 unique completed parameter combinations; all 48 checkpoints exist; recorded input hashes match the working files; the frozen winner is the maximum validation MRR@10; trial records contain no test metrics; the selected YAML loads the winning parameters. Training took 61.8 minutes in total.

The scoring optimization sums history embeddings before taking candidate dot products. `embedding_bag` skips padding in training, and a matrix-vector product avoids repeating the history for every candidate in full-sort prediction. The scoring equation, parameter shapes, checkpoint keys, and full-sort logits interface remain compatible.

The 48-trial validation search uses the original parameter grid and training budget under the current shared protocol. Test results are obtained only after validation selection is frozen. This run does not train or change the other five component caches or fit the Task 1.3 regression.

The historical `hyper-FISM.result` file contains 10 trials. Its baseline is the trial with the highest validation MRR@10, not the last trial or the highest test score. The new search covers all 48 grid combinations; the historical results also use a different split configuration, so their difference cannot be attributed solely to the logits correction.
