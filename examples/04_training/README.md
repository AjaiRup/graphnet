# Training examples

This subfolder contains two main training scripts:

**`01_train_dynedge.py`** Shows how to train a GNN on neutrino telescope data **without configuration files,** i.e., by programatically constructing the dataset and model used. This is good for debugging and experimenting with different dataset settings and model configurations, as it is easier to build the model using the API than by writing configuration files from scratch. **This is our recommended way of getting started with the library**. For instance, try running:

```bash
# Show the CLI
(graphnet) $ python examples/04_training/01_train_dynedge.py --help

# Train energy regression model
(graphnet) $ python examples/04_training/01_train_dynedge.py

# Train using a single GPU
(graphnet) $ python examples/04_training/01_train_dynedge.py --gpus 0

# Train using multiple GPUs
(graphnet) $ python examples/04_training/01_train_dynedge.py --gpus 0 1
```

**`03_train_model_dynedge_from_config.py`** Shows how to train a GNN on neutrino telescope data **using configuration files** to construct the dataset that loads the data and the model that is trained. This is the recommended way to configure standard dataset and models, as it is easier to ready and share than doing so in pure code. This example can be run using a few different models targeting different physics use cases. For instance, you can try running:

```bash
# Show the CLI
(graphnet) $ python examples/04_training/03_train_dynedge_from_config.py --help

# Train energy regression model
(graphnet) $ python examples/04_training/03_train_dynedge_from_config.py

# Same as above, as this is the default model config.
(graphnet) $ python examples/04_training/03_train_dynedge_from_config.py \
    --model-config configs/models/example_energy_reconstruction_model.yml

# Train a vertex position reconstruction model
(graphnet) $ python examples/04_training/03_train_dynedge_from_config.py \
    --model-config configs/models/example_vertex_position_reconstruction_model.yml

# Trains a direction (zenith, azimuth) reconstruction model. Note that the
# chosen `Task` in the model config file also returns estimated "kappa" values,
# i.e. inverse variance, for each predicted feature, meaning that we need to
# manually specify the names of these.
(graphnet) $ python examples/04_training/03_train_dynedge_from_config.py --gpus 0 \
    --model-config configs/models/example_direction_reconstruction_model.yml  \
    --prediction-names zenith_pred zenith_kappa_pred azimuth_pred azimuth_kappa_pred
```

## IceMix joint reconstruction

IceMix uses one shared CLS token, vertex-only LayerNorm, global vertex targets
scaled by 100 m, and continuous RDE features. Architecture and pulse sampling
are fixed; the model config contains only loss, optimizer, scheduler, and
dropout settings. Explicit CLI tuning arguments override that config.
Training defaults to 80 epochs, rotation enabled, and no early stopping.

From the checkout root:

```bash
export DATA_ROOT=/path/to/databases
mkdir -p slurm/logs
RUN_NAME=icemix-seed42 sbatch slurm/train_icemix.sh \
    --split-cache-dir outputs/vertex-ablation/rde-vertex-ablation-v2/cache/splits \
    --length-cache-dir outputs/vertex-ablation/rde-vertex-ablation-v2/cache/lengths
```

Keep the existing seed-42 80/10/10 split cache. `--resume-from` restores an
interrupted single-stage run. Use `--help` for training controls; model config
values apply unless explicitly overridden. `--smoke-test` uses the small
fixture with the full model and belongs on a compute node.

The direction-best checkpoint minimizes median angular error at true energy
>=10 GeV. Post-training predictions use the last epoch and record that choice;
evaluate a selected checkpoint explicitly for comparisons:

```bash
sbatch slurm/predict_eval.sh \
    --checkpoint outputs/my-run/checkpoints/best-direction-76.ckpt \
    --model-config outputs/my-run/model_config.yml --split test \
    --split-cache-dir outputs/vertex-ablation/rde-vertex-ablation-v2/cache/splits \
    --output outputs/my-run/predictions/selected.csv
python examples/04_training/11_plot_icemix_results.py \
    --results outputs/my-run/predictions/selected.csv \
    --metrics-csv outputs/my-run/logs/training/version_0/metrics.csv
```

Prediction exports include truth, event IDs, source database, and a metadata
sidecar recording checkpoint/config, split, and seed. Evaluation uses FP32.
`python diagnostics/predict_eval.py --analyze-only results.csv` summarizes an
existing export without importing the model.

Legacy shared-query configs require one-time conversion on a compute node. Other
architectures are rejected; original files are never overwritten:

```bash
python diagnostics/migrate_icemix.py --model-config /path/to/legacy-run/model_config.yml \
    --checkpoint /path/to/legacy-run/checkpoints/best-direction-76.ckpt \
    --output-dir outputs/migrated-icemix
```

The converter writes `model_config.yml`, `state_dict.pth`, and provenance.
It migrates inference weights, not optimizer/trainer state.

```bash
sbatch slurm/verify_cleanup.sh
```

This bounded verification job needs no production databases. It compares the
saved pre-cleanup source against the migrated model, runs regression tests, and runs
the five-event fixture smoke test. Results appear in
`outputs/cleanup_validation/run-<job>-*/`: `parity.json`, `tests.log`,
`smoke.log`, and `prediction.log`. `REFERENCE_SOURCE`, `REFERENCE_CHECKPOINT`, and `CONDA_ENV` can
override the job defaults. Historical runs and source snapshots remain archived.
