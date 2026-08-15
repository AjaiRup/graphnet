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

`09_train_icemix_joint_reconstruction.py` trains direction and vertex position
with a shared IceMix backbone. Use the versioned Slurm launcher with a directory
containing `nue.db`, `numu.db`, and `nutau.db`:

```bash
export DATA_ROOT=/path/to/databases
export OUTPUT_ROOT=$PWD/outputs

RUN_NAME=icemix-stage1 \
sbatch -A <account> -q <qos> --gres=gpu:l40s:8 \
    slurm/train_icemix.sh --seed 42 --max-epochs 100 \
    --rotation-augmentation --length-cache-dir "$OUTPUT_ROOT/benchmarks" \
    --split-cache-dir "$OUTPUT_ROOT/splits"
```

Select the Stage-1 checkpoint by the median direction error for events with
energy at least 10 GeV, then calibrate vertex uncertainty without updating the
backbone:

```bash
RUN_NAME=icemix-stage2 \
sbatch -A <account> -q <qos> --gres=gpu:l40s:8 \
    slurm/train_icemix.sh --seed 42 --max-epochs 3 --lr 1e-3 \
    --stage2-from "$OUTPUT_ROOT/icemix-stage1/checkpoints/best-direction-65.ckpt" \
    --length-cache-dir "$OUTPUT_ROOT/benchmarks" \
    --split-cache-dir "$OUTPUT_ROOT/splits"
```

`configs/models/icemix_joint_reconstruction.yml` and
`configs/models/icemix_joint_reconstruction_stage2.yml` record the approved
architectures. Checkpoints, predictions, W&B logs, and caches remain external
artifacts. The approved runs are `final-seed42-rot-100` (W&B `2hg0nbcy`) and
`final-seed42-rot-100-stage2` (W&B `qrnot5gb`); both were created from commit
`aba079ec`.
