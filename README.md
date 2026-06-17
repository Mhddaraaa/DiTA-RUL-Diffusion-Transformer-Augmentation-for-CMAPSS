# Efficient Diffusion-Based Data Augmentation with Structured Latent Representations and Lifecycle-Aware Conditioning for Remaining Useful Life Prediction

This repository trains a CMAPSS RUL pipeline in four stages:

1. Preprocess raw CMAPSS text files into sliding-window `.npy` arrays.
2. Train the DTE/TSHAE latent model.
3. Train the diffusion model conditioned on DTE latents and lifecycle alpha.
4. Generate augmented windows and train RUL prediction models.

The plotting scripts can then visualize the DTE latent space and the diffusion denoising output for selected engines.

## Data Layout

Place the raw CMAPSS files in `./CMAPSSData`:

```text
CMAPSSData/
  train_FD001.txt
  test_FD001.txt
  RUL_FD001.txt
  train_FD002.txt
  test_FD002.txt
  RUL_FD002.txt
  ...
```

The default output directory is `./output`.

## One-Command Training Job

`run_training_job.py` runs the full training pipeline for `FD001` and `FD002` by default:

```bash
python run_training_job.py
```

Run all four CMAPSS subsets:

```bash
python run_training_job.py --fds FD001 FD002 FD003 FD004
```

Use shorter runs while testing the setup:

```bash
python run_training_job.py --fds FD001 FD002 --epochs 2 --batch-size 32
```

Run only part of the pipeline, for example diffusion through RUL training after DTE checkpoints already exist:

```bash
python run_training_job.py --fds FD001 FD002 --start-at diffusion --stop-after rul
```

Run training and then create both plot types:

```bash
python run_training_job.py --fds FD001 FD002 --plots --plot-units 99 46
```

Useful runner options:

```bash
python run_training_job.py \
  --fds FD001 FD002 \
  --data-dir ./CMAPSSData \
  --output-dir ./output \
  --window-size 30 \
  --step 1 \
  --dte-epochs 50 \
  --diffusion-epochs 50 \
  --rul-epochs 50 \
  --rul-eval-mode engine \
  --use-huber
```

Preview the commands without running them:

```bash
python run_training_job.py --fds FD001 FD002 --dry-run
```

## Step-by-Step Training

You can also run each stage manually.

### 1. Preprocess CMAPSS

```bash
python preprocess_cmapss.py \
  --fds FD001 FD002 \
  --data-dir ./CMAPSSData \
  --out-dir ./output/preprocessed \
  --window-size 30 \
  --step 1
```

Outputs include:

```text
output/preprocessed/preprocessed_data_FD001_train_ws30_step1.npy
output/preprocessed/preprocessed_ruls_FD001_train_ws30_step1.npy
output/preprocessed/preprocessed_alpha_FD001_train_ws30_step1.npy
output/preprocessed/preprocessed_units_FD001_train_ws30_step1.npy
```

### 2. Train DTE

```bash
python DTE_main.py \
  --fds FD001 FD002 \
  --output-dir ./output \
  --preprocessed-dir ./output/preprocessed \
  --window-size 30 \
  --step 1 \
  --epochs 50 \
  --batch-size 32 \
  --lr 1e-4
```

Checkpoints are saved as:

```text
output/best_vae_model_FD001.pt
output/best_vae_model_FD002.pt
```

### 3. Train Diffusion

```bash
python train_diffusion.py \
  --fds FD001 FD002 \
  --output-dir ./output \
  --preprocessed-dir ./output/preprocessed \
  --window-size 30 \
  --step 1 \
  --epochs 50 \
  --batch-size 32 \
  --lr 1e-4
```

Checkpoints are saved as:

```text
output/best_diff_model_FD001.pt
output/best_diff_model_FD002.pt
```

### 4. Train RUL Models

```bash
python train_rul_models.py \
  --fds FD001 FD002 \
  --output-dir ./output \
  --window-size 30 \
  --step 1 \
  --eval-mode engine \
  --epochs 50 \
  --batch-size 32 \
  --lr 1e-4 \
  --use-huber
```

This stage builds or reuses generated training windows under:

```text
output/augmented/
```

RUL model checkpoints and metrics are saved under:

```text
output/rul_models_real_plus_gen/FD001/engine/
output/rul_models_real_plus_gen/FD002/engine/
```

## Plotting

### Plot DTE Latent Space

```bash
python plot_DTE_all.py \
  --fds FD001 FD002 \
  --output-dir ./output \
  --preprocessed-dir ./output/preprocessed \
  --window-size 30 \
  --step 1 \
  --units 86 103 \
  --output ./output/all_z_DTE_FD001_FD002.png
```

If `--units` is omitted, the script uses its default highlighted unit for each FD subset.

### Plot Diffusion Denoising for Selected Units

```bash
python plot_unit_denoising_generated.py \
  --fds FD001 FD002 \
  --output-dir ./output \
  --preprocessed-dir ./output/preprocessed \
  --plot-dir ./output \
  --window-size 30 \
  --step 1 \
  --split train \
  --units 99 46 \
  --t-fixed 25
```

Example outputs:

```text
output/FD001_unit99_denoise_align_t25.png
output/FD002_unit46_denoise_align_t25.png
```

Add `--show` to either plotting script if you also want an interactive plot window.

## Notes

- `--fds` accepts any subset of `FD001`, `FD002`, `FD003`, and `FD004`.
- `--window-size` and `--step` must match across preprocessing, training, and plotting.
- `config.py` still provides default architecture and training values. CLI arguments override the most common run settings.
- RUL evaluation defaults to `--eval-mode engine`, which uses the last window for each test engine.
