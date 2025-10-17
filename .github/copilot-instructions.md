# BioEncoder – AI assistant instructions

Purpose: Help AI coding agents work effectively in this repo by explaining how the package is structured, how training flows work (two-stage, metrics), project-specific patterns, and how to run builds/tests/train including multi-GPU.

## Architecture overview
- Package layout: `bioencoder/` contains core modules and user-facing scripts. Key modules:
  - `core/models.py`: defines `BioEncoderModel` with encoder backbone (torchvision/timm) and two heads:
    - Stage 1: projection head for supervised contrastive learning (SupCon). Toggle via `model.use_projection_head(bool)`.
    - Stage 2: frozen encoder + linear classifier.
  - `core/utils.py`: orchestration utilities:
    - Config helpers: `load_yaml`, `restore_config`, `set_seed`.
    - Data: `build_transforms()` via Albumentations; `build_loaders()` constructs train/valid DataLoaders. Supports `distributed=True` using `DistributedSampler` and exposes `train_sampler`, `train_supcon_sampler`, and a non-sampled `train_eval_loader` for stage-1 validation.
    - Model: `build_model()` creates `BioEncoderModel` and optionally loads checkpoints; `build_optim()` wires losses, optimizer, scheduler, and optional loss optimizer for stage 1.
    - Train/valid: `train_epoch_constructive` (SupCon), `train_epoch_ce` (CrossEntropy), `validation_constructive` (embeddings + pytorch-metric-learning `AccuracyCalculator`), `validation_ce` (F1/accuracy).
    - EMA helpers and augmentation sample saver.
  - `core/*` also includes registries for losses (`losses.py`), optimizers, schedulers, and augmentations.
- Scripts in `scripts/` are thin CLIs that call into `core.utils`:
  - `train.py`: main training entry. Reads YAML, sets up dirs, logging, TensorBoard, builds model/loaders/optim, and runs epoch loops. Now supports torch.distributed (DDP). Only rank 0 logs and saves.
  - Other scripts: `swa.py` (stochastic weight averaging), `lr_finder.py`, `inference.py`, `interactive_plots.py`, `model_explorer*.py`, `split_dataset.py`, `configure.py`.
- Config-driven workflow: global paths in `bioencoder/config.py` and per-run hyperparams in YAMLs under `bioencoder_configs/`.

## Training flow (what to know)
- Two-stage process:
  - Stage 1 (metric learning): `criterion.name == "SupCon"`; training uses pair views (`TwoCropTransform`). Validation computes embeddings on `valid_loader` vs full train set using `train_eval_loader`. Projection head toggled off to evaluate encoder space.
  - Stage 2 (classification): encoder frozen; linear `classifier` head trained with CrossEntropy.
- EMA: optional via `torch-ema`. In DDP, EMA is applied and restored only on rank 0 using the underlying module parameters.
- Checkpoints: saved per epoch when `target_metric` improves (rank 0 only) to `weights/<run>/<stage>/epoch{N}` with model/optimizer state.

## Distributed training (DDP)
- Launch with torchrun; env variables (LOCAL_RANK, RANK, WORLD_SIZE) are detected automatically in `scripts/train.py`.
- Device assignment: each process uses `cuda(LOCAL_RANK)`; model wrapped with `DistributedDataParallel`.
- Samplers: `build_loaders(..., distributed=True)` returns `DistributedSampler`s; call `set_epoch(epoch)` each epoch (already handled in train.py).
- Rank 0 only: logging, TensorBoard, augmentation sample saving, validation summaries, and checkpoint writes. A `dist.barrier()` synchronizes before validation.
- Backend: tries NCCL; falls back to Gloo (useful on Windows). Prefer Linux with NCCL for performance.

## Conventions and patterns
- Always build components through `core.utils` (not ad-hoc in scripts):
  - Transforms: `build_transforms(hparams)`; DataLoaders via `build_loaders()`; Model via `build_model()`; Optim/Sched/Loss via `build_optim()`.
- Data dirs: derived from `config.root_dir` and `config.run_name` creating `data/`, `runs/`, `logs/`, `weights/` subfolders; scripts accept `--overwrite` to reset outputs.
- Stage switching uses the YAML `train.stage` and toggles `BioEncoderModel.use_projection_head()` during validation when stage 1.
- Metrics:
  - Stage 1: computed with pytorch-metric-learning `AccuracyCalculator(k=1, exclude=[...])` on embeddings.
  - Stage 2: `validation_ce` calculates accuracy, per-class F1, and macro F1.
- Mixed precision: set via `train.amp`; uses `torch.cuda.amp.GradScaler` and autocast in utils.

## How to run
- Interactive (from Python):
  - `bioencoder.configure(...)`, then `bioencoder.train(config_path=...)` for each stage. See `help/03-training.md`.
- CLI:
  - `bioencoder_train --config-path "bioencoder_configs/train_stage1.yml"` (and stage 2). Logs in `logs/`, TensorBoard in `runs/`.
- Multi-GPU:
  - Example: `torchrun --standalone --nproc_per_node 4 -m bioencoder.scripts.train --config-path "bioencoder_configs/train_stage1.yml"`.

## When modifying or extending
- Add new backbones in `core/backbones.py` or via timm prefix `timm_` handled by `create_encoder()`.
- Losses/optimizers/schedulers are registered in their respective `core/*` modules; `build_optim()` expects a dict with `name` and `params`.
- Make changes to training logic in `core/utils.py` first; keep scripts as orchestrators.
- If you add a new metric or validation strategy, ensure rank-0 only logging and consider distributed synchronization.

## Key files
- `bioencoder/core/utils.py`: data/model/optim builders, training+validation loops, DDP loader support.
- `bioencoder/core/models.py`: model definition, projection head toggle, stage switching.
- `bioencoder/scripts/train.py`: end-to-end training orchestration, DDP init, logging, checkpoints.
- `bioencoder_configs/*.yml`: example hyperparameter configs for both stages.
- `help/03-training.md`: usage guide including DDP launch examples.
