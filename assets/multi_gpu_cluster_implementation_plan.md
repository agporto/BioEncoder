# Multi-GPU and Cluster Training Plan

This document outlines a phased plan to add distributed training support to BioEncoder with minimal disruption.

## Goals

- Add reliable single-node multi-GPU training first.
- Extend to multi-node cluster training after single-node is stable.
- Keep current single-GPU behavior as default.
- Preserve reproducibility, logging quality, and checkpoint usability.

## Non-goals (initially)

- Rewriting visualization tooling for distributed contexts.
- Supporting every distributed strategy from day one (focus on DDP first).

---

## Phase 0 - Baseline and design decisions

### Deliverables

- Inventory and document all CUDA assumptions in core and scripts.
- Decide launcher strategy (`torchrun`) and backend defaults (`nccl` for GPU).
- Define distributed config surface in YAML/CLI.

### Key decisions

- Process model: one process per GPU with PyTorch DDP.
- Keep single-process single-GPU path as default fallback.
- Only rank 0 writes logs/checkpoints/plots.

### Exit criteria

- Clear architecture note checked into repo.
- Finalized config keys and CLI flags for distributed mode.

---

## Phase 1 - Device and process abstraction cleanup

### Scope

- Refactor direct `.cuda()` calls to device-aware `.to(device)`.
- Add utilities for:
  - rank/world-size detection
  - distributed initialization and teardown
  - rank-aware logging helpers

### Likely file impact

- `bioencoder/core/utils.py`
- `bioencoder/scripts/train.py`
- `bioencoder/scripts/swa.py`
- `bioencoder/scripts/lr_finder.py`
- `bioencoder/scripts/interactive_plots.py`
- `bioencoder/scripts/inference.py`

### Exit criteria

- Single-GPU path still works unchanged.
- Code can run with explicit `device` argument.

---

## Phase 2 - Single-node DDP for stage-2 training (classification)

### Scope

- Add DDP wrapping in training script.
- Use `DistributedSampler` for train/validation.
- Ensure per-rank deterministic seeding.
- Aggregate validation metrics across ranks correctly.
- Gate side effects to rank 0 (tensorboard, checkpoint save, prints).

### Notes

- Start with stage-2 because metrics are simpler to reduce (counts/loss sums).
- Keep EMA handling explicit: either rank-local EMA with synchronized model or rank-0-only checkpoint EMA policy.

### Exit criteria

- Stage-2 trains on N GPUs in one node with expected scaling.
- Checkpoints load in current inference path.

---

## Phase 3 - Stage-1 DDP and metric-learning validation

### Scope

- Support distributed stage-1 constructive training.
- Rework embedding validation path:
  - gather embeddings/labels across ranks before metric calculator
  - run metric computation once (rank 0) and broadcast results

### Risk points

- Embedding all-gather memory cost on large datasets.
- Maintaining parity between projection-head and encoder-only validation.

### Exit criteria

- Stage-1 and stage-2 both supported in single-node DDP.
- Validation metrics stable and match single-GPU baseline within tolerance.

---

## Phase 4 - Multi-node cluster support

### Scope

- Add multi-node initialization (`MASTER_ADDR`, `MASTER_PORT`, `RANK`, `WORLD_SIZE`).
- Document launch recipes for scheduler environments (e.g., Slurm).
- Ensure checkpoint/log paths are shared-filesystem safe.
- Add resume semantics robust to pre-emption.

### Operational concerns

- Network topology and NCCL environment tuning.
- Fault behavior and restart policy.
- Runtime observability (per-rank logs, health checks).

### Exit criteria

- Multi-node train run completes on at least one real cluster environment.
- Resume from checkpoint works after interruption.

---

## Phase 5 - Hardening and developer UX

### Scope

- Add smoke tests for:
  - distributed init/teardown
  - sampler sharding correctness
  - metric reduction correctness
- Add integration checks for stage-1/stage-2 in 2-GPU local runs.
- Improve docs/config templates for distributed use.

### Exit criteria

- CI or scripted validation for distributed smoke tests.
- User docs include canonical single-node and multi-node launch commands.

---

## Suggested config/CLI additions

### YAML candidates

- `distributed.enabled: bool`
- `distributed.backend: "nccl" | "gloo"`
- `distributed.find_unused_parameters: bool`
- `distributed.sync_bn: bool`
- `distributed.grad_accum_steps: int`

### CLI candidates

- `--distributed`
- `--backend`
- `--local-rank` (for `torchrun`)

---

## Validation strategy by phase

- Compare single-GPU vs DDP loss curves for the same seed/config.
- Validate checkpoint compatibility with existing inference.
- Benchmark throughput scaling (1 -> 2 -> 4 GPUs).
- Confirm rank-0-only side effects (no duplicated logs/checkpoints).

---

## Effort estimate (rough)

- Phase 0-1: medium
- Phase 2: medium-high
- Phase 3: high
- Phase 4: high
- Phase 5: medium

Overall complexity: high for full cluster-grade support; medium-high for high-quality single-node DDP.
