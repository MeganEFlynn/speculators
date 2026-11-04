# Gemma-4-31B LiLiCorr — gemma.md execution log (intermediate results)

**Started:** 2026-11-03 (evening PT). This file is pushed after each completed milestone.
Plan source: `~/gemma.md` (two-epoch Nemotron-transfer plan over the regenerated
PerfectBlend responses). Working branch: `lilicorr-qwen38-fresh-2ep-argmax` on
`MeganEFlynn/speculators` (experiment commits land here).

## Provenance (step 1 — pinned)

| item | value |
| ---- | ----- |
| Training code | `MeganEFlynn/speculators` branch `lilicorr-qwen38-fresh-2ep-argmax`, from `vllm-project/speculators:lilicorr`; base `24bdd58`, now `654ecf0` (softcap fix) |
| Serving code | `MeganEFlynn/vllm` `lilicorr` @ `2bf157bd17` (vLLM 0.32.0 base, `/tmp/vllm032`) |
| Target model (data manifest) | `google/gemma-4-31B-it` — matches local `/tmp/models/gemma-4-31b-it` used by extraction |
| Data | `nm-research/gemma-4-31b-it-responses-open-perfectblend`, file `gemma_response_regen_open_perfectblend.v2.jsonl.gz` (10.5 GB, sha256 pending; gunzip -t OK) |
| Generation log | `main_run.log`: source `mlabonne/open-perfectblend` @ `af60f3c18201652a83a93f46fcfee1b646ba3df7`, sampling T=1.0 / top_p=0.95 / top_k=64 / seed=42, reasoning efforts low/medium/xhigh |
| Generation result | 1,417,995 conversations → 1,868,233 turns (rows), 2,914 errors dropped, 2.52B completion tokens |
| Local rows | 1,868,233 (matches manifest exactly) |

Note vs plan: the plan says "1.4M regenerated sampled PerfectBlend examples" — the unit is
conversations (1.42M); rows after turn-split are 1.87M. Split will be by `primary_id`
(conversation) so turns from one conversation cannot straddle train/validation.

## Data description (downloaded and verified)

- Schema per row: `id`, `primary_id`, `input_ids`, `loss_mask`, `text`, `metadata` —
  pretokenized with response-only loss masks (row 0: 218 supervised of 660).
- `input_ids` length: min 56, median 1,439, max 13,099 (sampled first 20k rows).
- Staged at `/tmp/ab4b/gemma_perfectblend_v2.jsonl` (53 GB) + `.gz` (10.5 GB).

## Softcap prerequisite — already fixed on this branch (`654ecf0`)

gemma.md §4 requires teacher-distribution parity including `30*tanh(logits/30)` exactly
once. That bug (softcap omitted from all teacher-target reconstruction) was found, fixed,
and verified earlier today; see `DIAGNOSTIC_TIED_UNTIED_AND_GEMMA_SOFTCAP.md` in this
directory. Parity on real extracted hidden states: TV 0.090 → 0.014 (BF16 noise band);
Qwen identity-verified unaffected. The plan's §4 checks (single plain-gain RMSNorm, unscaled
tied head, no embed-scale leakage, alignment) will be re-run against the KL/calibration path
once the loss lands.

## Status / next

- [x] Download + verify data, pin provenance (this report)
- [x] §2: group-shuffled dataset written (seed 42, group order only) — `prepare-data` passthrough running
- [x] §2: WSD scheduler addition (`schema.py`, `cli.py`, `trainer.py`) — commit `a437683`

## §2a WSD scheduler — done (`a437683`)

- `scheduler_type="wsd"` with `--scheduler-stable-fraction` (0.8) and
  `--scheduler-min-lr-ratio` (0.1), built on `transformers.get_wsd_schedule`
  (linear warmup, cosine decay), wired schema → CLI flags → `TrainerConfig` →
  `make_scheduler`. `num_decay_steps = S − floor(0.8·S)` per the plan.
- Verified numerically (S=1000, W=40): LR 6e-5 → 6e-4 by step 40, held through
  step 799, cosine to 6e-5 at step 999. Warmup W will be
  `min(1000, ceil(0.04·S))` resolved from actual S once the dataloader is final.
- Resume: WSD `state_dict` roundtrips through the existing checkpointer load
  path (which re-applies `get_last_lr()` to param groups); 9/9 existing
  scheduler tests pass.
- [x] §3: `kl_div` body loss + candidate calibration KL (0.05 CE + 0.45 calib), penalty 0 — `506d06e`
- [x] §4: teacher parity vs KL path — verified (see §4 section)
- [x] §2: prepare-data complete — see §2 data prep status below
- [x] §5: 50-update preflight — done (see §5 section below)
- [ ] §6: two-epoch run + serving eval + selection
- [ ] §6: two-epoch run + serving eval (7 spec tokens, T=1/top-p 0.95 and T=0) + selection

## §3 Calibration KL — done (`506d06e`)

- `lilicorr_w_calib` (default 0.0; old recipe 0.25/0.25 unchanged) adds
  KL(teacher‖head) restricted to the draft's top-k candidates, teacher-forced
  on observed predecessors (same supervised mask / reachable-prefix semantics
  as the lattice CE). Gradients flow through the head's `log_p`; candidate ids
  and teacher targets are detached. CLI flags:
  `--lilicorr-w-ce/--lilicorr-w-pen/--lilicorr-w-calib/--lilicorr-target-source`.
- Chunked and unchunked paths wired; new metric `lilicorr_calibration_sum`.
- Verified: chunked == unchunked exactly (diff 0.000000 at fixed anchor seed).
  An apparent 0.176 discrepancy between two unseeded calls was anchor
  re-selection (`torch.randperm` in `select_anchors`) — pre-existing on base,
  correct training behavior, not a chunking bug. Calibration math
  unit-checked (positive with reachable prefix; ~1e-10 for matched
  distributions). 8/8 lilicorr tests pass.

## §4 Gemma teacher parity — done

| Check | Result |
|---|---|
| Final-norm gain | `gemma4` correctly NOT in `GEMMA_STYLE_FINAL_NORM_MODEL_TYPES`; Gemma4RMSNorm is `x·w` (init ones), matches the `Qwen3RMSNorm` used for `verifier_norm`. eps 1e-6 matches. |
| Unscaled head | `verifier_lm_head` is a plain `nn.Linear` — no `sqrt(5376)` embed-scale. |
| Softcap once | Teacher: `tanh(logits/30)·30` via `_apply_verifier_logit_softcap` (`654ecf0`), applied in both the body-KL targets and candidate calibration (shared `targets_s`). Draft: no softcap in draft config; serving applies soft_cap only when the draft config sets it — none, matching training. |
| Reconstruction | top-1 agreement 1.0000, TV 0.00156, mean |Δ| 0.0217 (bf16, pre-norm h). |
| Alignment / masks | `(rows−1) % seq` gathers pre-norm verifier states; loss_mask from dataset (assistant-only); verified in softcap work. |

Caveat documented: this transformers' `Gemma4TextModel` returns
`hidden_states[-1]` POST final norm; training inputs are the vLLM-extracted
PRE-norm layer-60 states, so parity is checked against a pre-norm capture
(an initial tv≈0.999 was this double-norm artifact in a diagnostic script).

## §2b Gradient accumulation — done (`8070a03`)

- The trainer stepped once per packed window per rank — "global batch 64
  windows" impossible at dp=6. Added `gradient_accumulation_steps` (schema →
  CLI → `TrainerConfig`): loss scaled by 1/N; window = micro-batches
  [(k−1)·N, k·N); zero-grad on window start; clip + opt/sched step +
  global_step + logging on window end; partial windows flush at epoch end
  and the max_steps cutoff.
- Verified on a stub trainer: accum=4 over 4 micro-batches reproduces the
  mean-of-4 gradient exactly (0.0 diff) in 1 optimizer step; 6 → 2 steps with
  tail = sum/4; accum=1 = old behavior. dp=6 × accum 11 → 66 windows/update
  (closest to 64).

## §2 data prep — status

- Group stats: 3,907,366,330 tokens (2,518,432,942 supervised); 13.16% of
  rows ≥ 4096 tokens; after right-truncation 88.04% / 84.97% retained.
- Group-shuffled file verified: 1,868,233 rows, group heads in seeded order
  (seed 42, group order only); 99% boundary (row 1,849,550) lands exactly on
  a group edge → group-disjoint split. Source composition drifts by file
  position (tail deciles MetaMathQA-heavy), so the shuffle also balances the
  val split.
- `prepare-data` failures were infra: disk quota on the default HF cache
  (fixed with local `HF_DATASETS_CACHE`) and the 4,096 cgroup PID ceiling
  (pids.peak hit exactly 4096; pyarrow spawns ~78 threads/pool; stale test
  processes consumed headroom). Fixed by killing stale processes, capping
  pyarrow threads (`pa.set_cpu_count(8)`, `ARROW_IO_THREADS=4`), 2 workers.
  Staging cleanly at ~2.9k rows/s.


## §2 prepare-data — done

`/tmp/ab4b/gemma_pb_v2_4096` (101 arrow shards, 77.4 GB): **1,791,560 rows** =
1,868,233 input − 76,673 rows dropped for zero in-window supervised tokens.
3,125,821,337 total tokens / 2,139,890,739 supervised (88.04% / 84.92% of the
raw file, matching the earlier projection). seq_len 46–4096, mean 1744.75.
Schema `input_ids/loss_mask/seq_len`; `load_from_disk` verified (the
`dataset_info.json` split metadata is stale from an earlier write — arrow
tables are the source of truth and load correctly).

**Sampler / S computation** (real `MultipackDistributedBatchSamplerV2`,
batch_max_length=4096, dp=6, seed 42, epoch 0, contiguous 99% train split of
1,773,644 rows): **138,768 micro-batches per rank per epoch**, 90.72% bin
fill. With accum 11 → 12,616 optimizer steps/epoch → **S = 25,232** for two
epochs. WSD: warmup min(1000, ⌈0.04·S⌉) = **1000 steps**, hold to 0.8·S =
20,185, cosine to 6e-5.

**Scheduler horizon fix**: the scheduler steps once per optimizer step, but
`_resolve_scheduler_steps` defaulted its horizon to `num_epochs ×
train_loader_len` (micro-batches) — with accum 11 the WSD decay would have run
11× too slowly. `setup_optimizer` now divides the micro-batch count by the
accumulation window (ceil) before resolving steps. 9/9 scheduler tests pass.


## §5 preflight — done (2026-11-04)

**Run**: `/tmp/ab4b/preflight` — 50 optimizer updates on real data, dp=6 (GPUs 2–7),
accum 11, fresh 5-layer draft + LiLiCorr head, B=8/K=8, KL body + 0.05 CE + 0.45
calibration, γ=4, 4,096-token windows, 512 anchors, chunk 64, AdamW 6e-4, WSD,
grad-ckpt + sdpa. EXIT 0.

- **Finite, falling losses**: train loss 138.8 (init) → 8.5 by step 49; final
  `val/loss_epoch=8.854`, `val/dflash_loss=7.859`, `val/lilicorr_ce=1.999`,
  `val/lilicorr_calibration=1.988` — calibration KL active throughout (~2.0).
- **Checkpoint**: `epoch0_end` written with model (104 tensors, lilicorr head 46
  params present), optimizer state (104 entries, all finite), scheduler state
  (`last_epoch=50` — matches optimizer steps), training_state (`global_step=50`).
  Reload via `SpeculatorModel.from_pretrained` verified.
- **Known deviation**: AdamW `exp_avg`/`exp_avg_sq` are stored **bf16**, not the
  plan's FP32 master/update state. Decision needed before §6 (cast to fp32 in the
  checkpointer, or accept bf16 moments with BF16 model weights).
- **Infra fixes required for 4,096-token data**:
  - Render server restarted with `--max-model-len 4104`: 169,239 rows (9.4%) are
    exactly 4,096 tokens; at 4,096 the 1-token render request exceeds the context
    and vLLM rejects it (`max_tokens must be at least 1, got 0`), burning 4 retries
    per row. New server config: `/tmp/ab4b/render4097/serve.sh` (GPUs 0–1, same
    extract_hidden_states spec, aux ids `[4,16,28,40,52,60]`).
  - `PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True` required: first launch OOM'd
    at step 1 allocating the [5376, 262144] lm_head workspace (262k vocab × 4,096
    window); with expandable segments 78–79 GB/GPU steady, zero OOMs over 50 steps.
  - Render server launch MUST set `OPENBLAS/OMP/MKL_NUM_THREADS=1` — without them
    OpenBLAS spawns 78 threads/proc and engine-core spawn dies at the 4,096 PID
    cgroup ceiling.
- **Throughput** (matters for §6 planning): ~45 s/optimizer update; render fetch is
  68% of step time (fwd 487 ms, bwd 1.4 s per micro-batch). At S = 25,232 → ~13
  days for two epochs.

**Chunked-vs-unchunked**: fixed-seed parity was verified at §3 time (diff 0.000000
on synthetic batch). A re-verification on the trained checkpoint was attempted;
first run had a test-harness shape bug (fed 6 aux layers into a 5-layer fc) — to
be rerun with 5 aux + 1 final layout.
