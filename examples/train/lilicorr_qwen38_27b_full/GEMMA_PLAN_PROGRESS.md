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
- [ ] §2: split by `primary_id` 99/1, pack to 4,096-token windows, report retained tokens
- [ ] §2: WSD scheduler addition (`schema.py`, `cli.py`, `trainer.py`)
- [ ] §3: `kl_div` body loss + candidate calibration KL (0.05 CE + 0.45 calib), penalty 0,
      B=8/K=8, γ=4, both chunked and unchunked paths
- [ ] §4: teacher parity vs KL path
- [ ] §5: 50-update preflight (finite losses, calibration active, chunked==unchunked,
      save/reload FP32)
- [ ] §6: two-epoch run + serving eval (7 spec tokens, T=1/top-p 0.95 and T=0) + selection
