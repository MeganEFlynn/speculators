# Diagnostic report: LiLiCorr 27B underperformance (tied vs untied audit) + Gemma-4 softcap fix

**Date:** 2026-11-03
**Question:** Why does LiLiCorr train well on Qwen3.5-4B (tied embeddings) but underperform on
Qwen3.8-27B (untied)? Plus a follow-on: is Gemma-4's `final_logit_softcapping` handled correctly
in the reconstructed teacher targets?

**Short answer:** No bug in the tied/untied loading paths — teacher-logit reconstruction passes
byte-level parity on both layouts, and the penalty is elevated but not pathological on 27B. The
supported explanation for the 27B gap is step starvation (7,001 optimizer steps at dp=6 vs 40,090
at dp=2). The Gemma-4 softcap **was** a real bug in the training targets; it is fixed and verified
in this branch.

## 1. Teacher-logit reconstruction parity — PASSES on both layouts

Method: native verifier forward vs `verifier_lm_head(verifier_norm(extracted_final_hidden))`
at identical supervised positions, before any temperature/top-k/top-p. Frozen head/norm built
through the trainer's own construction path (`create_transformer_layer_config` →
`from_training_args`). Input = stale vLLM extraction payloads hash-matched to dataset rows.

| metric | Qwen3.5-4B (tied) | Qwen3.8-27B (untied) |
| ----- | ----------------- | -------------------- |
| sequences / positions | 1 / 2,991 | 14 / 23,003 |
| top-1 agreement | 99.57% | 99.22% |
| mean abs logit error | 0.028 | 0.027 |
| max abs logit error | 10.46 | 12.31 |
| mean TV distance | 0.0038 | 0.0069 |

Identical error magnitude on both — BF16/backend noise, not a reconstruction bug.

Weight sanity (same runs): 27B frozen head `==` checkpoint `lm_head.weight`, `!=` embed
(the untied head is preserved, not replaced by embeddings); 4B head `==` embed;
`verifier_norm` `==` checkpoint norm on both; final norm applies the `1+w` (Qwen3.5/Gemma3)
convention on both; the one-token target shift (`(rows-1) % seq_len`) is correct.

## 2. Penalty relative strength — elevated on 27B but not pathological

Fixed matched batches, run defaults (`w_ce=0.25`, `w_pen=0.25`, `target_source=argmax`,
γ=7), identical script both models. Loss terms and gradient norms (via `autograd.grad` on
shared backbone params: trunk layers, `fc`, `hidden_norm`):

| metric | 27B (row 58843 / 63292) | 4B (row 91644) |
| ----- | ----------------------- | -------------- |
| dflash CE | 1.17 / 3.81 | 1.16 |
| weighted lattice CE | 0.089 / 0.339 | 0.062 |
| weighted penalty | 0.281 / 0.748 | 0.134 |
| penalty share of loss | 18% / 15% | 10% |
| ‖∇penalty‖ / ‖∇total‖ | 45% / 53% | 43% |
| teacher top1−top2 gap median / p95 | 10.0 / 16.8 | 9.5 / 16.6 |

Teacher-logit gaps are the same scale on both models — the 27B's larger penalty values come
from the draft's weaker fit (more distractor mass), not from bigger logits or a reconstruction
error. Gradient share is within family. This is a symptom of undertraining, not a cause.

Per the gating criteria (parity passes; penalty not disproportionate), the 200–500-update
penalty-off ablation was **not run**.

## 3. Gemma-4 `final_logit_softcapping` — demonstrated bug, fixed

**Finding:** native Gemma-4 applies `tanh(logits/30)*30` after the lm_head
(`modeling_gemma4.py`), but the training-side teacher targets
(`verifier_lm_head(verifier_norm(h))` in dflash/lilicorr/eagle3/peagle) omitted it. Measured
impact on real extracted data: top-1 flip rate 0% (softcap is monotonic) but TV between
capped/uncapped teacher distributions ≈ 0.08 mean / 0.29 p95, and top1−top2 logit gaps
compressed to 61% of uncapped — i.e. the draft was trained against a systematically sharper
distribution than the one the served verifier uses for rejection sampling.

**Fix (this branch):**

- `src/speculators/model.py` — `_resolve_verifier_logit_softcap()` reads
  `final_logit_softcapping` from the verifier's effective (text) config during
  `_load_verifier_weights`; `_apply_verifier_logit_softcap()` applies
  `tanh(logits/cap)*cap` (exact identity when unset).
- All five teacher-target call sites wrapped: `models/dflash/core.py` (×2),
  `models/lilicorr/core.py`, `models/eagle3/core.py`, `models/peagle/core.py`.

**Verification:**

- Gemma, real vLLM-extracted hidden states (the actual training-data path): teacher-target
  TV vs native verifier drops **0.090 → 0.014** with the fix — the same BF16-noise band as the
  Qwen parity results. Top-1 agreement 97.6–98.8%.
- Qwen regression: softcap resolves to `None` on both Qwen3.5-4B and Qwen3.8-27B;
  `_apply_verifier_logit_softcap` is an exact identity (`torch.equal`). Existing checkpoints
  and Qwen retraining are unaffected.
- Serve side needs no change: the verifier applies its softcap in vLLM
  (`gemma4.py` LogitsProcessor), and the draft's candidate head is trained-owned (it now
  learns softcap-shaped logits), so its LogitsProcessor correctly stays uncapped.

**Red herring resolved:** transformers' `hidden_states[-1]` for Gemma-4 is *post*-final-norm,
but the vLLM extraction server captures layer 60 *pre*-norm (`hidden_states + residual`
before `self.norm`), so the extracted training data was never double-normed — only the
softcap was missing.

## 4. Conclusion

| category | finding |
| -------- | ------- |
| Demonstrated bugs | Gemma-4 softcap omission in teacher targets (fixed here). One corrupt hidden-states file found in the 27B stale payloads (3050 ≠ 3072 rows; would have been rejected by `check_hidden_states`, not silent). |
| Supported hypotheses | 27B underperformance is undertraining: 7,001 steps (dp=6) vs 40,090 (dp=2). Penalty share elevation (15–18% vs 10%) is a consequence. |
| Checks passed | Head/norm weight identity (both layouts), `1+w` norm convention, one-token shift, teacher-logit parity (top-1 ≥ 99.2% both), penalty gradient share within family, teacher gap scale within family, Qwen softcap-None identity. |

**Recommendation:** no code change for the 27B tied/untied question; close it with a
step-matched 27B continuation if the undertraining hypothesis needs definitive proof. Retrain
Gemma with this fix before drawing conclusions from the 100k run — its targets were wrong.

## Provenance

| item | value |
| ---- | ----- |
| Training code | speculators `lilicorr` branch, parent `b4830c0` + this fix |
| Serving code | `MeganEFlynn/vllm` `lilicorr` @ `2bf157bd17` (vLLM 0.32.0 base) |
| 27B run | `run-27b-matched-6r/checkpoints/checkpoint_best` (dp=6, 7,001 steps, val full_acc 0.276) |
| 4B control | `run-new/checkpoints/checkpoint_best` (dp=2, 40,090 steps, val full_acc 0.457) |
| Gemma runs | `gemma4-100k` (4,759 steps, val full_acc 0.226 — targets affected by the softcap bug), `gemma4-smoke` |
| Parity data | stale extraction payloads hash-matched to dataset rows (27B: 21 files / 14 usable; 4B: 1 file) |
| Diagnostic scripts | `/tmp/ab4b/diag/{parity_test2,penalty_measure,softcap_test,softcap_fix_verify}.py` |
