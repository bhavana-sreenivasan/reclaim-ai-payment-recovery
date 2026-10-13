# ReClaim v2: cost-aware payment-recovery policy evaluation (semi-synthetic)

> **Status: simulated data.** No real payment data is used. The results show how recovery policies
> rank *under the stated assumptions*. They are not estimates of real-world recovery rates.

## Quick start

```bash
pip install numpy pandas scipy scikit-learn xgboost
python generate_data.py --out data                              # ~1 s, deterministic (seed 2026)
python run_pipeline.py --seeds 2026 1 2 3 4 --out results       # ~1-2 min on one CPU core
```

`generate_data.py` writes two files:
- `failed_payments_v2.csv.gz`: what a production system could observe (features + the one action actually taken and its outcome).
- `oracle_potential_outcomes.csv.gz`: simulator ground truth for *every* channel. Used **only** for evaluation, never for training.

## What the pipeline does

1. **Data.** 100k failed payments, Jan-Jun 2026. Failure cause depends on payment method, bank reliability
   (anonymised banks, with outage windows), hour/weekday, and the salary cycle. Repeat-failing customers.
   Channel outcomes share a latent "stubbornness" term, so they are correlated. Recovery = natural OR intervention-induced.
   Drift in the last third (one bank degrades, UPI share rises).
2. **Root cause.** TF-IDF + logistic regression on the gateway message (about 40% of messages are vague by design).
   A confidence gate (threshold picked on validation for >=90% precision) outputs a cause or `UNKNOWN`.
   Cross-fitted on train so no row is scored by a model that saw it.
3. **Recovery model.** XGBoost trained **only on logged outcomes of the action actually taken**
   (uniform-random exploration over 8 channels + a no-action control arm), then isotonic-calibrated.
   Features use the *gated* cause, never the true cause.
4. **Policies.** Rules ladder, value-weighted rules, cost-blind reason playbook, a **no-ML segment lookup**,
   the predictive EV agent (cascade and single-shot), plus two ceilings: a policy that knows the true
   probabilities, and a hindsight oracle.
5. **Evaluation.** Time-based split (train Jan-Mar, validate Apr, **test May-Jun**). Value =
   *incremental net contribution over doing nothing* = margin x recovered amount - attempt costs (incl. a contact-fatigue penalty).
   The agent's EV uses estimated *uplift* (P(paid | channel) - P(paid | no action)).

## Results (5 data seeds, test period, per 1,000 failed payments; margin 30%)

| Policy | Recovery rate | Incremental net (₹) | Attempt cost (₹) |
|---|---|---|---|
| No action | 13.8% | 0 | 0 |
| Rules ladder (retry > email > SMS > WhatsApp) | 23.7% | 35,853 ± 548 | 191 |
| Value-weighted rules | 24.2% | 47,532 ± 590 | 5,723 |
| Reason playbook (cost-blind) | 40.0% | 67,403 ± 3,488 | 28,622 |
| Segment lookup (no ML, EV-gated) | 37.4% | 83,773 ± 1,414 | 5,416 |
| **Predictive EV agent (cascade)** | **37.5%** | **84,350 ± 1,615** | 4,319 |
| *True-probability policy (ceiling for any model)* | 42.0% | 101,372 ± 1,232 | 5,104 |
| *Hindsight oracle (unreachable)* | 59.0% | 159,464 | 2,776 |

### What the results do and do not show

- **Cost-aware targeting beats the rule-based policies in all 5 seeds**, with paired-bootstrap 95% CIs excluding zero
  (vs rules ladder: +₹46.8k per 1,000, CI 42.3k-52.5k).
- **Recovery rate is the wrong metric.** The cost-blind playbook recovers *more* (40.0% vs 37.5%) but escalates to
  expensive human calls on ~61% of cases and earns ₹17k less per 1,000 failures.
- **XGBoost does not clearly beat a no-ML lookup table.** Agent minus segment lookup: +₹370 per 1,000,
  95% CI -1,235 to +1,917 (not significant; the agent wins in 3 of 5 seeds). The gain comes from diagnosing the
  cause and choosing by uplift and cost, not from the model class.
- **Diagnosis matters.** Root-cause ablation (incremental net per 1,000): no diagnosis ₹58.7k -> predicted + gated ₹84.4k -> perfect diagnosis ₹97.7k.
- **Headroom is modest.** The agent earns about 83% of what a policy knowing the true probabilities would.
- **The recovery model is weak but calibrated.** Test AUC 0.66, Brier 0.166 vs 0.176 for a constant predictor, ECE 0.007-0.010.
- **The EV gate rarely rejects.** Contact is cheap (₹0.2-0.5), so ~99.99% of failures get at least one attempt;
  the cost logic mainly decides *whether to escalate to a human call* (8% of cases).
- **The evaluation method is validated.** An inverse-propensity (IPS) estimate computed from logged data alone matched
  simulator ground truth within its CI (agent: 126.7k est. vs 124.1k true; no-action: 54.5k vs 57.2k).
- Sensitivity grid (margin x human-call cost x messaging cost, 18 cells): the best baseline is always the
  segment lookup, and the agent stays within about 1% of it in every cell. It wins 14 of 18 (by up to +₹1,274 per 1,000)
  and trails in 4 (by ₹95-1,221 per 1,000). Treat the two as tied. The grid does not compare against the rule-based
  policies cell by cell; that comparison is the 5-seed table above.

Full tables: `results/`. Console log of the final run: `results/run_log.txt`.

## Assumptions and provenance

`[SRC]` = taken from a public source (secondary sources; verify before citing). `[ASSUME]` = my parameter, not measured.

| Parameter | Value | Provenance |
|---|---|---|
| UPI share of retail digital payments | ~80% of volume (failure mix 56-61%) | [SRC] press/NPCI-derived; failure mix [ASSUME] |
| WhatsApp utility message cost | Rs0.115 base + BSP markup + GST, ~Rs0.40 all-in | [SRC] Meta/BSP pricing pages, 2026 |
| Single dunning-email recovery | ~11-15% | [SRC] industry blogs (secondary) |
| Good full-stack recovery | ~45-70% | [SRC] industry blogs (secondary) |
| Success-probability table, natural recovery, correlation (0.55 / 0.60), later-attempt decay (0.85) | see `generate_data.py` | [ASSUME] |
| SMS / IVR / human-call cost; contact-fatigue penalty; margin 30% | Rs0.20 / 2.50 / 45; 0.25-5 | [ASSUME] |
| Banks | anonymised BANK_A-H; reliability invented | [ASSUME] (no claim about any real bank) |

## Limitations

- Everything is simulated; conclusions are conditional on the assumptions above. The sensitivity grid varies only margin and costs, not the success-probability table.
- The seed-to-seed spread reflects sampling noise, not parameter uncertainty.
- Cascade attempts are evaluated at the logged action time with a flat 15% per-attempt decay; natural recoveries do not stop later attempts (conservative for all policies).
- Outcomes for unchosen channels exist only because it is a simulator. The IPS check shows the logged-data route works, but real deployment would need a randomised holdout.

## Files

`generate_data.py` generator (assumptions documented inline) | `run_pipeline.py` models, policies, evaluation |
`results/` all tables (`policy_across_seeds.csv`, `bootstrap_ci.csv`, `sensitivity_grid.csv`, `ope_check.csv`, `ablation_root_cause.csv`, `model_diagnostics.csv`) |
`DATA_AUDIT.md` what was wrong with the v1 dataset
