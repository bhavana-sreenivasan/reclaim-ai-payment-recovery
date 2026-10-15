---
title: ReClaim v2
emoji: 💳
sdk: gradio
app_file: app.py
---

# ReClaim v2: cost-aware payment-recovery policy (semi-synthetic)

> **Status: simulated data.** No real payment data is used. The results show how recovery policies
> rank *under the stated assumptions*. They are not estimates of real-world recovery rates.

**Headline (simulated):** a cost-aware recovery policy earns **₹84k incremental net per 1,000 failed payments**,
versus **₹36k** for a retry-and-remind ladder (recovery 37.5% vs 23.7%; 13.8% if you do nothing).
It decides *which channel to use, in what order, and whether a ₹45 human call is worth it*, and it
**says "UNKNOWN" and stays on safe channels when it is not confident about why the payment failed.**

**Live demo:** [Hugging Face Space](https://huggingface.co/spaces/YOUR-USERNAME/YOUR-SPACE-NAME)

## Run the demo

```bash
pip install -r requirements.txt
python train_and_save.py        # writes ./artifacts/ (seed 2026, May-Jun held-out test)
python app.py                   # open http://127.0.0.1:7860/
```

Enter a failed payment. The app diagnoses the cause from the gateway message (with a confidence gate),
estimates each channel's **uplift** over doing nothing, subtracts its cost, builds the attempt plan, and
shows **why** (SHAP, see below).

### 90-second demo script

Defaults: Card, BANK_G, Medium tier, margin 30%, human-call cost ₹45.

| # | Set | You should see |
|---|---|---|
| 1 | ₹15,000, Card, "Card has expired" | Cause named (ERR_EXPIRED_CARD, 99%). Retries are blocked. Plan: **HUMAN_CALL → ALT_METHOD**. |
| 2 | Same, message "Transaction failed" | **UNKNOWN** (top guess only 24%, below the 0.39 gate). Retries stay blocked; only safe customer-facing channels. |
| 3 | Same as #1 but amount ₹500 | The ₹45 call no longer pays for itself. Plan: **ALT_METHOD → WHATSAPP**. |
| 4 | Amount ₹1 | **Do nothing.** No channel has positive expected value. |
| 5 | ₹15,000, NetBanking, "Bank server did not respond in time" | Transient cause, so retrying is safe. Plan: **RETRY_NOW → RETRY_LATER**. |

The gateway-message dropdown only offers messages that can occur for the chosen payment method
(e.g. no expired-card message on UPI or NetBanking).

## Why this plan? (explainability)

- **Diagnosis:** the cause model is TF-IDF + logistic regression, so each word (and the payment method)
  adds or subtracts an *exact* amount of evidence (coefficient × feature value).
- **Recovery score:** SHAP values (`shap.TreeExplainer`) from the XGBoost recovery model for the planned channel,
  plus an "extra vs. no action" column (the same attribution minus the no-action row), which is what the
  expected-value calculation actually uses.
- Units are the model's **raw log-odds score before isotonic calibration**, not percentage points.
- On simulated data these explanations mostly recover relationships built into the simulator. They are a
  sanity check on the model, not a discovery about real customers. The cause model also learns template
  fingerprints (only 3 specific messages per cause), so its 99% confidence on a clean message would not
  carry over to real gateway text.

## How this would plug into Razorpay (design sketch, not built)

```
payment.failed webhook ──> diagnose ──> confidence gate ──> channel economics ──> plan ──> action
 (error_code,               (cause or     (UNKNOWN => safe      (uplift x margin x      (cascade,   (WhatsApp / SMS / email /
  error_description,         UNKNOWN)      channels only)        amount - cost)          budgeted)   IVR / call / alt-method link)
  error_reason, method,
  bank, amount)
```

- Razorpay's `payment.failed` webhook carries the payment entity with `status`, `error_code`, `error_description`
  and `error_reason`. `error_description`/`error_reason` would take the place of the simulated gateway message.
  Real error text is coarser and messier than this simulator's, so the cause model **must be retrained on real
  failures**; the 94% gate precision here is a simulation number.
- The `payment.downtime` webhook is a natural input for the "bank failing right now" feature (`bank_fail_6h`).
- **Real deployment needs a randomised holdout** (a small traffic slice with random channel assignment plus a
  no-action control). That is what gives honest uplift labels and prevents the policy from learning its own past
  choices. The IPS check below shows the logged-data route works in simulation.
- Not modelled: customer consent/DND for calls and WhatsApp, contact-frequency caps, and RBI rules on retrying
  cards and mandates. These must be hard constraints in front of the policy, not model features.

## Quick start (research pipeline)

```bash
pip install numpy pandas scipy scikit-learn xgboost joblib
python generate_data.py --out data                              # deterministic (seed 2026)
python run_pipeline.py --seeds 2026 1 2 3 4 --out results       # a few minutes on one CPU core
```

`generate_data.py` writes two files:
- `failed_payments_v2.csv.gz`: what a production system could observe (features + the one action actually taken and its outcome).
- `oracle_potential_outcomes.csv.gz`: simulator ground truth for *every* channel. Used **only** for evaluation, never for training.

## What the pipeline does

1. **Data.** 100k failed payments, Jan-Jun 2026. Failure cause depends on payment method, bank reliability
   (anonymised banks, with outage windows), hour/weekday, and the salary cycle. Repeat-failing customers.
   Channel outcomes share a latent "stubbornness" term, so they are correlated. Recovery = natural OR intervention-induced.
   Drift in the last third (one bank degrades, UPI share rises).
2. **Root cause.** TF-IDF on the gateway message plus the payment method, fed to logistic regression
   (about 40% of messages are vague by design). A confidence gate (threshold picked on validation for >=90% precision)
   outputs a cause or `UNKNOWN`. Cross-fitted on train so no row is scored by a model that saw it.
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
- **The recovery model is weak but calibrated.** Means over 5 seeds: test AUC 0.66, Brier 0.166 vs 0.176 for a constant
  predictor, calibrated ECE 0.010 (per-seed range 0.007-0.014).
- **The root-cause gate (5-seed mean):** precision 94% at 69% coverage (per-seed 92-97% at 66-72%).
  The demo app also shows seed 2026 alone (AUC 0.67, 97% at 66%); it is the same models, one seed.
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
| Direct cost per attempt (INR): email 0.05, SMS 0.20, WhatsApp 0.40, alt-method link 0.40, IVR 2.50, human call 45 | `DIRECT_COST` in `generate_data.py` | [ASSUME] (WhatsApp [SRC]) |
| Contact-fatigue penalty per attempt (INR): email 0.25, SMS 0.50, WhatsApp 0.50, alt-method 0.50, IVR 2.00, human call 5.00 | `FATIGUE` in `generate_data.py` | [ASSUME] |
| **All-in cost shown in the app** = direct + fatigue: email 0.30, SMS 0.70, WhatsApp/alt-method 0.90, IVR 4.50, human call 50 | | derived |
| Contribution margin | 30% (slider 5-60%) | [ASSUME] |
| Banks | anonymised BANK_A-H; reliability invented | [ASSUME] (no claim about any real bank) |

## Limitations

- Everything is simulated; conclusions are conditional on the assumptions above. The sensitivity grid varies only margin and costs, not the success-probability table.
- The seed-to-seed spread reflects sampling noise, not parameter uncertainty.
- Gateway messages come from a small template set (3 specific messages per cause plus 4 generic ones). The cause
  model's confidence on real, free-form gateway text would be much lower.
- Cascade attempts are evaluated at the logged action time with a flat 15% per-attempt decay; natural recoveries do not stop later attempts (conservative for all policies).
- Outcomes for unchosen channels exist only because it is a simulator. The IPS check shows the logged-data route works, but real deployment would need a randomised holdout.
- Consent, DND and RBI retry rules are not modelled (see the integration sketch).

## Files

`app.py` Gradio demo (with SHAP panels) | `train_and_save.py` trains and saves `artifacts/` for the app |
`generate_data.py` generator (assumptions documented inline) | `run_pipeline.py` models, policies, evaluation |
`results/` all tables (`policy_across_seeds.csv`, `bootstrap_ci.csv`, `sensitivity_grid.csv`, `ope_check.csv`, `ablation_root_cause.csv`, `model_diagnostics.csv`) |
`DATA_AUDIT.md` what was wrong with the v1 dataset
