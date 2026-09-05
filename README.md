# ReClaim — AI-Driven Payment Recovery Agent

An agent that decides **whether** a failed payment is worth trying to recover, and **which**
bounded action is most economically likely to succeed — instead of blindly retrying or spamming
every channel on every failure.

## Pipeline

```
Root-Cause AI (XGBoost + TF-IDF, confidence-gated)
        ↓
Recovery Context (predicted error code, or UNKNOWN below the confidence gate)
        ↓
XGBoost Recovery Probability (per channel)
        ↓
Probability Floor + Expected Value Gate
        ↓
Rank Channels by EV → Tier-based Attempt Budget → Sequential Cascade
        ↓
State Machine (CREATED → TRIAGED → DISPATCHED → RECOVERED / FAILED / EXPIRED / NOT_PURSUED)
```

## Results

Benchmarked against 4 baseline policies on 100,000 synthetic transactions (₹135.1M total value):

| Strategy | ₹ Recovered | Recovery Cost ₹ | ₹ Recovered/Attempt | Recovery Rate | Unresolved ₹ |
|---|---|---|---|---|---|
| No Intervention | ₹18.5M | ₹0 | — | 13.7% | ₹116.7M |
| Rules-Only Baseline | ₹104.2M | ₹37,830 | ₹688.86 | 77.2% | ₹31.0M |
| Reason-Aware Agent | ₹86.3M | ₹17,761 | ₹862.81 | 63.4% | ₹48.9M |
| Value-Weighted Agent | ₹108.2M | ₹20,734 | ₹745.63 | 80.0% | ₹26.9M |
| **⭐ Predictive AI Agent** | **₹116.6M** | ₹33,646 | **₹878.38** | **85.1%** | **₹18.5M** |

The Predictive Agent wins on total recovered, recovery rate, and unresolved balance — while using
*fewer* attempts than either rule-based agent (132,752 vs. 151,216 / 145,118) — and its root-cause
classifier captures 99.5% of the theoretically achievable discrimination on this dataset (AUC 0.714
vs. an oracle ceiling of 0.7173), meaning the remaining gap is irreducible outcome noise, not a
modeling shortfall.

## Repository structure

```
.
├── notebook/
│   └── razorpay_dataset_pipeline.ipynb   # full pipeline: data gen → training → benchmark
├── dashboard/
│   ├── app.py                            # Streamlit dashboard (Stage 7)
│   └── requirements.txt
├── data/                                  # CSVs produced by the notebook (gitignored by default)
└── README.md
```

## Running the pipeline

Open `notebook/razorpay_dataset_pipeline.ipynb` in Colab or Jupyter and run top to bottom. It will
write all intermediate + final CSVs (`recovery_actions.csv`, `recovery_events.csv`,
`root_cause_diagnostics.csv`, `stage4_five_strategy_benchmark.csv`,
`stage5_exception_clusters.csv`, `stage6_recalibration_eval.csv`, etc.) to your configured
`WORKDIR`.

## Running the dashboard

```bash
cd dashboard
pip install -r requirements.txt
streamlit run app.py
```

In the sidebar, point **Data source** at the same folder the notebook wrote its CSVs to.

## Key design decisions

- **No leakage across the confidence gate**: every stage downstream of root-cause classification
  only ever sees the *predicted* error code (`UNKNOWN` below the confidence threshold) — never the
  ground-truth label used to generate the data.
- **Independent RNG streams**: the no-intervention baseline and the per-channel intervention
  outcomes are drawn independently, so the benchmark can't accidentally correlate "what would have
  happened anyway" with "what the AI achieved."
- **EV-gated, not just probability-gated**: an action is only taken if predicted probability
  clears a floor *and* expected value clears a cost buffer — this is what lets the agent
  intentionally skip cheap-but-hopeless cases while still pursuing expensive-but-justified ones.
- **Primary metric is ₹ recovered per ₹ spent, not gross ₹ recovered** — a policy that contacts
  everyone can look better on raw recovery while being far less efficient; see the benchmark table
  and `dashboard/app.py`'s top panel for how we surface this trade-off honestly, including where
  the Predictive Agent's *marginal* returns hold up even when its *blended* ratio looks lower than
  a cheaper agent's.

## Known limitations

- All outcomes are synthetic (Bernoulli draws from a modeled `true_success_probability`), so the
  benchmark is a **simulation metric for comparing policies fairly**, not a causal estimate of
  production impact.
- The root-cause classifier's AUC (~0.71) is close to the dataset's theoretical ceiling
  (~0.7173) — see the oracle-check cell in the notebook for the derivation. Further AUC
  improvements would require reducing the outcome noise built into the generator, not just
  better modeling.
