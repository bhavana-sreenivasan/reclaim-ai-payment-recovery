# Audit of the v1 dataset (`data_optimized.csv.gz`) and what v2 changed

All v1 figures below were measured directly from the committed file and notebook.

## What was wrong with v1

| # | Finding (v1) | Why it matters | v2 |
|---|---|---|---|
| 1 | Payment method, bank and gateway were near-uniform (~25% / 20% / 33% each). | UPI is roughly 80% of Indian digital payment volume; uniform mixes are not plausible. | Method mix 56-61% UPI (failures over-index on cards/netbanking); bank shares skewed; gateways 50/30/20. |
| 2 | Failure causes were statistically independent of method, bank and gateway (e.g. `ERR_EXPIRED_CARD` on UPI at the same ~15% rate as on cards). | Any model that learns "cause given method" is learning noise; the data contains impossible events. | Cause mix is conditional on method (0.0% expired-card on UPI/NetBanking), bank reliability, outages, hour, salary cycle. |
| 3 | Hour-of-day was flat; no month-end or weekday structure. | Real traffic and failure patterns are strongly time-dependent. | Evening peak (about 7x peak-to-trough), weekday shape, month-end cash crunch, subscription billing batches. |
| 4 | Channel outcomes were drawn **independently** within a transaction (residual correlation -0.0007; P(at least one of email/SMS/WhatsApp) = 0.840 observed vs 0.841 expected under independence). | Any 2-3 attempt cascade nearly always finds a success, which inflates every policy. This is why the rules-only baseline recovered 77%. Industry write-ups put good recovery stacks around 45-70%. | Shared latent stubbornness: mean pairwise outcome correlation 0.17; hindsight ceiling 59%. |
| 5 | Per-attempt costs were Rs0-0.50 against a median failed amount of Rs898 (about 0.02%). | The EV gate pursued 99.15% of cases and so never really gated anything. | Added human-call and IVR channels plus a contact-fatigue penalty so costs bind on escalation decisions. |
| 6 | Models trained on all five channels' outcomes for every transaction. | Production only reveals the action taken. | Training uses logged outcomes of one randomly assigned action (incl. a no-action arm); counterfactuals live in a separate oracle file used only for scoring. |
| 7 | Only 6 gateway-message templates per cause. | The root-cause task was trivial (99.9% precision at the gate). | About 40% of messages are vague and shared across causes; gate precision 94% at ~69% coverage. |
| 8 | Evaluation used a random group split; the benchmark then scored all 100k rows, including training rows. | Optimistic and not time-aware. | Strict time split; benchmark scores only the held-out May-Jun period, which includes drift. |
| 9 | `Data_synthesis.ipynb` did not reproduce the committed CSV (7 banks / 8 failure types / Jan-Aug in the notebook vs 5 / 5 / August-only in the data). | The dataset could not be regenerated from the repo. | `generate_data.py` is deterministic and writes the data directly. |

## Related issue in the v1 notebook (not a data problem)

The recovery-probability model (cell 10, AUC 0.714) takes the **true** `error_code` as an input feature, and its output
feeds the EV gate and the benchmark. So "downstream stages only see predicted error codes" does not hold for v1.
The 0.714 AUC belongs to that recovery model, not to the root-cause classifier (88.2% held-out accuracy).
v2 uses only the gated, predicted cause as a model input.

## What v2 still is

A simulator with invented parameters. It supports claims of the form "policy A beat policy B under these stated
assumptions". It does not support claims about real-world recovery rates.
