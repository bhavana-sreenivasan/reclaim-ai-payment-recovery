"""ReClaim v2 core logic (no UI framework): load models, diagnose, plan, explain.
SIMULATED DATA ONLY. Models were trained on a simulator with invented parameters."""
import json
from pathlib import Path

import joblib
import numpy as np
import pandas as pd

try:
    import shap
except Exception:                      # the dashboard still works without SHAP
    shap = None

import run_pipeline as rp
from generate_data import (CHANNELS, ERRS, ERR_MIX, GENERIC_MSG, METHODS, SPECIFIC_MSG, TIER_BUDGET,
                           total_cost_vector)

HERE = Path(__file__).parent


def _find(name):
    """Look in artifacts/ first, then next to this file, so folder or flat uploads both work."""
    for d in (HERE / "artifacts", HERE):
        if (d / name).exists():
            return d / name
    raise FileNotFoundError(f"{name} not found in artifacts/ or next to the app")


RC = joblib.load(_find("root_cause.joblib")); RC_MODEL, TAU = RC["model"], RC["tau"]
REC = joblib.load(_find("recovery_model.joblib")); MODEL, ISO = REC["model"], REC["iso"]
META = json.load(open(_find("meta.json")))
# 5-seed means from results/model_diagnostics.csv (python run_pipeline.py --seeds 2026 1 2 3 4)
FIVE_SEED = dict(auc=0.6623, prec=0.9412, cov=0.6935)

# Causes the generator never produces for a payment method (ERR_MIX == 0).
IMPOSSIBLE = {m: {ERRS[j] for j in range(len(ERRS)) if ERR_MIX[i, j] == 0} for i, m in enumerate(METHODS)}


def messages_for(method):
    bad = IMPOSSIBLE.get(method, set())
    specific = [m for cause, msgs in SPECIFIC_MSG.items() if cause not in bad for m in msgs]
    return specific + list(GENERIC_MSG)          # generic messages are valid for every method


# ------------------------------------------------------------------ explainability
try:
    EXPLAINER = shap.TreeExplainer(MODEL) if shap is not None else None
except Exception:
    EXPLAINER = None

FEATURE_LABELS = {
    "arm": "Channel used", "ctx": "Diagnosed cause", "method": "Payment method", "bank": "Bank",
    "gateway": "Gateway", "device": "Device", "network": "Network", "tier": "Customer tier",
    "log_amount": "Amount", "hsf": "Hours since failure", "hour": "Hour of day", "dow": "Day of week",
    "dom": "Day of month", "hist_pay": "Past payment success rate", "hist_resp": "Past response rate",
    "tenure": "Tenure (days)", "hist_cnt": "Past transaction count",
    "cust_f30": "Customer failures (30d)", "bank_f6": "Bank failures (6h)"}
DOW = ["Mon", "Tue", "Wed", "Thu", "Fri", "Sat", "Sun"]


def _fmt_value(name, v):
    if name == "log_amount":
        return f"INR {np.exp(float(v)):,.0f}"
    if name == "dow":
        return DOW[int(v)]
    if name in ("hist_pay", "hist_resp"):
        return f"{float(v):.2f}"
    if isinstance(v, (float, np.floating)):
        return f"{float(v):g}"
    return str(v)


def explain_recovery(df, ctx_arr, ch_idx, top=8):
    """SHAP on the XGBoost recovery model for the chosen channel, plus the no-action row, so
    'extra' shows what the channel adds over doing nothing. Units: raw log-odds (before calibration).
    Returns None if SHAP is unavailable."""
    if EXPLAINER is None:
        return None
    try:
        Xc = rp.feats(df, ctx_arr, np.array([CHANNELS[ch_idx]]))
        Xn = rp.feats(df, ctx_arr, np.array(["NONE"]))
        sc, sn = EXPLAINER.shap_values(Xc)[0], EXPLAINER.shap_values(Xn)[0]
    except Exception:
        return None
    order = np.argsort(-np.abs(sc))[:top]
    return pd.DataFrame([dict(
        Factor=FEATURE_LABELS.get(Xc.columns[i], Xc.columns[i]), Value=_fmt_value(Xc.columns[i], Xc.iloc[0, i]),
        Effect=float(sc[i]), Extra=float(sc[i] - sn[i])) for i in order])


def explain_diagnosis(df, top=6):
    """Exact linear attribution for TF-IDF + logistic regression: coef x tfidf value per signal."""
    try:
        pre, clf = RC_MODEL[0], RC_MODEL[-1]
        x = pre.transform(df).tocsr(); names = pre.get_feature_names_out()
        k = int(np.argmax(RC_MODEL.predict_proba(df)[0]))
        contrib = x.data * clf.coef_[k][x.indices]
        rows = []
        for j in np.argsort(-np.abs(contrib))[:top]:
            n = names[x.indices[j]]
            n = (f'word: "{n[3:]}"' if n.startswith("t__") else
                 "payment method = " + n.split("payment_method_", 1)[-1] if n.startswith("m__") else n)
            rows.append({"Signal": n, "Top candidate cause": clf.classes_[k], "Evidence": float(contrib[j])})
        return pd.DataFrame(rows)
    except Exception:
        return None


# ------------------------------------------------------------------ the decision
def recommend(amount, method, bank, tier, message, hsf, margin_pct, human_cost,
              gateway, device, network, hour, dom, hist_pay, hist_resp, tenure, hist_cnt, cust_f30, bank_f6):
    """Returns a dict. If 'error' is set, show it and stop."""
    if amount is None or amount <= 0:
        return dict(error="Enter an amount above 0.")
    if any(v is None or v < 0 for v in (tenure, hist_cnt, cust_f30, bank_f6)):
        return dict(error="Tenure, counts and failure numbers cannot be negative.")
    if message not in messages_for(method):
        return dict(error="Gateway message doesn't match this payment method. Pick one from the list.")
    margin = margin_pct / 100
    ts = pd.Timestamp(year=2026, month=5, day=int(dom), hour=int(hour))   # May 2026 calendar
    df = pd.DataFrame([dict(
        timestamp=ts, amount=float(amount), payment_method=method, bank=bank, gateway=gateway,
        device_type=device, network_type=network, customer_tier=tier, gateway_message=message,
        hours_since_failure=float(hsf), tenure_days=float(tenure), hist_txn_count=float(hist_cnt),
        hist_pay_success_rate=float(hist_pay), hist_response_rate=float(hist_resp),
        customer_failures_prior_30d=float(cust_f30), bank_fail_6h=float(bank_f6))])

    # 1) root cause + confidence gate
    pr = RC_MODEL.predict_proba(df)[0]; cls = RC_MODEL.classes_
    order = np.argsort(-pr)[:3]
    pred, conf = cls[order[0]], float(pr[order[0]])
    ctx = pred if conf >= TAU else "UNKNOWN"
    diag = pd.DataFrame({"Likely cause": cls[order], "Probability": pr[order].round(3)})
    ctx_arr = np.array([ctx], dtype=object)

    # 2) uplift per channel -> EV -> cascade
    P = rp.predict_arms(MODEL, ISO, df, ctx_arr)[0]            # 8 channels + NONE
    uplift = P[:8] - P[8]
    cost = total_cost_vector(1.0, float(human_cost))
    ev = margin * float(amount) * uplift - cost
    elig = rp.elig_from_ctx(ctx_arr)[0]
    plan = rp.orders_from_ev(ev[None, :], elig[None, :], np.array([TIER_BUDGET[tier]]))[0]
    step = {int(c): i + 1 for i, c in enumerate(plan) if c >= 0}

    table = pd.DataFrame({
        "Channel": CHANNELS,
        "Allowed": np.where(elig, "yes", "no"),
        "P(paid) if used": np.where(elig, np.round(P[:8].astype(np.float64), 3), np.nan),
        "Uplift vs. doing nothing": np.where(elig, np.round(uplift.astype(np.float64), 3), np.nan),
        "Cost incl. fatigue (INR)": np.round(cost.astype(np.float64), 2),
        "Expected net value (INR)": np.where(elig, np.round(ev.astype(np.float64), 2), np.nan),
        "Plan step": [str(step.get(i, "")) for i in range(8)]})
    table = table.sort_values("Expected net value (INR)", ascending=False, na_position="last").reset_index(drop=True)

    plan_seq = [CHANNELS[int(c)] for c in plan if c >= 0]
    ch = int(plan[0]) if step else int(np.argmax(np.where(elig, ev, -np.inf)))
    return dict(error=None, ctx=ctx, pred=pred, conf=conf, tau=TAU, diag=diag, table=table,
                plan=plan_seq, no_action=float(P[8]), budget=TIER_BUDGET[tier],
                why_rec=explain_recovery(df, ctx_arr, ch), why_diag=explain_diagnosis(df),
                explained_channel=CHANNELS[ch], explained_planned=bool(step))
