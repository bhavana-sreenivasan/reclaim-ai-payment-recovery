"""ReClaim v2 -- one-screen demo: failed payment in -> diagnosed cause -> cost-aware recovery plan out.
SIMULATED DATA ONLY. The model was trained on a simulator with invented parameters."""
import json
from pathlib import Path
import gradio as gr, joblib, numpy as np, pandas as pd

try:
    import shap
except Exception:                      # app still runs without SHAP; the explanation panels say so
    shap = None

import run_pipeline as rp
from generate_data import (BANKS, CHANNELS, DEVICES, GATEWAYS, METHODS, NETWORKS, TIERS, TIER_BUDGET,
                           SPECIFIC_MSG, GENERIC_MSG, ERRS, ERR_MIX, total_cost_vector)

ART = Path(__file__).parent / "artifacts"
RC = joblib.load(ART / "root_cause.joblib"); RC_MODEL, TAU = RC["model"], RC["tau"]
REC = joblib.load(ART / "recovery_model.joblib"); MODEL, ISO = REC["model"], REC["iso"]
META = json.load(open(ART / "meta.json"))
# 5-seed means from results/model_diagnostics.csv (python run_pipeline.py --seeds 2026 1 2 3 4)
FIVE_SEED = dict(auc=0.6623, prec=0.9412, cov=0.6935)

# Causes the generator never produces for a payment method (ERR_MIX == 0), derived so it cannot drift.
IMPOSSIBLE = {m: {ERRS[j] for j in range(len(ERRS)) if ERR_MIX[i, j] == 0}
              for i, m in enumerate(METHODS)}


def messages_for(method):
    bad = IMPOSSIBLE.get(method, set())
    specific = [m for cause, msgs in SPECIFIC_MSG.items() if cause not in bad for m in msgs]
    return specific + list(GENERIC_MSG)          # generic messages are valid for every method


def on_method(method, current_msg):
    choices = messages_for(method)
    value = current_msg if current_msg in choices else choices[0]
    return gr.update(choices=choices, value=value)


# ------------------------------------------------------------------ explainability (SHAP)
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


def _bar(x, scale):
    n = int(round(min(abs(x) / scale, 1.0) * 8)) if scale > 0 else 0
    return ("\u25b2 " if x >= 0 else "\u25bc ") + f"{x:+.2f} " + "\u2588" * max(n, 1 if abs(x) >= 0.005 else 0)


def explain_recovery(df, ctx_arr, ch_idx, top=8):
    """SHAP for the XGBoost recovery model on the row for the chosen channel, plus the same row with
    no action, so the second column shows what the channel adds over doing nothing (the uplift the EV uses).
    Units are the model's raw log-odds score (before isotonic calibration)."""
    empty = pd.DataFrame({"Factor": ["SHAP unavailable"], "Value": [""], "Effect on score": [""],
                          "Extra vs. no action": [""]})
    if EXPLAINER is None:
        return empty
    try:
        Xc = rp.feats(df, ctx_arr, np.array([CHANNELS[ch_idx]]))
        Xn = rp.feats(df, ctx_arr, np.array(["NONE"]))
        sc, sn = EXPLAINER.shap_values(Xc)[0], EXPLAINER.shap_values(Xn)[0]
    except Exception:
        return empty
    delta = sc - sn
    order = np.argsort(-np.abs(sc))[:top]
    scale = float(np.abs(sc).max())
    rows = [dict(Factor=FEATURE_LABELS.get(Xc.columns[i], Xc.columns[i]),
                 Value=_fmt_value(Xc.columns[i], Xc.iloc[0, i]),
                 **{"Effect on score": _bar(float(sc[i]), scale),
                    "Extra vs. no action": f"{float(delta[i]):+.2f}"}) for i in order]
    return pd.DataFrame(rows)


def explain_diagnosis(df, top=6):
    """Exact linear attribution for the TF-IDF + logistic-regression diagnosis: each word/phrase (and the
    payment method) contributes coef x tfidf-value to the top cause's score."""
    empty = pd.DataFrame({"Signal": ["unavailable"], "Top candidate cause": [""], "Evidence": [""]})
    try:
        pre, clf = RC_MODEL[0], RC_MODEL[-1]
        x = pre.transform(df).tocsr(); names = pre.get_feature_names_out()
        k = int(np.argmax(RC_MODEL.predict_proba(df)[0]))
        contrib = x.data * clf.coef_[k][x.indices]
        order = np.argsort(-np.abs(contrib))[:top]
        scale = float(np.abs(contrib).max())
        rows = []
        for j in order:
            n = names[x.indices[j]]
            n = (f'word: "{n[3:]}"' if n.startswith("t__") else
                 "payment method = " + n.split("payment_method_", 1)[-1] if n.startswith("m__") else n)
            rows.append({"Signal": n, "Top candidate cause": clf.classes_[k], "Evidence": _bar(float(contrib[j]), scale)})
        return pd.DataFrame(rows)
    except Exception:
        return empty


def recommend(amount, method, bank, tier, message, hsf, margin_pct, human_cost,
              gateway, device, network, hour, dom, hist_pay, hist_resp, tenure, hist_cnt, cust_f30, bank_f6):
    if any(v is None for v in (amount, hsf, margin_pct, human_cost, tenure, hist_cnt, cust_f30, bank_f6)):
        return (gr.update(),) * 6
    if amount <= 0:
        return ("**Enter an amount above 0.**",) + (gr.update(),) * 5
    if any(v < 0 for v in (tenure, hist_cnt, cust_f30, bank_f6)):
        return ("**Tenure, counts and failure numbers cannot be negative.**",) + (gr.update(),) * 5
    if message not in messages_for(method):
        return ("**Gateway message doesn't match this payment method.** Pick one from the list.",) + (gr.update(),) * 5
    margin = margin_pct / 100
    # day-of-month + hour define the timestamp (May 2026 calendar; weekday follows from it)
    ts = pd.Timestamp(year=2026, month=5, day=int(dom), hour=int(hour))
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

    P64, up64 = P.astype(np.float64), uplift.astype(np.float64)
    table = pd.DataFrame({
        "Channel": CHANNELS,
        "Allowed": np.where(elig, "yes", "no"),
        "P(paid) if used": np.where(elig, np.round(P64[:8], 3), np.nan),
        "Uplift vs. doing nothing": np.where(elig, np.round(up64, 3), np.nan),
        "Cost incl. fatigue (INR)": np.round(cost.astype(np.float64), 2),
        "Expected net value (INR)": np.where(elig, np.round(ev.astype(np.float64), 2), np.nan),
        "Plan step": [str(step.get(i, "")) for i in range(8)]})
    table = table.sort_values("Expected net value (INR)", ascending=False, na_position="last")

    if ctx == "UNKNOWN":
        gate = (f"**Diagnosis: UNKNOWN** (top guess *{pred}* at {conf:.0%} is below the gate threshold {TAU:.2f}, "
                f"so only safe customer-facing channels are allowed).")
    else:
        gate = f"**Diagnosis: {ctx}** (confidence {conf:.0%}, gate threshold {TAU:.2f})."
    if step:
        seq = " → ".join(CHANNELS[int(c)] for c in plan if c >= 0)
        verdict = f"### Plan: {seq}\nNo-action recovery chance: **{float(P[8]):.0%}**. Attempt budget for a {tier} tier customer: {TIER_BUDGET[tier]}."
    else:
        verdict = "### Plan: do nothing\nNo allowed channel has positive expected value at this amount, margin and cost."

    # 3) explanations: the first planned channel, or the best allowed channel if the plan is empty
    ch = int(plan[0]) if step else int(np.argmax(np.where(elig, ev, -np.inf)))
    why_rec = explain_recovery(df, ctx_arr, ch)
    why_diag = explain_diagnosis(df)
    note = (f"Explaining {CHANNELS[ch]}" + ("" if step else " (not planned: no channel has positive value)")
            + f" for diagnosis {ctx}.")
    return gate, diag, verdict, table, why_diag, gr.update(value=why_rec, label="Why this score? " + note)


with gr.Blocks(title="ReClaim v2: cost-aware payment recovery") as demo:
    gr.Markdown(
        "# ReClaim v2: cost-aware payment recovery\n"
        "> **Simulated data.** The models were trained on a simulator with invented parameters. "
        "This shows how the policy *reasons*, not real recovery rates.\n\n"
        "Enter a failed payment. The app diagnoses the cause from the gateway message (with a confidence gate), "
        "estimates each channel's **uplift** over doing nothing, subtracts its cost, and builds the attempt plan.")
    with gr.Row():
        with gr.Column():
            amount = gr.Number(1500, label="Amount (INR)", minimum=1)
            method = gr.Dropdown(METHODS, value="Card", label="Payment method")
            bank = gr.Dropdown(BANKS, value="BANK_G", label="Bank (anonymised)")
            tier = gr.Dropdown(TIERS, value="Medium", label="Customer tier")
            message = gr.Dropdown(messages_for("Card"), value="Insufficient funds in account",
                                  label="Gateway message")
            hsf = gr.Slider(0.05, 72, value=2, label="Hours since failure")
            margin = gr.Slider(5, 60, value=30, step=1, label="Contribution margin (%)")
            human = gr.Slider(10, 120, value=45, step=5, label="Human-call cost (INR)")
            with gr.Accordion("Advanced inputs", open=False):
                gateway = gr.Dropdown(GATEWAYS, value="GW_1", label="Gateway")
                device = gr.Dropdown(DEVICES, value="Android", label="Device")
                network = gr.Dropdown(NETWORKS, value="4G", label="Network")
                hour = gr.Slider(0, 23, value=20, step=1, label="Hour of failure")
                dom = gr.Slider(1, 31, value=27, step=1, label="Day of month")
                hist_pay = gr.Slider(0, 1, value=0.9, label="Historical payment success rate")
                hist_resp = gr.Slider(0, 1, value=0.4, label="Historical response rate")
                tenure = gr.Number(400, label="Tenure (days)")
                hist_cnt = gr.Number(30, label="Historical transaction count")
                cust_f30 = gr.Number(0, label="Customer failures, prior 30 days")
                bank_f6 = gr.Number(3, label="Bank failures, last 6 hours")
        with gr.Column():
            gate_md = gr.Markdown(); verdict_md = gr.Markdown()
            diag_df = gr.Dataframe(label="Root-cause top 3", interactive=False)
    table_df = gr.Dataframe(label="Per-channel economics (best first)", interactive=False, wrap=True)
    with gr.Accordion("Why this plan? (SHAP explanations)", open=True):
        gr.Markdown(
            "**Diagnosis:** the cause model is linear, so each word (and the payment method) adds or subtracts "
            "an exact amount of evidence. **Recovery score:** SHAP values from the XGBoost recovery model for the "
            "planned channel, in the model's raw log-odds score (before calibration, so not percentage points). "
            "*Extra vs. no action* is the same attribution minus the no-action row, i.e. what the channel adds. "
            "On simulated data these mostly recover the relationships built into the simulator.")
        why_diag_df = gr.Dataframe(label="Why this diagnosis? (top signals)", interactive=False, wrap=True)
        why_rec_df = gr.Dataframe(label="Why this score?", interactive=False, wrap=True)
    gr.Markdown(
        "**Cost incl. fatigue** = direct cost + a contact-fatigue penalty (opt-outs, annoyance); "
        "for a human call it is your slider value + INR 5. "
        "**Expected net value** = margin x amount x uplift - cost. A negative value means the attempt "
        "loses money, so the policy skips it.")
    gr.Markdown(
        f"Held-out (simulated) test metrics, May-Jun, seed {META.get('seed', 2026)}: recovery-model AUC {META['test_auc']:.2f}, "
        f"root-cause gate precision {META['test_gate_precision']:.0%} at {META['test_gate_coverage']:.0%} coverage. "
        f"Mean over 5 seeds: AUC {FIVE_SEED['auc']:.2f}, gate precision {FIVE_SEED['prec']:.0%} at {FIVE_SEED['cov']:.0%} coverage. "
        "Findings and limitations are in the project README.")
    ins = [amount, method, bank, tier, message, hsf, margin, human,
           gateway, device, network, hour, dom, hist_pay, hist_resp, tenure, hist_cnt, cust_f30, bank_f6]
    outs = [gate_md, diag_df, verdict_md, table_df, why_diag_df, why_rec_df]
    method.change(on_method, [method, message], message)   # restrict messages to ones valid for the method
    gr.on(triggers=[c.change for c in ins], fn=recommend, inputs=ins, outputs=outs,
          trigger_mode="always_last", show_progress="hidden")
    demo.load(recommend, ins, outs)

if __name__ == "__main__":
    demo.launch()