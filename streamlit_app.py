"""ReClaim v2 -- Streamlit dashboard: failed payment in -> diagnosed cause -> cost-aware recovery plan out.
SIMULATED DATA ONLY. The models were trained on a simulator with invented parameters."""
from pathlib import Path

import pandas as pd
import streamlit as st

import core
from generate_data import BANKS, CHANNELS, DEVICES, GATEWAYS, METHODS, NETWORKS, TIERS

st.set_page_config(page_title="ReClaim v2: cost-aware payment recovery", page_icon="💳", layout="wide")
RESULTS = Path(__file__).parent / "results"

st.title("💳 ReClaim v2: cost-aware payment recovery")
st.warning("**Simulated data.** The models were trained on a simulator with invented parameters. "
           "This shows how the policy *reasons*, not real recovery rates.", icon="⚠️")

# ------------------------------------------------------------------ inputs (sidebar)
with st.sidebar:
    st.header("Failed payment")
    amount = st.number_input("Amount (INR)", min_value=1.0, value=15000.0, step=100.0)
    method = st.selectbox("Payment method", METHODS, index=METHODS.index("Card"))
    bank = st.selectbox("Bank (anonymised)", BANKS, index=BANKS.index("BANK_G"))
    tier = st.selectbox("Customer tier", TIERS, index=TIERS.index("Medium"))
    msgs = core.messages_for(method)          # only messages that can occur for this method
    default_msg = "Card has expired" if "Card has expired" in msgs else msgs[0]
    message = st.selectbox("Gateway message", msgs, index=msgs.index(default_msg))
    hsf = st.slider("Hours since failure", 0.05, 72.0, 2.0)
    st.header("Business assumptions")
    margin_pct = st.slider("Contribution margin (%)", 5, 60, 30)
    human_cost = st.slider("Human-call cost (INR)", 10, 120, 45, step=5)
    with st.expander("Advanced inputs"):
        gateway = st.selectbox("Gateway", GATEWAYS)
        device = st.selectbox("Device", DEVICES)
        network = st.selectbox("Network", NETWORKS, index=NETWORKS.index("4G"))
        hour = st.slider("Hour of failure", 0, 23, 20)
        dom = st.slider("Day of month", 1, 31, 27)
        hist_pay = st.slider("Historical payment success rate", 0.0, 1.0, 0.9)
        hist_resp = st.slider("Historical response rate", 0.0, 1.0, 0.4)
        tenure = st.number_input("Tenure (days)", min_value=0.0, value=400.0)
        hist_cnt = st.number_input("Historical transaction count", min_value=0.0, value=30.0)
        cust_f30 = st.number_input("Customer failures, prior 30 days", min_value=0.0, value=0.0)
        bank_f6 = st.number_input("Bank failures, last 6 hours", min_value=0.0, value=3.0)

tab_plan, tab_why, tab_eval = st.tabs(["Recovery planner", "Why this plan? (SHAP)", "Evaluation results"])

res = core.recommend(amount, method, bank, tier, message, hsf, margin_pct, human_cost,
                     gateway, device, network, hour, dom, hist_pay, hist_resp, tenure, hist_cnt, cust_f30, bank_f6)

# ------------------------------------------------------------------ tab 1: planner
with tab_plan:
    if res["error"]:
        st.error(res["error"])
    else:
        c1, c2, c3 = st.columns([1.2, 1.6, 1])
        c1.metric("Diagnosis", res["ctx"], help=f"Top guess {res['pred']} at {res['conf']:.0%}; gate threshold {res['tau']:.2f}")
        c2.metric("Plan", " → ".join(res["plan"]) if res["plan"] else "Do nothing")
        c3.metric("No-action recovery chance", f"{res['no_action']:.0%}")
        if res["ctx"] == "UNKNOWN":
            st.info(f"Top guess **{res['pred']}** at {res['conf']:.0%} is below the gate threshold {res['tau']:.2f}, "
                    "so the cause is withheld and only safe customer-facing channels are allowed.")
        else:
            st.success(f"Confidence {res['conf']:.0%} (gate threshold {res['tau']:.2f}).")
        if not res["plan"]:
            st.info("No allowed channel has positive expected value at this amount, margin and cost.")
        else:
            st.caption(f"Attempt budget for a {tier} tier customer: {core.TIER_BUDGET[tier]}.")

        left, right = st.columns([1.5, 1])
        with left:
            st.subheader("Per-channel economics (best first)")
            st.dataframe(res["table"], hide_index=True, width="stretch")
        with right:
            st.subheader("Expected net value (INR)")
            chart = res["table"].dropna(subset=["Expected net value (INR)"]).set_index("Channel")["Expected net value (INR)"]
            st.bar_chart(chart)
            st.subheader("Root-cause top 3")
            st.dataframe(res["diag"], hide_index=True, width="stretch")
        st.caption("**Cost incl. fatigue** = direct cost + a contact-fatigue penalty (opt-outs, annoyance); for a human "
                   "call it is your slider value + INR 5. **Expected net value** = margin x amount x uplift - cost. "
                   "A negative value means the attempt loses money, so the policy skips it.")

# ------------------------------------------------------------------ tab 2: SHAP
with tab_why:
    if res["error"]:
        st.error(res["error"])
    else:
        st.markdown(
            "**Diagnosis:** the cause model is linear, so each word (and the payment method) adds or subtracts an exact "
            "amount of evidence. **Recovery score:** SHAP values from the XGBoost recovery model for the planned channel, "
            "in the model's raw log-odds score (before calibration, so not percentage points). *Extra vs. no action* is "
            "the same attribution minus the no-action row, i.e. what the channel adds. On simulated data these mostly "
            "recover the relationships built into the simulator.")
        a, b = st.columns(2)
        with a:
            st.subheader("Why this diagnosis?")
            if res["why_diag"] is None:
                st.info("Explanation unavailable.")
            else:
                st.bar_chart(res["why_diag"].set_index("Signal")["Evidence"])
                st.dataframe(res["why_diag"].round(2), hide_index=True, width="stretch")
        with b:
            note = f"Explaining {res['explained_channel']}" + ("" if res["explained_planned"] else " (not planned: no channel has positive value)")
            st.subheader("Why this score?")
            st.caption(f"{note} for diagnosis {res['ctx']}.")
            if res["why_rec"] is None:
                st.info("SHAP is unavailable in this environment.")
            else:
                st.bar_chart(res["why_rec"].set_index("Factor")["Effect"])
                st.dataframe(res["why_rec"].rename(columns={"Effect": "Effect on score", "Extra": "Extra vs. no action"}).round(2),
                             hide_index=True, width="stretch")

# ------------------------------------------------------------------ tab 3: evaluation
with tab_eval:
    st.subheader("Policy comparison (5 simulated seeds, test period, per 1,000 failed payments, margin 30%)")
    f = RESULTS / "policy_across_seeds.csv"
    if f.exists():
        pol = pd.read_csv(f)
        view = pol.rename(columns={"policy": "Policy", "recovery_rate_mean": "Recovery rate",
                                   "incr_net_per_1k_mean": "Incremental net (INR)", "incr_net_per_1k_sd": "± sd",
                                   "cost_per_1k_mean": "Attempt cost (INR)", "human_call_pct_mean": "Human-call share"})
        st.bar_chart(view.set_index("Policy")["Incremental net (INR)"])
        st.dataframe(view[["Policy", "Recovery rate", "Incremental net (INR)", "± sd", "Attempt cost (INR)"]]
                     .style.format({"Recovery rate": "{:.1%}", "Incremental net (INR)": "{:,.0f}",
                                    "± sd": "{:,.0f}", "Attempt cost (INR)": "{:,.0f}"}),
                     hide_index=True, width="stretch")
    else:
        st.info("Add the `results/` folder (from `run_pipeline.py`) to see the policy comparison here.")

    g1, g2 = st.columns(2)
    with g1:
        st.subheader("Does diagnosis matter? (root-cause ablation)")
        f = RESULTS / "ablation_root_cause.csv"
        if f.exists():
            ab = pd.read_csv(f)
            st.bar_chart(ab.set_index("variant")["incr_net_per_1k_mean"])
    with g2:
        st.subheader("Paired-bootstrap differences")
        f = RESULTS / "bootstrap_ci.csv"
        if f.exists():
            bo = pd.read_csv(f)
            st.dataframe(bo.round(0), hide_index=True, width="stretch")

    st.markdown(
        f"Held-out (simulated) test metrics, May-Jun, seed {core.META.get('seed', 2026)}: recovery-model AUC "
        f"{core.META['test_auc']:.2f}, root-cause gate precision {core.META['test_gate_precision']:.0%} at "
        f"{core.META['test_gate_coverage']:.0%} coverage. Mean over 5 seeds: AUC {core.FIVE_SEED['auc']:.2f}, gate "
        f"precision {core.FIVE_SEED['prec']:.0%} at {core.FIVE_SEED['cov']:.0%} coverage. "
        "Findings and limitations are in the project README.")
