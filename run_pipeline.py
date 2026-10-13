"""
ReClaim v2 pipeline: root-cause inference -> uplift model -> cost-aware policies -> honest evaluation.

Honest-evaluation design
------------------------
* TIME-BASED split: train Jan-Mar, validate Apr, test May-Jun (the test period contains drift).
* The recovery model learns ONLY from logged outcomes of the single action actually taken
  (uniform-random exploration, incl. a no-action control arm) -- never from all-channel
  counterfactuals, which do not exist in production.
* Policy value is the INCREMENTAL net contribution over doing nothing:
      net = margin * recovered_amount - attempt_costs      (recovered = natural OR intervention)
* The agent's EV uses estimated UPLIFT (P(paid|channel) - P(paid|no action)), so it does not
  take credit for customers who would have paid anyway.
* Baselines include a no-ML segment-lookup table, to test whether XGBoost adds anything.
* Reported with paired bootstrap CIs, 5 independent data seeds, and a cost/margin sensitivity grid.
* An off-policy-evaluation (IPS) check estimates a policy's value from the logged data alone and
  compares it with the simulator's ground truth.
"""
import argparse, json, os, time
import numpy as np, pandas as pd, xgboost as xgb
from sklearn.compose import ColumnTransformer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.isotonic import IsotonicRegression
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import brier_score_loss, roc_auc_score
from sklearn.model_selection import KFold
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import OneHotEncoder

from generate_data import (ARMS, BANKS, CHANNELS, DEVICES, ELIGIBLE, ERRS, GATEWAYS, METHODS, NETWORKS,
                           SAFE_CHANNELS, TIER_BUDGET, TIERS, generate, total_cost_vector)

MARGIN = 0.30            # [ASSUME] contribution margin on recovered revenue
THIN = 0.85              # [ASSUME] each later attempt in a cascade is 15% less likely to convert
B = 9                    # number of logging arms (8 channels + NONE)
CH = {c: i for i, c in enumerate(CHANNELS)}
CODE = {e: i for i, e in enumerate(ERRS)}


# ============================================================ root cause (text -> failure cause)
def fit_root_cause(tr, va, te, target_precision=0.90):
    def mk():
        pre = ColumnTransformer([("t", TfidfVectorizer(ngram_range=(1, 2), min_df=2), "gateway_message"),
                                 ("m", OneHotEncoder(handle_unknown="ignore"), ["payment_method"])])
        return make_pipeline(pre, LogisticRegression(max_iter=500, C=3.0))
    full = mk().fit(tr, tr.true_error_code)
    classes = full.classes_
    oof = np.zeros((len(tr), len(classes)))
    for a, b in KFold(5, shuffle=True, random_state=0).split(tr):       # cross-fit: no self-prediction on train
        oof[b] = mk().fit(tr.iloc[a], tr.true_error_code.iloc[a]).predict_proba(tr.iloc[b])
    pv, pt = full.predict_proba(va), full.predict_proba(te)
    pred_v, conf_v = classes[pv.argmax(1)], pv.max(1)
    ok_v = pred_v == va.true_error_code.values
    tau = 0.90
    for t in np.arange(0.20, 0.97, 0.01):                               # smallest threshold meeting precision target
        g = conf_v >= t
        if g.sum() > 50 and ok_v[g].mean() >= target_precision:
            tau = float(t); break
    out = {}
    for name, p in (("tr", oof), ("va", pv), ("te", pt)):
        pred, conf = classes[p.argmax(1)], p.max(1)
        out[name] = dict(pred=pred, conf=conf, ctx=np.where(conf >= tau, pred, "UNKNOWN"))
    gate_te = out["te"]["conf"] >= tau
    diag = dict(gate_tau=round(tau, 3),
                test_accuracy_all=float((out["te"]["pred"] == te.true_error_code.values).mean()),
                test_gate_coverage=float(gate_te.mean()),
                test_gate_precision=float((out["te"]["pred"] == te.true_error_code.values)[gate_te].mean()))
    return out, diag


def elig_from_ctx(ctx):
    M = np.zeros((len(ctx), 8), bool)
    M[ctx == "UNKNOWN"] = SAFE_CHANNELS
    for e, i in CODE.items():
        M[ctx == e] = ELIGIBLE[i]
    return M


# ============================================================ recovery (uplift) model
def feats(df, ctx, arm=None):
    n = len(df)
    arm = np.full(n, "NONE") if arm is None else arm
    ts = df.timestamp.dt
    return pd.DataFrame({
        "arm": pd.Categorical(arm, categories=ARMS), "ctx": pd.Categorical(ctx, categories=ERRS + ["UNKNOWN"]),
        "method": pd.Categorical(df.payment_method.values, categories=METHODS),
        "bank": pd.Categorical(df.bank.values, categories=BANKS),
        "gateway": pd.Categorical(df.gateway.values, categories=GATEWAYS),
        "device": pd.Categorical(df.device_type.values, categories=DEVICES),
        "network": pd.Categorical(df.network_type.values, categories=NETWORKS),
        "tier": pd.Categorical(df.customer_tier.values, categories=TIERS),
        "log_amount": np.log(df.amount.values), "hsf": df.hours_since_failure.values,
        "hour": ts.hour.values, "dow": ts.dayofweek.values, "dom": ts.day.values,
        "hist_pay": df.hist_pay_success_rate.values, "hist_resp": df.hist_response_rate.values,
        "tenure": df.tenure_days.values, "hist_cnt": df.hist_txn_count.values,
        "cust_f30": df.customer_failures_prior_30d.values, "bank_f6": df.bank_fail_6h.values})


def ece(y, p, bins=10):
    q = np.quantile(p, np.linspace(0, 1, bins + 1)); q[-1] += 1e-9
    idx = np.digitize(p, q[1:-1]); tot = 0.0
    for b in range(bins):
        m = idx == b
        if m.any():
            tot += m.mean() * abs(y[m].mean() - p[m].mean())
    return float(tot)


def fit_recovery_model(tr, va, ctx_tr, ctx_va):
    Xtr, Xva = feats(tr, ctx_tr, tr.logged_channel.values), feats(va, ctx_va, va.logged_channel.values)
    m = xgb.XGBClassifier(n_estimators=700, max_depth=5, learning_rate=0.04, subsample=0.8, colsample_bytree=0.8,
                          min_child_weight=10, reg_lambda=5, enable_categorical=True, tree_method="hist",
                          eval_metric="logloss", early_stopping_rounds=40, n_jobs=1, random_state=0)
    m.fit(Xtr, tr.logged_paid.values, eval_set=[(Xva, va.logged_paid.values)], verbose=False)
    iso = IsotonicRegression(out_of_bounds="clip", y_min=0, y_max=1).fit(m.predict_proba(Xva)[:, 1], va.logged_paid.values)
    return m, iso


def predict_arms(m, iso, df, ctx):
    """(N, 9) calibrated P(paid) for every arm (8 channels + NONE), whatever was actually logged."""
    base = feats(df, ctx)
    X = pd.concat([base.assign(arm=pd.Categorical([a] * len(df), categories=ARMS)) for a in ARMS], ignore_index=True)
    p = iso.transform(m.predict_proba(X)[:, 1])
    return p.reshape(len(ARMS), len(df)).T


# ============================================================ policies -> attempt orders
def orders_from_ev(ev, elig, budget, min_ev=0.0, K=3):
    e = np.where(elig, ev, -np.inf)
    o = np.argsort(-e, axis=1)[:, :K]
    top = np.take_along_axis(e, o, 1)
    ok = (top > min_ev) & (np.arange(K)[None, :] < budget[:, None])
    return np.where(ok, o, -1)


def ladder_orders(budget):
    lad = np.array([CH["RETRY_NOW"], CH["EMAIL"], CH["SMS"], CH["WHATSAPP"]])
    return np.where(np.arange(3)[None, :] < budget[:, None], lad[None, :3], -1)


def value_orders(amount, budget):
    lo, mid, hi = [CH["EMAIL"], CH["SMS"], CH["WHATSAPP"]], [CH["WHATSAPP"], CH["SMS"], CH["EMAIL"]], \
        [CH["HUMAN_CALL"], CH["WHATSAPP"], CH["SMS"]]
    seq = np.where((amount < 500)[:, None], np.array(lo)[None], np.where((amount < 3000)[:, None],
                                                                        np.array(mid)[None], np.array(hi)[None]))
    return np.where(np.arange(3)[None, :] < budget[:, None], seq, -1)


class SegmentTable:
    """No-ML baseline: empirical P(paid) by (cause context, amount tercile, arm) from TRAIN logs."""
    def __init__(self, tr, ctx_tr):
        self.cuts = np.quantile(tr.amount, [1 / 3, 2 / 3])
        d = pd.DataFrame({"ctx": ctx_tr, "ab": np.digitize(tr.amount, self.cuts), "arm": tr.logged_channel.values,
                          "paid": tr.logged_paid.values})
        self.glob = d.groupby("arm").paid.mean().reindex(ARMS)
        self.seg = d.groupby(["ctx", "ab", "arm"]).paid.mean().unstack("arm").reindex(columns=ARMS)
        self.ctx_only = d.groupby(["ctx", "arm"]).paid.mean().unstack("arm").reindex(columns=ARMS)

    def rates(self, df, ctx, with_amount_bin=True):
        if with_amount_bin:
            idx = pd.MultiIndex.from_arrays([ctx, np.digitize(df.amount.values, self.cuts)])
            R = self.seg.reindex(idx).values
        else:
            R = self.ctx_only.reindex(ctx).values
        return np.where(np.isnan(R), self.glob.values[None, :], R)


# ============================================================ simulator evaluation
def run_orders(orders, amt, nat, raw, thin, cost_vec, margin):
    N = len(amt); done = np.zeros(N, bool); cost = np.zeros(N); att = np.zeros(N, int)
    for k in range(orders.shape[1]):
        ch = orders[:, k]; idx = np.where((ch >= 0) & ~done)[0]
        if len(idx) == 0:
            continue
        c = ch[idx]; cost[idx] += cost_vec[c]; att[idx] += 1
        ok = raw[idx, c].astype(bool) & (thin[idx, c] < THIN ** k)
        done[idx[ok]] = True
    rec = nat | done
    return dict(rec=rec, cost=cost, att=att, net=margin * amt * rec - cost)


def oracle_ceiling(amt, nat, raw, cost_vec, margin):
    cand = raw.astype(bool) & ((margin * amt[:, None] - cost_vec[None, :]) > 0)
    cheapest = np.where(cand, cost_vec[None, :], np.inf).min(1)
    use = (~nat) & np.isfinite(cheapest)
    rec = nat | use; cost = np.where(use, cheapest, 0.0)
    return dict(rec=rec, cost=cost, att=use.astype(int), net=margin * amt * rec - cost)


def summarise(name, r, amt, nat, margin):
    base_net = margin * amt * nat
    n = len(amt)
    return dict(policy=name, recovery_rate=r["rec"].mean(), uplift_pp=100 * (r["rec"].mean() - nat.mean()),
                pursued_pct=100 * (r["att"] > 0).mean(), attempts_per_1k=1000 * r["att"].sum() / n,
                cost_per_1k=1000 * r["cost"].sum() / n, net_per_1k=1000 * r["net"].sum() / n,
                incr_net_per_1k=1000 * (r["net"].sum() - base_net.sum()) / n)


def all_policies(ctx, elig, budget, df, P, seg, cost_vec, margin):
    """Build attempt orders for every policy under a given margin/cost vector."""
    amt = df.amount.values
    unc = P[:, :8] - P[:, [8]]                                    # estimated uplift per channel
    R_seg = seg.rates(df, ctx, True); R_ctx = seg.rates(df, ctx, False)
    pol = {
        "No action": np.full((len(df), 3), -1),
        "Rules ladder (retry>email>SMS>WA)": ladder_orders(budget),
        "Value-weighted rules": value_orders(amt, budget),
        "Reason playbook (cost-blind)": orders_from_ev(R_ctx[:, :8] - R_ctx[:, [8]], elig, budget, min_ev=-np.inf),
        "Segment lookup (no ML, EV-gated)": orders_from_ev(
            margin * amt[:, None] * (R_seg[:, :8] - R_seg[:, [8]]) - cost_vec[None, :], elig, budget),
        "Predictive EV agent (cascade)": orders_from_ev(margin * amt[:, None] * unc - cost_vec[None, :], elig, budget),
        "Predictive EV agent (single-shot)": orders_from_ev(margin * amt[:, None] * unc - cost_vec[None, :], elig,
                                                          np.ones(len(df), int), K=1),
    }
    return pol


# ============================================================ one full run
def run_one(seed, detailed=False, rng_boot=None):
    t0 = time.time()
    obs, orc = generate(seed)
    ts = obs.timestamp
    m_tr, m_va, m_te = ts < "2026-04-01", (ts >= "2026-04-01") & (ts < "2026-05-01"), ts >= "2026-05-01"
    tr, va, te = [obs[m].reset_index(drop=True) for m in (m_tr, m_va, m_te)]
    otr, ote = orc[m_te].reset_index(drop=True), None
    ote = otr
    rc, rc_diag = fit_root_cause(tr, va, te)
    model, iso = fit_recovery_model(tr, va, rc["tr"]["ctx"], rc["va"]["ctx"])

    # --- recovery-model diagnostics on held-out logged rows (the only outcomes observable in production)
    Xte_log = feats(te, rc["te"]["ctx"], te.logged_channel.values)
    p_raw = model.predict_proba(Xte_log)[:, 1]; p_cal = iso.transform(p_raw); y = te.logged_paid.values
    diag = dict(seed=seed, n_train=len(tr), n_val=len(va), n_test=len(te), **rc_diag,
                test_auc=float(roc_auc_score(y, p_cal)), test_brier_raw=float(brier_score_loss(y, p_raw)),
                test_brier_calibrated=float(brier_score_loss(y, p_cal)), test_ece_raw=ece(y, p_raw),
                test_ece_calibrated=ece(y, p_cal), test_base_rate=float(y.mean()),
                brier_of_constant_baseline=float(brier_score_loss(y, np.full(len(y), tr.logged_paid.mean()))))

    P = predict_arms(model, iso, te, rc["te"]["ctx"])
    ctx = rc["te"]["ctx"]; elig = elig_from_ctx(ctx)
    budget = te.customer_tier.map(TIER_BUDGET).values
    seg = SegmentTable(tr, rc["tr"]["ctx"])
    amt = te.amount.values
    nat = ote.paid_nat.values.astype(bool)
    raw = ote[[f"raw_{c}" for c in CHANNELS]].values
    thin = ote[[f"thin_{c}" for c in CHANNELS]].values
    cost_vec = total_cost_vector()

    pols = all_policies(ctx, elig, budget, te, P, seg, cost_vec, MARGIN)
    # ceiling for ANY probability-based policy: knows the true success probabilities (not the random outcomes)
    p_nat_t = ote.p_nat.values; p_raw_t = ote[[f"p_{c}" for c in CHANNELS]].values
    ev_true = MARGIN * amt[:, None] * ((1 - p_nat_t)[:, None] * p_raw_t) - cost_vec[None, :]
    pols["True-probability policy (knows p, not outcomes)"] = orders_from_ev(ev_true, p_raw_t > 0, budget)
    res = {k: run_orders(v, amt, nat, raw, thin, cost_vec, MARGIN) for k, v in pols.items()}
    res["Oracle ceiling (hindsight, single-shot)"] = oracle_ceiling(amt, nat, raw, cost_vec, MARGIN)
    table = pd.DataFrame([summarise(k, v, amt, nat, MARGIN) for k, v in res.items()])
    table["human_call_pct"] = [100 * (pols[k] == CH["HUMAN_CALL"]).any(1).mean() if k in pols else np.nan
                               for k in res]
    table["seed"] = seed
    # ---------- ablation: what is the root-cause (diagnosis) stage worth?
    def variant(ctx_tr, ctx_va, ctx_te):
        m2, i2 = fit_recovery_model(tr, va, ctx_tr, ctx_va)
        P2 = predict_arms(m2, i2, te, ctx_te)
        o2 = orders_from_ev(MARGIN * amt[:, None] * (P2[:, :8] - P2[:, [8]]) - cost_vec[None, :],
                            elig_from_ctx(ctx_te), budget)
        r2 = run_orders(o2, amt, nat, raw, thin, cost_vec, MARGIN)
        return 1000 * (r2["net"].sum() - (MARGIN * amt * nat).sum()) / len(amt), r2["rec"].mean()
    unk = lambda d: np.full(len(d), "UNKNOWN", dtype=object)
    abl = []
    for name, c3 in (("No diagnosis (context = UNKNOWN)", (unk(tr), unk(va), unk(te))),
                     ("Predicted + confidence-gated (the pipeline)", (rc["tr"]["ctx"], rc["va"]["ctx"], rc["te"]["ctx"])),
                     ("Perfect diagnosis (true cause)", (tr.true_error_code.values, va.true_error_code.values,
                                                        te.true_error_code.values))):
        v, rr = variant(*c3)
        abl.append(dict(variant=name, incr_net_per_1k=v, recovery_rate=rr, seed=seed))
    out = dict(table=table, diag=diag, ablation=pd.DataFrame(abl), secs=round(time.time() - t0, 1))
    if not detailed:
        return out

    # ---------- paired bootstrap CIs (per 1,000 failed payments)
    rng = rng_boot or np.random.default_rng(0)
    pred = res["Predictive EV agent (cascade)"]["net"]
    boots = []
    n = len(amt)
    for other in ["Rules ladder (retry>email>SMS>WA)", "Value-weighted rules", "Reason playbook (cost-blind)",
                  "Segment lookup (no ML, EV-gated)", "Predictive EV agent (single-shot)"]:
        d = pred - res[other]["net"]
        bm = np.array([d[rng.integers(0, n, n)].mean() for _ in range(1000)]) * 1000
        boots.append(dict(comparison=f"Predictive cascade minus {other}", mean_diff_per_1k=d.mean() * 1000,
                          ci95_lo=np.percentile(bm, 2.5), ci95_hi=np.percentile(bm, 97.5),
                          significant=bool(np.percentile(bm, 2.5) > 0 or np.percentile(bm, 97.5) < 0)))
    out["bootstrap"] = pd.DataFrame(boots)

    # ---------- sensitivity grid: does the ranking survive different economics?
    rows = []
    for margin in (0.10, 0.30, 0.50):
        for human in (20.0, 45.0, 90.0):
            for mm in (1.0, 5.0):
                cv = total_cost_vector(msg_mult=mm, human_cost=human)
                pl = all_policies(ctx, elig, budget, te, P, seg, cv, margin)
                rr = {k: run_orders(v, amt, nat, raw, thin, cv, margin) for k, v in pl.items()}
                base_net = margin * amt * nat
                inc = {k: 1000 * (v["net"].sum() - base_net.sum()) / n for k, v in rr.items()}
                best_base = max(v for k, v in inc.items() if not k.startswith("Predictive") and k != "No action")
                bb_name = [k for k, v in inc.items() if v == best_base][0]
                rows.append(dict(margin=margin, human_call_cost=human, messaging_cost_x=mm,
                                 predictive_incr_per_1k=inc["Predictive EV agent (cascade)"],
                                 best_baseline=bb_name, best_baseline_incr_per_1k=best_base,
                                 predictive_minus_best_baseline=inc["Predictive EV agent (cascade)"] - best_base,
                                 predictive_wins=inc["Predictive EV agent (cascade)"] > best_base))
    out["sensitivity"] = pd.DataFrame(rows)

    # ---------- off-policy evaluation check (IPS) of the single-shot predictive policy
    # Can the logged data ALONE estimate a policy's value? Compare with simulator ground truth.
    pi = np.where(pols["Predictive EV agent (single-shot)"][:, 0] >= 0, pols["Predictive EV agent (single-shot)"][:, 0], 8)
    arm_idx = te.logged_channel.map({a: i for i, a in enumerate(ARMS)}).values
    cost_ext = np.append(cost_vec, 0.0)
    r_logged = MARGIN * amt * te.logged_paid.values - cost_ext[arm_idx]
    w = (arm_idx == pi) * B
    ips_terms = w * r_logged
    ips = ips_terms.mean() * 1000
    noact_terms = (arm_idx == 8) * B * (MARGIN * amt * te.logged_paid.values)
    truth = res["Predictive EV agent (single-shot)"]["net"].mean() * 1000
    truth_noact = (MARGIN * amt * nat).mean() * 1000
    bs = lambda t: np.percentile([t[rng.integers(0, n, n)].mean() * 1000 for _ in range(500)], [2.5, 97.5])
    lo, hi = bs(ips_terms); lo2, hi2 = bs(noact_terms)
    out["ope"] = pd.DataFrame([
        dict(policy="Predictive single-shot", ips_estimate_net_per_1k=ips, ips_ci95_lo=lo, ips_ci95_hi=hi,
             simulator_truth_net_per_1k=truth, truth_inside_ci=bool(lo <= truth <= hi)),
        dict(policy="No action", ips_estimate_net_per_1k=noact_terms.mean() * 1000, ips_ci95_lo=lo2, ips_ci95_hi=hi2,
             simulator_truth_net_per_1k=truth_noact, truth_inside_ci=bool(lo2 <= truth_noact <= hi2))])

    # ---------- where does the agent send people? (sanity check, per failure cause)
    first = pols["Predictive EV agent (cascade)"][:, 0]
    mix = pd.crosstab(te.true_error_code, np.where(first >= 0, np.array(CHANNELS, dtype=object)[np.maximum(first, 0)],
                                                  "NOT_PURSUED"), normalize="index").round(3)
    out["first_action_mix"] = mix
    out["gate_pursued_pct"] = float(100 * (first >= 0).mean())

    # ---------- v2 data realism snapshot (for the audit table)
    obs_all = obs
    hr = obs_all.timestamp.dt.hour.value_counts(normalize=True)
    out["realism"] = dict(
        upi_share=float((obs_all.payment_method == "UPI").mean()),
        expired_card_on_upi_pct=float(100 * ((obs_all.payment_method == "UPI") & (obs_all.true_error_code == "ERR_EXPIRED_CARD")).mean()
                                      / max((obs_all.payment_method == "UPI").mean(), 1e-9)),
        hour_peak_to_trough=float(hr.max() / hr.min()),
        natural_recovery_rate=float(orc.paid_nat.mean()),
        ceiling_any_channel_recovery=float((orc.paid_nat.astype(bool) |
                                            (orc[[f"raw_{c}" for c in CHANNELS]].sum(1) > 0)).mean()),
        mean_cross_channel_corr=float(np.corrcoef(orc[[f"raw_{c}" for c in
                                                       ["EMAIL", "SMS", "WHATSAPP", "ALT_METHOD", "VOICE_IVR", "HUMAN_CALL"]]].values.T.astype(float))
                                      [np.triu_indices(6, 1)].mean()))
    return out


# ============================================================ driver
def md(df, floatfmt=2):
    d = df.copy()
    for c in d.columns:
        if d[c].dtype.kind == "f":
            d[c] = d[c].map(lambda x: f"{x:,.{floatfmt}f}")
    head = "| " + " | ".join(map(str, d.columns)) + " |\n|" + "|".join(["---"] * len(d.columns)) + "|\n"
    return head + "\n".join("| " + " | ".join(map(str, r)) + " |" for r in d.values)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--seeds", type=int, nargs="+", default=[2026, 1, 2, 3, 4])
    ap.add_argument("--out", default="results")
    a = ap.parse_args()
    os.makedirs(a.out, exist_ok=True)
    runs = []
    for i, s in enumerate(a.seeds):
        r = run_one(s, detailed=(i == 0))
        runs.append(r); print(f"seed {s} done in {r['secs']}s", flush=True)
    prim = runs[0]
    prim["table"].drop(columns="seed").to_csv(f"{a.out}/policy_comparison_primary_seed.csv", index=False)
    allt = pd.concat([r["table"] for r in runs])
    allt.to_csv(f"{a.out}/policy_all_seeds.csv", index=False)
    agg = allt.groupby("policy", sort=False).agg(
        recovery_rate_mean=("recovery_rate", "mean"), recovery_rate_sd=("recovery_rate", "std"),
        incr_net_per_1k_mean=("incr_net_per_1k", "mean"), incr_net_per_1k_sd=("incr_net_per_1k", "std"),
        cost_per_1k_mean=("cost_per_1k", "mean"), attempts_per_1k_mean=("attempts_per_1k", "mean"),
        human_call_pct_mean=("human_call_pct", "mean")).reset_index()
    agg.to_csv(f"{a.out}/policy_across_seeds.csv", index=False)
    pred = allt[allt.policy == "Predictive EV agent (cascade)"].set_index("seed").incr_net_per_1k
    wins = {p: int((pred > allt[allt.policy == p].set_index("seed").incr_net_per_1k).sum())
            for p in allt.policy.unique() if not p.startswith(("Predictive", "Oracle", "True")) and p != "No action"}
    pd.concat([r["ablation"] for r in runs]).groupby("variant", sort=False).agg(
        incr_net_per_1k_mean=("incr_net_per_1k", "mean"), incr_net_per_1k_sd=("incr_net_per_1k", "std"),
        recovery_rate_mean=("recovery_rate", "mean")).reset_index().to_csv(f"{a.out}/ablation_root_cause.csv", index=False)
    prim["bootstrap"].to_csv(f"{a.out}/bootstrap_ci.csv", index=False)
    prim["sensitivity"].to_csv(f"{a.out}/sensitivity_grid.csv", index=False)
    prim["ope"].to_csv(f"{a.out}/ope_check.csv", index=False)
    prim["first_action_mix"].to_csv(f"{a.out}/first_action_by_failure_cause.csv")
    diags = pd.DataFrame([r["diag"] for r in runs]); diags.to_csv(f"{a.out}/model_diagnostics.csv", index=False)
    json.dump(dict(realism=prim["realism"], predictive_beats_in_n_of_seeds=wins, n_seeds=len(a.seeds),
                   gate_pursued_pct=prim["gate_pursued_pct"]), open(f"{a.out}/summary.json", "w"), indent=2)
    print("\nPRIMARY SEED\n", prim["table"].round(2).to_string(index=False))
    print("\nACROSS SEEDS\n", agg.round(2).to_string(index=False))
    print("\nWINS (predictive > baseline) by seed:", wins)
    print("\nBOOTSTRAP\n", prim["bootstrap"].round(1).to_string(index=False))
    print("\nOPE\n", prim["ope"].round(1).to_string(index=False))
    print("\nDIAG\n", diags.round(3).T.to_string())
    print("\nSENS wins:", int(prim["sensitivity"].predictive_wins.sum()), "/", len(prim["sensitivity"]))
