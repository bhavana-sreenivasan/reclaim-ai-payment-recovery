"""
ReClaim v2 -- semi-synthetic failed-payment generator.

WHAT THIS IS (and is not)
-------------------------
Still simulated data. No public dataset records "we nudged this failed payment via
channel X and it recovered". v2 makes the simulation *structurally realistic* and
*scientifically honest*:

  1. Realistic structure: payment-method mix, method-dependent failure taxonomy
     (no expired-card failures on UPI), bank reliability + outage windows, hour-of-day /
     weekday / salary-cycle patterns, subscription billing batches, repeat-failing customers.
  2. Correlated outcomes: a shared latent "stubbornness" per transaction means a customer who
     ignores SMS usually also ignores WhatsApp (v1 drew every channel independently, so any
     multi-attempt cascade almost always found a success).
  3. Monotone, incremental effects: recovered = naturally_recovered OR intervention_succeeds.
     An intervention can only add recoveries, and uplift is measurable against a no-action arm.
  4. Realistic data access: the observed log contains ONLY the outcome of the action actually
     taken (uniform-random exploration incl. a no-action control arm). The all-actions
     "potential outcomes" live in a separate ORACLE file used only for evaluation.
  5. Real time structure: Jan-Jun 2026, with drift in the later months (one bank's reliability
     degrades, UPI share rises) so a time-based split actually means something.
  6. Costs that make the economics bind (voice / human-call channels, contact-fatigue penalty).

Every tunable number is either (a) SOURCED (see comments tagged [SRC]) or (b) an explicit
ASSUMPTION (tagged [ASSUME]). Nothing here is measured from Razorpay or any bank.
Bank names are anonymised on purpose: failure behaviour here is invented and must not be
read as a claim about any real bank.
"""
import numpy as np
import pandas as pd
from scipy.special import expit, logit
from scipy.stats import norm

# --------------------------------------------------------------------------- constants
CHANNELS = ["RETRY_NOW", "RETRY_LATER", "EMAIL", "SMS", "WHATSAPP", "ALT_METHOD", "VOICE_IVR", "HUMAN_CALL"]
NONE = "NONE"                       # no-action control arm
ARMS = CHANNELS + [NONE]            # logging policy randomises uniformly over these 9 arms
ERRS = ["ERR_INSUFFICIENT_FUNDS", "ERR_BANK_TIMEOUT", "ERR_USER_ABORT", "ERR_EXPIRED_CARD",
        "ERR_GATEWAY_TIMEOUT", "ERR_AUTH_FAILED", "ERR_LIMIT_EXCEEDED", "ERR_ISSUER_DECLINE"]
METHODS = ["UPI", "Card", "NetBanking", "Subscription"]
BANKS = [f"BANK_{c}" for c in "ABCDEFGH"]
GATEWAYS = ["GW_1", "GW_2", "GW_3"]
DEVICES = ["Android", "iOS", "Web"]
NETWORKS = ["4G", "5G", "WiFi"]
TIERS = ["High", "Medium", "Low"]
TIER_BUDGET = {"High": 3, "Medium": 2, "Low": 1}   # max attempts per failed payment (kept from v1)

# Per-attempt direct cost in INR.
#  [SRC] WhatsApp utility-category template (payment reminder) is ~Rs0.115/msg from Meta + BSP
#        markup (~Rs0.10-0.30) + 18% GST  =>  ~Rs0.40.  Marketing-category would be ~Rs0.86+GST.
#  [ASSUME] SMS (DLT), email, IVR, human-call costs.
DIRECT_COST = dict(RETRY_NOW=0.0, RETRY_LATER=0.0, EMAIL=0.05, SMS=0.20, WHATSAPP=0.40,
                   ALT_METHOD=0.40, VOICE_IVR=2.50, HUMAN_CALL=45.0)
# [ASSUME] per-contact "fatigue" penalty (opt-outs / annoyance / lost goodwill), INR.
FATIGUE = dict(RETRY_NOW=0.0, RETRY_LATER=0.0, EMAIL=0.25, SMS=0.50, WHATSAPP=0.50,
               ALT_METHOD=0.50, VOICE_IVR=2.0, HUMAN_CALL=5.0)
MESSAGING = ["EMAIL", "SMS", "WHATSAPP", "ALT_METHOD"]


def total_cost_vector(msg_mult=1.0, human_cost=None):
    """Total per-attempt cost (direct + fatigue) as an array aligned with CHANNELS."""
    c = {k: DIRECT_COST[k] + FATIGUE[k] for k in CHANNELS}
    for k in MESSAGING:
        c[k] *= msg_mult
    if human_cost is not None:
        c["HUMAN_CALL"] = human_cost + FATIGUE["HUMAN_CALL"]
    return np.array([c[k] for k in CHANNELS])


# Intervention-induced success probability given NOT naturally recovered.  NaN = channel is
# structurally ineligible for that failure cause (e.g. immediate retry of an expired card).
# [ASSUME] magnitudes. Email row is anchored to the ~11-15% single-dunning-email recovery
# figures cited in industry write-ups [SRC: secondary industry blogs, not peer-reviewed].
nan = np.nan
#            RETRY_NOW RETRY_LATER EMAIL  SMS   WHATSAPP ALT   IVR   HUMAN
BASE_P = np.array([
    [nan,  0.30, 0.08, 0.10, 0.12, 0.22, 0.12, 0.25],   # INSUFFICIENT_FUNDS
    [0.50, 0.40, 0.06, 0.08, 0.10, 0.18, 0.08, 0.20],   # BANK_TIMEOUT
    [nan,  nan,  0.10, 0.12, 0.20, 0.18, 0.10, 0.28],   # USER_ABORT
    [nan,  nan,  0.14, 0.14, 0.18, 0.30, 0.12, 0.35],   # EXPIRED_CARD
    [0.60, 0.45, 0.05, 0.06, 0.08, 0.15, 0.06, 0.15],   # GATEWAY_TIMEOUT
    [nan,  nan,  0.12, 0.14, 0.20, 0.28, 0.15, 0.32],   # AUTH_FAILED (needs the customer)
    [nan,  0.35, 0.10, 0.12, 0.15, 0.30, 0.12, 0.28],   # LIMIT_EXCEEDED (limit resets)
    [0.15, 0.25, 0.08, 0.10, 0.12, 0.25, 0.10, 0.25],   # ISSUER_DECLINE (soft decline)
])
ELIGIBLE = ~np.isnan(BASE_P)                      # (8 errors, 8 channels)
# What a policy may do when the root cause is UNKNOWN: only "safe" customer-facing channels.
SAFE_CHANNELS = np.array([c in ("EMAIL", "SMS", "WHATSAPP", "ALT_METHOD", "VOICE_IVR", "HUMAN_CALL")
                          for c in CHANNELS])
# Natural (no-action) recovery probability by failure cause. [ASSUME]
NAT_BASE = np.array([0.08, 0.30, 0.12, 0.03, 0.30, 0.20, 0.12, 0.08])

# Failure-cause mix by payment method (rows sum to 1). [ASSUME] -- structure, not measurements:
# UPI/NetBanking cannot fail with an expired card; subscriptions are non-interactive (no abort).
#                INSUF  BANK_TO ABORT EXPIRED GW_TO  AUTH   LIMIT  ISSUER
ERR_MIX = np.array([
    [0.34, 0.22, 0.14, 0.00, 0.04, 0.14, 0.08, 0.04],   # UPI
    [0.18, 0.10, 0.14, 0.08, 0.03, 0.28, 0.05, 0.14],   # Card
    [0.20, 0.30, 0.25, 0.00, 0.06, 0.15, 0.04, 0.00],   # NetBanking
    [0.45, 0.07, 0.00, 0.17, 0.03, 0.08, 0.08, 0.12],   # Subscription (auto-debit)
])

# Failed-payment mix by method. [SRC] UPI is ~80% of retail digital-payment *volume* in India
# (NPCI/RBI data via press). Failures over-index on cards/netbanking (more auth friction), so the
# mix among FAILURES is less UPI-heavy. [ASSUME] the exact split; it is a merchant-mix parameter.
METHOD_MIX_START = np.array([0.56, 0.24, 0.09, 0.11])
METHOD_MIX_END = np.array([0.61, 0.21, 0.07, 0.11])       # drift: UPI share rises over the period

# Hour-of-day demand shape [ASSUME]: morning ramp, midday plateau, evening peak (~8-10pm).
HOUR_W = np.array([1.0, .7, .5, .4, .4, .6, 1.0, 1.8, 2.6, 3.2, 3.6, 3.8, 3.6, 3.4, 3.2, 3.2,
                   3.4, 3.8, 4.2, 4.8, 5.2, 5.0, 3.8, 2.0])
SUB_HOUR_W = np.full(24, 0.02); SUB_HOUR_W[0:4] = 0.20; SUB_HOUR_W[9:11] = 0.10   # billing batches
SUB_DOM = {1: .35, 5: .20, 10: .15, 15: .15, 20: .08, 25: .07}                    # billing days
WEEKDAY_W = np.array([1.0, 1.0, 1.0, 1.0, 1.1, 1.15, 1.05])                        # Mon..Sun [ASSUME]

# Messages: ~62% of failures carry a specific message, the rest a generic one shared across causes
# (real gateways often return vague text). This is what makes root-cause inference non-trivial.
SPECIFIC_MSG = {
    "ERR_INSUFFICIENT_FUNDS": ["Insufficient funds in account", "Your account balance is low for this transaction",
                               "Debit failed: balance not sufficient"],
    "ERR_BANK_TIMEOUT": ["Bank server did not respond in time", "Issuer timed out while processing",
                         "Bank is not responding, try later"],
    "ERR_USER_ABORT": ["Customer cancelled the payment", "Payment window closed before completion",
                       "Collect request expired without approval"],
    "ERR_EXPIRED_CARD": ["Card has expired", "Card expiry date invalid", "Declined: expired card"],
    "ERR_GATEWAY_TIMEOUT": ["Gateway timeout while contacting acquirer", "Payment gateway took too long to respond",
                            "Upstream service timeout"],
    "ERR_AUTH_FAILED": ["Authentication failed", "Incorrect OTP or PIN entered", "3-D Secure verification failed"],
    "ERR_LIMIT_EXCEEDED": ["Transaction limit exceeded", "Daily limit reached for this account",
                           "Amount exceeds permitted limit"],
    "ERR_ISSUER_DECLINE": ["Do not honor", "Declined by issuing bank", "Transaction not permitted for this card"],
}
GENERIC_MSG = ["Transaction failed", "Payment failed. Please try again", "Transaction could not be completed",
               "Declined by bank"]


def _window_count(keys, window, block):
    """# of earlier items with the same group in [key-window, key). `keys` = group*block + t, sorted."""
    lo = np.maximum(keys - window, (keys // block) * block)
    return np.searchsorted(keys, keys, side="left") - np.searchsorted(keys, lo, side="left")


def generate(seed=2026, n_txn=100_000, n_cust=30_000, start="2026-01-01", n_days=181):
    rng = np.random.default_rng(seed)
    start_ts = pd.Timestamp(start)
    start_wd = start_ts.weekday()
    day_idx = np.arange(n_days)
    dom_of_day = np.array([(start_ts + pd.Timedelta(days=int(d))).day for d in day_idx])
    wd_of_day = (day_idx + start_wd) % 7

    # ---------------------------------------------------------------- customers (latent traits)
    tier_c = rng.choice(3, size=n_cust, p=[0.10, 0.40, 0.50])               # High/Med/Low [ASSUME]
    fin_health = rng.normal(size=n_cust)                                    # latent
    responsiveness = rng.normal(size=n_cust)                                # latent
    bank_share = np.array([.26, .20, .15, .12, .09, .07, .06, .05])         # [ASSUME]
    bank_c = rng.choice(8, size=n_cust, p=bank_share / bank_share.sum())
    # [ASSUME] iOS is a small share of Indian smartphones; "Web" = desktop/mobile-web checkout.
    dev_c = rng.choice(3, size=n_cust, p=[0.66, 0.07, 0.27])
    net_c = rng.choice(3, size=n_cust, p=[0.55, 0.30, 0.15])
    pref_method_c = rng.choice(4, size=n_cust, p=METHOD_MIX_START)
    tenure_c = np.clip(rng.gamma(2.0, 180, size=n_cust), 1, 2500).astype(int)
    hist_cnt_c = np.clip(rng.poisson(40, size=n_cust) + 2, 2, 200)
    hist_pay_rate_c = expit(1.6 + 0.6 * fin_health + 0.4 * rng.normal(size=n_cust))
    hist_resp_rate_c = expit(-1.6 + 0.7 * responsiveness + 0.4 * rng.normal(size=n_cust))
    # customers with weaker finances (and heavy users) fail more often -> repeat failers
    fail_w = np.exp(-0.4 * fin_health + 0.5 * rng.normal(size=n_cust))
    cust = rng.choice(n_cust, size=n_txn, p=fail_w / fail_w.sum())

    # ---------------------------------------------------------------- method (with drift)
    # draw time position first for drift, then method
    pos = rng.random(n_txn)                                                 # 0..1 through the period
    mix_t = METHOD_MIX_START[None, :] * (1 - pos[:, None]) + METHOD_MIX_END[None, :] * pos[:, None]
    cum = mix_t.cumsum(1)
    method_global = (rng.random(n_txn)[:, None] > cum).sum(1).clip(0, 3)
    use_pref = rng.random(n_txn) < 0.80
    method = np.where(use_pref, pref_method_c[cust], method_global)

    # ---------------------------------------------------------------- timestamps
    growth = 1 + 0.15 * day_idx / n_days
    dom_boost = np.where(np.isin(dom_of_day, [1, 2, 3, 28, 29, 30, 31]), 1.15, 1.0)
    w_day = WEEKDAY_W[wd_of_day] * growth * dom_boost
    w_day = w_day / w_day.sum()
    sub_w_day = np.array([SUB_DOM.get(int(d), 0.0) for d in dom_of_day]) * growth
    sub_w_day = sub_w_day / sub_w_day.sum()
    is_sub = method == 3
    day = rng.choice(n_days, size=n_txn, p=w_day)
    day[is_sub] = rng.choice(n_days, size=int(is_sub.sum()), p=sub_w_day)
    hour = rng.choice(24, size=n_txn, p=HOUR_W / HOUR_W.sum())
    hour[is_sub] = rng.choice(24, size=int(is_sub.sum()), p=SUB_HOUR_W / SUB_HOUR_W.sum())
    secs = day * 86400 + hour * 3600 + rng.integers(0, 3600, size=n_txn)

    # ---------------------------------------------------------------- bank outage windows
    bank = bank_c[cust].copy()
    flip = rng.random(n_txn) < 0.15                                          # not always primary bank
    bank[flip] = rng.choice(8, size=int(flip.sum()), p=bank_share / bank_share.sum())
    windows = {b: [] for b in range(8)}
    for b in range(8):
        n_train = 5
        n_test = 14 if b == 4 else 5       # DRIFT: BANK_E degrades in the last third of the period
        for lo, hi, n in ((0, 120, n_train), (121, n_days - 1, n_test)):
            for _ in range(n):
                s = rng.uniform(lo * 86400, hi * 86400)
                windows[b].append((s, s + rng.uniform(2, 9) * 3600))
    # [ASSUME] 7% of failures are pulled into an outage window of their own bank
    pull = np.where(rng.random(n_txn) < 0.07)[0]
    for i in pull:
        s, e = windows[bank[i]][rng.integers(len(windows[bank[i]]))]
        secs[i] = int(rng.uniform(s, e))
    secs = np.clip(secs, 0, n_days * 86400 - 1)
    in_outage = np.zeros(n_txn, bool)
    for b in range(8):
        m = bank == b
        for s, e in windows[b]:
            in_outage |= m & (secs >= s) & (secs <= e)

    order = np.argsort(secs, kind="stable")                                   # sort everything by time
    cust, method, bank, secs, in_outage = cust[order], method[order], bank[order], secs[order], in_outage[order]
    is_sub = method == 3
    ts = start_ts + pd.to_timedelta(secs, unit="s")
    hour = ts.hour.values
    dow = ts.dayofweek.values
    dom = ts.day.values

    # ---------------------------------------------------------------- amount, device, network
    amount = np.empty(n_txn)
    p0, p1, p2 = (method == 0), (method == 1), (method == 2)
    amount[p0] = np.clip(np.exp(rng.normal(np.log(450), 0.95, p0.sum())), 10, 100_000)   # UPI cap [ASSUME ~1L]
    amount[p1] = np.clip(np.exp(rng.normal(np.log(1400), 1.05, p1.sum())), 50, 200_000)
    amount[p2] = np.clip(np.exp(rng.normal(np.log(2200), 1.10, p2.sum())), 100, 300_000)
    sub_prices = np.array([149, 199, 299, 499, 799, 1199, 2999])
    amount[is_sub] = rng.choice(sub_prices, size=int(is_sub.sum()), p=[.12, .20, .25, .20, .12, .07, .04])
    tier = tier_c[cust]
    # subscriptions keep their discrete list prices; other methods scale mildly with customer tier
    amount = np.round(amount * np.where(is_sub, 1.0, np.array([1.30, 1.0, 0.85])[tier]), 2)
    dev = np.where(rng.random(n_txn) < 0.85, dev_c[cust], rng.choice(3, size=n_txn, p=[.66, .07, .27]))
    net = np.where(rng.random(n_txn) < 0.85, net_c[cust], rng.choice(3, size=n_txn, p=[.55, .30, .15]))
    gateway = rng.choice(3, size=n_txn, p=[0.5, 0.3, 0.2])
    f, r = fin_health[cust], responsiveness[cust]

    # ---------------------------------------------------------------- failure cause
    P = ERR_MIX[method].copy()                                              # (N, 8)
    pre_salary = (dom >= 24) | (dom <= 2)                                   # [ASSUME] month-end cash crunch
    P[:, 0] *= np.exp(-0.35 * f) * np.where(pre_salary, 1.35, 1.0)
    bank_to_mult = np.array([0.8, 0.9, 1.0, 1.0, 1.1, 1.2, 1.4, 1.7])      # [ASSUME] issuer reliability varies
    P[:, 1] *= bank_to_mult[bank] * np.where(in_outage, 6.0, 1.0) * np.where(hour <= 5, 1.6, 1.0) \
        * np.array([1.1, 0.9, 1.0])[net]
    P[:, 4] *= np.array([1.0, 1.0, 1.5])[gateway]
    P[:, 2] *= np.array([1.15, 0.9, 1.0])[net] * np.where(dev == 2, 1.2, 1.0)
    P = P / P.sum(1, keepdims=True)
    err = (rng.random(n_txn)[:, None] > P.cumsum(1)).sum(1).clip(0, 7)

    # ---------------------------------------------------------------- gateway message
    msg = np.empty(n_txn, dtype=object)
    use_specific = rng.random(n_txn) < 0.62
    for k, e in enumerate(ERRS):
        m = np.where(err == k)[0]
        spec = rng.random(len(m)) < 0.62
        pool = SPECIFIC_MSG[e]
        msg[m] = np.where(spec, np.array(pool, dtype=object)[rng.integers(len(pool), size=len(m))],
                          np.array(GENERIC_MSG, dtype=object)[rng.integers(len(GENERIC_MSG), size=len(m))])

    # ---------------------------------------------------------------- real-time / history features
    t_h = np.clip(np.exp(rng.normal(np.log(2.0), 1.0, n_txn)), 0.05, 72)    # hours from failure to action
    BLK = 10 ** 8
    kc = np.sort(cust.astype(np.int64) * BLK + secs)
    cust_prior_30d = np.empty(n_txn)
    # compute in sorted-by-(cust,time) order then map back
    ordc = np.argsort(cust.astype(np.int64) * BLK + secs, kind="stable")
    cnt = _window_count(kc, 30 * 86400, BLK)
    cust_prior_30d[ordc] = cnt
    kb = np.sort(bank.astype(np.int64) * BLK + secs)
    ordb = np.argsort(bank.astype(np.int64) * BLK + secs, kind="stable")
    bank_fail_6h = np.empty(n_txn)
    bank_fail_6h[ordb] = _window_count(kb, 6 * 3600, BLK)

    # ---------------------------------------------------------------- success probabilities
    logit_base = logit(np.where(np.isnan(BASE_P), 0.5, BASE_P))[err]        # (N, 8)
    log_amt = np.log(amount / 900.0)
    amt_coef = np.array([0.35, 0.10, 0.10, 0.10, 0.10, 0.10, 0.35, 0.15])[err]
    amt_ch = np.array([0.3, 0.3, 1, 1, 1, 1, 1, 0.3])
    adj = -(amt_coef * log_amt)[:, None] * amt_ch[None, :]
    contact = np.array([0, 0, 1, 1, 1, 1, 1, 1.0])
    adj += (np.array([0.25, 0.0, -0.25])[tier])[:, None] * contact[None, :]           # valued customers engage more
    adj += r[:, None] * np.array([0, 0, 0.30, 0.30, 0.45, 0.40, 0.30, 0.25])[None, :]  # responsiveness (latent)
    adj += (np.where(np.isin(err, [0, 6]), 0.30, 0.10) * f)[:, None]                   # financial health (latent)
    dev_adj = np.array([[0, 0, -.05, .10, .05, 0, 0, 0],      # Android
                        [0, 0, .10, -.05, .15, 0, 0, 0],      # iOS
                        [0, 0, .20, 0, -.50, 0, 0, .10]])     # Web
    adj += dev_adj[dev]
    t_decay = np.array([0.25, 0.10, 0.55, 0.55, 0.55, 0.55, 0.45, 0.25])
    adj -= t_decay[None, :] * np.log1p(t_h / 4.0)[:, None]                              # act fast: success decays
    salary = np.where(dom <= 7, 0.5, np.where(dom >= 22, -0.4, 0.0)) * np.isin(err, [0, 6])
    adj += salary[:, None] * np.array([1, 0, 1, 1, 1, 1, 1, 1.0])[None, :]
    adj[:, 0] -= 1.4 * (in_outage & np.isin(err, [1, 4]))                               # retry during outage fails
    adj[:, 2] += 0.2 * is_sub; adj[:, 4] -= 0.2 * is_sub; adj[:, 1] += 0.3 * is_sub; adj[:, 7] += 0.1 * is_sub
    p_raw = expit(logit_base + adj)
    p_raw = np.where(ELIGIBLE[err], p_raw, 0.0)                                         # ineligible => cannot succeed

    nat_logit = logit(NAT_BASE[err]) + 0.30 * f + np.array([0.15, 0, -0.15])[tier] - 0.4 * is_sub
    p_nat = expit(nat_logit)

    # ---------------------------------------------------------------- joint potential outcomes
    rho, rho_n = 0.55, 0.60
    S = rng.normal(size=n_txn)                                                           # shared stubbornness
    Z = rho * S[:, None] + np.sqrt(1 - rho ** 2) * rng.normal(size=(n_txn, 8))
    raw = norm.cdf(Z) < p_raw                                                            # (N, 8) bool
    Zn = rho_n * S + np.sqrt(1 - rho_n ** 2) * rng.normal(size=n_txn)
    paid_nat = norm.cdf(Zn) < p_nat
    thin_u = rng.random((n_txn, 8))

    # ---------------------------------------------------------------- logged data: uniform exploration
    arm = rng.integers(0, 9, size=n_txn)                                                 # 0..7 channel, 8 = NONE
    raw_pad = np.concatenate([raw, np.zeros((n_txn, 1), bool)], axis=1)
    logged_paid = paid_nat | raw_pad[np.arange(n_txn), arm]
    arm_names = np.array(ARMS, dtype=object)

    ids = np.array([f"TX_{i + 1:07d}" for i in range(n_txn)])
    observed = pd.DataFrame({
        "transaction_id": ids,
        "customer_id": np.array([f"CUST_{c:06d}" for c in cust]),
        "timestamp": ts,
        "amount": amount,
        "payment_method": np.array(METHODS, dtype=object)[method],
        "bank": np.array(BANKS, dtype=object)[bank],
        "gateway": np.array(GATEWAYS, dtype=object)[gateway],
        "device_type": np.array(DEVICES, dtype=object)[dev],
        "network_type": np.array(NETWORKS, dtype=object)[net],
        "gateway_message": msg,
        "hours_since_failure": np.round(t_h, 3),
        "customer_tier": np.array(TIERS, dtype=object)[tier],
        "tenure_days": tenure_c[cust],
        "hist_txn_count": hist_cnt_c[cust],
        "hist_pay_success_rate": np.round(hist_pay_rate_c[cust], 4),
        "hist_response_rate": np.round(hist_resp_rate_c[cust], 4),
        "customer_failures_prior_30d": cust_prior_30d.astype(int),
        "bank_fail_6h": bank_fail_6h.astype(int),
        "true_error_code": np.array(ERRS, dtype=object)[err],     # reconciled label; training label ONLY
        "logged_channel": arm_names[arm],
        "logged_propensity": round(1 / 9, 6),
        "logged_paid": logged_paid.astype(int),
    })
    oracle = pd.DataFrame({"transaction_id": ids, "p_nat": p_nat.round(5), "paid_nat": paid_nat.astype(int),
                           "in_outage": in_outage.astype(int), "fin_health": f.round(4),
                           "responsiveness": r.round(4)})
    for j, c in enumerate(CHANNELS):
        oracle[f"p_{c}"] = p_raw[:, j].round(5)
        oracle[f"raw_{c}"] = raw[:, j].astype(int)
        oracle[f"thin_{c}"] = thin_u[:, j].round(5)
    return observed, oracle


if __name__ == "__main__":
    import argparse, os
    ap = argparse.ArgumentParser()
    ap.add_argument("--seed", type=int, default=2026)
    ap.add_argument("--out", default="data")
    a = ap.parse_args()
    os.makedirs(a.out, exist_ok=True)
    obs, orc = generate(a.seed)
    obs.to_csv(f"{a.out}/failed_payments_v2.csv.gz", index=False)
    orc.to_csv(f"{a.out}/oracle_potential_outcomes.csv.gz", index=False)
    print(obs.shape, orc.shape)
