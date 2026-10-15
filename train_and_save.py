"""Train ReClaim v2 once (seed 2026, same time split as the paper results) and save artifacts for the app.
Usage:  python train_and_save.py        -> writes ./artifacts/
"""
import json, sys
from pathlib import Path
import joblib, numpy as np
from sklearn.metrics import roc_auc_score, brier_score_loss

import run_pipeline as rp
from generate_data import generate

SEED = 2026
OUT = Path("artifacts"); OUT.mkdir(exist_ok=True)

obs, orc = generate(SEED)
ts = obs.timestamp
m_tr, m_va, m_te = ts < "2026-04-01", (ts >= "2026-04-01") & (ts < "2026-05-01"), ts >= "2026-05-01"
tr, va, te = [obs[m].reset_index(drop=True) for m in (m_tr, m_va, m_te)]

rc, rc_diag = rp.fit_root_cause(tr, va, te)
model, iso = rp.fit_recovery_model(tr, va, rc["tr"]["ctx"], rc["va"]["ctx"])

# held-out metrics (logged rows only, as in the pipeline)
X = rp.feats(te, rc["te"]["ctx"], te.logged_channel.values)
p = iso.transform(model.predict_proba(X)[:, 1]); y = te.logged_paid.values
meta = dict(seed=SEED, **rc_diag, test_auc=float(roc_auc_score(y, p)),
            test_brier=float(brier_score_loss(y, p)),
            test_brier_constant=float(brier_score_loss(y, np.full(len(y), tr.logged_paid.mean()))))

joblib.dump(dict(model=rc["model"], tau=rc["tau"]), OUT / "root_cause.joblib")
joblib.dump(dict(model=model, iso=iso), OUT / "recovery_model.joblib")
json.dump(meta, open(OUT / "meta.json", "w"), indent=2)
print(json.dumps(meta, indent=2))
