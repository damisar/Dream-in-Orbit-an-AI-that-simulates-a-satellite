"""Usage: python predict.py --test_dir DOSSIER --out predictions.csv [--model_dir model]"""
import argparse, json
import joblib
import numpy as np
import pandas as pd
import torch
from features import build_features
from worldmodel import make_mlp, StepNet, simulate

ap = argparse.ArgumentParser()
ap.add_argument("--test_dir", required=True)
ap.add_argument("--out", required=True)
ap.add_argument("--model_dir", default="model")
ap.add_argument("--members_out", default=None)   # optional: save every member (for the demo)
args = ap.parse_args()

meta = json.load(open(f"{args.model_dir}/meta.json"))
circuits, nn_cols, all_cols = meta["circuits"], meta["nn_cols"], meta["all_cols"]
mu = pd.Series(meta["mu"], index=nn_cols)
sd = pd.Series(meta["sd"], index=nn_cols)
ymean = pd.Series(meta["ymean"], index=circuits)
scale = meta["scale"]
lo, hi = torch.tensor(meta["lo"]), torch.tensor(meta["hi"])
nets = torch.load(f"{args.model_dir}/nets.pth")

# Hours to predict, and the continuous hourly grid covering them
heures = pd.read_csv(f"{args.test_dir}/heures_a_predire.csv")
wanted = pd.DatetimeIndex(pd.to_datetime(heures["ut_ms"], unit="ms"))
grid = pd.date_range(wanted.min(), wanted.max(), freq="1h")

# Context features, built only from the files of the test folder
X = build_features(args.test_dir, grid, meta["ev_kinds"], meta["cmd_fams"])
Xt = torch.tensor(np.nan_to_num(((X[nn_cols] - mu) / sd).values.astype("float32")))

# Initial state: the last real hour of power_init.csv
pi = pd.read_csv(f"{args.test_dir}/power_init.csv")
pi["time"] = pd.to_datetime(pi["ut_ms"], unit="ms")
pi_h = pi.set_index("time")[circuits].resample("1h").mean().ffill()
s0 = np.nan_to_num(((pi_h.iloc[-1] - ymean) / scale).values.astype("float32"))

unscale = lambda a: a * scale + ymean.values

# Members: MLPs (no memory), autoregressive world models (the "dream"), gradient boosting
mlp_all, ar_all = [], []
for state in nets["mlp"]:
    m = make_mlp(meta["n_in"], len(circuits)); m.load_state_dict(state); m.eval()
    with torch.no_grad():
        mlp_all.append(unscale(m(Xt).numpy()))
for state in nets["ar"]:
    m = StepNet(meta["n_in"], len(circuits)); m.load_state_dict(state)
    ar_all.append(unscale(simulate(m, Xt, s0, lo, hi)))
members = [np.mean(mlp_all, axis=0), np.mean(ar_all, axis=0)]
if meta["has_gb"]:
    gb = joblib.load(f"{args.model_dir}/gb.joblib")
    gb_pred = np.column_stack([gb[c].predict(X[all_cols]) for c in circuits])
    members.append(gb_pred)

blend = np.mean(members, axis=0)       # equal-weight vote

pred = pd.DataFrame(blend, index=grid, columns=circuits).reindex(wanted)
pred = pred.fillna(ymean)              # safety: never leave a missing value
out = pred.reset_index(drop=True)
out.insert(0, "ut_ms", heures["ut_ms"].values)
out.to_csv(args.out, index=False)
print("saved", args.out, out.shape, "missing:", int(out.isna().sum().sum()))

if args.members_out:
    np.savez(args.members_out, hours=grid.strftime("%Y-%m-%d %H:%M:%S").values,
             mlp=np.array(mlp_all), ar=np.array(ar_all),
             gb=gb_pred if meta["has_gb"] else np.zeros(1))
