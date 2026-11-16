"""Train everything from the kit. Usage: python train.py [--train_dirs annee1 annee2] [--out_dir model]"""
import argparse, json, os
import joblib
import numpy as np
import pandas as pd
import torch
from sklearn.ensemble import HistGradientBoostingRegressor
from features import top_lists, build_features
from worldmodel import TREND_COLS, load_hourly, train_mlp, train_ar

ap = argparse.ArgumentParser()
ap.add_argument("--train_dirs", nargs="+", default=["annee1", "annee2"])
ap.add_argument("--out_dir", default="model")
ap.add_argument("--n_seeds", type=int, default=3)     # networks per family (MLP, AR)
ap.add_argument("--no_gb", action="store_true")       # neural-only version
args = ap.parse_args()
os.makedirs(args.out_dir, exist_ok=True)

# Lists of frequent event kinds / command families, fixed from the first training folder
ev_kinds, cmd_fams = top_lists(args.train_dirs[0])

# Hourly power (targets) and hourly context features, for every training folder
Xs, Ys = [], []
for d in args.train_dirs:
    print("building features:", d, flush=True)
    y = load_hourly(d)
    Xs.append(build_features(d, y.index, ev_kinds, cmd_fams))
    Ys.append(y)
X_all, Y_all = pd.concat(Xs), pd.concat(Ys)
circuits = list(Y_all.columns)
all_cols = list(X_all.columns)
nn_cols = [c for c in all_cols if c not in TREND_COLS]   # nets do not see the daily trends

# Standardize features; targets: centre per circuit, ONE global scale (matches the RMSE metric)
mu = X_all[nn_cols].mean()
sd = X_all[nn_cols].std().replace(0, 1)
Xt = torch.tensor(((X_all[nn_cols] - mu) / sd).values.astype("float32"))
ymean = Y_all.mean()
scale = float((Y_all - ymean).values.std())
Yt = torch.tensor(((Y_all - ymean) / scale).values.astype("float32"))
lo, hi = Yt.min(0).values, Yt.max(0).values              # clamp range for the dream

# Gradient boosting: one model per circuit, all features (component of the blend)
if not args.no_gb:
    print("training gradient boosting", flush=True)
    gb = {}
    for c in circuits:
        m = HistGradientBoostingRegressor(max_iter=200, learning_rate=0.05, random_state=0)
        gb[c] = m.fit(X_all[all_cols], Y_all[c])
    joblib.dump(gb, f"{args.out_dir}/gb.joblib")

# Neural networks: several seeds each
print("training MLPs", flush=True)
mlps = [train_mlp(Xt, Yt, seed=s) for s in range(args.n_seeds)]
print("training autoregressive world models", flush=True)
ars = [train_ar(Xt, Yt, lo, hi, seed=s) for s in range(args.n_seeds)]
torch.save({"mlp": [m.state_dict() for m in mlps],
            "ar": [m.state_dict() for m in ars]}, f"{args.out_dir}/nets.pth")

meta = dict(circuits=circuits, all_cols=all_cols, nn_cols=nn_cols,
            ev_kinds=list(ev_kinds), cmd_fams=list(cmd_fams),
            mu=mu.tolist(), sd=sd.tolist(), ymean=ymean.tolist(), scale=scale,
            lo=lo.tolist(), hi=hi.tolist(), n_in=len(nn_cols),
            has_gb=not args.no_gb, train_dirs=args.train_dirs)
json.dump(meta, open(f"{args.out_dir}/meta.json", "w"))
print("done. saved in", args.out_dir)
