import numpy as np
import pandas as pd
import torch
import torch.nn as nn

# Slow-moving daily columns that act like a calendar: dropped from the neural nets
TREND_COLS = ["sunmars_km", "earthmars_km", "sunmarsearthangle_deg",
              "solarconstantmars", "eclipseduration_min", "occultationduration_min"]


def load_hourly(data_dir):
    """power.csv -> one row per hour, empty hours filled by interpolation."""
    p = pd.read_csv(f"{data_dir}/power.csv")
    p["time"] = pd.to_datetime(p["ut_ms"], unit="ms")
    h = p.set_index("time").drop(columns="ut_ms").resample("1h").mean()
    return h.interpolate(method="time").bfill().ffill()


def make_mlp(n_in, n_out, hidden=64):
    """Network without memory: context of one hour -> 33 circuits."""
    return nn.Sequential(nn.Linear(n_in, hidden), nn.ReLU(), nn.Dropout(0.2),
                         nn.Linear(hidden, hidden), nn.ReLU(),
                         nn.Linear(hidden, n_out))


class StepNet(nn.Module):
    """World model step: (context of this hour, previous hour's power) -> this hour's power."""
    def __init__(self, n_ctx, n_out, hidden=64):
        super().__init__()
        self.net = nn.Sequential(nn.Linear(n_ctx + n_out, hidden), nn.ReLU(), nn.Dropout(0.1),
                                 nn.Linear(hidden, hidden), nn.ReLU(),
                                 nn.Linear(hidden, n_out))

    def forward(self, ctx, prev):
        return self.net(torch.cat([ctx, prev], dim=-1))


def train_mlp(Xt, Yt, seed, steps=1500, hidden=64, wd=1e-3):
    torch.manual_seed(seed); np.random.seed(seed)
    m = make_mlp(Xt.shape[1], Yt.shape[1], hidden)
    opt = torch.optim.Adam(m.parameters(), lr=2e-3, weight_decay=wd)
    for _ in range(steps):
        m.train()
        idx = torch.randint(0, len(Xt), (512,))
        loss = ((m(Xt[idx]) - Yt[idx]) ** 2).mean()
        opt.zero_grad(); loss.backward(); opt.step()
    return m


def train_ar(Xt, Yt, lo, hi, seed, steps=1200, K=48, B=64, hidden=64, wd=1e-3):
    """Autoregressive training with scheduled sampling (own predictions fed back, 0% -> 100%)."""
    torch.manual_seed(seed); np.random.seed(seed)
    m = StepNet(Xt.shape[1], Yt.shape[1], hidden)
    opt = torch.optim.Adam(m.parameters(), lr=2e-3, weight_decay=wd)
    for step in range(steps):
        m.train()
        p_own = min(1.0, step / (0.7 * steps))
        s = torch.tensor(np.random.randint(1, len(Xt) - K, size=B))
        prev = Yt[s - 1]
        loss = 0
        for k in range(K):
            out = m(Xt[s + k], prev)
            loss = loss + ((out - Yt[s + k]) ** 2).mean()
            own = torch.maximum(torch.minimum(out.detach(), hi), lo)   # clamp to training range
            use_own = (torch.rand(B, 1) < p_own).float()
            prev = use_own * own + (1 - use_own) * Yt[s + k]
        loss = loss / K
        opt.zero_grad(); loss.backward(); opt.step()
        if (step + 1) % 300 == 0:
            print(f"  AR seed {seed} step {step + 1}: loss {loss.item():.4f}", flush=True)
    return m


def simulate(m, Xt, s0, lo, hi):
    """Free-running dream: every hour uses the model's OWN previous prediction."""
    m.eval(); prev = torch.tensor(s0)[None]; out = []
    with torch.no_grad():
        for t in range(len(Xt)):
            y = torch.maximum(torch.minimum(m(Xt[t][None], prev), hi), lo)
            out.append(y[0].numpy()); prev = y
    return np.vstack(out)
