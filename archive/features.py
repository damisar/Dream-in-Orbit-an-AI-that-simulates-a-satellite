import numpy as np
import pandas as pd

H = 3_600_000  # one hour in milliseconds
FTL_TYPES = ["SLEW", "EARTH", "INERTIAL", "NADIR", "WARMUP",
             "MAINTENANCE", "RADIO_SCIENCE", "ACROSS_TRACK"]


def event_kind(desc):
    return desc.str.split(r"[_ ]").str[:2].str.join("_")


def top_lists(data_dir, n_ev=15, n_cmd=12):
    """Most frequent event kinds and command families (computed on training data only)."""
    evtf = pd.read_csv(f"{data_dir}/evtf.csv")
    dmop = pd.read_csv(f"{data_dir}/dmop.csv")
    kinds = event_kind(evtf["description"]).value_counts().head(n_ev).index.tolist()
    fams = dmop["subsystem"].str[:4].value_counts().head(n_cmd).index.tolist()
    return kinds, fams


def overlap_fraction(starts, ends, t0, n):
    """Fraction of each hour covered by the periods [start, end]."""
    out = np.zeros(n)
    for s, e in zip(starts, ends):
        s, e = s - t0, e - t0
        s, e = max(s, 0), min(e, n * H)
        h = s // H
        while h * H < e:
            out[h] += (min(e, (h + 1) * H) - max(s, h * H)) / H
            h += 1
    return np.clip(out, 0, 1)


def events_to_intervals(evtf, prefix):
    """Pair each _START with the next _END for events starting with prefix."""
    ev = evtf[evtf["description"].str.startswith(prefix)].sort_values("ut_ms")
    starts, ends, open_start = [], [], None
    for t, d in zip(ev["ut_ms"], ev["description"]):
        if d.endswith("_START"):
            open_start = t
        elif d.endswith("_END") and open_start is not None:
            starts.append(open_start)
            ends.append(t)
            open_start = None
    return starts, ends


def build_features(data_dir, index, ev_kinds, cmd_fams):
    """One row per hour of `index`, built only from the context files in data_dir."""
    idx = pd.date_range(index.min(), index.max(), freq="1h")
    n = len(idx)
    t0 = int(idx[0].value // 10**6)
    f = pd.DataFrame(index=idx)

    # 1) Pointing modes and communication (share of the hour)
    ftl = pd.read_csv(f"{data_dir}/ftl.csv")
    for t in FTL_TYPES:
        sub = ftl[ftl["type"] == t]
        f["ftl_" + t] = overlap_fraction(sub["utb_ms"], sub["ute_ms"], t0, n)
    comms = ftl[ftl["flagcomms"].astype(str) == "True"]
    f["ftl_comms"] = overlap_fraction(comms["utb_ms"], comms["ute_ms"], t0, n)

    # 2) Sun angles (hourly mean)
    saaf = pd.read_csv(f"{data_dir}/saaf.csv")
    saaf["time"] = pd.to_datetime(saaf["ut_ms"], unit="ms")
    saaf_h = saaf.set_index("time")[["sa", "sx", "sy", "sz"]].resample("1h").mean()
    f = f.join(saaf_h.reindex(idx).interpolate().bfill().ffill())

    # 3) Daily trends, interpolated to hourly
    lt = pd.read_csv(f"{data_dir}/ltdata.csv")
    lt["time"] = pd.to_datetime(lt["ut_ms"], unit="ms")
    lt = lt.set_index("time").drop(columns="ut_ms")
    lt_h = lt.reindex(lt.index.union(idx)).interpolate(method="time").reindex(idx).bfill().ffill()
    f = f.join(lt_h)

    # 4) Event counts per hour and eclipses
    evtf = pd.read_csv(f"{data_dir}/evtf.csv")
    evtf["kind"] = event_kind(evtf["description"])
    evtf["hour"] = (evtf["ut_ms"] - t0) // H
    ev = evtf[(evtf["hour"] >= 0) & (evtf["hour"] < n)]
    cnt = ev.groupby(["hour", "kind"]).size().unstack(fill_value=0)
    cnt = cnt.reindex(index=range(n), columns=ev_kinds, fill_value=0)
    cnt.index = idx
    f = f.join(cnt.add_prefix("evt_"))
    for name in ["MAR_UMBRA", "MAR_PENUMBRA"]:
        s, e = events_to_intervals(evtf, name)
        f["ecl_" + name] = overlap_fraction(s, e, t0, n)

    # 5) Command counts per hour (most common families)
    dmop = pd.read_csv(f"{data_dir}/dmop.csv")
    dmop["fam"] = dmop["subsystem"].str[:4]
    dmop["hour"] = (dmop["ut_ms"] - t0) // H
    dm = dmop[(dmop["hour"] >= 0) & (dmop["hour"] < n)]
    dm_counts = dm.groupby(["hour", "fam"]).size().unstack(fill_value=0)
    dm_counts = dm_counts.reindex(index=range(n), columns=cmd_fams, fill_value=0)
    dm_counts.index = idx
    f = f.join(dm_counts.add_prefix("cmd_"))

    return f.reindex(index)
