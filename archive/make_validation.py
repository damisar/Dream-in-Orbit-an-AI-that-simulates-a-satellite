"""
Construit un dossier de validation au format exact de l'évaluation finale.

Usage :
    python make_validation.py --annee annee2 --out validation

Produit :
    validation/test/        dmop.csv, evtf.csv, ftl.csv, saaf.csv, ltdata.csv,
                            power_init.csv (7 premiers jours), heures_a_predire.csv
    validation/verite_horaire.csv   vraie consommation moyenne par heure

Ensuite :
    python predict.py --test_dir validation/test --out pred.csv
    python evaluate.py --pred pred.csv --verite validation/verite_horaire.csv --heures validation/test/heures_a_predire.csv

L'organisation utilise exactement la même procédure sur l'année de test.
"""
import argparse
import shutil
from pathlib import Path

import pandas as pd

CONTEXTES = ["dmop", "evtf", "ftl", "saaf", "ltdata"]
JOURS_INIT = 7
H = 3600 * 1000


def construire(power_csv: Path, contexte_dir: Path, out: Path):
    test = out / "test"
    test.mkdir(parents=True, exist_ok=True)
    for c in CONTEXTES:
        shutil.copy(contexte_dir / f"{c}.csv", test / f"{c}.csv")

    raw = pd.read_csv(power_csv)
    debut = raw["ut_ms"].min()
    fin_init = debut + JOURS_INIT * 24 * H
    raw[raw["ut_ms"] < fin_init].to_csv(test / "power_init.csv", index=False)

    raw["ut_ms"] = (raw["ut_ms"] // H) * H
    verite = raw.groupby("ut_ms").mean().sort_index()
    premiere_heure = ((fin_init // H) + 1) * H
    verite = verite[verite.index >= premiere_heure]
    verite.to_csv(out / "verite_horaire.csv")

    heures = pd.DataFrame({"ut_ms": range(int(premiere_heure), int(verite.index.max()) + H, H)})
    heures.to_csv(test / "heures_a_predire.csv", index=False)
    return len(heures), premiere_heure, int(verite.index.max())


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--annee", required=True, help="dossier contenant power.csv et les 5 fichiers de contexte")
    ap.add_argument("--out", default="validation")
    a = ap.parse_args()
    d = Path(a.annee)
    n, deb, fin = construire(d / "power.csv", d, Path(a.out))
    print(f"{n} heures à prédire, de {pd.to_datetime(deb, unit='ms')} à {pd.to_datetime(fin, unit='ms')}")


if __name__ == "__main__":
    main()
