"""
Évaluation Dream in Orbit : RMSE entre prédictions horaires et vraie consommation.

Usage :
    python evaluate.py --pred predictions.csv --verite verite_annee3_horaire.csv [--heures heures_a_predire.csv]

Format attendu de predictions.csv :
    - colonne ut_ms (début de l'heure, en millisecondes UTC) + les 33 colonnes NPWDxxxx
    - une ligne par heure de heures_a_predire.csv, aucune valeur manquante
Score : RMSE sur toutes les cellules (heure x circuit) où la vérité existe. Plus bas = mieux.
"""
import argparse
import sys

import numpy as np
import pandas as pd


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--pred", required=True)
    ap.add_argument("--verite", required=True)
    ap.add_argument("--heures", default=None)
    a = ap.parse_args()

    v = pd.read_csv(a.verite).set_index("ut_ms")
    p = pd.read_csv(a.pred)

    erreurs = []
    if "ut_ms" not in p.columns:
        erreurs.append("colonne ut_ms absente")
    manquantes = [c for c in v.columns if c not in p.columns]
    if manquantes:
        erreurs.append(f"{len(manquantes)} colonnes manquantes, ex : {manquantes[:3]}")
    if erreurs:
        sys.exit("FORMAT INVALIDE : " + " ; ".join(erreurs))

    p = p.set_index("ut_ms")[v.columns]
    if a.heures:
        attendues = pd.read_csv(a.heures)["ut_ms"]
        absentes = set(attendues) - set(p.index)
        if absentes:
            sys.exit(f"FORMAT INVALIDE : {len(absentes)} heures attendues absentes")
    if p.isna().any().any():
        sys.exit("FORMAT INVALIDE : valeurs manquantes dans les prédictions")

    communes = v.index.intersection(p.index)
    if len(communes) < 0.99 * len(v):
        sys.exit("FORMAT INVALIDE : moins de 99 % des heures notées sont présentes")
    vv = v.loc[communes].to_numpy()
    pp = p.loc[communes].to_numpy()
    masque = ~np.isnan(vv)
    rmse = float(np.sqrt(np.mean((pp[masque] - vv[masque]) ** 2)))
    print(f"RMSE = {rmse:.6f}  ({masque.sum()} valeurs notées sur {len(communes)} heures)")


if __name__ == "__main__":
    main()
