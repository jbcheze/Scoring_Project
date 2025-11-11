import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.stats import chi2_contingency
import numpy as np

def plot_bad_rate_by_quantile(df: pd.DataFrame, var: str, q: int = 10, target: str = "loan_status"):

 
    if var not in df.columns or target not in df.columns:
        raise ValueError("Variable ou target introuvable dans le DataFrame.")
    
   
    col_bin = f"{var}_bin"
    df[col_bin] = pd.qcut(df[var], q=q, duplicates="drop")

    bad_rate = (
        df.groupby(col_bin)[target]
        .mean()
        .reset_index()
        .rename(columns={target: "bad_rate"})
    )
    

    plt.figure(figsize=(10,5))
    sns.barplot(data=bad_rate, x=col_bin, y="bad_rate", color="skyblue")
    plt.xticks(rotation=45)
    plt.title(f"Taux de défaut par quantile de {var}")
    plt.xlabel(f"{var} (en {q} quantiles)")
    plt.ylabel("Taux de défaut moyen")
    plt.tight_layout()
    
    return bad_rate

def prepare_chimerge(df, var: str, q: int = 10, target: str = "loan_status"):
    """
    Découpe la variable en q bins et calcule le nombre de bons et mauvais par bin.
    """
    col_bin = f"{var}_bin"
    df[col_bin] = pd.qcut(df[var], q=q, duplicates="drop")

    counts = (
        df.groupby(col_bin)[target]
        .agg(["count", "sum"])
        .rename(columns={"count": "total", "sum": "bad"})
        .reset_index()
    )
    counts["good"] = counts["total"] - counts["bad"]
    return counts



def apply_chimerge(df_counts: pd.DataFrame, max_bins: int = 5, significance: float = 0.05):
    """
    Applique la fusion ChiMerge sur un tableau de bons/mauvais préparé avec `prepare_chimerge`.
    
    Paramètres :
    ------------
    df_counts : DataFrame
        Doit contenir les colonnes ['total', 'bad', 'good']
    max_bins : int
        Nombre maximal de classes finales
    significance : float
        Seuil de p-value pour fusionner (par défaut 0.05)
    """
    df = df_counts.copy()
    df["min"] = df.iloc[:, 0].apply(lambda x: x.left)
    df["max"] = df.iloc[:, 0].apply(lambda x: x.right)
    df = df[["min", "max", "good", "bad"]].reset_index(drop=True)

    # Boucle principale de fusion
    while len(df) > max_bins:
        chi2_list = []
        for i in range(len(df) - 1):
            table = np.array([
                [df.loc[i, "good"], df.loc[i, "bad"]],
                [df.loc[i+1, "good"], df.loc[i+1, "bad"]]
            ])
            chi2, p, _, _ = chi2_contingency(table)
            chi2_list.append((p, i))

        # Trouve la paire la plus similaire (p-value la plus grande)
        p_max, idx = max(chi2_list, key=lambda x: x[0])

        # Si les deux bins ne sont pas significativement différents → on fusionne
        if p_max > significance:
            df.loc[idx, "max"] = df.loc[idx + 1, "max"]
            df.loc[idx, "good"] += df.loc[idx + 1, "good"]
            df.loc[idx, "bad"] += df.loc[idx + 1, "bad"]
            df = df.drop(idx + 1).reset_index(drop=True)
        else:
            break  # toutes les classes sont significativement différentes

    # Calcul du taux de défaut final
    df["bad_rate"] = df["bad"] / (df["good"] + df["bad"])
    return df


import pandas as pd

def apply_binning(df: pd.DataFrame, var: str, intervals: pd.DataFrame) -> pd.DataFrame:
    """
    Crée une variable discrète (catégorielle) à partir des intervalles issus du ChiMerge.

    Paramètres :
    ------------
    df : pd.DataFrame
        Jeu de données contenant la variable à discrétiser.
    var : str
        Nom de la variable continue (ex: "person_age").
    intervals : pd.DataFrame
        Résultat de la fonction apply_chimerge(), contenant les colonnes 'min' et 'max'.

    Retour :
    --------
    df : pd.DataFrame
        Le DataFrame original avec une nouvelle colonne '{var}_bin'.
    """

    # Construction des bornes à partir du tableau ChiMerge
    bins = intervals["min"].tolist() + [intervals["max"].iloc[-1]]
    labels = [f"{var}_Bin{i+1}" for i in range(len(intervals))]

    # Création de la nouvelle variable catégorielle
    df[f"{var}_bin"] = pd.cut(
        df[var],
        bins=bins,
        labels=labels,
        include_lowest=True
    )

    return df



def check_binning_quality(df: pd.DataFrame, var: str, target: str = "loan_status", plot: bool = True) -> pd.DataFrame:
    """
    Vérifie la qualité et la cohérence d'une variable discrétisée :
    - nombre d'observations par bin
    - taux de défaut moyen par bin
    - visualisation optionnelle du profil de risque

    Paramètres :
    ------------
    df : pd.DataFrame
        Jeu de données contenant la variable discrétisée et la cible.
    var : str
        Nom de la variable d'origine (ex : "person_age").
    target : str
        Nom de la variable cible binaire (0 = bon, 1 = défaut).
    plot : bool
        Si True, affiche le graphique du taux de défaut par bin.

    Retour :
    --------
    summary : pd.DataFrame
        Tableau récapitulatif avec effectif et taux de défaut par bin.
    """

    var_bin = f"{var}_bin"

    if var_bin not in df.columns:
        raise ValueError(f"La variable {var_bin} n'existe pas dans le DataFrame. Exécute d'abord apply_binning().")

    # Calcul du nombre d'observations et du taux de défaut moyen
    summary = (
        df.groupby(var_bin)[target]
        .agg(["count", "mean"])
        .rename(columns={"count": "effectif", "mean": "taux_defaut"})
        .reset_index()
    )

    print(f"\n=== Vérification de la discrétisation : {var} ===")
    print(summary)
    print("\n→ Vérifie que le taux de défaut évolue de manière cohérente (monotone).")

    # Visualisation
    if plot:
        plt.figure(figsize=(8, 4))
        plt.bar(summary[var_bin], summary["taux_defaut"], color="skyblue")
        plt.title(f"Taux de défaut par bin de {var}")
        plt.ylabel("Taux de défaut moyen")
        plt.xticks(rotation=45)
        plt.tight_layout()
        plt.show()

    return summary

def compute_woe_iv(df: pd.DataFrame, var_bin: str, target: str = "loan_status"):
    """
    Calcule le WOE et l'IV d'une variable déjà discrétisée (ex: person_income_bin).

    Paramètres
    ----------
    df : pd.DataFrame
        Jeu de données contenant la variable discrétisée et la cible.
    var_bin : str
        Nom de la variable discrétisée (catégorielle).
    target : str
        Nom de la variable cible (1 = défaut, 0 = bon).

    Retour
    -------
    woe_df : pd.DataFrame
        Tableau contenant les stats par bin (good, bad, dist, WOE, IV partiel).
    iv : float
        Valeur totale d'Information Value pour la variable.
    """

    # Agrégation par bin
    grouped = (
        df.groupby(var_bin)[target]
        .agg(["count", "sum"])
        .rename(columns={"count": "total", "sum": "bad"})
    )
    grouped["good"] = grouped["total"] - grouped["bad"]

    # Totaux globaux
    total_good = grouped["good"].sum()
    total_bad = grouped["bad"].sum()

    # Proportions
    grouped["dist_good"] = grouped["good"] / total_good
    grouped["dist_bad"] = grouped["bad"] / total_bad

    # Éviter division par zéro
    grouped["WOE"] = np.log((grouped["dist_good"] + 1e-9) / (grouped["dist_bad"] + 1e-9))

    # IV partiel et total
    grouped["IV_partial"] = (grouped["dist_good"] - grouped["dist_bad"]) * grouped["WOE"]
    iv_total = grouped["IV_partial"].sum()

    return grouped.reset_index(), iv_total


def prepare_chimerge_auto(df: pd.DataFrame, var: str, q: int = 10, target: str = "loan_status",
                          seuil_anormal: float = 3.0, verbose: bool = True) -> pd.DataFrame:
    """
    Prépare la table de comptage pour ChiMerge, avec détection et correction automatique des bins anormaux.

    Paramètres
    ----------
    df : pd.DataFrame
        Jeu de données complet
    var : str
        Variable numérique à discrétiser
    q : int
        Nombre initial de quantiles (par défaut 10)
    target : str
        Variable cible binaire (1 = défaut, 0 = bon)
    seuil_anormal : float
        Facteur multiplicatif du taux de défaut médian au-delà duquel un bin est considéré anormal
    verbose : bool
        Si True, affiche les étapes de correction

    Retour
    ------
    counts : pd.DataFrame
        Table de fréquence corrigée prête pour le ChiMerge
    """

    # Découpage initial en quantiles
    col_bin = f"{var}_bin"
    df[col_bin] = pd.qcut(df[var], q=q, duplicates="drop")

    # Calcul des effectifs et taux de défaut
    counts = (
        df.groupby(col_bin)[target]
        .agg(["count", "sum"])
        .rename(columns={"count": "total", "sum": "bad"})
        .reset_index()
    )
    counts["good"] = counts["total"] - counts["bad"]
    counts["bad_rate"] = counts["bad"] / counts["total"]

    # Détection du bin anormal
    median_rate = counts["bad_rate"].median()
    high_bins = counts[counts["bad_rate"] > seuil_anormal * median_rate]

    if not high_bins.empty and verbose:
        print(f"\n⚠️ {var}: détection de {len(high_bins)} bin(s) anormal(s) avec taux de défaut > {seuil_anormal}× la médiane.")
        print(high_bins[["count", "bad_rate"]])

    # Correction : fusion du dernier bin s’il est anormal
    if not high_bins.empty:
        max_bin_idx = counts["bad_rate"].idxmax()
        if max_bin_idx == counts.index[-1]:
            if verbose:
                print(f"➡️ Fusion du dernier bin de {var} avec son précédent pour stabiliser la discrétisation.")
            counts.loc[max_bin_idx - 1, "bad"] += counts.loc[max_bin_idx, "bad"]
            counts.loc[max_bin_idx - 1, "good"] += counts.loc[max_bin_idx, "good"]
            counts.loc[max_bin_idx - 1, "total"] += counts.loc[max_bin_idx, "total"]
            counts.loc[max_bin_idx - 1, "bad_rate"] = counts.loc[max_bin_idx - 1, "bad"] / counts.loc[max_bin_idx - 1, "total"]
            counts = counts.drop(max_bin_idx).reset_index(drop=True)

    return counts
