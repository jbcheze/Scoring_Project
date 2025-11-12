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

# def prepare_chimerge(df, var: str, q: int = 10, target: str = "loan_status"):
#     """
#     Découpe la variable en q bins et calcule le nombre de bons et mauvais par bin.
#     """
#     col_bin = f"{var}_bin"
#     df[col_bin] = pd.qcut(df[var], q=q, duplicates="drop")

#     counts = (
#         df.groupby(col_bin)[target]
#         .agg(["count", "sum"])
#         .rename(columns={"count": "total", "sum": "bad"})
#         .reset_index()
#     )
#     counts["good"] = counts["total"] - counts["bad"]
#     return counts



# def apply_chimerge(df_counts: pd.DataFrame, max_bins: int = 5, significance: float = 0.05):
#     """
#     Applique la fusion ChiMerge sur un tableau de bons/mauvais préparé avec `prepare_chimerge`.
    
#     Paramètres :
#     ------------
#     df_counts : DataFrame
#         Doit contenir les colonnes ['total', 'bad', 'good']
#     max_bins : int
#         Nombre maximal de classes finales
#     significance : float
#         Seuil de p-value pour fusionner (par défaut 0.05)
#     """
#     df = df_counts.copy()
#     df["min"] = df.iloc[:, 0].apply(lambda x: x.left)
#     df["max"] = df.iloc[:, 0].apply(lambda x: x.right)
#     df = df[["min", "max", "good", "bad"]].reset_index(drop=True)

#     # Boucle principale de fusion
#     while len(df) > max_bins:
#         chi2_list = []
#         for i in range(len(df) - 1):
#             table = np.array([
#                 [df.loc[i, "good"], df.loc[i, "bad"]],
#                 [df.loc[i+1, "good"], df.loc[i+1, "bad"]]
#             ])
#             chi2, p, _, _ = chi2_contingency(table)
#             chi2_list.append((p, i))

#         # Trouve la paire la plus similaire (p-value la plus grande)
#         p_max, idx = max(chi2_list, key=lambda x: x[0])

#         # Si les deux bins ne sont pas significativement différents → on fusionne
#         if p_max > significance:
#             df.loc[idx, "max"] = df.loc[idx + 1, "max"]
#             df.loc[idx, "good"] += df.loc[idx + 1, "good"]
#             df.loc[idx, "bad"] += df.loc[idx + 1, "bad"]
#             df = df.drop(idx + 1).reset_index(drop=True)
#         else:
#             break  # toutes les classes sont significativement différentes

#     # Calcul du taux de défaut final
#     df["bad_rate"] = df["bad"] / (df["good"] + df["bad"])
#     return df


# import pandas as pd

# def apply_binning(df: pd.DataFrame, var: str, intervals: pd.DataFrame) -> pd.DataFrame:
#     """
#     Crée une variable discrète (catégorielle) à partir des intervalles issus du ChiMerge.

#     Paramètres :
#     ------------
#     df : pd.DataFrame
#         Jeu de données contenant la variable à discrétiser.
#     var : str
#         Nom de la variable continue (ex: "person_age").
#     intervals : pd.DataFrame
#         Résultat de la fonction apply_chimerge(), contenant les colonnes 'min' et 'max'.

#     Retour :
#     --------
#     df : pd.DataFrame
#         Le DataFrame original avec une nouvelle colonne '{var}_bin'.
#     """

#     # Construction des bornes à partir du tableau ChiMerge
#     bins = intervals["min"].tolist() + [intervals["max"].iloc[-1]]
#     labels = [f"{var}_Bin{i+1}" for i in range(len(intervals))]

#     # Création de la nouvelle variable catégorielle
#     df[f"{var}_bin"] = pd.cut(
#         df[var],
#         bins=bins,
#         labels=labels,
#         include_lowest=True
#     )

#     return df



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


# def prepare_chimerge_auto(df: pd.DataFrame, var: str, q: int = 10, target: str = "loan_status",
#                           seuil_anormal: float = 3.0, verbose: bool = True) -> pd.DataFrame:
#     """
#     Prépare la table de comptage pour ChiMerge, avec détection et correction automatique des bins anormaux.

#     Paramètres
#     ----------
#     df : pd.DataFrame
#         Jeu de données complet
#     var : str
#         Variable numérique à discrétiser
#     q : int
#         Nombre initial de quantiles (par défaut 10)
#     target : str
#         Variable cible binaire (1 = défaut, 0 = bon)
#     seuil_anormal : float
#         Facteur multiplicatif du taux de défaut médian au-delà duquel un bin est considéré anormal
#     verbose : bool
#         Si True, affiche les étapes de correction

#     Retour
#     ------
#     counts : pd.DataFrame
#         Table de fréquence corrigée prête pour le ChiMerge
#     """

#     # Découpage initial en quantiles
#     col_bin = f"{var}_bin"
#     df[col_bin] = pd.qcut(df[var], q=q, duplicates="drop")

#     # Calcul des effectifs et taux de défaut
#     counts = (
#         df.groupby(col_bin)[target]
#         .agg(["count", "sum"])
#         .rename(columns={"count": "total", "sum": "bad"})
#         .reset_index()
#     )
#     counts["good"] = counts["total"] - counts["bad"]
#     counts["bad_rate"] = counts["bad"] / counts["total"]

#     # Détection du bin anormal
#     median_rate = counts["bad_rate"].median()
#     high_bins = counts[counts["bad_rate"] > seuil_anormal * median_rate]

#     if not high_bins.empty and verbose:
#         print(f"\n⚠️ {var}: détection de {len(high_bins)} bin(s) anormal(s) avec taux de défaut > {seuil_anormal}× la médiane.")
#         print(high_bins[["count", "bad_rate"]])

#     # Correction : fusion du dernier bin s’il est anormal
#     if not high_bins.empty:
#         max_bin_idx = counts["bad_rate"].idxmax()
#         if max_bin_idx == counts.index[-1]:
#             if verbose:
#                 print(f"➡️ Fusion du dernier bin de {var} avec son précédent pour stabiliser la discrétisation.")
#             counts.loc[max_bin_idx - 1, "bad"] += counts.loc[max_bin_idx, "bad"]
#             counts.loc[max_bin_idx - 1, "good"] += counts.loc[max_bin_idx, "good"]
#             counts.loc[max_bin_idx - 1, "total"] += counts.loc[max_bin_idx, "total"]
#             counts.loc[max_bin_idx - 1, "bad_rate"] = counts.loc[max_bin_idx - 1, "bad"] / counts.loc[max_bin_idx - 1, "total"]
#             counts = counts.drop(max_bin_idx).reset_index(drop=True)

#     return counts


# ######################################
# ######################################
# ######################################

# def chimerge_categorical(df: pd.DataFrame, var: str, target: str = "loan_status",
#                          max_bins: int = 5, min_pct: float = 0.05, verbose: bool = True):
#     """
#     Version améliorée du ChiMerge pour variables catégorielles.
#     Regroupe les modalités selon leur taux de défaut tout en respectant :
#     - un nombre maximum de groupes (max_bins)
#     - un effectif minimal par groupe (min_pct)
#     - une stabilité du risque (pas de groupes avec < 5 % du total)
#     """

#     total_obs = len(df)

#     # 1️⃣ Calcul du taux de défaut et tri des modalités
#     stats = (
#         df.groupby(var)[target]
#         .agg(["count", "sum"])
#         .rename(columns={"count": "total", "sum": "bad"})
#         .assign(good=lambda d: d["total"] - d["bad"])
#     ).reset_index()

#     stats["bad_rate"] = stats["bad"] / stats["total"]
#     stats = stats.sort_values("bad_rate").reset_index(drop=True)
#     stats["group"] = stats[var]

#     # 2️⃣ Boucle de fusion ChiMerge (catégoriel)
#     while len(stats) > max_bins:
#         chi_values = []
#         for i in range(len(stats) - 1):
#             table = np.array([
#                 [stats.loc[i, "good"], stats.loc[i, "bad"]],
#                 [stats.loc[i + 1, "good"], stats.loc[i + 1, "bad"]]
#             ])
#             chi2, p, _, _ = chi2_contingency(table)
#             chi_values.append((p, i))

#         # Trouver la paire la plus proche
#         p_max, idx = max(chi_values, key=lambda x: x[0])

#         # Fusionner les deux modalités les plus proches
#         stats.loc[idx, "group"] = f"{stats.loc[idx, 'group']}_{stats.loc[idx + 1, 'group']}"
#         stats.loc[idx, "good"] += stats.loc[idx + 1, "good"]
#         stats.loc[idx, "bad"] += stats.loc[idx + 1, "bad"]
#         stats.loc[idx, "total"] += stats.loc[idx + 1, "total"]
#         stats = stats.drop(idx + 1).reset_index(drop=True)

#     # 3️⃣ Regroupement final
#     regroupement = {}
#     for _, row in stats.iterrows():
#         for cat in row["group"].split("_"):
#             regroupement[cat] = row["group"]

#     df[f"{var}_bin"] = df[var].map(regroupement)

#     # 4️⃣ Vérification des contraintes
#     summary = (
#         df.groupby(f"{var}_bin")[target]
#         .agg(["mean", "count"])
#         .rename(columns={"mean": "taux_defaut", "count": "effectif"})
#         .reset_index()
#     )
#     summary["pct_total"] = summary["effectif"] / total_obs

#     # Condition : ≥ 5% d’effectif par groupe
#     small_groups = summary[summary["pct_total"] < min_pct]

#     if not small_groups.empty:
#         if verbose:
#             print(f"\n⚠️ Fusion de groupes trop petits (< {min_pct*100:.1f}% des observations)")
#         # Fusionner le plus petit avec le plus proche en taux de défaut
#         while (summary["pct_total"] < min_pct).any() and len(summary) > 1:
#             small_idx = summary["pct_total"].idxmin()
#             if small_idx == 0:
#                 merge_with = 1
#             else:
#                 prev_diff = abs(summary.loc[small_idx, "taux_defaut"] - summary.loc[small_idx - 1, "taux_defaut"])
#                 next_diff = abs(summary.loc[small_idx, "taux_defaut"] - summary.loc[small_idx + 1, "taux_defaut"]) if small_idx + 1 < len(summary) else np.inf
#                 merge_with = small_idx - 1 if prev_diff <= next_diff else small_idx + 1

#             g1, g2 = summary.loc[[small_idx, merge_with], f"{var}_bin"].tolist()
#             new_group = f"{g1}_{g2}"

#             # Mise à jour du regroupement
#             for k, v in regroupement.items():
#                 if v in [g1, g2]:
#                     regroupement[k] = new_group

#             # Appliquer au DataFrame
#             df[f"{var}_bin"] = df[var].map(regroupement)

#             # Recalcul du summary
#             summary = (
#                 df.groupby(f"{var}_bin")[target]
#                 .agg(["mean", "count"])
#                 .rename(columns={"mean": "taux_defaut", "count": "effectif"})
#                 .reset_index()
#             )
#             summary["pct_total"] = summary["effectif"] / total_obs

#     if verbose:
#         print(f"\n✅ Regroupement final pour {var} :")
#         print(summary[["effectif", "pct_total", "taux_defaut"]])

#     return regroupement, df
import pandas as pd
import numpy as np
from scipy.stats import chi2_contingency
import matplotlib.pyplot as plt


# ----------------------------------------------------------------------
# 1️⃣ Préparation initiale des bins
# ----------------------------------------------------------------------
def prepare_chimerge(df, var: str, q: int = 10, target: str = "loan_status"):
    """
    Découpe la variable en q quantiles et calcule les effectifs bons/mauvais.
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
    counts["bad_rate"] = counts["bad"] / counts["total"]

    return counts.sort_values(by=col_bin).reset_index(drop=True)


# ----------------------------------------------------------------------
# 2️⃣ Application du ChiMerge amélioré avec contraintes
# ----------------------------------------------------------------------
def apply_chimerge(df_counts: pd.DataFrame, max_bins: int = 5, significance: float = 0.05,
                   min_pct: float = 0.05, enforce_monotonic: bool = True, total_obs: int = None):
    """
    Applique la fusion ChiMerge en respectant :
    - max_bins (≤ 5)
    - min_pct (≥ 5% du total)
    - monotonie du taux de défaut
    """

    df = df_counts.copy()
    df["min"] = df.iloc[:, 0].apply(lambda x: x.left)
    df["max"] = df.iloc[:, 0].apply(lambda x: x.right)
    df = df[["min", "max", "good", "bad"]].reset_index(drop=True)

    if total_obs is None:
        total_obs = df["good"].sum() + df["bad"].sum()

    # --- Boucle ChiMerge classique ---
    while len(df) > max_bins:
        chi2_list = []
        for i in range(len(df) - 1):
            table = np.array([
                [df.loc[i, "good"], df.loc[i, "bad"]],
                [df.loc[i+1, "good"], df.loc[i+1, "bad"]]
            ])
            chi2, p, _, _ = chi2_contingency(table)
            chi2_list.append((p, i))

        p_max, idx = max(chi2_list, key=lambda x: x[0])
        if p_max > significance:
            # Fusion si bins similaires
            df.loc[idx, "max"] = df.loc[idx + 1, "max"]
            df.loc[idx, ["good", "bad"]] += df.loc[idx + 1, ["good", "bad"]]
            df = df.drop(idx + 1).reset_index(drop=True)
        else:
            break

    df["total"] = df["good"] + df["bad"]
    df["bad_rate"] = df["bad"] / df["total"]
    df["pct_total"] = df["total"] / total_obs

    # --- Fusion des bins trop petits (<5%) ---
    changed = True
    while changed and len(df) > 1:
        changed = False
        small_bins = df[df["pct_total"] < min_pct]
        if not small_bins.empty:
            idx = small_bins.index[0]
            merge_with = idx - 1 if idx > 0 else 1
            df.loc[merge_with, "max"] = max(df.loc[merge_with, "max"], df.loc[idx, "max"])
            df.loc[merge_with, ["good", "bad"]] += df.loc[idx, ["good", "bad"]]
            df = df.drop(idx).reset_index(drop=True)
            df["total"] = df["good"] + df["bad"]
            df["pct_total"] = df["total"] / total_obs
            df["bad_rate"] = df["bad"] / df["total"]
            changed = True

    # --- Correction de la monotonie ---
    if enforce_monotonic and len(df) > 2:
        changed = True
        while changed:
            changed = False
            direction = np.sign(df["bad_rate"].iloc[-1] - df["bad_rate"].iloc[0])  # croissant ou décroissant
            for i in range(len(df) - 1):
                if direction > 0 and df.loc[i+1, "bad_rate"] < df.loc[i, "bad_rate"]:
                    # Fusion si incohérence
                    df.loc[i, "max"] = df.loc[i+1, "max"]
                    df.loc[i, ["good", "bad"]] += df.loc[i+1, ["good", "bad"]]
                    df = df.drop(i+1).reset_index(drop=True)
                    df["total"] = df["good"] + df["bad"]
                    df["bad_rate"] = df["bad"] / df["total"]
                    df["pct_total"] = df["total"] / total_obs
                    changed = True
                    break
                elif direction < 0 and df.loc[i+1, "bad_rate"] > df.loc[i, "bad_rate"]:
                    df.loc[i, "max"] = df.loc[i+1, "max"]
                    df.loc[i, ["good", "bad"]] += df.loc[i+1, ["good", "bad"]]
                    df = df.drop(i+1).reset_index(drop=True)
                    df["total"] = df["good"] + df["bad"]
                    df["bad_rate"] = df["bad"] / df["total"]
                    df["pct_total"] = df["total"] / total_obs
                    changed = True
                    break

    return df


# ----------------------------------------------------------------------
# 3️⃣ Application du binning
# ----------------------------------------------------------------------
def apply_binning(df: pd.DataFrame, var: str, intervals: pd.DataFrame) -> pd.DataFrame:
    """
    Crée une variable discrète à partir des bornes validées.
    """
    bins = intervals["min"].tolist() + [intervals["max"].iloc[-1]]
    labels = [f"{var}_Bin{i+1}" for i in range(len(intervals))]
    df[f"{var}_bin"] = pd.cut(df[var], bins=bins, labels=labels, include_lowest=True)
    return df


# ----------------------------------------------------------------------
# 4️⃣ ChiMerge catégoriel amélioré
# ----------------------------------------------------------------------
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.stats import chi2_contingency

def chimerge_categorical(df: pd.DataFrame, var: str, target: str = "loan_status",
                         max_bins: int = 5, min_pct: float = 0.05,
                         verbose: bool = True, enforce_monotonicity: bool = True):
    """
    ChiMerge amélioré :
    - Fusionne les modalités selon Chi2
    - Imposant une monotonie stricte du taux de défaut
    - Affiche les graphes avant/après
    """
    total_obs = len(df)

    # === 1️⃣ Calcul de base
    stats = (
        df.groupby(var)[target]
        .agg(["count", "sum"])
        .rename(columns={"count": "total", "sum": "bad"})
        .assign(good=lambda d: d["total"] - d["bad"])
    ).reset_index()

    stats["bad_rate"] = stats["bad"] / stats["total"]
    stats = stats.sort_values("bad_rate").reset_index(drop=True)
    stats["group"] = stats[var]

    # === 2️⃣ Fusion ChiMerge
    while len(stats) > max_bins:
        chi_values = []
        for i in range(len(stats) - 1):
            table = np.array([
                [stats.loc[i, "good"], stats.loc[i, "bad"]],
                [stats.loc[i+1, "good"], stats.loc[i+1, "bad"]]
            ])
            chi2, p, _, _ = chi2_contingency(table)
            chi_values.append((p, i))
        p_max, idx = max(chi_values, key=lambda x: x[0])
        stats.loc[idx, "group"] = f"{stats.loc[idx, 'group']}_{stats.loc[idx+1, 'group']}"
        stats.loc[idx, ["good", "bad", "total"]] += stats.loc[idx+1, ["good", "bad", "total"]]
        stats = stats.drop(idx+1).reset_index(drop=True)
        stats["bad_rate"] = stats["bad"] / stats["total"]

    # === 3️⃣ Fusion bins trop petits
    changed = True
    while changed:
        changed = False
        stats["pct_total"] = stats["total"] / total_obs
        small_bins = stats[stats["pct_total"] < min_pct]
        if not small_bins.empty:
            idx = small_bins.index[0]
            merge_with = idx - 1 if idx > 0 else 1
            stats.loc[merge_with, "group"] += "_" + stats.loc[idx, "group"]
            stats.loc[merge_with, ["good", "bad", "total"]] += stats.loc[idx, ["good", "bad", "total"]]
            stats = stats.drop(idx).reset_index(drop=True)
            stats["bad_rate"] = stats["bad"] / stats["total"]
            changed = True

    # === 4️⃣ Correction stricte de la monotonie ===
    if enforce_monotonicity:
        direction = np.sign(stats["bad_rate"].iloc[-1] - stats["bad_rate"].iloc[0])

        def is_monotone(arr, direction):
            return np.all(np.diff(arr) >= 0) if direction >= 0 else np.all(np.diff(arr) <= 0)

        while not is_monotone(stats["bad_rate"].values, direction) and len(stats) > 1:
            # Trouver la première inversion
            bad_rate = stats["bad_rate"].values
            if direction >= 0:
                inv_idx = np.where(np.diff(bad_rate) < 0)[0]
            else:
                inv_idx = np.where(np.diff(bad_rate) > 0)[0]
            if len(inv_idx) == 0:
                break
            i = inv_idx[0]
            # Fusionner le bin fautif avec le suivant
            stats.loc[i, "group"] += "_" + stats.loc[i+1, "group"]
            stats.loc[i, ["good", "bad", "total"]] += stats.loc[i+1, ["good", "bad", "total"]]
            stats = stats.drop(i+1).reset_index(drop=True)
            stats["bad_rate"] = stats["bad"] / stats["total"]

    # === 5️⃣ Attribution des noms bin1, bin2...
    stats = stats.sort_values("bad_rate").reset_index(drop=True)
    stats["bin_name"] = [f"bin{i+1}" for i in range(len(stats))]

    regroupement = {}
    for _, row in stats.iterrows():
        for cat in row["group"].split("_"):
            regroupement[cat] = row["bin_name"]

    df[f"{var}_bin"] = df[var].map(regroupement)

    # === 6️⃣ Graphiques
    plt.figure(figsize=(8, 5))
    sns.countplot(data=df, x=var, order=df[var].value_counts().index, palette="viridis")
    plt.title(f"Répartition de {var} avant discrétisation")
    plt.xlabel(var)
    plt.ylabel("Effectif")
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.show()

    plt.figure(figsize=(8, 5))
    sns.countplot(data=df, x=f"{var}_bin", order=sorted(df[f"{var}_bin"].unique()), palette="mako")
    plt.title(f"Répartition de {var} après discrétisation (ChiMerge monotone)")
    plt.xlabel(f"{var}_bin")
    plt.ylabel("Effectif")
    plt.tight_layout()
    plt.show()

    # === 7️⃣ Courbe du taux de défaut par bin
    summary = (
        df.groupby(f"{var}_bin")[target]
        .agg(["mean", "count"])
        .rename(columns={"mean": "taux_defaut", "count": "effectif"})
        .reset_index()
    )
    plt.figure(figsize=(7, 4))
    sns.lineplot(data=summary, x=f"{var}_bin", y="taux_defaut", marker="o")
    plt.title(f"Taux de défaut par bin ({var}) – Vérification de la monotonie")
    plt.xlabel(f"{var}_bin")
    plt.ylabel("Taux de défaut moyen")
    plt.tight_layout()
    plt.show()

    # === 8️⃣ Affichage final
    if verbose:
        summary["pct_total"] = summary["effectif"] / total_obs
        print(f"\n✅ Regroupement final monotone pour {var} :")
        print(summary)
        print("\n📊 Détails des regroupements :")
        for old_cat, new_bin in regroupement.items():
            print(f" - {old_cat} → {new_bin}")

    return regroupement, df
