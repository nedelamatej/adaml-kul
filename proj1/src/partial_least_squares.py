# [ADAML] Advanced Data Analysis and Machine Learning
#         Partial Least Squares
#
# Lappeenranta-Lahti University of Technology
# School of Engineering Science
#
# Name: partial_least_squares.py
# Aut.: Moriom Akter
#       Matej Nedela
#       Monowarul Sabbir
# Date: 11/10/2026
# Ver.: 1.0

import textwrap

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.patches import Rectangle
from sklearn.cross_decomposition import PLSRegression
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.model_selection import GroupKFold

names = {
    "unit_id": "Unit number",
    "cycle": "Time [cycle]",
    "altitude": "Altitude [K ft]",
    "mach": "Mach number",
    "throttle": "Throttle resolver angle",
    "T2": "Total temperature at fan inlet [°R]",
    "T24": "Total temperature at LPC outlet [°R]",
    "T30": "Total temperature at HPC outlet [°R]",
    "T50": "Total temperature at LPT outlet [°R]",
    "P2": "Pressure at fan inlet [psia]",
    "P15": "Total pressure in bypass-duct [psia]",
    "P30": "Total pressure at HPC outlet [psia]",
    "Nf": "Physical fan speed [rpm]",
    "Nc": "Physical core speed [rpm]",
    "epr": "Engine pressure ratio (P50/P2)",
    "Ps30": "Static pressure at HPC outlet [psia]",
    "phi": "Ratio of fuel flow to Ps30 [pps/psi]",
    "NRf": "Corrected fan speed [rpm]",
    "NRc": "Corrected core speed [rpm]",
    "BPR": "Bypass ratio",
    "farB": "Burner fuel-air ratio",
    "htBleed": "Bleed enthalpy",
    "Nf_dmd": "Demanded fan speed [rpm]",
    "PCNfR_dmd": "Demanded corrected fan speed [rpm]",
    "W31": "HPT coolant bleed [lbm/s]",
    "W32": "LPT coolant bleed [lbm/s]",
}

op_conds = pd.DataFrame(
    [
        {"altitude": 0.0, "mach": 0.00, "throttle": 100.0},
        {"altitude": 10.0, "mach": 0.25, "throttle": 100.0},
        {"altitude": 20.0, "mach": 0.70, "throttle": 100.0},
        {"altitude": 25.0, "mach": 0.62, "throttle": 60.0},
        {"altitude": 35.0, "mach": 0.84, "throttle": 100.0},
        {"altitude": 42.0, "mach": 0.84, "throttle": 100.0},
    ]
).reset_index(names="idx")

subsets = ["FD001", "FD002", "FD003", "FD004"]
sensors = list(names)[5:]


# Data pretreatment


def read_df(subset, split):
    df = pd.read_csv(f"data/{split}_{subset}.txt", sep=r"\s+", names=list(names))
    df.insert(0, "subset", subset)

    return df


def read_rul(subset):
    rul = pd.read_csv(f"data/RUL_{subset}.txt", sep=r"\s+", names=["rul"])["rul"]
    rul.index += 1

    return rul


def add_operational_condition(df):
    df["altitude"] = 0 + df["altitude"].round(0)
    df["mach"] = 0 + df["mach"].round(2)
    df["throttle"] = 0 + df["throttle"].round(0)

    op_cond = df.merge(op_conds, on=["altitude", "mach", "throttle"], how="left")["idx"].values

    df.insert(3, "op_cond", op_cond)

    df = df.drop(columns=["altitude", "mach", "throttle"])

    return df


def add_remaining_useful_life(df, rul=None):
    if rul is None:
        rul = df.groupby("unit_id")["cycle"].transform("max") - df["cycle"]
    else:
        rul = df.groupby("unit_id")["cycle"].transform("max") - df["cycle"] + df["unit_id"].map(rul)

    df.insert(4, "rul", rul.clip(upper=125))

    return df


def load_df(split):
    dfs = []

    for subset in subsets:
        df = read_df(subset, split)
        df = add_operational_condition(df)
        df = add_remaining_useful_life(df, read_rul(subset) if split == "test" else None)

        dfs.append(df)

    return pd.concat(dfs, ignore_index=True)


def fit_z_scores(df):
    mean, std = df.groupby("op_cond")[sensors].mean(), df.groupby("op_cond")[sensors].std()

    cols = (df.groupby(["subset", "op_cond"])[sensors].std() >= 1e-3).all()[lambda keep: keep].index

    return mean.loc[:, cols], std.loc[:, cols]


def apply_z_scores(df, mean, std):
    cols = std.columns

    df = df.drop(columns=[sensor for sensor in sensors if sensor not in cols])

    df[cols] = (df[cols] - mean.loc[df["op_cond"]].values) / std.loc[df["op_cond"]].values

    return df


def compute_rolling_mean(df, cols):
    df[cols] = df.groupby(["subset", "unit_id"])[cols].transform(lambda x: x.rolling(10, min_periods=1).mean())

    return df


df_train = load_df("train")
df_test = load_df("test")

mean, std = fit_z_scores(df_train)

df_train = apply_z_scores(df_train, mean, std)
df_test = apply_z_scores(df_test, mean, std)

sensors = list(mean.columns)

df_train = compute_rolling_mean(df_train, sensors)
df_test = compute_rolling_mean(df_test, sensors)


# Variable importance in projection


def compute_variable_importance_in_projection(model):
    t, w, q = model.x_scores_, model.x_weights_, model.y_loadings_
    p, h = w.shape

    ssy = np.array([q[0, j] ** 2 * (t[:, j] @ t[:, j]) for j in range(h)])

    vips = np.zeros(p)

    for i in range(p):
        weight = sum((w[i, j] / np.linalg.norm(w[:, j])) ** 2 * ssy[j] for j in range(h))
        vips[i] = np.sqrt(p * weight / ssy.sum())

    return vips


vip = {}

for subset, df in df_train.groupby("subset"):
    pls = PLSRegression(n_components=3).fit(df[sensors], df["rul"])

    vip[subset] = compute_variable_importance_in_projection(pls)

vip = pd.DataFrame(vip, index=sensors)
vip["mean"] = vip.mean(axis=1)

print(f"{'-' * 3} Variable Importance in Projection {'-' * 3}")
print(vip.sort_values("mean", ascending=False).round(2))


# R squared


def compute_r_squared(x, y, groups):
    r2_folds = []

    for train_idx, validation_idx in GroupKFold(n_splits=5).split(x, y, groups=groups):
        pls = PLSRegression(n_components=1).fit(x[train_idx], y[train_idx])
        y_pred = pls.predict(x[validation_idx]).ravel()

        r2_folds.append(r2_score(y[validation_idx], y_pred))

    return np.mean(r2_folds)


r2 = {}

for subset, df in df_train.groupby("subset"):
    y, train_ids = df["rul"].values, df["unit_id"].values

    r2[subset] = [compute_r_squared(df[[sensor]].values, y, train_ids) for sensor in sensors]

r2 = pd.DataFrame(r2, index=sensors)
r2["mean"] = r2.mean(axis=1)

print(f"\n{'-' * 15} R Squared {'-' * 15}")
print(r2.sort_values("mean", ascending=False).round(2))


# Sensors selection


selected = [sensors[i] for i in np.where((vip["mean"] > 0.5) & (r2["mean"] > 0.0))[0]]

print(f"\n{'-' * 16} Selected {'-' * 16}")
print(textwrap.fill(f"{', '.join(selected)} ({len(selected)} / {len(sensors)})", width=42))


# Variable importance in projection plot


fig, ax = plt.subplots(2, 2, figsize=(12, 5.5))
ax = ax.flatten()

for idx, subset in enumerate(subsets):
    ax[idx].axis([-0.5, len(sensors) - 0.5, 0, 1.4])
    ax[idx].grid(alpha=0.5, axis="y")

    colors = ["green" if val > 0.5 else "orange" for val in vip[subset]]

    ax[idx].axhline(0.5, color="red", alpha=0.75, linestyle="--")

    ax[idx].bar(range(len(sensors)), vip[subset], color=colors, edgecolor="k", alpha=0.5)

    ax[idx].set_title(f"{subset}")
    ax[idx].set_ylabel("Variable importance in projection")
    ax[idx].set_xticks(range(len(sensors)))
    ax[idx].set_xticklabels(sensors, fontsize=6)

plt.subplots_adjust(left=0.075, right=0.935, bottom=0.075, top=0.95)

plt.savefig("figs/vip_score.svg")


# R squared plot


fig, ax = plt.subplots(2, 2, figsize=(12, 5.5))
ax = ax.flatten()

for idx, subset in enumerate(subsets):
    ax[idx].axis([-0.5, len(sensors) - 0.5, -0.6, 0.8])
    ax[idx].grid(alpha=0.5, axis="y")

    colors = ["purple" if val > 0.0 else "gray" for val in r2[subset]]

    ax[idx].axhline(0.0, color="red", alpha=0.75, linestyle="--")

    ax[idx].bar(range(len(sensors)), r2[subset], color=colors, edgecolor="k", alpha=0.5)

    ax[idx].set_title(f"{subset}")
    ax[idx].set_ylabel("R squared")
    ax[idx].set_xticks(range(len(sensors)))
    ax[idx].set_xticklabels(sensors, fontsize=6)

plt.subplots_adjust(left=0.075, right=0.935, bottom=0.075, top=0.95)

plt.savefig("figs/r2_score.svg")


# Correlation heatmap


fig, ax = plt.subplots(2, 2, figsize=(12, 5.5))
ax = ax.flatten()

for idx, subset in enumerate(subsets):
    ax[idx].imshow(
        df_train.loc[df_train["subset"] == subset][selected].corr(), vmin=-1, vmax=+1, cmap="coolwarm", aspect="auto"
    )

    ax[idx].set_title(subset)
    ax[idx].set_xticks(range(len(selected)), selected, fontsize=6)
    ax[idx].set_yticks(range(len(selected)), selected, fontsize=6)

plt.subplots_adjust(left=0.075, right=0.99, bottom=0.075, top=0.95)

cbar = fig.colorbar(ax[0].images[0], ax=ax, fraction=0.04, pad=0.02, aspect=30)
cbar.set_ticks([-1, 0, 1])
cbar.set_ticklabels(["-1", "0", "+1"])

plt.savefig("figs/correlation_heatmap.svg")


# Partial least squares


results = {}

lv_performance_r2 = {subset: [] for subset in subsets}
lv_performance_q2 = {subset: [] for subset in subsets}

for subset in subsets:
    train = df_train.loc[df_train["subset"] == subset]
    test = df_test.loc[df_test["subset"] == subset].groupby("unit_id").tail(1)

    train_ids = train["unit_id"].values

    x_train = train[selected].values
    y_train = train["rul"].values
    x_test = test[selected].values
    y_test = test["rul"].values

    for lv in range(1, len(selected)):
        # Q squared (Q2)

        q2_scores = []

        for train_idx, validation_idx in GroupKFold(n_splits=5).split(x_train, y_train, groups=train_ids):
            x_fold, x_validation = x_train[train_idx], x_train[validation_idx]
            y_fold, y_validation = y_train[train_idx], y_train[validation_idx]

            y_pred = PLSRegression(n_components=lv).fit(x_fold, y_fold).predict(x_validation).ravel().clip(0, 125)

            q2_scores.append(r2_score(y_validation, y_pred))

        lv_performance_q2[subset].append(np.mean(q2_scores))

        # R squared (R2)

        y_pred = PLSRegression(n_components=lv).fit(x_train, y_train).predict(x_test).ravel().clip(0, 125)

        lv_performance_r2[subset].append(r2_score(y_test, y_pred))

    q2 = np.array(lv_performance_q2[subset])
    lv = int(np.argmax(q2 >= q2.max() - 0.01)) + 1

    model = PLSRegression(n_components=lv).fit(x_train, y_train)
    y_pred = model.predict(x_test).ravel().clip(0, 125)

    results[subset] = {
        "model": model,
        "lv": lv,
        "r2": r2_score(y_test, y_pred),
        "rmse": np.sqrt(mean_squared_error(y_test, y_pred)),
        "y_test": y_test,
        "y_pred": y_pred,
    }

    print(f"\n{'-' * 17} {subset} {'-' * 17}")
    print(f"LVs:     {lv:0>2}")
    print(f"R2:      {results[subset]['r2']:.2f}")
    print(f"RMSE:    {results[subset]['rmse']:.2f}")


# R squared heatmap


df_r2 = pd.DataFrame(lv_performance_r2).T
df_r2.columns = range(1, df_r2.shape[1] + 1)

fig, ax = plt.subplots(figsize=(12, 3))

ax.imshow(df_r2, vmin=0.6, vmax=0.8, aspect="auto")

for row, col in np.ndindex(df_r2.shape):
    ax.text(col, row, f"{df_r2.iloc[row, col]:.3f}", ha="center", va="center")

for row, subset in enumerate(df_r2.index):
    col = results[subset]["lv"] - 1

    ax.add_patch(Rectangle((col - 0.5, row - 0.5), 1, 1, fill=False, edgecolor="red", lw=2))

ax.set_xlabel("Number of latent variables")
ax.set_xticks(range(df_r2.shape[1]), df_r2.columns)
ax.set_yticks(range(df_r2.shape[0]), df_r2.index)

fig.colorbar(ax.images[0], ax=ax, fraction=0.04, pad=0.02, aspect=30)

plt.subplots_adjust(left=0.075, right=0.99, bottom=0.15, top=0.95)
plt.savefig("figs/r2_heatmap.svg")


# Q squared heatmap


df_q2 = pd.DataFrame(lv_performance_q2).T
df_q2.columns = range(1, df_q2.shape[1] + 1)

fig, ax = plt.subplots(figsize=(12, 3))

ax.imshow(df_q2, vmin=0.6, vmax=0.8, aspect="auto")

for row, col in np.ndindex(df_q2.shape):
    ax.text(col, row, f"{df_q2.iloc[row, col]:.3f}", ha="center", va="center")

for row, subset in enumerate(df_q2.index):
    col = results[subset]["lv"] - 1

    ax.add_patch(Rectangle((col - 0.5, row - 0.5), 1, 1, fill=False, edgecolor="red", lw=2))

ax.set_xlabel("Number of latent variables")
ax.set_xticks(range(df_q2.shape[1]), df_q2.columns)
ax.set_yticks(range(df_q2.shape[0]), df_q2.index)

fig.colorbar(ax.images[0], ax=ax, fraction=0.04, pad=0.02, aspect=30)

plt.subplots_adjust(left=0.075, right=0.99, bottom=0.15, top=0.95)
plt.savefig("figs/q2_heatmap.svg")


# Predicted vs observed


fig, ax = plt.subplots(2, 2, figsize=(12, 5.5))
ax = ax.flatten()

for idx, subset in enumerate(subsets):
    y_test = results[subset]["y_test"]
    y_pred = results[subset]["y_pred"]

    ax[idx].axis([-10, 135, -10, 135])
    ax[idx].grid(alpha=0.5)

    ax[idx].axline((0, 0), slope=1, color="red", alpha=0.75, linestyle="--")

    ax[idx].scatter(y_test, y_pred, color="blue", edgecolors="k", alpha=0.5, label="Predicted")

    ax[idx].set_title(f"{subset} (R2 = {results[subset]['r2']:.2f}, RMSE = {results[subset]['rmse']:.2f})")
    if idx >= 2:
        ax[idx].set_xlabel("True remaining useful life")
    ax[idx].set_ylabel("Predicted remaining useful life")
    ax[idx].legend(loc="lower right")

plt.subplots_adjust(left=0.075, right=0.935, bottom=0.075, top=0.95)
plt.savefig("figs/predicted_vs_observed.svg")


# Residual analysis


fig, ax = plt.subplots(2, 2, figsize=(12, 5.5))
ax = ax.flatten()

for idx, subset in enumerate(subsets):
    y_test = results[subset]["y_test"]
    y_pred = results[subset]["y_pred"]

    residuals = y_test - y_pred

    ax[idx].axis([-10, 135, -70, 70])
    ax[idx].grid(alpha=0.5)

    ax[idx].axhline(0, color="red", alpha=0.75, linestyle="--")

    ax[idx].scatter(y_pred, residuals, color="purple", edgecolors="k", alpha=0.5, label="Residuals")

    ax[idx].set_title(f"{subset} (R2 = {results[subset]['r2']:.2f}, RMSE = {results[subset]['rmse']:.2f})")
    if idx >= 2:
        ax[idx].set_xlabel("Predicted remaining useful life")
    ax[idx].set_ylabel("Residuals")
    ax[idx].legend(loc="lower left")

plt.subplots_adjust(left=0.075, right=0.935, bottom=0.075, top=0.95)
plt.savefig("figs/residual_analysis.svg")
