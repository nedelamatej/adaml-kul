# [ADAML] Advanced Data Analysis and Machine Learning
#         Data Pretreatment
#
# Lappeenranta-Lahti University of Technology
# School of Engineering Science
#
# Name: data_pretreatment.py
# Aut.: Moriom Akter
#       Matej Nedela
#       Monowarul Sabbir
# Date: 26/09/2026
# Ver.: 1.1

import matplotlib.pyplot as plt
import pandas as pd
from sklearn.decomposition import PCA

names = {
  "unit_id":   "Unit number",
  "cycle":     "Time [cycle]",
  "altitude":  "Altitude [K ft]",
  "mach":      "Mach number",
  "throttle":  "Throttle resolver angle",
  "T2":        "Total temperature at fan inlet [°R]",
  "T24":       "Total temperature at LPC outlet [°R]",
  "T30":       "Total temperature at HPC outlet [°R]",
  "T50":       "Total temperature at LPT outlet [°R]",
  "P2":        "Pressure at fan inlet [psia]",
  "P15":       "Total pressure in bypass-duct [psia]",
  "P30":       "Total pressure at HPC outlet [psia]",
  "Nf":        "Physical fan speed [rpm]",
  "Nc":        "Physical core speed [rpm]",
  "epr":       "Engine pressure ratio (P50/P2)",
  "Ps30":      "Static pressure at HPC outlet [psia]",
  "phi":       "Ratio of fuel flow to Ps30 [pps/psi]",
  "NRf":       "Corrected fan speed [rpm]",
  "NRc":       "Corrected core speed [rpm]",
  "BPR":       "Bypass ratio",
  "farB":      "Burner fuel-air ratio",
  "htBleed":   "Bleed enthalpy",
  "Nf_dmd":    "Demanded fan speed [rpm]",
  "PCNfR_dmd": "Demanded corrected fan speed [rpm]",
  "W31":       "HPT coolant bleed [lbm/s]",
  "W32":       "LPT coolant bleed [lbm/s]",
}

op_conds = pd.DataFrame([
  { 'altitude':  0.0, 'mach': 0.00, 'throttle': 100.0 },
  { 'altitude': 10.0, 'mach': 0.25, 'throttle': 100.0 },
  { 'altitude': 20.0, 'mach': 0.70, 'throttle': 100.0 },
  { 'altitude': 25.0, 'mach': 0.62, 'throttle':  60.0 },
  { 'altitude': 35.0, 'mach': 0.84, 'throttle': 100.0 },
  { 'altitude': 42.0, 'mach': 0.84, 'throttle': 100.0 },
]).reset_index(names='idx')

df_fd001 = pd.read_csv('data/train_FD001.txt', sep=r'\s+', names=names.keys())
df_fd002 = pd.read_csv('data/train_FD002.txt', sep=r'\s+', names=names.keys())
df_fd003 = pd.read_csv('data/train_FD003.txt', sep=r'\s+', names=names.keys())
df_fd004 = pd.read_csv('data/train_FD004.txt', sep=r'\s+', names=names.keys())

dfs = [df_fd001, df_fd002, df_fd003, df_fd004]

cols = list(pd.concat(dfs).columns[5:])

for idx, df in enumerate(dfs):
  df['altitude'] = 0 + df['altitude'].round(0)
  df['mach']     = 0 + df['mach'].round(2)
  df['throttle'] = 0 + df['throttle'].round(0)

  df['op_cond']  = df.merge(op_conds, on=['altitude', 'mach', 'throttle'], how='left')['idx'].values
  df['rul']      = df.groupby('unit_id')['cycle'].transform('max') - df['cycle']

  df.drop(columns=['altitude', 'mach', 'throttle'], inplace=True)

  dfs[idx] = df[['unit_id', 'op_cond', 'cycle', 'rul'] + cols].copy()

df_concat = pd.concat(dfs, ignore_index=True)

for df in dfs:
  df.drop(columns=[col for col in cols if df_concat.groupby(['op_cond'])[col].std().mean() < 1e-3], inplace=True)

cols = list(pd.concat(dfs).columns[4:])

mean = df_concat.groupby('op_cond')[cols].mean()
std  = df_concat.groupby('op_cond')[cols].std()

for df in dfs:
  for col in cols:
    df[f'{col}_raw'] = df[col]
    df[f'{col}'] = (df[col] - df['op_cond'].map(mean[col])) / df['op_cond'].map(std[col])

df_concat = pd.concat(dfs, ignore_index=True)

# Box plot of static pressure at HPC outlet (Ps30) by operating condition

plt.figure(figsize=(12, 3))
plt.axis([35.3, 50.7, 0.5, 6.5])

plt.boxplot([df_concat[df_concat['op_cond'] == op_cond]['Ps30_raw'] for op_cond in op_conds['idx'][::-1]], orientation='horizontal', widths=0.6)

plt.xlabel('Static pressure at HPC outlet [psia]')
plt.xticks(range(36, 50 + 1, 14 // 7))
plt.yticks(range(1, len(op_conds['idx']) + 1), [f'Op. cond. {op_cond}' for op_cond in op_conds['idx'][::-1]])

plt.subplots_adjust(left=0.075, right=0.975, bottom=0.15, top=0.95)
plt.savefig('figs/box_plot.svg')

# Violin plot of key sensors before standardization

plt.figure(figsize=(12, 3))
plt.axis([-120, 2520, 0.5, 5.5])

plt.violinplot([df_concat[f'{col}_raw'] for col in ['NRf', 'Nf', 'Ps30', 'P30', 'T30']], orientation='horizontal', widths=0.8)

plt.xlabel('Raw value (temperature, pressure, static pressure or speed)')
plt.xticks(range(0, 2400 + 1, 2400 // 10))
plt.yticks(range(1, 6), ['NRf', 'Nf', 'Ps30', 'P30', 'T30'])

plt.subplots_adjust(left=0.075, right=0.975, bottom=0.15, top=0.95)
plt.savefig('figs/violin_plot_before.svg')

# Violin plot of key sensors after standardization

plt.figure(figsize=(12, 3))
plt.axis([-11, 11, 0.5, 5.5])

plt.violinplot([df_concat[f'{col}'] for col in ['NRf', 'Nf', 'Ps30', 'P30', 'T30']], orientation='horizontal', widths=0.8)

plt.xlabel('Standardized value (z-score)')
plt.xticks(range(-10, 10 + 1, 20 // 10))
plt.yticks(range(1, 6), ['NRf', 'Nf', 'Ps30', 'P30', 'T30'])

plt.subplots_adjust(left=0.075, right=0.975, bottom=0.15, top=0.95)
plt.savefig('figs/violin_plot_after.svg')

# Biplots after data pretreatment

fig, ax = plt.subplots(2, 2, figsize=(12, 5.5))
ax = ax.flatten()

explained_variances = []

for idx, df in enumerate(dfs):
  pca = PCA()
  X = pca.fit_transform(df[cols].values)

  explained_variances.append(pca.explained_variance_ratio_)

  ax[idx].axis([-9, 15, -8, 12])
  ax[idx].grid(alpha=0.5)

  ax[idx].axhline(0, color='grey', alpha=0.75, linestyle='--')
  ax[idx].axvline(0, color='grey', alpha=0.75, linestyle='--')

  sc = ax[idx].scatter(X[:, 0], X[:, 1], s=1, c=df['rul'], alpha=0.5, vmin=0, vmax=90, cmap='YlOrRd_r')

  for i, _ in enumerate(cols):
    ax[idx].arrow(0, 0, 10 * pca.components_[0, i], 10 * pca.components_[1, i], color='black', alpha=0.5, head_width=0.1, length_includes_head=True)

  ax[idx].set_xticks(range(-9, 16, 3))
  ax[idx].set_yticks(range(-8, 13, 4))

ax[0].set_title('FD001')
ax[1].set_title('FD002')
ax[2].set_title('FD003')
ax[3].set_title('FD004')
ax[2].set_xlabel(f'Principal comp. 1 (PC1), FD001: {explained_variances[0][0]:.2%}, FD003: {explained_variances[2][0]:.2%}')
ax[3].set_xlabel(f'Principal comp. 1 (PC1), FD002: {explained_variances[1][0]:.2%}, FD004: {explained_variances[3][0]:.2%}')
ax[0].set_ylabel(f'Principal comp. 2 (PC2), {explained_variances[0][1]:.2%}')
ax[1].set_ylabel(f'Principal comp. 2 (PC2), {explained_variances[1][1]:.2%}')
ax[2].set_ylabel(f'Principal comp. 2 (PC2), {explained_variances[2][1]:.2%}')
ax[3].set_ylabel(f'Principal comp. 2 (PC2), {explained_variances[3][1]:.2%}')

plt.subplots_adjust(left=0.075, right=0.99, bottom=0.075, top=0.95)

cbar = fig.colorbar(sc, ax=ax.tolist(), fraction=0.04, pad=0.02, aspect=30)
cbar.set_ticks([0, 20, 70, 90])
cbar.set_ticklabels(['0', '20', '70', '>90'])
cbar.set_label('Remaining useful life', labelpad=-17.5)

plt.savefig('figs/biplots_after.png', dpi=900)
