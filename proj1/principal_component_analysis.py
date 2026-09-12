# [ADAML] Advanced Data Analysis and Machine Learning
#         Principal Component Analysis
#
# Lappeenranta-Lahti University of Technology
# School of Engineering Science
#
# Name: principal_component_analysis.py
# Aut.: Moriom Akter
#       Matej Nedela
#       Monowarul Sabbir
# Date: 13/09/2026
# Ver.: 1.0

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler

np.random.seed(42)

names = {
  "unit_id":   "Unit number",
  "cycle":     "Time [cycle]",
  "altitude":  "Altitude [K ft]",
  "mach":      "Mach number",
  "tra":       "Throttle resolver angle",
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

subsets = ['FD001', 'FD002', 'FD003', 'FD004']
results = {}

for subset in subsets:
  df = pd.read_csv(f'data/train_{subset}.txt', sep=r'\s+', names=names.keys())

  X = df.iloc[:, 5:26].values

  # Standardize PCA
  scaler = StandardScaler()
  X_scaled = scaler.fit_transform(X)

  # Apply PCA
  pca = PCA()
  X_pca = pca.fit_transform(X_scaled)

  results[subset] = {
    'explained_variance': pca.explained_variance_ratio_,
    'cumulative_variance': np.cumsum(pca.explained_variance_ratio_),
    'loadings': pca.components_[:2, :],
    'X_pca': X_pca
  }

  loadings_matrix = pca.components_[:2, :]

  print(f'{'-' * 33} {subset} {'-' * 33}', end='\n\n')
  print(f'Cumulative explained variance =\n{results[subset]["cumulative_variance"].round(4)}', end='\n\n')
  print(f'PC1 =\n{loadings_matrix[0, :].round(4)}', end='\n\n')
  print(f'PC2 =\n{loadings_matrix[1, :].round(4)}', end='\n\n')

# Biplots

fig, ax = plt.subplots(2, 2, figsize=(12, 5.5))
ax = ax.flatten()

for idx, subset in enumerate(subsets):
  X_pca = results[subset]['X_pca']
  loadings = results[subset]['loadings']

  ax[idx].axis([-7.5, 12.5, -4, 8])
  ax[idx].grid(alpha=0.5)

  ax[idx].axhline(0, color='grey', alpha=0.75, linestyle='--')
  ax[idx].axvline(0, color='grey', alpha=0.75, linestyle='--')

  # randomly sample 1000 points for better visualization in the biplot
  sample_idx = np.random.choice(X_pca.shape[0], min(1000, X_pca.shape[0]), replace=False)
  ax[idx].scatter(X_pca[sample_idx, 0], X_pca[sample_idx, 1], color='blue', alpha=0.25)

  for i, name in enumerate(list(names.keys())[5:]):
    ax[idx].arrow(0, 0, 10 * loadings[0, i], 10 * loadings[1, i], color='red', alpha=0.75, head_width=0.1, length_includes_head=True)

  ax[idx].set_title(subset)
  if (idx >= 2): ax[idx].set_xlabel('Principal component 1 (PC1)')
  ax[idx].set_ylabel('Principal component 2 (PC2)')

plt.subplots_adjust(left=0.075, right=0.975, bottom=0.075, top=0.95)
plt.savefig('figs/biplots.svg')

# Loading plots

fig, ax = plt.subplots(2, 2, figsize=(12, 5.5))
ax = ax.flatten()

for idx, file in enumerate(subsets):
  loadings = results[file]['loadings']

  ax[idx].axis([-0.4, 0.4, -0.4, 0.8])
  ax[idx].grid(alpha=0.5)

  ax[idx].axhline(0, color='grey', alpha=0.75, linestyle='--')
  ax[idx].axvline(0, color='grey', alpha=0.75, linestyle='--')

  for i, name in enumerate(list(names.keys())[5:]):
    ax[idx].arrow(0, 0, loadings[0, i], loadings[1, i], color='red', alpha=0.75, head_width=0.01, length_includes_head=True)
    ax[idx].text(1.1 * loadings[0, i], 1.1 * loadings[1, i], name, color='green', alpha=0.5, fontsize=7, ha='center', va='center')

  ax[idx].set_title(file)
  if (idx >= 2): ax[idx].set_xlabel('Principal component 1 (PC1)')
  ax[idx].set_ylabel('Principal component 2 (PC2)')

plt.subplots_adjust(left=0.075, right=0.975, bottom=0.075, top=0.95)
plt.savefig('figs/loading_plots.svg')
