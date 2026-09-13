# [ADAML] Advanced Data Analysis and Machine Learning
#         Exploratory Analysis
#
# Lappeenranta-Lahti University of Technology
# School of Engineering Science
#
# Name: exploratory_analysis.py
# Aut.: Moriom Akter
#       Matej Nedela
#       Monowarul Sabbir
# Date: 13/09/2026
# Ver.: 1.0

import matplotlib.pyplot as plt
import pandas as pd

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

df_fd001 = pd.read_csv('data/train_FD001.txt', sep=r'\s+', names=names.keys())
df_fd002 = pd.read_csv('data/train_FD002.txt', sep=r'\s+', names=names.keys())
df_fd003 = pd.read_csv('data/train_FD003.txt', sep=r'\s+', names=names.keys())
df_fd004 = pd.read_csv('data/train_FD004.txt', sep=r'\s+', names=names.keys())

df = pd.concat([df_fd001, df_fd002, df_fd003, df_fd004], ignore_index=True)

print(f'FD001 rows: {df_fd001.shape[0]}')
print(f'      cols: {df_fd001.shape[1]}', end='\n\n')
print(f'FD002 rows: {df_fd002.shape[0]}')
print(f'      cols: {df_fd002.shape[1]}', end='\n\n')
print(f'FD003 rows: {df_fd003.shape[0]}')
print(f'      cols: {df_fd003.shape[1]}', end='\n\n')
print(f'FD004 rows: {df_fd004.shape[0]}')
print(f'      cols: {df_fd004.shape[1]}', end='\n\n')

df['altitude'] = df['altitude'].round(0)
df['mach'] = df['mach'].round(2)
df['tra'] = df['tra'].round(0)

print(f'Oper. set.: {df.groupby(["altitude", "mach", "tra"]).ngroups}', end='\n\n')

zero_sigma = []

for name in list(names.keys())[5:]:
  if df.groupby(['altitude', 'mach', 'tra'])[name].std().mean() == 0:
    zero_sigma.append(name)

print(f'Zero sigma: {", ".join(zero_sigma)}', end='\n\n')

hpc_related = []

for (key, value) in names.items():
  if "HPC" in value:
    hpc_related.append(key)

print(f'HPC relat.: {", ".join(hpc_related)}', end='\n\n')

fan_related = []

for (key, value) in names.items():
  if "fan" in value:
    fan_related.append(key)

print(f'Fan relat.: {", ".join(fan_related)}', end='\n\n')

print(f' Symbol    |    Mean |    Std. |    Min. |    Max. ')
print(f'-----------+---------+---------+---------+---------')

for name in list(names.items())[5:]:
  print(f' {name[0]:<9}', end=' | ')
  print(f'{df[name[0]].mean():>7.2f}', end=' | ')
  print(f'{df[name[0]].std():>7.2f}', end=' | ')
  print(f'{df[name[0]].min():>7.2f}', end=' | ')
  print(f'{df[name[0]].max():>7.2f}', end='\n')

# Histogram of unit lifetimes

plt.figure(figsize=(12, 2.25))
plt.axis([100, 400, 0, 20])

plt.hist(df_fd001.groupby('unit_id')['cycle'].max(), bins=[i / 30 * 300 + 100 for i in range(31)])

plt.xlabel('Lifetime (number of cycles)')
plt.ylabel('Number of units')

plt.subplots_adjust(left=0.075, right=0.975, bottom=0.2, top=0.95)
plt.savefig('figs/histogram.svg')

# Total temperature at HPC outlet (T30)

plt.figure(figsize=(6, 2.25))
plt.axis([-10, 310, 1570, 1610])
plt.grid(alpha=0.5)

plt.plot(df_fd001[df_fd001['unit_id'] == 1]['cycle'], df_fd001[df_fd001['unit_id'] == 1]['T30'], label='Unit 1')
plt.plot(df_fd001[df_fd001['unit_id'] == 2]['cycle'], df_fd001[df_fd001['unit_id'] == 2]['T30'], label='Unit 2')

plt.xlabel('Cycle')
plt.ylabel('T30 [°R]')
plt.legend(loc='upper left')

plt.subplots_adjust(left=0.15, right=0.95, bottom=0.2, top=0.95)
plt.savefig('figs/total_temperature.svg')

# Static pressure at HPC outlet (Ps30)

plt.figure(figsize=(6, 2.25))
plt.axis([-10, 310, 46.5, 48.5])
plt.grid(alpha=0.5)

plt.plot(df_fd001[df_fd001['unit_id'] == 1]['cycle'], df_fd001[df_fd001['unit_id'] == 1]['Ps30'], label='Unit 1')
plt.plot(df_fd001[df_fd001['unit_id'] == 2]['cycle'], df_fd001[df_fd001['unit_id'] == 2]['Ps30'], label='Unit 2')

plt.xlabel('Cycle')
plt.ylabel('Ps30 [psia]')
plt.legend(loc='upper left')

plt.subplots_adjust(left=0.15, right=0.95, bottom=0.2, top=0.95)
plt.savefig('figs/static_pressure.svg')
