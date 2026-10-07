# ==============================================================================
# PROJECT: NASA Turbofan Engine Remaining Useful Life (RUL) Prediction
# Feature Selection, VIP Plots, Heatmaps & using Linear PLS
# ==============================================================================

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib.patches import Rectangle
from sklearn.preprocessing import StandardScaler
from sklearn.cross_decomposition import PLSRegression
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.model_selection import KFold
from sklearn.model_selection import GroupKFold

# ==============================================================================
# 1: Data loading & VIP-based feature selection (Threshold >= 0.5)
# ==============================================================================

train_files = ['data/train_FD001.txt', 
               'data/train_FD002.txt', 
               'data/train_FD003.txt', 
               'data/train_FD004.txt']

sensor_names = [
    "T2", "T24", "T30", "T50", "P2", "P15", "P30", 
    "Nf", "Nc", "epr", "Ps30", "phi", "NRf", "NRc", 
    "BPR", "farB", "htBleed", "Nf_dmd", "PCNfR_dmd", "W31", "W32"
]

dataset_ids = ["FD001", "FD002", "FD003", "FD004"]

def compute_train_rul(df, max_life=125):
    rul_list = []
    for engine_id, group in df.groupby(0):
        max_cycle = group[1].max()
        ruls = max_cycle - group[1].values
        ruls = np.where(ruls > max_life, max_life, ruls)
        rul_list.extend(ruls)
    return np.array(rul_list)

def calculate_pls_vip(model):
    t = model.x_scores_
    w = model.x_weights_
    q = model.y_loadings_
    p, h = w.shape
    vips = np.zeros(p)
    ssy = np.array([q[0, j]**2 * np.dot(t[:, j], t[:, j]) for j in range(h)])
    total_ssy = np.sum(ssy)
    if total_ssy == 0:
        return np.ones(p)
    for i in range(p):
        weight = np.sum([(w[i, j] / np.linalg.norm(w[:, j]))**2 * ssy[j] for j in range(h)])
        vips[i] = np.sqrt(p * weight / total_ssy)
    return vips

print("Calculating VIP Scores for each Dataset...")
dataset_vips = {}
all_vips = []

for file, ds_id in zip(train_files, dataset_ids):
    df = pd.read_csv(file, header=None, sep=r'\s+')
    X = df.iloc[:, 5:26].values
    y = compute_train_rul(df)
    
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    
    pls_temp = PLSRegression(n_components=5)
    pls_temp.fit(X_scaled, y)
    vips = calculate_pls_vip(pls_temp)
    
    dataset_vips[ds_id] = vips
    all_vips.append(vips)

mean_vips = np.mean(all_vips, axis=0)
selected_sensor_indices = np.where(mean_vips >= 0.5)[0]
selected_sensor_names = [sensor_names[i] for i in selected_sensor_indices]

print(f"\n--- Selected Important Sensors (Global Mean VIP >= 0.5) ---")
print(selected_sensor_names)
print(f"Total Selected Sensors: {len(selected_sensor_names)} out of {len(sensor_names)}\n")


# ==============================================================================
# 2: VIP score plotting
# ==============================================================================

fig, axes = plt.subplots(2, 2, figsize=(14, 10))
axes = axes.flatten()

for idx, ds_id in enumerate(dataset_ids):
    vips = dataset_vips[ds_id]
    x_pos = np.arange(len(sensor_names))
    colors = ['green' if v >= 0.5 else 'coral' for v in vips]
    
    axes[idx].bar(x_pos, vips, color=colors, alpha=0.7, edgecolor='k')
    axes[idx].axhline(y=0.5, color='blue', linestyle='--', linewidth=1.5, label='Threshold (VIP = 0.5)')
    axes[idx].set_xticks(x_pos)
    axes[idx].set_xticklabels(sensor_names, rotation=45, fontsize=8)
    axes[idx].set_title(f"VIP Scores - {ds_id}", fontweight="bold")
    axes[idx].set_ylabel("VIP Score", fontweight="bold")
    axes[idx].grid(axis='y', linestyle='--', alpha=0.5)
    axes[idx].legend(loc="upper right", fontsize=8)

plt.suptitle("Variable Importance in Projection (VIP) across All Datasets", fontsize=14, fontweight="bold")
plt.tight_layout(rect=[0, 0, 1, 0.95])
plt.show()


# ==============================================================================
# 3: Correlation Heatmaps for Selected Sensors (VIP >= 0.5)
# ==============================================================================

fig, axes = plt.subplots(2, 2, figsize=(14, 11))
axes = axes.flatten()

for idx, (file, ds_id) in enumerate(zip(train_files, dataset_ids)):
    df = pd.read_csv(file, header=None, sep=r'\s+')
    
    ds_vips = dataset_vips[ds_id]
    ds_selected_indices = np.where(ds_vips >= 0.5)[0]
    
    X_filtered = df.iloc[:, 5:26].values[:, ds_selected_indices]
    sensor_df = pd.DataFrame(X_filtered)
    
    num_cols = [str(i+1) for i in range(sensor_df.shape[1])]
    sensor_df.columns = num_cols
    corr_matrix = sensor_df.corr()
    
    sns.heatmap(corr_matrix, ax=axes[idx], annot=False, cmap='Blues', vmin=-1, vmax=1, 
                linewidths=0.5, linecolor='white', cbar=False,
                xticklabels=num_cols, yticklabels=num_cols)
    
    axes[idx].set_title(f"Correlation Matrix - {ds_id} ({len(num_cols)} Features)", fontweight="bold")
    axes[idx].set_xticklabels(num_cols, rotation=0, fontsize=9)
    axes[idx].set_yticklabels(num_cols, rotation=0, fontsize=9)

# Global Variable Mapping Legend
global_legend_text = "Global Variables Legend:\n" + "\n".join([f"{i+1}: {name}" for i, name in enumerate(sensor_names)])
fig.text(0.82, 0.91, global_legend_text, fontsize=7.5, verticalalignment='top', 
         bbox=dict(boxstyle='round', facecolor='white', edgecolor='gray', alpha=0.9))

# Colorbar legend for heatmap 
cbar_ax = fig.add_axes([0.86, 0.12, 0.02, 0.30])
sm = plt.cm.ScalarMappable(cmap='Blues', norm=plt.Normalize(vmin=-1, vmax=1))
sm.set_array([])
cbar = fig.colorbar(sm, cax=cbar_ax, orientation='vertical')
cbar.set_label('Correlation Coefficient', fontsize=8, fontweight='bold')

plt.suptitle("Sensor Correlation Matrices (Numbered Features)", fontsize=14, fontweight="bold")
plt.tight_layout(rect=[0, 0, 0.80, 0.95])
plt.show()


# ==============================================================================
# 4: Linear PLS Modeling & LV Performance with R2 and Q2
# ==============================================================================

test_files = ['data/test_FD001.txt', 'data/test_FD002.txt', 'data/test_FD003.txt', 'data/test_FD004.txt']
rul_files = ['data/RUL_FD001.txt', 'data/RUL_FD002.txt', 'data/RUL_FD003.txt', 'data/RUL_FD004.txt']

predictions = {}
max_lv_allowed = len(selected_sensor_indices)
optimal_lvs = {"FD001": min(4, max_lv_allowed), "FD002": min(10, max_lv_allowed), 
               "FD003": min(4, max_lv_allowed), "FD004": min(12, max_lv_allowed)}
lv_range = list(range(1, 15)) # 14 LVs

lv_performance_r2 = {ds: [] for ds in dataset_ids}
lv_performance_q2 = {ds: [] for ds in dataset_ids}

for train_file, test_file, rul_file, ds_id in zip(train_files, test_files, rul_files, dataset_ids):
    print(f"\n================ Running PLS Modeling for Dataset: {ds_id} ================")

    train_data = pd.read_csv(train_file, sep=r"\s+", header=None)
    test_data = pd.read_csv(test_file, sep=r"\s+", header=None)
    rul_data = pd.read_csv(rul_file, sep=r"\s+", header=None)

    X_train = train_data.iloc[:, 5:26].values[:, selected_sensor_indices]
    y_train = compute_train_rul(train_data)
    y_test = rul_data.values.ravel()

    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)

    X_test_last = []
    for engine_id, group in test_data.groupby(0):
        last_row = group.iloc[-1, 5:26].values[selected_sensor_indices]
        X_test_last.append(last_row)
    X_test_last = np.array(X_test_last)
    X_test_last_scaled = scaler.transform(X_test_last)

    train_engine_ids = train_data.iloc[:, 0].values
    gkf = GroupKFold(n_splits=5)
    
    for lvs in lv_range:
        q2_scores = []
        
        for train_idx, val_idx in gkf.split(X_train_scaled, y_train, groups=train_engine_ids):
            X_tr, X_val = X_train_scaled[train_idx], X_train_scaled[val_idx]
            y_tr, y_val = y_train[train_idx], y_train[val_idx]
            
            cv_pls = PLSRegression(n_components=lvs)
            cv_pls.fit(X_tr, y_tr)
            y_val_pred = cv_pls.predict(X_val).ravel()
            q2_scores.append(r2_score(y_val, y_val_pred))
        
        lv_performance_q2[ds_id].append(np.mean(q2_scores))

        pls_temp = PLSRegression(n_components=lvs)
        pls_temp.fit(X_train_scaled, y_train)
        y_p_temp = pls_temp.predict(X_test_last_scaled).ravel()
        lv_performance_r2[ds_id].append(r2_score(y_test, y_p_temp))

    selected_lv = optimal_lvs[ds_id]
    print(f"[{ds_id}] Selected Latent Variables (LV) = {selected_lv}")
    
    pls_model = PLSRegression(n_components=selected_lv)
    pls_model.fit(X_train_scaled, y_train)

    y_pred = pls_model.predict(X_test_last_scaled).ravel()
    predictions[ds_id] = {"y_test": y_test, "y_pred": y_pred}

    rmse = np.sqrt(mean_squared_error(y_test, y_pred))
    r2 = r2_score(y_test, y_pred)

    print(f"[{ds_id}] Model Test RMSE: {rmse:.4f}")
    print(f"[{ds_id}] Model Test R2 Score: {r2:.4f}")

# ==============================================================================
# 6: R2 heatmap with chosen LV indication
# ==============================================================================

df_r2 = pd.DataFrame(lv_performance_r2, index=lv_range).T
df_r2.columns = [str(col) for col in lv_range]

plt.figure(figsize=(12, 6))
ax_r2 = sns.heatmap(df_r2, annot=True, fmt=".4f", cmap='Blues', vmin=0, vmax=1, 
                    linewidths=0.5, linecolor='white', cbar=True, annot_kws={"size": 9})

# Highlight optimal LV cells with red border boxes
for row_idx, ds_id in enumerate(dataset_ids):
    opt_lv = optimal_lvs[ds_id]
    if opt_lv in lv_range:
        col_idx = lv_range.index(opt_lv)
        ax_r2.add_patch(Rectangle((col_idx, row_idx), 1, 1, fill=False, edgecolor='red', lw=3))

plt.title("Test $R^2$ Performance across Datasets and Latent Variables (Red box = Chosen LV)", fontsize=13, fontweight="bold")
plt.xlabel("No. components in the model (Latent Variables)", fontweight="bold", fontsize=11)
plt.ylabel("Datasets", fontweight="bold", fontsize=11)
plt.tight_layout()
plt.show()


# ==============================================================================
# 7: Q2 heatmap with chosen LV indication
# ==============================================================================

df_q2 = pd.DataFrame(lv_performance_q2, index=lv_range).T
df_q2.columns = [str(col) for col in lv_range]

plt.figure(figsize=(12, 6))
ax_q2 = sns.heatmap(df_q2, annot=True, fmt=".4f", cmap='Blues', vmin=0, vmax=1, 
                    linewidths=0.5, linecolor='white', cbar=True, annot_kws={"size": 9})

# Highlight optimal LV cells with red border boxes
for row_idx, ds_id in enumerate(dataset_ids):
    opt_lv = optimal_lvs[ds_id]
    if opt_lv in lv_range:
        col_idx = lv_range.index(opt_lv)
        ax_q2.add_patch(Rectangle((col_idx, row_idx), 1, 1, fill=False, edgecolor='red', lw=3))

plt.title("Cross-Validation $Q^2$ Performance across Datasets and Latent Variables (Red box = Chosen LV)", fontsize=13, fontweight="bold")
plt.xlabel("No. components in the model (Latent Variables)", fontweight="bold", fontsize=11)
plt.ylabel("Datasets", fontweight="bold", fontsize=11)
plt.tight_layout()
plt.show()


# ==============================================================================
# 8: Predicted VS Observed RUL plotting
# ==============================================================================

fig, axes = plt.subplots(2, 2, figsize=(12, 10))
axes = axes.flatten()

for idx, ds_id in enumerate(dataset_ids):
    y_test_true = predictions[ds_id]["y_test"]
    y_pred_val = predictions[ds_id]["y_pred"]
    
    r2_val = r2_score(y_test_true, y_pred_val)
    rmse_val = np.sqrt(mean_squared_error(y_test_true, y_pred_val))
    
    axes[idx].scatter(y_test_true, y_pred_val, color='blue', alpha=0.7, edgecolors='k', label='Predictions')
    axes[idx].plot([y_test_true.min(), y_test_true.max()], [y_test_true.min(), y_test_true.max()], 'r--', lw=2, label='Ideal Fit')
    
    axes[idx].set_title(f"Predicted vs True RUL - {ds_id}\n($R^2$: {r2_val:.2f}, RMSE: {rmse_val:.1f})", fontweight="bold")
    axes[idx].set_xlabel("True RUL", fontweight="bold")
    axes[idx].set_ylabel("Predicted RUL", fontweight="bold")
    axes[idx].grid(True, alpha=0.3)
    axes[idx].legend(loc="upper left", fontsize=9)

plt.suptitle("Predicted vs Observed RUL across All Datasets", fontsize=14, fontweight="bold")
plt.tight_layout(rect=[0, 0, 1, 0.95])
plt.show()