import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_selection import mutual_info_classif
from sklearn.model_selection import train_test_split, cross_val_score
import glob
import tkinter as tk
from tkinter import filedialog
import sys
import os

# Set style
sns.set_style("whitegrid")
plt.rcParams['figure.figsize'] = (15, 10)

# ====================================================================
# FUNGSI EKSTRAKSI FITUR (HANYA 6 FITUR TD)
# ====================================================================
def extract_emg_features(signal, label, window_size=200, threshold_ssc=5):
    """
    Menghitung 6 fitur Time-Domain (TD) dari data sinyal EMG
    RMS, VAR, MAV, SSC, ZC, WL
    """
    features = []
    N_signal = len(signal)
    
    # Iterasi melalui sinyal dengan windowing (tanpa overlap)
    for i in range(0, N_signal, window_size):
        window = signal[i:i + window_size]
        
        # Hanya proses jika jendela memiliki data yang cukup
        if len(window) < window_size * 0.5:
            continue
            
        # 1. Root Mean Square (RMS)
        rms = np.sqrt(np.mean(window**2))
        
        # 2. Variance (VAR)
        var = np.var(window)
        
        # 3. Mean Absolute Value (MAV)
        mav = np.mean(np.abs(window))
        
        # 4. Slope Sign Change (SSC)
        ssc = np.sum(np.logical_and(
            (window[1:-1] > window[0:-2]) & (window[1:-1] > window[2:]),
            np.abs(window[1:-1] - window[0:-2]) > threshold_ssc
        )) + np.sum(np.logical_and(
            (window[1:-1] < window[0:-2]) & (window[1:-1] < window[2:]),
            np.abs(window[1:-1] - window[0:-2]) > threshold_ssc
        ))
        
        # 5. Zero Crossing (ZC)
        threshold_zc = 0 
        zc = np.sum(np.diff(np.sign(window - threshold_zc)) != 0)
        
        # 6. Waveform Length (WL)
        wl = np.sum(np.abs(np.diff(window)))
        
        features.append({
            'label': label,
            'RMS': rms,
            'VAR': var,
            'MAV': mav,
            'SSC': ssc,
            'ZC': zc,
            'WL': wl
        })
        
    return pd.DataFrame(features)

# ====================================================================
# SCRIPT UTAMA
# ====================================================================

print("="*70)
print("ANALISIS FEATURE IMPORTANCE - 3 GERAKAN TANGAN (GENGGAM, RELAKS, TEKUK)")
print("="*70)

# ===== 1. LOAD DATA (3 FOLDER) =====
print("\n[1] Loading data dari 3 folder...")

root = tk.Tk()
root.withdraw()

gesture_folders = {}
gesture_names = ['GENGGAM', 'RELAKS', 'TEKUK']

for gesture in gesture_names:
    print(f"\n🚨 PILIH FOLDER {gesture}: Pilih folder yang berisi semua file data {gesture}.")
    folder_path = filedialog.askdirectory(title=f"Pilih Folder Data {gesture}")
    if folder_path:
        gesture_folders[gesture.lower()] = folder_path
    else:
        print(f" ❌ Folder {gesture} tidak dipilih. Program dihentikan.")
        sys.exit(1)

raw_data = []

# Load semua data dari 3 folder
for gesture_name, folder_path in gesture_folders.items():
    csv_files = glob.glob(os.path.join(folder_path, '*.csv'))
    print(f"\n✓ Ditemukan {len(csv_files)} file CSV di folder {gesture_name.upper()}")
    
    for i, file in enumerate(csv_files):
        try:
            df = pd.read_csv(file)
            
            # ⚠️ BUANG 10 BARIS PERTAMA (SENSOR SENSITIF)
            if len(df) > 10:
                df = df.iloc[10:].reset_index(drop=True)
                print(f"   ⚠️ Membuang 10 baris pertama dari {os.path.basename(file)}")
            
            df['label'] = gesture_name
            raw_data.append(df)
            print(f"   ✓ Loaded {gesture_name.upper()} File #{i+1}: {os.path.basename(file)} ({len(df)} rows setelah buang 10 baris)")
        except Exception as e:
            print(f" ❌ Gagal memuat file {os.path.basename(file)}. Error: {e}")

if not raw_data:
    print("\n 🛑 FATAL ERROR: Tidak ada data yang berhasil dimuat. Program dihentikan.")
    sys.exit(1)

all_raw_data = pd.concat(raw_data, ignore_index=True)
print(f"\n   Total data mentah: {len(all_raw_data)} rows (setelah buang 10 baris awal per file)")

# ===== 2. EKSTRAKSI FITUR TD (6 FITUR SAJA) =====
print("\n[2] Mengekstrak 6 fitur TD (RMS, VAR, MAV, SSC, ZC, WL)...")

feature_list = []
WINDOW_SIZE = 200 

for df in raw_data:
    if 'Nilai ADC' not in df.columns:
        print(f" ❌ Peringatan: Kolom 'Nilai ADC' tidak ditemukan di salah satu file.")
        continue
        
    signal = df['Nilai ADC'].values
    label = df['label'].iloc[0]
    
    features_df = extract_emg_features(signal, label, window_size=WINDOW_SIZE)
    if not features_df.empty:
        feature_list.append(features_df)

if not feature_list:
    print("\n 🛑 FATAL ERROR: Gagal mengekstrak fitur. Program dihentikan.")
    sys.exit(1)

final_data = pd.concat(feature_list, ignore_index=True)

# Pisahkan fitur dan label
X = final_data.drop(['label'], axis=1)
y = final_data['label']

print(f"\n   Total sampel fitur: {len(X)}")
for gesture in gesture_names:
    count = len(X[y == gesture.lower()])
    print(f"   {gesture} Sampel: {count}")

print(f"\n   Features: {list(X.columns)}")
print(f"   Total features: {len(X.columns)}")

# Encode label (3 kelas)
label_mapping = {'genggam': 0, 'relaks': 1, 'tekuk': 2}
y_encoded = y.map(label_mapping)

# ===== 3. ANALISIS STATISTIK DASAR (3 KELAS) =====
print("\n[3] Analisis Statistik Dasar (3 Kelas)...")

stats_comparison = pd.DataFrame()
for col in X.columns:
    stats_row = {'Feature': col}
    
    for gesture in gesture_names:
        vals = X[y == gesture.lower()][col]
        stats_row[f'{gesture}_Mean'] = vals.mean()
        stats_row[f'{gesture}_Std'] = vals.std()
    
    # Hitung range (max - min) dari ketiga mean
    means = [stats_row[f'{g}_Mean'] for g in gesture_names]
    stats_row['Mean_Range'] = max(means) - min(means)
    
    stats_comparison = pd.concat([stats_comparison, pd.DataFrame([stats_row])], ignore_index=True)

# ===== 4. RANDOM FOREST FEATURE IMPORTANCE =====
print("\n[4] Random Forest Feature Importance (Multi-Class)...")

# Split data
X_train, X_test, y_train, y_test = train_test_split(X, y_encoded, test_size=0.2, random_state=42, stratify=y_encoded)

# Train Random Forest
rf = RandomForestClassifier(n_estimators=100, random_state=42, max_depth=10)
rf.fit(X_train, y_train)

# Get feature importance
rf_importance = pd.DataFrame({
    'Feature': X.columns,
    'RF_Importance': rf.feature_importances_
}).sort_values('RF_Importance', ascending=False)

# Cross-Validation Score
try:
    cv_scores = cross_val_score(rf, X, y_encoded, cv=min(5, len(X) // 10))
    print(f"   Model Cross-Validation Accuracy: {cv_scores.mean():.4f} (±{cv_scores.std():.4f})")
except Exception as e:
    print(f"   ⚠️ Gagal menghitung CV Score. Error: {e}")
    print(f"   Model Test Accuracy: {rf.score(X_test, y_test):.4f}")

# ===== 5. MUTUAL INFORMATION =====
print("\n[5] Mutual Information Score (Multi-Class)...")
mi_scores = mutual_info_classif(X, y_encoded, random_state=42)
mi_importance = pd.DataFrame({'Feature': X.columns, 'MI_Score': mi_scores}).sort_values('MI_Score', ascending=False)

# ===== 6. SEPARATION SCORE (ANOVA F-statistic) =====
print("\n[6] Separation Score (ANOVA F-statistic untuk Multi-Class)...")

from scipy import stats

separation_scores = []
for col in X.columns:
    groups = [X[y == gesture.lower()][col].values for gesture in gesture_names]
    
    # Gunakan ANOVA F-test untuk multi-class
    f_stat, p_value = stats.f_oneway(*groups)
    separation_scores.append(f_stat)

separation_df = pd.DataFrame({
    'Feature': X.columns, 
    'Separation_Score': separation_scores
}).sort_values('Separation_Score', ascending=False)

# ===== 7. COMBINE ALL SCORES =====
print("\n[7] Combining all metrics...")

final_ranking = rf_importance.copy()
final_ranking = final_ranking.merge(mi_importance, on='Feature')
final_ranking = final_ranking.merge(separation_df, on='Feature')
final_ranking = final_ranking.merge(stats_comparison, on='Feature')

def safe_normalize(series):
    min_val = series.min()
    max_val = series.max()
    if max_val == min_val:
        return np.zeros(len(series))
    return (series - min_val) / (max_val - min_val)

final_ranking['RF_Norm'] = safe_normalize(final_ranking['RF_Importance'])
final_ranking['MI_Norm'] = safe_normalize(final_ranking['MI_Score'])
final_ranking['Sep_Norm'] = safe_normalize(final_ranking['Separation_Score'])

final_ranking['Combined_Score'] = (final_ranking['RF_Norm'] + final_ranking['MI_Norm'] + final_ranking['Sep_Norm']) / 3
final_ranking = final_ranking.sort_values('Combined_Score', ascending=False).reset_index(drop=True)

# ===== 8. PRINT RESULTS =====
print("\n" + "="*70)
print("HASIL ANALISIS - RANKING FITUR TERPENTING (6 FITUR)")
print("="*70)

print("\n📊 RANKING BERDASARKAN COMBINED SCORE:")
print("-" * 70)
for idx, row in final_ranking.iterrows():
    print(f"\n🏆 Rank {idx + 1}: {row['Feature']}")
    print(f"   Combined Score    : {row['Combined_Score']:.4f}")
    print(f"   RF Importance     : {row['RF_Importance']:.4f}")
    print(f"   MI Score          : {row['MI_Score']:.4f}")
    print(f"   Separation (ANOVA): {row['Separation_Score']:.4f}")
    print(f"   GENGGAM: {row['GENGGAM_Mean']:.2f} ± {row['GENGGAM_Std']:.2f}")
    print(f"   RELAKS : {row['RELAKS_Mean']:.2f} ± {row['RELAKS_Std']:.2f}")
    print(f"   TEKUK  : {row['TEKUK_Mean']:.2f} ± {row['TEKUK_Std']:.2f}")
    print(f"   Range  : {row['Mean_Range']:.2f}")

# ===== 9. VISUALIZATIONS =====
print("\n[9] Generating visualizations...")

fig, axes = plt.subplots(2, 2, figsize=(16, 12))
fig.suptitle('EMG Feature Importance - 3 Gestures (GENGGAM, RELAKS, TEKUK)', fontsize=16, fontweight='bold')

# Plot 1: Combined Score
axes[0, 0].barh(final_ranking['Feature'], final_ranking['Combined_Score'], color='steelblue')
axes[0, 0].set_xlabel('Combined Score')
axes[0, 0].set_title('All Features - Combined Score')
axes[0, 0].invert_yaxis()

# Plot 2: Random Forest Importance
axes[0, 1].barh(rf_importance['Feature'], rf_importance['RF_Importance'], color='green')
axes[0, 1].set_xlabel('RF Importance')
axes[0, 1].set_title('All Features - Random Forest')
axes[0, 1].invert_yaxis()

# Plot 3: Mutual Information
axes[1, 0].barh(mi_importance['Feature'], mi_importance['MI_Score'], color='orange')
axes[1, 0].set_xlabel('MI Score')
axes[1, 0].set_title('All Features - Mutual Information')
axes[1, 0].invert_yaxis()

# Plot 4: Separation Score
axes[1, 1].barh(separation_df['Feature'], separation_df['Separation_Score'], color='red')
axes[1, 1].set_xlabel('Separation Score (ANOVA F-stat)')
axes[1, 1].set_title('All Features - Separation Score')
axes[1, 1].invert_yaxis()

plt.tight_layout()
plt.savefig('feature_importance_3gestures.png', dpi=300, bbox_inches='tight')
print("   ✓ Saved: feature_importance_3gestures.png")

# Plot 5: Distribution Comparison (All 6 Features)
fig2, axes2 = plt.subplots(2, 3, figsize=(18, 10))
fig2.suptitle('6 Features - Distribution Comparison (3 Gestures)', fontsize=16, fontweight='bold')

axes2 = axes2.flatten()

for idx, feature in enumerate(final_ranking['Feature'].values):
    genggam_vals = X[y == 'genggam'][feature]
    relaks_vals = X[y == 'relaks'][feature]
    tekuk_vals = X[y == 'tekuk'][feature]
    
    axes2[idx].hist(genggam_vals, alpha=0.5, label='Genggam', bins=20, color='blue')
    axes2[idx].hist(relaks_vals, alpha=0.5, label='Relaks', bins=20, color='green')
    axes2[idx].hist(tekuk_vals, alpha=0.5, label='Tekuk', bins=20, color='red')
    axes2[idx].set_title(feature)
    axes2[idx].legend()
    axes2[idx].set_xlabel('Value')
    axes2[idx].set_ylabel('Frequency')

plt.tight_layout()
plt.savefig('feature_distribution_3gestures.png', dpi=300, bbox_inches='tight')
print("   ✓ Saved: feature_distribution_3gestures.png")

# Plot 6: Boxplot Comparison (All 6 Features)
fig3, axes3 = plt.subplots(2, 3, figsize=(18, 10))
fig3.suptitle('6 Features - Boxplot Comparison (3 Gestures)', fontsize=16, fontweight='bold')

axes3 = axes3.flatten()

for idx, feature in enumerate(final_ranking['Feature'].values):
    data_to_plot = [
        X[y == 'genggam'][feature],
        X[y == 'relaks'][feature],
        X[y == 'tekuk'][feature]
    ]
    
    axes3[idx].boxplot(data_to_plot, labels=['Genggam', 'Relaks', 'Tekuk'])
    axes3[idx].set_title(feature)
    axes3[idx].set_ylabel('Value')
    axes3[idx].grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig('feature_boxplot_3gestures.png', dpi=300, bbox_inches='tight')
print("   ✓ Saved: feature_boxplot_3gestures.png")

# ===== 10. SAVE RESULTS TO CSV =====
final_ranking.to_csv('feature_importance_3gestures.csv', index=False)
print("   ✓ Saved: feature_importance_3gestures.csv")

print("\n" + "="*70)
print("✅ ANALISIS SELESAI!")
print("="*70)

print("\n💡 FITUR PALING BERPENGARUH (berdasarkan Combined Score):")
for i, row in final_ranking.iterrows():
    print(f"   {i+1}. {row['Feature']} - Score: {row['Combined_Score']:.4f}")

print("\n📝 INTERPRETASI:")
print("   - Fitur dengan Combined Score tinggi = pembeda terbaik antar gerakan")
print("   - Lihat Mean_Range untuk melihat perbedaan nilai antar gerakan")
print("   - Cek visualisasi (PNG files) untuk melihat distribusi tiap fitur!")

print("\n🎯 REKOMENDASI:")
top_feature = final_ranking.iloc[0]['Feature']
print(f"   Fitur TERBAIK: {top_feature}")
print(f"   Gunakan fitur ini sebagai prioritas utama untuk klasifikasi!")

print("\n" + "="*70)

plt.show()