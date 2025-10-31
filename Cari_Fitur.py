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
# FUNGSI EKSTRAKSI FITUR (TD & FD)
# ====================================================================
def extract_emg_features(signal, label, fs=200, window_size=200, threshold_ssc=5, threshold_wamp=15):
    """
    Menghitung fitur Time-Domain (TD) dan Frequency-Domain (FD)
    dari data sinyal EMG ('Nilai ADC') per jendela (window).
    """
    features = []
    N_signal = len(signal)
    
    # Iterasi melalui sinyal dengan windowing (tanpa overlap)
    for i in range(0, N_signal, window_size):
        window = signal[i:i + window_size]
        
        # Hanya proses jika jendela memiliki data yang cukup (misalnya > 50% window_size)
        if len(window) < window_size * 0.5:
            continue
            
        N = len(window)
        
        # 1. TIME-DOMAIN (TD)
        rms = np.sqrt(np.mean(window**2))
        mav = np.mean(np.abs(window))
        var = np.var(window)
        
        # Zero Crossing (ZC)
        threshold_zc = 0 
        zc = np.sum(np.diff(np.sign(window - threshold_zc)) != 0)
        
        # Slope Sign Change (SSC)
        ssc = np.sum(np.logical_and(
            (window[1:-1] > window[0:-2]) & (window[1:-1] > window[2:]),
            np.abs(window[1:-1] - window[0:-2]) > threshold_ssc
        )) + np.sum(np.logical_and(
            (window[1:-1] < window[0:-2]) & (window[1:-1] < window[2:]),
            np.abs(window[1:-1] - window[0:-2]) > threshold_ssc
        ))
        
        # Waveform Length (WL)
        wl = np.sum(np.abs(np.diff(window)))
        
        # Willison Amplitude (WAMP)
        wamp = np.sum(np.abs(np.diff(window)) >= threshold_wamp)
        
        # 2. FREQUENCY-DOMAIN (FD)
        
        if N > 0:
            fft_values = np.fft.fft(window)
            psd = np.abs(fft_values[:N//2])**2
            f = np.fft.fftfreq(N, 1/fs)[:N//2]
            
            # Mean Frequency (MNF)
            mnf = np.sum(f * psd) / (np.sum(psd) + 1e-10)
            
            # Median Frequency (MDF)
            cummulative_psd = np.cumsum(psd)
            median_power = cummulative_psd[-1] / 2
            mdf_index = np.where(cummulative_psd >= median_power)[0]
            mdf = f[mdf_index[0]] if len(mdf_index) > 0 else 0
        else:
            mnf, mdf = 0, 0
            
        features.append({
            'label': label,
            'RMS': rms,
            'MAV': mav,
            'VAR': var,
            'ZC': zc,
            'SSC': ssc,
            'WL': wl,
            'WAMP': wamp,
            'MNF': mnf,
            'MDF': mdf
        })
        
    return pd.DataFrame(features)

# ====================================================================
# SCRIPT UTAMA
# ====================================================================

print("="*60)
print("ANALISIS FEATURE IMPORTANCE - EMG GENGGAM VS TEKUK (FITUR BARU)")
print("="*60)

# ===== 1. LOAD DATA (MODIFIKASI: Memilih Folder) =====
print("\n[1] Loading data (Menggunakan Folder Explorer)...")

root = tk.Tk()
root.withdraw()

# --- Pilih Folder Genggam ---
print("\n🚨 [PILIH FOLDER GENGGAM]: Pilih folder yang berisi semua file data Genggam.")
genggam_folder = filedialog.askdirectory(title="Pilih Folder Data Genggam (EMG)")

# --- Pilih Folder Tekuk ---
print("\n🚨 [PILIH FOLDER TEKUK]: Pilih folder yang berisi semua file data Tekuk.")
tekuk_folder = filedialog.askdirectory(title="Pilih Folder Data Tekuk (EMG)")

raw_data = []

# --- Load Genggam dari Folder ---
genggam_files = []
if genggam_folder:
    genggam_files = glob.glob(os.path.join(genggam_folder, '*.csv'))

print(f"   ✓ Ditemukan {len(genggam_files)} file CSV di folder Genggam.")
for i, file in enumerate(genggam_files):
    try:
        df = pd.read_csv(file)
        df['label'] = 'genggam'
        raw_data.append(df)
        print(f"   ✓ Loaded Genggam File #{i+1}: {os.path.basename(file)} ({len(df)} rows)")
    except Exception as e:
        print(f" ❌ Gagal memuat file {os.path.basename(file)}. Error: {e}")

# --- Load Tekuk dari Folder ---
tekuk_files = []
if tekuk_folder:
    tekuk_files = glob.glob(os.path.join(tekuk_folder, '*.csv'))
    
print(f"   ✓ Ditemukan {len(tekuk_files)} file CSV di folder Tekuk.")
for i, file in enumerate(tekuk_files):
    try:
        df = pd.read_csv(file)
        df['label'] = 'tekuk'
        raw_data.append(df)
        print(f"   ✓ Loaded Tekuk File #{i+1}: {os.path.basename(file)} ({len(df)} rows)")
    except Exception as e:
        print(f" ❌ Gagal memuat file {os.path.basename(file)}. Error: {e}")

if not raw_data:
    print("\n 🛑 FATAL ERROR: Tidak ada data yang berhasil dimuat. Program dihentikan.")
    sys.exit(1)

all_raw_data = pd.concat(raw_data, ignore_index=True)
print(f"\n   Total data mentah: {len(all_raw_data)} rows")

# ===== 2. EKSTRAKSI FITUR TD dan FD (BERBASIS WINDOWING) =====
print("\n[2] Mengekstrak fitur TD dan FD (per jendela)...")

feature_list = []
# Sesuaikan WINDOW_SIZE jika Anda ingin lebih banyak sampel fitur!
WINDOW_SIZE = 200 
FS = 200          

for df in raw_data:
    if 'Nilai ADC' not in df.columns:
        print(f" ❌ Peringatan: Kolom 'Nilai ADC' tidak ditemukan di salah satu file.")
        continue
        
    signal = df['Nilai ADC'].values
    label = df['label'].iloc[0]
    
    features_df = extract_emg_features(signal, label, fs=FS, window_size=WINDOW_SIZE)
    if not features_df.empty:
        feature_list.append(features_df)

if not feature_list:
    print("\n 🛑 FATAL ERROR: Gagal mengekstrak fitur. Program dihentikan.")
    sys.exit(1)

final_data = pd.concat(feature_list, ignore_index=True)

# Pisahkan fitur dan label
X = final_data.drop(['label'], axis=1)
y = final_data['label']

genggam_count = len(X[y == 'genggam'])
tekuk_count = len(X[y == 'tekuk'])

print(f"   Total baris fitur (sampel): {len(X)}")
print(f"   Genggam Sampel: {genggam_count}")
print(f"   Tekuk Sampel: {tekuk_count}")
print(f"   Features: {list(X.columns)}")
print(f"   Total features: {len(X.columns)}")

if genggam_count < 2 or tekuk_count < 2:
     print("\n ⚠️ PERINGATAN: Jumlah sampel untuk salah satu kelas (< 2) terlalu sedikit. Hasil analisis tidak stabil.")
     MIN_SAMPLES_OK = False
else:
    MIN_SAMPLES_OK = True

# Encode label
y_encoded = y.map({'genggam': 0, 'tekuk': 1})

# ===== 3. ANALISIS STATISTIK DASAR =====
print("\n[3] Analisis Statistik Dasar...")

stats_comparison = pd.DataFrame()
for col in X.columns:
    genggam_vals = X[y == 'genggam'][col]
    tekuk_vals = X[y == 'tekuk'][col]
    
    stats_comparison = pd.concat([stats_comparison, pd.DataFrame({
        'Feature': [col],
        'Genggam_Mean': [genggam_vals.mean()],
        'Genggam_Std': [genggam_vals.std()],
        'Tekuk_Mean': [tekuk_vals.mean()],
        'Tekuk_Std': [tekuk_vals.std()],
        'Mean_Diff': [abs(genggam_vals.mean() - tekuk_vals.mean())],
        'Std_Ratio': [genggam_vals.std() / (tekuk_vals.std() + 1e-10)]
    })], ignore_index=True)

# ===== 4. RANDOM FOREST FEATURE IMPORTANCE =====
print("\n[4] Random Forest Feature Importance...")

if MIN_SAMPLES_OK:
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
    
    # Gunakan Cross-Validation untuk akurasi yang lebih robust
    try:
        cv_scores = cross_val_score(rf, X, y_encoded, cv=min(5, len(X) // 2))
        print(f"   Model Cross-Validation Score: {cv_scores.mean():.4f}")
    except Exception as e:
        print(f"   ⚠️ Peringatan: Gagal menghitung CV Score. Error: {e}")
        print(f"   Model Accuracy (Test Set): {rf.score(X_test, y_test):.4f}")

else:
    # Jika sampel terlalu sedikit, latih dengan semua data dan beri peringatan
    rf = RandomForestClassifier(n_estimators=100, random_state=42, max_depth=10)
    rf.fit(X, y_encoded)
    
    rf_importance = pd.DataFrame({
        'Feature': X.columns,
        'RF_Importance': rf.feature_importances_
    }).sort_values('RF_Importance', ascending=False)
    
    print("   ⚠️ Peringatan: Sampel terlalu sedikit. Dilatih dengan SEMUA data. Akurasi model tidak valid.")


# ===== 5. MUTUAL INFORMATION =====
print("\n[5] Mutual Information Score...")
mi_scores = mutual_info_classif(X, y_encoded, random_state=42)
mi_importance = pd.DataFrame({'Feature': X.columns, 'MI_Score': mi_scores}).sort_values('MI_Score', ascending=False)

# ===== 6. SEPARATION SCORE (Cohen's d) =====
print("\n[6] Separation Score (Cohen's d)...")
separation_scores = []
for col in X.columns:
    genggam_vals = X[y == 'genggam'][col].values
    tekuk_vals = X[y == 'tekuk'][col].values
    
    mean_diff = abs(np.mean(genggam_vals) - np.mean(tekuk_vals))
    n1 = len(genggam_vals)
    n2 = len(tekuk_vals)
    s1_sq = np.var(genggam_vals, ddof=1)
    s2_sq = np.var(tekuk_vals, ddof=1)
    
    if n1 > 1 and n2 > 1:
        # Rumus Pooled Standard Deviation yang benar
        pooled_std = np.sqrt(((n1 - 1) * s1_sq + (n2 - 1) * s2_sq) / (n1 + n2 - 2))
    elif n1 == 1 and n2 >= 2:
        # Jika n1=1, gunakan Std dari kelas n2 (Tekuk) sebagai estimasi pooled std
        pooled_std = np.std(tekuk_vals)
    elif n2 == 1 and n1 >= 2:
        # Jika n2=1, gunakan Std dari kelas n1 (Genggam) sebagai estimasi pooled std
        pooled_std = np.std(genggam_vals)
    else:
        # Fallback jika kedua kelas terlalu sedikit (n<=1)
        pooled_std = np.sqrt((np.std(genggam_vals)**2 + np.std(tekuk_vals)**2) / 2)

    cohen_d = mean_diff / (pooled_std + 1e-10)
    separation_scores.append(cohen_d)

separation_df = pd.DataFrame({'Feature': X.columns, 'Separation_Score': separation_scores}).sort_values('Separation_Score', ascending=False)

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
        return np.zeros(len(series)) # Gunakan 0 jika semua nilainya sama
    return (series - min_val) / (max_val - min_val)

final_ranking['RF_Norm'] = safe_normalize(final_ranking['RF_Importance'])
final_ranking['MI_Norm'] = safe_normalize(final_ranking['MI_Score'])
final_ranking['Sep_Norm'] = safe_normalize(final_ranking['Separation_Score'])

final_ranking['Combined_Score'] = (final_ranking['RF_Norm'] + final_ranking['MI_Norm'] + final_ranking['Sep_Norm']) / 3
final_ranking = final_ranking.sort_values('Combined_Score', ascending=False).reset_index(drop=True)

# ===== 8. PRINT RESULTS & VISUALIZATION (sama seperti sebelumnya) =====
print("\n" + "="*60)
print("HASIL ANALISIS - TOP 10 FITUR TERPENTING")
print("="*60)

print("\n📊 RANKING BERDASARKAN COMBINED SCORE:")
print("-" * 60)
for idx, row in final_ranking.head(10).iterrows():
    print(f"\n🏆 Rank {idx + 1}: {row['Feature']}")
    print(f"   Combined Score    : {row['Combined_Score']:.4f}")
    print(f"   RF Importance     : {row['RF_Importance']:.4f}")
    print(f"   MI Score          : {row['MI_Score']:.4f}")
    print(f"   Separation Score  : {row['Separation_Score']:.4f}")
    print(f"   Genggam: {row['Genggam_Mean']:.4f} \u00B1 {row['Genggam_Std']:.4f}")
    print(f"   Tekuk  : {row['Tekuk_Mean']:.4f} \u00B1 {row['Tekuk_Std']:.4f}")
    print(f"   Diff   : {row['Mean_Diff']:.4f}")

# [8] Generating visualizations...
print("\n[8] Generating visualizations...")

if len(final_ranking) >= 1:
    fig, axes = plt.subplots(2, 2, figsize=(16, 12))
    fig.suptitle('EMG Feature Importance Analysis - Genggam vs Tekuk', fontsize=16, fontweight='bold')

    top_features = final_ranking.head(10)
    
    # Plot 1: Combined Score
    axes[0, 0].barh(top_features['Feature'], top_features['Combined_Score'], color='steelblue')
    axes[0, 0].set_xlabel('Combined Score')
    axes[0, 0].set_title('Top Features - Combined Score')
    axes[0, 0].invert_yaxis()

    # Plot 2: Random Forest Importance
    rf_plot = rf_importance.head(10)
    axes[0, 1].barh(rf_plot['Feature'], rf_plot['RF_Importance'], color='green')
    axes[0, 1].set_xlabel('RF Importance')
    axes[0, 1].set_title('Top Features - Random Forest')
    axes[0, 1].invert_yaxis()

    # Plot 3: Mutual Information
    mi_plot = mi_importance.head(10)
    axes[1, 0].barh(mi_plot['Feature'], mi_plot['MI_Score'], color='orange')
    axes[1, 0].set_xlabel('MI Score')
    axes[1, 0].set_title('Top Features - Mutual Information')
    axes[1, 0].invert_yaxis()

    # Plot 4: Separation Score
    sep_plot = separation_df.head(10)
    axes[1, 1].barh(sep_plot['Feature'], sep_plot['Separation_Score'], color='red')
    axes[1, 1].set_xlabel("Separation Score (Cohen's d)")
    axes[1, 1].set_title('Top Features - Separation Score')
    axes[1, 1].invert_yaxis()

    plt.tight_layout()
    plt.savefig('feature_importance_analysis_new_features.png', dpi=300, bbox_inches='tight')
    print("   ✓ Saved: feature_importance_analysis_new_features.png")

    # Plot 5: Comparison Plot untuk Top 6 Features
    fig2, axes2 = plt.subplots(2, 3, figsize=(18, 10))
    fig2.suptitle('Top 6 Features - Distribution Comparison', fontsize=16, fontweight='bold')

    top_n = min(6, len(final_ranking))
    top_features_to_plot = final_ranking.head(top_n)['Feature'].values
    axes2 = axes2.flatten()

    for idx, feature in enumerate(top_features_to_plot):
        genggam_vals = X[y == 'genggam'][feature]
        tekuk_vals = X[y == 'tekuk'][feature]
        
        bins = min(30, len(genggam_vals.unique()) // 2, len(tekuk_vals.unique()) // 2)
        bins = max(5, bins)
        
        axes2[idx].hist(genggam_vals, alpha=0.6, label='Genggam', bins=bins, color='blue')
        axes2[idx].hist(tekuk_vals, alpha=0.6, label='Tekuk', bins=bins, color='red')
        axes2[idx].set_title(feature)
        axes2[idx].legend()
        axes2[idx].set_xlabel('Value')
        axes2[idx].set_ylabel('Frequency')

    for i in range(top_n, 6):
        fig2.delaxes(axes2[i])

    plt.tight_layout()
    plt.savefig('feature_distribution_comparison_new_features.png', dpi=300, bbox_inches='tight')
    print("   ✓ Saved: feature_distribution_comparison_new_features.png")
else:
    print(" ⚠️ Peringatan: Data tidak memadai untuk menghasilkan visualisasi.")


# ===== 10. SAVE RESULTS TO CSV =====
final_ranking.to_csv('feature_importance_results_new_features.csv', index=False)
print("   ✓ Saved: feature_importance_results_new_features.csv")

print("\n" + "="*60)
print("✅ ANALISIS SELESAI!")
print("="*60)
print("\n💡 REKOMENDASI:")
print("\nFitur-fitur yang PALING BERPENGARUH untuk klasifikasi:")
top_reco = final_ranking.head(5)['Feature'].values
for i, feature in enumerate(top_reco, 1):
    print(f"   {i}. {feature}")

print("\n📝 Gunakan fitur-fitur di atas untuk model klasifikasi kamu!")
print("📊 Cek hasil visualisasi di file PNG yang sudah di-generate!")
print("\n" + "="*60)

plt.show()