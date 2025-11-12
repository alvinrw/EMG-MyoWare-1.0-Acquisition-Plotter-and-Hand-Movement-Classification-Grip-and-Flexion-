import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_selection import mutual_info_classif
import glob
import os
from scipy import stats

# ====================================================================
# FUNGSI EKSTRAKSI FITUR
# ====================================================================
def extract_emg_features(signal, label, window_size=200):
    """Ekstrak 6 fitur TD: RMS, VAR, MAV, SSC, ZC, WL"""
    features = []
    N_signal = len(signal)
    # Ubah threshold_ssc menjadi nilai yang lebih sesuai, 
    # misalnya berdasarkan rata-rata atau standar deviasi sinyal, 
    # atau biarkan 0 untuk uji coba jika threshold 5 menyebabkan SSC selalu 0.
    # threshold_ssc = 5 
    
    # Untuk menghindari SSC selalu 0 jika sinyal mentah terlalu kecil, kita set 0.
    # Namun, 5 adalah nilai umum. Kita pertahankan 5.
    threshold_ssc = 5 

    for i in range(0, N_signal, window_size):
        window = signal[i:i + window_size]
        
        if len(window) < window_size * 0.5:
            continue
            
        rms = np.sqrt(np.mean(window**2))
        var = np.var(window)
        mav = np.mean(np.abs(window))
        
        # Slope Sign Changes (SSC)
        diff_window = np.diff(window)
        
        ssc = np.sum(np.logical_and(
            (window[1:-1] > window[0:-2]) & (window[1:-1] > window[2:]),
            np.abs(window[1:-1] - window[0:-2]) > threshold_ssc
        )) + np.sum(np.logical_and(
            (window[1:-1] < window[0:-2]) & (window[1:-1] < window[2:]),
            np.abs(window[1:-1] - window[0:-2]) > threshold_ssc
        ))
        
        # Zero Crossing (ZC)
        zc = np.sum(np.diff(np.sign(window)) != 0)
        
        # Waveform Length (WL)
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
# MAIN
# ====================================================================

print("="*60)
print("ANALISIS FITUR TERBAIK")
print("="*60)

# Input folder paths
print("\nMasukkan path folder:")
gesture_folders = {}
gesture_names = ['GENGGAM', 'RELAKS', 'TEKUK']

for gesture in gesture_names:
    path = input(f"{gesture}: ").strip().replace('"', '').replace("'", '')
    if not os.path.exists(path):
        print(f"Folder {gesture} tidak ditemukan!")
        exit()
    gesture_folders[gesture.lower()] = path

# Load dan ekstrak fitur
print("\nProses data...")
feature_list = []

for gesture_name, folder_path in gesture_folders.items():
    csv_files = glob.glob(os.path.join(folder_path, '*.csv'))
    
    for file in csv_files:
        try:
            df = pd.read_csv(file)
            # Perhatikan: Data yang terlalu singkat bisa menyebabkan masalah. 
            # Pastikan window_size (200) lebih kecil dari jumlah baris setelah slicing.
            if len(df) > 10:
                df = df.iloc[10:].reset_index(drop=True)
            
            if 'Nilai ADC' not in df.columns:
                continue
            
            signal = df['Nilai ADC'].values
            features_df = extract_emg_features(signal, gesture_name, window_size=200)
            
            if not features_df.empty:
                feature_list.append(features_df)
        except Exception as e:
            # Opsional: Tampilkan error untuk debugging file mana yang bermasalah
            # print(f"Error memproses file {file}: {e}") 
            continue

if not feature_list:
    print("Tidak ada data yang valid!")
    exit()

final_data = pd.concat(feature_list, ignore_index=True)
X = final_data.drop(['label'], axis=1)
y = final_data['label']

# Encode label
label_mapping = {'genggam': 0, 'relaks': 1, 'tekuk': 2}
y_encoded = y.map(label_mapping)

# Pastikan data tidak kosong sebelum menghitung
if X.empty or len(y_encoded.dropna()) == 0:
    print("Data fitur kosong atau label tidak valid setelah pemrosesan.")
    exit()

# Hitung 3 metode
# 1. Random Forest Importance
rf = RandomForestClassifier(n_estimators=100, random_state=42, max_depth=10)
rf.fit(X, y_encoded)
rf_importance = pd.DataFrame({'Feature': X.columns, 'Score': rf.feature_importances_})

# 2. Mutual Information
mi_scores = mutual_info_classif(X, y_encoded, random_state=42)
mi_importance = pd.DataFrame({'Feature': X.columns, 'Score': mi_scores})

# 3. F-ANOVA (Separation Score)
separation_scores = []
for col in X.columns:
    # Hanya ambil grup yang ada di data
    groups = [X[y == g.lower()][col].values for g in gesture_names if g.lower() in y.unique()]
    
    if len(groups) >= 2: # F-ANOVA memerlukan minimal 2 grup
        f_stat, _ = stats.f_oneway(*groups)
        separation_scores.append(f_stat)
    else:
        # Jika hanya ada 1 atau 0 grup, set score ke 0
        separation_scores.append(0.0) 
        
separation_df = pd.DataFrame({'Feature': X.columns, 'Score': separation_scores})

# Combine
final_ranking = rf_importance.merge(mi_importance, on='Feature', suffixes=('_RF', '_MI'))
final_ranking = final_ranking.merge(separation_df, on='Feature')
final_ranking.columns = ['Feature', 'RF_Score', 'MI_Score', 'Sep_Score']

# === SOLUSI UNTUK ERROR NaN: Mengisi NaN dengan 0 ===
final_ranking[['RF_Score', 'MI_Score', 'Sep_Score']] = final_ranking[['RF_Score', 'MI_Score', 'Sep_Score']].fillna(0)
# ======================================================

def safe_normalize(series):
    """Fungsi normalisasi yang aman dari pembagian nol"""
    min_val = series.min()
    max_val = series.max()
    if max_val == min_val:
        return np.zeros(len(series))
    return (series - min_val) / (max_val - min_val)

final_ranking['RF_Norm'] = safe_normalize(final_ranking['RF_Score'])
final_ranking['MI_Norm'] = safe_normalize(final_ranking['MI_Score'])
final_ranking['Sep_Norm'] = safe_normalize(final_ranking['Sep_Score'])

final_ranking['Final_Score'] = (
    final_ranking['RF_Norm'] + 
    final_ranking['MI_Norm'] + 
    final_ranking['Sep_Norm']
) / 3

final_ranking = final_ranking.sort_values('Final_Score', ascending=False).reset_index(drop=True)

# Print hasil
print("\n" + "="*60)
print("RANKING FITUR TERBAIK:")
print("="*60)
for idx, row in final_ranking.iterrows():
    # Pastikan Final_Score bukan NaN sebelum dikonversi
    final_score = row['Final_Score']
    if pd.isna(final_score):
        final_score = 0.0 # Jika masih ada NaN (seharusnya tidak terjadi setelah fillna)
        
    # Baris ini sekarang aman karena Final_Score sudah pasti bukan NaN
    bar = "█" * int(final_score * 50) 
    print(f"{idx+1}. {row['Feature']:6s} [{bar:<50}] {final_score:.3f}")

print("\n" + "="*60)
print(f"FITUR TERBAIK: {final_ranking.iloc[0]['Feature']}")
print("="*60)

# Visualisasi simple
fig, ax = plt.subplots(figsize=(10, 6))
colors = ['#2E86AB', '#A23B72', '#F18F01', '#C73E1D', '#6A994E', '#BC4B51']
bars = ax.barh(final_ranking['Feature'], final_ranking['Final_Score'], color=colors[:len(final_ranking)])

ax.set_xlabel('Score', fontsize=12)
ax.set_title('Ranking Fitur Terbaik untuk Klasifikasi', fontsize=14, fontweight='bold')
ax.invert_yaxis()

# Tambah nilai di ujung bar
for i, (bar, score) in enumerate(zip(bars, final_ranking['Final_Score'])):
    ax.text(score + 0.01, i, f'{score:.3f}', va='center', fontsize=10)

plt.tight_layout()
plt.savefig('feature_ranking.png', dpi=300, bbox_inches='tight')
print("\nSaved: feature_ranking.png")

# Save CSV
final_ranking[['Feature', 'Final_Score']].to_csv('feature_ranking.csv', index=False)
print("Saved: feature_ranking.csv")

plt.show()