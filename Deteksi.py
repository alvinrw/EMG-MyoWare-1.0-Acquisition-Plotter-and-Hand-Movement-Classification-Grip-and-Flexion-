import pandas as pd
import numpy as np
import os
from collections import Counter

# ====================================================================
# KONFIGURASI THRESHOLD
# ====================================================================
VAR_RELAKS_MAX = 50000
VAR_TEKUK_MAX = 500000

# Parameter windowing
WINDOW_SIZE = 50
MIN_WINDOW_PERCENT = 0.8

# ====================================================================
# FUNGSI EKSTRAKSI FITUR
# ====================================================================
def extract_var(signal):
    """Menghitung Variance"""
    if len(signal) < 2:
        return 0
    return np.var(signal)

# ====================================================================
# FUNGSI KLASIFIKASI
# ====================================================================
def classify_emg(var):
    """Klasifikasi berdasarkan VAR"""
    if var < VAR_RELAKS_MAX:
        return "RELAKS"
    elif var < VAR_TEKUK_MAX:
        return "TEKUK"
    else:
        return "GENGGAM"

# ====================================================================
# PROSES FILE CSV
# ====================================================================
def process_csv():
    """Load CSV, ekstrak VAR, dan prediksi"""
    
    print("\n" + "="*60)
    print("KLASIFIKASI EMG - 3 GERAKAN TANGAN")
    print("="*60)
    
    # Input path
    file_path = input("\nMasukkan path file CSV: ").strip()
    
    # Hapus tanda kutip jika ada (copy paste dari Windows Explorer)
    file_path = file_path.replace('"', '').replace("'", '')
    
    if not file_path or not os.path.exists(file_path):
        print("File tidak ditemukan!")
        return
    
    # Load data
    print(f"\nFile: {os.path.basename(file_path)}")
    
    try:
        df = pd.read_csv(file_path)
        if 'Nilai ADC' not in df.columns:
            print("Error: Kolom 'Nilai ADC' tidak ditemukan!")
            return
        
        signal = df['Nilai ADC'].values
        
        # Buang 10 baris pertama
        if len(signal) > 10:
            signal = signal[10:]
        
        print(f"Total samples: {len(signal)}")
        
    except Exception as e:
        print(f"Error: {e}")
        return
    
    # Proses per window
    print(f"\nProcessing (window size: {WINDOW_SIZE})...")
    print("-" * 60)
    
    results = []
    
    for i in range(0, len(signal), WINDOW_SIZE):
        window = signal[i:i + WINDOW_SIZE]
        
        if len(window) < WINDOW_SIZE * MIN_WINDOW_PERCENT:
            continue
        
        var = extract_var(window)
        prediction = classify_emg(var)
        
        results.append({
            'prediction': prediction,
            'var': var
        })
        
        print(f"Window {len(results):2d}: {prediction:8s} | VAR = {var:12,.2f}")
    
    print("-" * 60)
    
    # Ringkasan
    if results:
        print(f"\nTotal windows: {len(results)}")
        
        print("\nHasil Prediksi:")
        for pred_type in ['RELAKS', 'TEKUK', 'GENGGAM']:
            count = sum(1 for r in results if r['prediction'] == pred_type)
            percentage = (count / len(results)) * 100
            print(f"  {pred_type:8s}: {count:3d} ({percentage:5.1f}%)")
        
        # Prediksi dominan
        dominant = Counter([r['prediction'] for r in results]).most_common(1)[0][0]
        print(f"\nPrediksi Dominan: {dominant}")
        
        # Statistik VAR
        var_values = [r['var'] for r in results]
        print(f"\nStatistik VAR:")
        print(f"  Min   : {min(var_values):12,.2f}")
        print(f"  Max   : {max(var_values):12,.2f}")
        print(f"  Mean  : {np.mean(var_values):12,.2f}")
        
    else:
        print("\nTidak ada data yang diproses.")
    
    print("\n" + "="*60)

# ====================================================================
# MAIN
# ====================================================================
if __name__ == "__main__":
    print("="*60)
    print("KLASIFIKASI EMG - 3 GERAKAN TANGAN")
    print("="*60)
    print("Tekan Ctrl+C untuk keluar")
    print("="*60)
    
    try:
        while True:
            process_csv()
            
            print("\n\nSiap untuk file berikutnya...")
            input("Tekan Enter untuk pilih file lagi (atau Ctrl+C untuk keluar)...")
            print("\n")
            
    except KeyboardInterrupt:
        print("\n\nProgram dihentikan.")
    except Exception as e:
        print(f"\nError: {e}")