import pandas as pd
import numpy as np
import os
import matplotlib.pyplot as plt
from collections import Counter
from datetime import datetime # Diperlukan untuk timestamp folder

# ====================================================================
# KONFIGURASI THRESHOLD
# ====================================================================
# Nilai threshold ini sangat bergantung pada kalibrasi sensor dan subjek
VAR_RELAKS_MAX = 50000 
VAR_TEKUK_MAX = 500000

# Parameter windowing
WINDOW_SIZE = 50
MIN_WINDOW_PERCENT = 0.8

# Folder Output
OUTPUT_FOLDER = "HASIL"
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
    """Load CSV, ekstrak VAR, prediksi, plot hasil, dan simpan ke folder unik."""
    
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
    file_name = os.path.basename(file_path)
    print(f"\nFile: {file_name}")
    
    try:
        # Load seluruh data
        df = pd.read_csv(file_path)
        
        if 'Nilai ADC' not in df.columns:
            print("Error: Kolom 'Nilai ADC' tidak ditemukan!")
            return
            
        # Buang 10 baris pertama dan reset index (penting untuk plot dan sinyal)
        if len(df) > 10:
            df = df.iloc[10:].reset_index(drop=True)
            
        # Konversi kolom waktu ke datetime untuk plotting
        if 'Waktu (HH:MM:SS.ms)' in df.columns:
            # Menggunakan errors='coerce' untuk mengubah nilai yang tidak valid menjadi NaT
            df['Waktu (HH:MM:SS.ms)'] = pd.to_datetime(df['Waktu (HH:MM:SS.ms)'], format='%H:%M:%S.%f', errors='coerce')
        else:
            # Asumsi data tanpa timestamp, gunakan index sebagai waktu
            df['Waktu (HH:MM:SS.ms)'] = df.index
            print("Peringatan: Kolom 'Waktu (HH:MM:SS.ms)' tidak ditemukan, menggunakan index sebagai waktu.")

        # Ambil sinyal setelah slicing
        signal = df['Nilai ADC'].values
        
        print(f"Total samples: {len(signal)}")
        
    except Exception as e:
        print(f"Error saat memuat atau memproses data: {e}")
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
        
        # Hitung Ringkasan untuk Konsol dan File Teks
        summary_lines = "Hasil Prediksi:\n"
        results_counts = Counter([r['prediction'] for r in results])
        total_windows = len(results)

        print("\nHasil Prediksi:")
        for pred_type in ['RELAKS', 'TEKUK', 'GENGGAM']:
            count = results_counts.get(pred_type, 0)
            percentage = (count / total_windows) * 100
            
            # Print ke konsol
            print(f"   {pred_type:8s}: {count:3d} ({percentage:5.1f}%)")
            
            # Tambahkan ke string untuk file teks (Point 5)
            summary_lines += f"   {pred_type:8s}: {count:3d} ({percentage:5.1f}%)\n"
        
        # Prediksi dominan
        dominant = Counter([r['prediction'] for r in results]).most_common(1)[0][0]
        print(f"\nPrediksi Dominan: {dominant}")
        
        # Statistik VAR
        var_values = [r['var'] for r in results]
        print(f"\nStatistik VAR:")
        print(f"   Min   : {min(var_values):12,.2f}")
        print(f"   Max   : {max(var_values):12,.2f}")
        print(f"   Mean  : {np.mean(var_values):12,.2f}")
        
        # ------------------------------------------------------------
        # PENGATURAN FOLDER OUTPUT DAN PENYIMPANAN
        # ------------------------------------------------------------
        
        # Tentukan folder dasar berdasarkan prediksi dominan
        base_output_dir = os.path.join(OUTPUT_FOLDER, dominant.lower())
        
        # Buat subfolder unik untuk setiap kali tes (Point 4)
        base_name, _ = os.path.splitext(file_name)
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        run_folder = f"{base_name}_{timestamp}"
        target_dir = os.path.join(base_output_dir, run_folder)
        os.makedirs(target_dir, exist_ok=True)
        
        print(f"\n[OUTPUT] Semua hasil disimpan di folder: {target_dir}")

        # Simpan ringkasan prediksi ke file teks (Point 5)
        summary_path = os.path.join(target_dir, "Hasil_Prediksi.txt")
        with open(summary_path, 'w') as f:
            f.write(summary_lines)
        print(f"[SAVED] Ringkasan Prediksi: {summary_path}")

        # Hitung Tegangan (diperlukan untuk plot)
        df['Tegangan (V)'] = (df['Nilai ADC'] / 4095) * 3.3
        
        # 1. Plotting dan Penyimpanan Gambar
        print("Menampilkan dan menyimpan plot sinyal...")
        
        # Plot 1: ADC Value vs. Time (Point 1 & 2)
        plt.figure(figsize=(12, 5))
        plt.plot(df['Waktu (HH:MM:SS.ms)'], df['Nilai ADC'], color='#2E86AB', linewidth=1.2)
        plt.title("Plot Sinyal EMG (Nilai ADC)", fontsize=14)
        plt.xlabel("Waktu", fontsize=12)
        plt.ylabel("Nilai ADC (0-4095)", fontsize=12)
        plt.grid(True, linestyle='--', alpha=0.6)
        plt.xticks(rotation=45, ha='right')
        plt.tight_layout()
        
        plot_adc_path = os.path.join(target_dir, f"{base_name}_PLOT_ADC.png")
        plt.savefig(plot_adc_path)
        print(f"[SAVED] Plot ADC: {plot_adc_path}")


        # Plot 2: Voltage vs. Time (Point 1 & 2)
        plt.figure(figsize=(12, 5))
        plt.plot(df['Waktu (HH:MM:SS.ms)'], df['Tegangan (V)'], color='#A23B72', linewidth=1.2)
        plt.title("Plot Sinyal EMG (Tegangan Output)", fontsize=14)
        plt.xlabel("Waktu", fontsize=12)
        plt.ylabel("Tegangan (Volt)", fontsize=12)
        plt.grid(True, linestyle='--', alpha=0.6)
        plt.xticks(rotation=45, ha='right')
        plt.tight_layout()
        
        plot_voltage_path = os.path.join(target_dir, f"{base_name}_PLOT_VOLTAGE.png")
        plt.savefig(plot_voltage_path)
        print(f"[SAVED] Plot Tegangan: {plot_voltage_path}")
        
        plt.show() # Tampilkan kedua plot
        # ------------------------------------------------------------
        
    else:
        print("\nTidak ada data yang diproses.")
    
    print("\n" + "="*60)

# ====================================================================
# MAIN EXECUTION
# ====================================================================
if __name__ == "__main__":
    print("="*60)
    print("KLASIFIKASI EMG - 3 GERAKAN TANGAN (TERMINAL MODE)")
    print("="*60)
    print(f"Hasil akan disimpan di folder '{OUTPUT_FOLDER}/[GERAKAN]/[NAMA_FILE_WAKTU]'")
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
        print(f"\nError tak terduga: {e}")