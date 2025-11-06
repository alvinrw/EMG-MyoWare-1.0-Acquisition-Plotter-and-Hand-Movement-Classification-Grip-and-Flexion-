import pandas as pd
import numpy as np
import serial
import time
import csv
from datetime import datetime
import os
import sys

# ====================================================================
# THRESHOLD KONFIGURASI - DARI HASIL ANALISIS
# ====================================================================

# THRESHOLD VAR (Variance) - FITUR TERBAIK (Score: 1.0000)
VAR_RELAKS_MAX = 50000        # VAR < ini → RELAKS
VAR_TEKUK_MAX = 500000        # VAR < ini (tapi > RELAKS) → TEKUK
                               # VAR > ini → GENGGAM

# THRESHOLD SSC (Slope Sign Change) - FITUR KEDUA TERBAIK (Score: 0.7138)
SSC_RELAKS_MIN = 85           # SSC > ini → bukan RELAKS
SSC_GENGGAM_MAX = 60          # SSC < ini → cenderung GENGGAM
SSC_TEKUK_MIN = 60            # SSC > ini → cenderung TEKUK/RELAKS

# THRESHOLD WL (Waveform Length) - FITUR KETIGA TERBAIK (Score: 0.6935)
WL_RELAKS_MAX = 11000         # WL < ini → cenderung RELAKS
WL_GENGGAM_MIN = 13000        # WL > ini → cenderung GENGGAM/TEKUK

# Parameter windowing
WINDOW_SIZE = 50              # Ukuran window (sample) - 0.25 detik @ 200Hz
OVERLAP_PERCENTAGE = 50       # Overlap untuk realtime (0-99%)
SAMPLING_RATE = 200           # Hz
MIN_WINDOW_PERCENT = 0.8      # Minimal 80% dari window size untuk diproses

# Parameter Serial (untuk Mode Realtime)
DEFAULT_PORT = "COM11"        # Port default
BAUD_RATE = 9600             # Baud rate

# ====================================================================
# FUNGSI EKSTRAKSI FITUR - TOP 3 FEATURES
# ====================================================================

def extract_features(signal):
    """
    Ekstrak 3 fitur terbaik: VAR, SSC, WL
    
    Returns: dict dengan keys 'VAR', 'SSC', 'WL'
    """
    if len(signal) < 3:
        return {'VAR': 0, 'SSC': 0, 'WL': 0}
    
    # 1. VARIANCE (VAR) - FITUR TERBAIK
    var = np.var(signal)
    
    # 2. SLOPE SIGN CHANGE (SSC) - FITUR KEDUA TERBAIK
    threshold_ssc = 5
    ssc = np.sum(np.logical_and(
        (signal[1:-1] > signal[0:-2]) & (signal[1:-1] > signal[2:]),
        np.abs(signal[1:-1] - signal[0:-2]) > threshold_ssc
    )) + np.sum(np.logical_and(
        (signal[1:-1] < signal[0:-2]) & (signal[1:-1] < signal[2:]),
        np.abs(signal[1:-1] - signal[0:-2]) > threshold_ssc
    ))
    
    # 3. WAVEFORM LENGTH (WL) - FITUR KETIGA TERBAIK
    wl = np.sum(np.abs(np.diff(signal)))
    
    return {
        'VAR': var,
        'SSC': ssc,
        'WL': wl
    }

# ====================================================================
# FUNGSI KLASIFIKASI (RULE-BASED - 3 FITUR TERBAIK)
# ====================================================================

def classify_emg(features):
    """
    Klasifikasi EMG berdasarkan VAR, SSC, dan WL.
    
    Logika prioritas:
    1. VAR (paling berpengaruh) - deteksi GENGGAM vs lainnya
    2. SSC (sangat berpengaruh) - deteksi RELAKS vs TEKUK
    3. WL (konfirmasi tambahan)
    
    Returns: tuple (prediction, confidence_score)
    """
    var = features['VAR']
    ssc = features['SSC']
    wl = features['WL']
    
    # Counter voting untuk confidence
    votes = {'GENGGAM': 0, 'RELAKS': 0, 'TEKUK': 0}
    
    # ==== DETEKSI BERDASARKAN VAR (Bobot: 3 poin - fitur terbaik) ====
    if var > VAR_TEKUK_MAX:
        votes['GENGGAM'] += 3
    elif var < VAR_RELAKS_MAX:
        votes['RELAKS'] += 3
    else:
        votes['TEKUK'] += 3
    
    # ==== DETEKSI BERDASARKAN SSC (Bobot: 2 poin - fitur kedua) ====
    if ssc > SSC_RELAKS_MIN:
        # SSC tinggi = banyak perubahan slope = RELAKS atau TEKUK
        if ssc > 80:
            votes['RELAKS'] += 2
        else:
            votes['TEKUK'] += 2
    else:
        # SSC rendah = perubahan slope sedikit = GENGGAM
        votes['GENGGAM'] += 2
    
    # ==== DETEKSI BERDASARKAN WL (Bobot: 1 poin - fitur ketiga) ====
    if wl > WL_GENGGAM_MIN:
        # WL tinggi = total perubahan amplitudo besar
        votes['GENGGAM'] += 1
        votes['TEKUK'] += 1  # TEKUK juga bisa WL tinggi
    elif wl < WL_RELAKS_MAX:
        # WL rendah = perubahan amplitudo kecil
        votes['RELAKS'] += 1
    else:
        # WL sedang
        votes['TEKUK'] += 1
    
    # ==== FINAL DECISION ====
    # Pilih kelas dengan vote tertinggi
    prediction = max(votes, key=votes.get)
    max_votes = votes[prediction]
    confidence = (max_votes / 6.0) * 100  # Total bobot maksimal = 6
    
    return prediction, confidence

# ====================================================================
# MODE 1: PREDIKSI DARI FILE REKAMAN
# ====================================================================

def process_file_mode():
    """
    Mode 1: Load file CSV, ekstrak fitur per window, prediksi, simpan hasil.
    """
    print("\n" + "="*70)
    print("MODE 1: PREDIKSI DARI FILE REKAMAN")
    print("="*70)
    
    # Input nama file
    print("\nMasukkan path file CSV (atau tekan Enter untuk file dialog):")
    file_path = input("Path: ").strip()
    
    if not file_path:
        import tkinter as tk
        from tkinter import filedialog
        root = tk.Tk()
        root.withdraw()
        file_path = filedialog.askopenfilename(
            title="Pilih File EMG CSV",
            filetypes=[("CSV Files", "*.csv"), ("All Files", "*.*")]
        )
    
    if not file_path or not os.path.exists(file_path):
        print("❌ File tidak ditemukan!")
        return
    
    # Load data
    print(f"\n📂 Loading file: {os.path.basename(file_path)}")
    try:
        df = pd.read_csv(file_path)
        if 'Nilai ADC' not in df.columns:
            print("❌ Error: Kolom 'Nilai ADC' tidak ditemukan!")
            return
        
        signal = df['Nilai ADC'].values
        
        # Buang 10 baris pertama (sensor sensitif)
        if len(signal) > 10:
            signal = signal[10:]
            print(f"⚠️  Membuang 10 sample pertama (sensor sensitif)")
        
        print(f"✓ Data loaded: {len(signal)} samples")
        
    except Exception as e:
        print(f"❌ Error loading file: {e}")
        return
    
    # Proses per window
    print(f"\n🔄 Processing dengan window size {WINDOW_SIZE}...")
    results = []
    
    for i in range(0, len(signal), WINDOW_SIZE):
        window = signal[i:i + WINDOW_SIZE]
        
        if len(window) < WINDOW_SIZE * MIN_WINDOW_PERCENT:  # Skip jika window terlalu kecil
            continue
        
        # Ekstrak fitur
        features = extract_features(window)
        
        # Klasifikasi
        prediction, confidence = classify_emg(features)
        
        # Simpan hasil
        results.append({
            'Window': i // WINDOW_SIZE + 1,
            'Start_Sample': i,
            'End_Sample': min(i + WINDOW_SIZE, len(signal)),
            'Prediksi': prediction,
            'Confidence': f"{confidence:.1f}%",
            'VAR': features['VAR'],
            'SSC': features['SSC'],
            'WL': features['WL']
        })
        
        print(f"  Window {len(results):2d}: {prediction:8s} ({confidence:5.1f}%) | VAR={features['VAR']:10.0f} SSC={features['SSC']:3.0f} WL={features['WL']:6.0f}")
    
    # Simpan hasil ke CSV
    if results:
        results_df = pd.DataFrame(results)
        output_file = f"hasil_prediksi_{os.path.basename(file_path)}"
        results_df.to_csv(output_file, index=False)
        
        print(f"\n✅ Selesai! Total {len(results)} windows diproses")
        print(f"📊 Hasil disimpan ke: {output_file}")
        
        # Tampilkan ringkasan
        print("\n📈 RINGKASAN PREDIKSI:")
        for pred_type in ['GENGGAM', 'RELAKS', 'TEKUK']:
            count = len(results_df[results_df['Prediksi'] == pred_type])
            percentage = (count / len(results_df)) * 100
            print(f"   {pred_type:8s}: {count:3d} windows ({percentage:5.1f}%)")
        
        # Statistik fitur
        print("\n📊 STATISTIK FITUR:")
        for feat in ['VAR', 'SSC', 'WL']:
            print(f"\n   {feat}:")
            print(f"      Min  : {results_df[feat].min():.2f}")
            print(f"      Max  : {results_df[feat].max():.2f}")
            print(f"      Mean : {results_df[feat].mean():.2f}")
    else:
        print("\n⚠️ Tidak ada data yang berhasil diproses")

# ====================================================================
# MODE 2: PREDIKSI REALTIME DARI SERIAL
# ====================================================================

def process_realtime_mode():
    """
    Mode 2: Baca data dari serial, buffer per window, prediksi realtime.
    """
    print("\n" + "="*70)
    print("MODE 2: PREDIKSI REALTIME DARI SERIAL")
    print("="*70)
    
    # Input port
    port_input = input(f"\nMasukkan COM port (default: {DEFAULT_PORT}): ").strip()
    port = port_input if port_input else DEFAULT_PORT
    
    print(f"\n📡 Connecting ke {port} @ {BAUD_RATE} baud...")
    
    try:
        ser = serial.Serial(port, BAUD_RATE, timeout=1)
        time.sleep(2)  # Tunggu koneksi stabil
        print("✓ Koneksi berhasil!")
        
    except Exception as e:
        print(f"❌ Error koneksi serial: {e}")
        return
    
    # Siapkan buffer dan file output
    buffer = []
    overlap_step = int(WINDOW_SIZE * (1 - OVERLAP_PERCENTAGE / 100))
    
    timestamp_start = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_file = f"hasil_realtime_{timestamp_start}.csv"
    
    print(f"\n💾 Logging ke: {output_file}")
    print(f"🔧 Window: {WINDOW_SIZE}, Overlap: {OVERLAP_PERCENTAGE}%")
    print(f"🎯 Fitur: VAR (primary), SSC, WL")
    print(f"\n{'='*70}")
    print("🚀 MULAI MONITORING - Tekan Ctrl+C untuk stop")
    print(f"{'='*70}\n")
    
    window_count = 0
    
    # Buang 10 sample pertama
    skip_count = 0
    skip_samples = 10
    
    try:
        with open(output_file, 'w', newline='') as f:
            writer = csv.writer(f)
            writer.writerow(['Timestamp', 'Window', 'Prediksi', 'Confidence', 'VAR', 'SSC', 'WL'])
            
            while True:
                try:
                    # Baca data serial
                    line = ser.readline().decode('utf-8', errors='ignore').strip()
                    
                    if line.isdigit():
                        adc_value = int(line)
                        
                        # Skip 10 sample pertama
                        if skip_count < skip_samples:
                            skip_count += 1
                            continue
                        
                        buffer.append(adc_value)
                        
                        # Jika buffer sudah cukup untuk 1 window
                        if len(buffer) >= WINDOW_SIZE:
                            # Ambil window
                            window = np.array(buffer[:WINDOW_SIZE])
                            
                            # Ekstrak fitur
                            features = extract_features(window)
                            
                            # Klasifikasi
                            prediction, confidence = classify_emg(features)
                            window_count += 1
                            
                            # Timestamp
                            timestamp = datetime.now().strftime("%H:%M:%S.%f")[:-3]
                            
                            # Tampilkan hasil (dengan warna)
                            conf_str = f"{confidence:5.1f}%"
                            print(f"[{timestamp}] Window {window_count:3d}: {prediction:8s} ({conf_str}) | VAR={features['VAR']:10.0f} SSC={features['SSC']:3.0f} WL={features['WL']:6.0f}")
                            
                            # Simpan ke CSV
                            writer.writerow([
                                timestamp,
                                window_count,
                                prediction,
                                f"{confidence:.1f}",
                                f"{features['VAR']:.2f}",
                                features['SSC'],
                                f"{features['WL']:.2f}"
                            ])
                            f.flush()  # Force write ke file
                            
                            # Geser buffer (overlap)
                            buffer = buffer[overlap_step:]
                
                except UnicodeDecodeError:
                    continue  # Skip jika ada error decoding
                    
    except KeyboardInterrupt:
        print(f"\n\n{'='*70}")
        print("⏹️  MONITORING DIHENTIKAN")
        print(f"{'='*70}")
        print(f"✅ Total {window_count} windows diproses")
        print(f"💾 Data tersimpan di: {output_file}")
        
    finally:
        ser.close()
        print("🔌 Serial port ditutup")



# ====================================================================
# MENU UTAMA
# ====================================================================

def main():
    """
    Menu utama program.
    """
    while True:
        print("\n" + "="*70)
        print("       KLASIFIKASI EMG - 3 GERAKAN TANGAN")
        print("    (VAR + SSC + WL - Based on Analysis Results)")
        print("="*70)
        print("\n📊 KONFIGURASI THRESHOLD SAAT INI:")
        print(f"   VAR:  RELAKS < {VAR_RELAKS_MAX:,}  |  TEKUK < {VAR_TEKUK_MAX:,}  |  GENGGAM ≥ {VAR_TEKUK_MAX:,}")
        print(f"   SSC:  GENGGAM < {SSC_GENGGAM_MAX}  |  TEKUK/RELAKS ≥ {SSC_TEKUK_MIN}")
        print(f"   WL :  RELAKS < {WL_RELAKS_MAX:,}  |  GENGGAM/TEKUK ≥ {WL_GENGGAM_MIN:,}")
        print("="*70)
        print("\nPILIH MODE:")
        print("  1. Prediksi dari File Rekaman (CSV)")
        print("  2. Prediksi Realtime dari Serial")
        print("  3. Keluar")
        print("="*70)
        
        choice = input("\nPilihan (1/2/3): ").strip()
        
        if choice == '1':
            process_file_mode()
        elif choice == '2':
            process_realtime_mode()
        elif choice == '3':
            print("\n👋 Terima kasih! Program selesai.")
            sys.exit(0)
        else:
            print("\n❌ Pilihan tidak valid!")
        
        input("\n⏎ Tekan Enter untuk kembali ke menu...")

# ====================================================================
# RUN PROGRAM
# ====================================================================

if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\n\n👋 Program dihentikan oleh user.")
        sys.exit(0)