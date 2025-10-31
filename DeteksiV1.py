import pandas as pd
import numpy as np
import serial
import time
import csv
from datetime import datetime
import os
import sys

# ====================================================================
# THRESHOLD KONFIGURASI - UBAH SESUAI KEBUTUHAN
# ====================================================================

# Threshold untuk klasifikasi WAMP
WAMP_THRESHOLD_TEKUK = 100  # Jika WAMP > ini → TEKUK
WAMP_THRESHOLD_GENGGAM = 99  # Jika WAMP < ini → GENGGAM

# Threshold untuk deteksi RELAKSASI (sinyal sangat lemah)
RELAKSASI_WAMP_MAX = 50      # WAMP dibawah ini = kemungkinan relaksasi
RELAKSASI_MAV_MAX = 100      # MAV dibawah ini = kemungkinan relaksasi
RELAKSASI_RMS_MAX = 200      # RMS dibawah ini = kemungkinan relaksasi

# Parameter windowing
WINDOW_SIZE = 150            # Ukuran window (sample) - 1 detik @ 200Hz
OVERLAP_PERCENTAGE = 75      # Overlap untuk realtime (0-99%)
SAMPLING_RATE = 200          # Hz

# Parameter Serial (untuk Mode Realtime)
DEFAULT_PORT = "COM11"       # Port default
BAUD_RATE = 9600            # Baud rate

# ====================================================================
# FUNGSI EKSTRAKSI FITUR
# ====================================================================

def extract_emg_features(signal, fs=200, threshold_ssc=5, threshold_wamp=15):
    """
    Ekstraksi fitur Time-Domain (TD) dan Frequency-Domain (FD)
    dari sinyal EMG untuk 1 window.
    
    Returns: Dictionary berisi semua fitur
    """
    N = len(signal)
    
    if N == 0:
        return None
    
    # 1. TIME-DOMAIN FEATURES
    
    # Root Mean Square (RMS)
    rms = np.sqrt(np.mean(signal**2))
    
    # Mean Absolute Value (MAV)
    mav = np.mean(np.abs(signal))
    
    # Variance (VAR)
    var = np.var(signal)
    
    # Zero Crossing (ZC)
    zc = np.sum(np.diff(np.sign(signal)) != 0)
    
    # Slope Sign Change (SSC)
    if N > 2:
        ssc = np.sum(np.logical_and(
            (signal[1:-1] > signal[0:-2]) & (signal[1:-1] > signal[2:]),
            np.abs(signal[1:-1] - signal[0:-2]) > threshold_ssc
        )) + np.sum(np.logical_and(
            (signal[1:-1] < signal[0:-2]) & (signal[1:-1] < signal[2:]),
            np.abs(signal[1:-1] - signal[0:-2]) > threshold_ssc
        ))
    else:
        ssc = 0
    
    # Waveform Length (WL)
    wl = np.sum(np.abs(np.diff(signal)))
    
    # Willison Amplitude (WAMP) - FITUR UTAMA
    wamp = np.sum(np.abs(np.diff(signal)) >= threshold_wamp)
    
    # 2. FREQUENCY-DOMAIN FEATURES
    
    if N > 1:
        fft_values = np.fft.fft(signal)
        psd = np.abs(fft_values[:N//2])**2
        f = np.fft.fftfreq(N, 1/fs)[:N//2]
        
        # Mean Frequency (MNF)
        mnf = np.sum(f * psd) / (np.sum(psd) + 1e-10)
        
        # Median Frequency (MDF)
        cumulative_psd = np.cumsum(psd)
        median_power = cumulative_psd[-1] / 2
        mdf_index = np.where(cumulative_psd >= median_power)[0]
        mdf = f[mdf_index[0]] if len(mdf_index) > 0 else 0
    else:
        mnf, mdf = 0, 0
    
    return {
        'RMS': rms,
        'MAV': mav,
        'VAR': var,
        'ZC': zc,
        'SSC': ssc,
        'WL': wl,
        'WAMP': wamp,  # FITUR UTAMA UNTUK KLASIFIKASI
        'MNF': mnf,
        'MDF': mdf
    }

# ====================================================================
# FUNGSI KLASIFIKASI (RULE-BASED)
# ====================================================================

def classify_emg(features):
    """
    Klasifikasi EMG berdasarkan rule-based (threshold WAMP).
    
    Returns: 
        - "RELAKSASI" jika sinyal sangat lemah
        - "TEKUK" jika WAMP tinggi
        - "GENGGAM" jika WAMP rendah
    """
    
    wamp = features['WAMP']
    mav = features['MAV']
    rms = features['RMS']
    
    # 1. CEK RELAKSASI (sinyal sangat lemah)
    # Ubah threshold di bagian THRESHOLD KONFIGURASI di atas
    if wamp < RELAKSASI_WAMP_MAX and mav < RELAKSASI_MAV_MAX and rms < RELAKSASI_RMS_MAX:
        return "RELAKSASI"
    
    # 2. KLASIFIKASI UTAMA BERDASARKAN WAMP
    # WAMP tinggi = banyak perubahan amplitudo = TEKUK
    # WAMP rendah = perubahan amplitudo sedikit = GENGGAM
    if wamp > WAMP_THRESHOLD_TEKUK:
        return "TEKUK"
    else:
        return "GENGGAM"

# ====================================================================
# MODE 1: PREDIKSI DARI FILE REKAMAN
# ====================================================================

def process_file_mode():
    """
    Mode 1: Load file CSV, ekstrak fitur per window, prediksi, simpan hasil.
    """
    print("\n" + "="*60)
    print("MODE 1: PREDIKSI DARI FILE REKAMAN")
    print("="*60)
    
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
        print(f"✓ Data loaded: {len(signal)} samples")
        
    except Exception as e:
        print(f"❌ Error loading file: {e}")
        return
    
    # Proses per window
    print(f"\n🔄 Processing dengan window size {WINDOW_SIZE}...")
    results = []
    
    for i in range(0, len(signal), WINDOW_SIZE):
        window = signal[i:i + WINDOW_SIZE]
        
        if len(window) < WINDOW_SIZE * 0.5:  # Skip jika window terlalu kecil
            continue
        
        # Ekstrak fitur
        features = extract_emg_features(window, fs=SAMPLING_RATE)
        if features is None:
            continue
        
        # Klasifikasi
        prediction = classify_emg(features)
        
        # Simpan hasil
        results.append({
            'Window': i // WINDOW_SIZE + 1,
            'Start_Sample': i,
            'End_Sample': min(i + WINDOW_SIZE, len(signal)),
            'Prediksi': prediction,
            'WAMP': features['WAMP'],
            'MAV': features['MAV'],
            'RMS': features['RMS']
        })
        
        print(f"  Window {len(results)}: {prediction} (WAMP={features['WAMP']:.2f})")
    
    # Simpan hasil ke CSV
    if results:
        results_df = pd.DataFrame(results)
        output_file = f"hasil_prediksi_{os.path.basename(file_path)}"
        results_df.to_csv(output_file, index=False)
        
        print(f"\n✅ Selesai! Total {len(results)} windows diproses")
        print(f"📊 Hasil disimpan ke: {output_file}")
        
        # Tampilkan ringkasan
        print("\n📈 RINGKASAN:")
        for pred_type in ['GENGGAM', 'TEKUK', 'RELAKSASI']:
            count = len(results_df[results_df['Prediksi'] == pred_type])
            percentage = (count / len(results_df)) * 100
            print(f"   {pred_type}: {count} windows ({percentage:.1f}%)")
    else:
        print("\n⚠️ Tidak ada data yang berhasil diproses")

# ====================================================================
# MODE 2: PREDIKSI REALTIME DARI SERIAL
# ====================================================================

def process_realtime_mode():
    """
    Mode 2: Baca data dari serial, buffer per window, prediksi realtime.
    """
    print("\n" + "="*60)
    print("MODE 2: PREDIKSI REALTIME DARI SERIAL")
    print("="*60)
    
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
    print(f"\n{'='*60}")
    print("🚀 MULAI MONITORING - Tekan Ctrl+C untuk stop")
    print(f"{'='*60}\n")
    
    window_count = 0
    
    try:
        with open(output_file, 'w', newline='') as f:
            writer = csv.writer(f)
            writer.writerow(['Timestamp', 'Window', 'Prediksi', 'WAMP', 'MAV', 'RMS'])
            
            while True:
                try:
                    # Baca data serial
                    line = ser.readline().decode('utf-8', errors='ignore').strip()
                    
                    if line.isdigit():
                        adc_value = int(line)
                        buffer.append(adc_value)
                        
                        # Jika buffer sudah cukup untuk 1 window
                        if len(buffer) >= WINDOW_SIZE:
                            # Ambil window
                            window = np.array(buffer[:WINDOW_SIZE])
                            
                            # Ekstrak fitur
                            features = extract_emg_features(window, fs=SAMPLING_RATE)
                            
                            if features is not None:
                                # Klasifikasi
                                prediction = classify_emg(features)
                                window_count += 1
                                
                                # Timestamp
                                timestamp = datetime.now().strftime("%H:%M:%S.%f")[:-3]
                                
                                # Tampilkan hasil
                                print(f"[{timestamp}] Window {window_count}: {prediction:10s} | WAMP={features['WAMP']:6.1f} | MAV={features['MAV']:6.1f}")
                                
                                # Simpan ke CSV
                                writer.writerow([
                                    timestamp,
                                    window_count,
                                    prediction,
                                    f"{features['WAMP']:.2f}",
                                    f"{features['MAV']:.2f}",
                                    f"{features['RMS']:.2f}"
                                ])
                                f.flush()  # Force write ke file
                            
                            # Geser buffer (overlap)
                            buffer = buffer[overlap_step:]
                
                except UnicodeDecodeError:
                    continue  # Skip jika ada error decoding
                    
    except KeyboardInterrupt:
        print(f"\n\n{'='*60}")
        print("⏹️  MONITORING DIHENTIKAN")
        print(f"{'='*60}")
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
        print("\n" + "="*60)
        print("     KLASIFIKASI EMG - GENGGAM VS TEKUK")
        print("           (Rule-Based Classifier)")
        print("="*60)
        print("\nPILIH MODE:")
        print("  1. Prediksi dari File Rekaman (CSV)")
        print("  2. Prediksi Realtime dari Serial")
        print("  3. Keluar")
        print("="*60)
        
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