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
RELAKSASI_WAMP_MAX = 50      # WAMP dibawah ini = RELAKSASI

# Parameter windowing
WINDOW_SIZE = 150            # Ukuran window (sample) - 0.75 detik @ 200Hz
OVERLAP_PERCENTAGE = 75      # Overlap untuk realtime (0-99%)
SAMPLING_RATE = 200          # Hz

# Parameter Serial (untuk Mode Realtime)
DEFAULT_PORT = "COM11"       # Port default
BAUD_RATE = 9600            # Baud rate

# Parameter WAMP (threshold untuk mendeteksi perubahan signifikan)
WAMP_AMPLITUDE_THRESHOLD = 15  # Selisih minimal untuk dihitung sebagai "perubahan"

# ====================================================================
# FUNGSI EKSTRAKSI FITUR - WAMP ONLY
# ====================================================================

def extract_wamp(signal, threshold=15):
    """
    Menghitung WAMP (Willison Amplitude) - SATU-SATUNYA FITUR.
    
    WAMP = jumlah perubahan amplitudo yang signifikan (≥ threshold)
    
    Args:
        signal: Array nilai ADC
        threshold: Threshold perubahan amplitudo (default: 15)
    
    Returns: 
        WAMP value (integer)
    """
    if len(signal) < 2:
        return 0
    
    # Hitung selisih antar sample berurutan
    diff = np.abs(np.diff(signal))
    
    # Hitung berapa kali selisih ≥ threshold
    wamp = np.sum(diff >= threshold)
    
    return wamp

# ====================================================================
# FUNGSI KLASIFIKASI (RULE-BASED - WAMP ONLY)
# ====================================================================

def classify_emg(wamp):
    """
    Klasifikasi EMG berdasarkan WAMP saja.
    
    Returns: 
        - "RELAKSASI" jika WAMP sangat rendah (sinyal lemah)
        - "TEKUK" jika WAMP tinggi (gerakan dinamis)
        - "GENGGAM" jika WAMP rendah-sedang (kontraksi statis)
    """
    
    # 1. CEK RELAKSASI (sinyal sangat lemah)
    if wamp < RELAKSASI_WAMP_MAX:
        return "RELAKSASI"
    
    # 2. KLASIFIKASI UTAMA BERDASARKAN WAMP
    # WAMP tinggi = banyak perubahan amplitudo = TEKUK (dinamis)
    # WAMP rendah = perubahan amplitudo sedikit = GENGGAM (statis)
    if wamp > WAMP_THRESHOLD_TEKUK:
        return "TEKUK"
    else:
        return "GENGGAM"

# ====================================================================
# MODE 1: PREDIKSI DARI FILE REKAMAN
# ====================================================================

def process_file_mode():
    """
    Mode 1: Load file CSV, ekstrak WAMP per window, prediksi, simpan hasil.
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
        
        # Ekstrak WAMP
        wamp = extract_wamp(window, threshold=WAMP_AMPLITUDE_THRESHOLD)
        
        # Klasifikasi
        prediction = classify_emg(wamp)
        
        # Simpan hasil
        results.append({
            'Window': i // WINDOW_SIZE + 1,
            'Start_Sample': i,
            'End_Sample': min(i + WINDOW_SIZE, len(signal)),
            'Prediksi': prediction,
            'WAMP': wamp
        })
        
        print(f"  Window {len(results)}: {prediction:10s} (WAMP={wamp})")
    
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
        
        # Statistik WAMP
        print("\n📊 STATISTIK WAMP:")
        print(f"   Min  : {results_df['WAMP'].min()}")
        print(f"   Max  : {results_df['WAMP'].max()}")
        print(f"   Mean : {results_df['WAMP'].mean():.2f}")
        print(f"   Median: {results_df['WAMP'].median():.2f}")
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
    print(f"🎯 Threshold: TEKUK>{WAMP_THRESHOLD_TEKUK}, RELAKSASI<{RELAKSASI_WAMP_MAX}")
    print(f"\n{'='*60}")
    print("🚀 MULAI MONITORING - Tekan Ctrl+C untuk stop")
    print(f"{'='*60}\n")
    
    window_count = 0
    
    try:
        with open(output_file, 'w', newline='') as f:
            writer = csv.writer(f)
            writer.writerow(['Timestamp', 'Window', 'Prediksi', 'WAMP'])
            
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
                            
                            # Ekstrak WAMP
                            wamp = extract_wamp(window, threshold=WAMP_AMPLITUDE_THRESHOLD)
                            
                            # Klasifikasi
                            prediction = classify_emg(wamp)
                            window_count += 1
                            
                            # Timestamp
                            timestamp = datetime.now().strftime("%H:%M:%S.%f")[:-3]
                            
                            # Tampilkan hasil
                            print(f"[{timestamp}] Window {window_count}: {prediction:10s} | WAMP={wamp:4d}")
                            
                            # Simpan ke CSV
                            writer.writerow([
                                timestamp,
                                window_count,
                                prediction,
                                wamp
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
        print("         (WAMP-Only Classifier)")
        print("="*60)
        print("\n📊 KONFIGURASI SAAT INI:")
        print(f"   • WAMP > {WAMP_THRESHOLD_TEKUK} → TEKUK")
        print(f"   • WAMP < {WAMP_THRESHOLD_GENGGAM} → GENGGAM")
        print(f"   • WAMP < {RELAKSASI_WAMP_MAX} → RELAKSASI")
        print(f"   • Window Size: {WINDOW_SIZE} samples")
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