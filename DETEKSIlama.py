import pandas as pd
import numpy as np
import serial
import time
from datetime import datetime
import os
import sys

# ====================================================================
# THRESHOLD KONFIGURASI - UBAH SESUAI KEBUTUHAN
# ====================================================================

# THRESHOLD VAR (Variance) - SATU-SATUNYA FITUR!
VAR_RELAKS_MAX = 50000        # VAR < ini → RELAKS
VAR_TEKUK_MAX = 500000        # VAR antara RELAKS dan ini → TEKUK
                               # VAR > ini → GENGGAM

# Parameter windowing
WINDOW_SIZE = 50              # Ukuran window (sample) - 0.25 detik @ 200Hz
OVERLAP_PERCENTAGE = 75       # Overlap 75% untuk deteksi lebih halus
SAMPLING_RATE = 200           # Hz
MIN_WINDOW_PERCENT = 0.8      # Minimal 80% dari window size untuk diproses

# Parameter Voting System (untuk stabilitas)
VOTING_BUFFER_SIZE = 5        # Ambil 5 prediksi terakhir untuk voting
MIN_CONFIDENCE_VOTES = 3      # Minimal 3 dari 5 harus sama untuk "yakin"

# Parameter Serial (untuk Mode Realtime)
DEFAULT_PORT = "COM11"        # Port default
BAUD_RATE = 9600             # Baud rate

# ====================================================================
# FUNGSI EKSTRAKSI FITUR - VAR ONLY
# ====================================================================

def extract_var(signal):
    """
    Menghitung VAR (Variance) - SATU-SATUNYA FITUR.
    
    VAR = ukuran sebaran/variasi data dari mean-nya
    VAR tinggi = sinyal berfluktuasi besar (GENGGAM kuat)
    VAR rendah = sinyal stabil/lemah (RELAKS)
    
    Args:
        signal: Array nilai ADC
    
    Returns: 
        VAR value (float)
    """
    if len(signal) < 2:
        return 0
    
    return np.var(signal)

# ====================================================================
# FUNGSI KLASIFIKASI (SUPER SIMPLE - VAR ONLY)
# ====================================================================

def classify_emg(var):
    """
    Klasifikasi EMG berdasarkan VAR saja.
    
    Returns: 
        - "RELAKS" jika VAR sangat rendah (sinyal lemah/stabil)
        - "TEKUK" jika VAR sedang (gerakan dinamis)
        - "GENGGAM" jika VAR tinggi (kontraksi kuat)
    """
    
    if var < VAR_RELAKS_MAX:
        return "RELAKS"
    elif var < VAR_TEKUK_MAX:
        return "TEKUK"
    else:
        return "GENGGAM"

# ====================================================================
# MODE 1: PREDIKSI DARI FILE REKAMAN (TANPA SAVE CSV)
# ====================================================================

def process_file_mode():
    """
    Mode 1: Load file CSV, ekstrak VAR per window, prediksi, tampilkan hasil.
    TANPA SAVE CSV!
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
    print(f"{'='*70}")
    
    results = []
    
    for i in range(0, len(signal), WINDOW_SIZE):
        window = signal[i:i + WINDOW_SIZE]
        
        if len(window) < WINDOW_SIZE * MIN_WINDOW_PERCENT:  # Skip jika window terlalu kecil
            continue
        
        # Ekstrak VAR
        var = extract_var(window)
        
        # Klasifikasi
        prediction = classify_emg(var)
        
        # Simpan hasil untuk statistik
        results.append({
            'prediction': prediction,
            'var': var
        })
        
        # Tampilkan hasil dengan warna/emoji
        emoji = "💪" if prediction == "GENGGAM" else "🤏" if prediction == "TEKUK" else "😌"
        print(f"  Window {len(results):2d}: {emoji} {prediction:8s} | VAR = {var:12,.2f}")
    
    print(f"{'='*70}")
    
    # Tampilkan ringkasan (TANPA SAVE CSV!)
    if results:
        print(f"\n✅ Selesai! Total {len(results)} windows diproses")
        
        # Tampilkan ringkasan prediksi
        print("\n📈 RINGKASAN PREDIKSI:")
        for pred_type in ['GENGGAM', 'RELAKS', 'TEKUK']:
            count = sum(1 for r in results if r['prediction'] == pred_type)
            percentage = (count / len(results)) * 100
            print(f"   {pred_type:8s}: {count:3d} windows ({percentage:5.1f}%)")
        
        # Statistik VAR
        var_values = [r['var'] for r in results]
        print("\n📊 STATISTIK VAR:")
        print(f"   Min    : {min(var_values):12,.2f}")
        print(f"   Max    : {max(var_values):12,.2f}")
        print(f"   Mean   : {np.mean(var_values):12,.2f}")
        print(f"   Median : {np.median(var_values):12,.2f}")
        
        # Prediksi dominan
        from collections import Counter
        dominant = Counter([r['prediction'] for r in results]).most_common(1)[0][0]
        print(f"\n🎯 PREDIKSI DOMINAN: {dominant}")
        
    else:
        print("\n⚠️ Tidak ada data yang berhasil diproses")

# ====================================================================
# MODE 2: PREDIKSI REALTIME DARI SERIAL (TANPA SAVE CSV)
# ====================================================================

def process_realtime_mode():
    """
    Mode 2: Baca data dari serial, buffer per window, prediksi realtime.
    DENGAN VOTING SYSTEM untuk hasil lebih stabil!
    """
    print("\n" + "="*70)
    print("MODE 2: PREDIKSI REALTIME DARI SERIAL (DENGAN VOTING)")
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
    
    # Siapkan buffer
    buffer = []
    overlap_step = int(WINDOW_SIZE * (1 - OVERLAP_PERCENTAGE / 100))
    
    # Buffer untuk voting system
    prediction_history = []
    last_stable_prediction = None
    
    print(f"\n🔧 Window: {WINDOW_SIZE}, Overlap: {OVERLAP_PERCENTAGE}%")
    print(f"🗳️  Voting: Ambil {VOTING_BUFFER_SIZE} window terakhir, minimal {MIN_CONFIDENCE_VOTES} harus sama")
    print(f"🎯 Threshold: RELAKS<{VAR_RELAKS_MAX:,} | TEKUK<{VAR_TEKUK_MAX:,} | GENGGAM≥{VAR_TEKUK_MAX:,}")
    print(f"\n{'='*70}")
    print("🚀 MULAI MONITORING - Tekan Ctrl+C untuk stop")
    print(f"{'='*70}\n")
    
    window_count = 0
    
    # Buang 10 sample pertama
    skip_count = 0
    skip_samples = 10
    
    try:
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
                        
                        # Ekstrak VAR
                        var = extract_var(window)
                        
                        # Klasifikasi per window
                        raw_prediction = classify_emg(var)
                        window_count += 1
                        
                        # Tambahkan ke history untuk voting
                        prediction_history.append(raw_prediction)
                        
                        # Batasi history sesuai buffer size
                        if len(prediction_history) > VOTING_BUFFER_SIZE:
                            prediction_history.pop(0)
                        
                        # === VOTING SYSTEM ===
                        from collections import Counter
                        vote_counts = Counter(prediction_history)
                        most_common = vote_counts.most_common(1)[0]
                        voted_prediction = most_common[0]
                        vote_count = most_common[1]
                        
                        # Cek confidence (berapa dari N window yang setuju)
                        confidence_percent = (vote_count / len(prediction_history)) * 100
                        is_confident = vote_count >= MIN_CONFIDENCE_VOTES
                        
                        # Timestamp
                        timestamp = datetime.now().strftime("%H:%M:%S.%f")[:-3]
                        
                        # === TAMPILKAN HASIL ===
                        
                        # 1. Raw prediction (per window)
                        emoji_raw = "💪" if raw_prediction == "GENGGAM" else "🤏" if raw_prediction == "TEKUK" else "😌"
                        
                        # 2. Voted prediction (hasil voting)
                        emoji_voted = "💪" if voted_prediction == "GENGGAM" else "🤏" if voted_prediction == "TEKUK" else "😌"
                        
                        # Tampilkan dengan format yang jelas
                        if is_confident:
                            # Kalau yakin, tampilkan dengan tanda ✅
                            if voted_prediction != last_stable_prediction:
                                # Kalau ada perubahan gerakan, tampilkan dengan highlight
                                print(f"\n{'🔄'*35}")
                                print(f"[{timestamp}] 🎯 PERUBAHAN GERAKAN TERDETEKSI!")
                                print(f"               {emoji_voted} {voted_prediction} (Confidence: {confidence_percent:.0f}% - {vote_count}/{len(prediction_history)})")
                                print(f"{'🔄'*35}\n")
                                last_stable_prediction = voted_prediction
                            else:
                                # Kalau masih sama, tampilkan biasa
                                print(f"[{timestamp}] W{window_count:3d}: {emoji_raw} {raw_prediction:8s} → ✅ {emoji_voted} {voted_prediction:8s} ({vote_count}/{len(prediction_history)}) | VAR={var:12,.2f}")
                        else:
                            # Kalau belum yakin (transisi), tampilkan dengan tanda ⏳
                            print(f"[{timestamp}] W{window_count:3d}: {emoji_raw} {raw_prediction:8s} → ⏳ {emoji_voted} {voted_prediction:8s} ({vote_count}/{len(prediction_history)}) | VAR={var:12,.2f} [TRANSISI]")
                        
                        # Geser buffer (overlap)
                        buffer = buffer[overlap_step:]
            
            except UnicodeDecodeError:
                continue  # Skip jika ada error decoding
                
    except KeyboardInterrupt:
        print(f"\n\n{'='*70}")
        print("⏹️  MONITORING DIHENTIKAN")
        print(f"{'='*70}")
        print(f"✅ Total {window_count} windows diproses")
        if last_stable_prediction:
            print(f"🏁 Gerakan terakhir yang stabil: {last_stable_prediction}")
        
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
        print("           (VAR ONLY - Super Simple!)")
        print("="*70)
        print("\n📊 KONFIGURASI THRESHOLD SAAT INI:")
        print(f"   VAR < {VAR_RELAKS_MAX:,}           → RELAKS 😌")
        print(f"   VAR {VAR_RELAKS_MAX:,} - {VAR_TEKUK_MAX:,}  → TEKUK 🤏")
        print(f"   VAR ≥ {VAR_TEKUK_MAX:,}          → GENGGAM 💪")
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