import pandas as pd
import matplotlib.pyplot as plt



# Ganti sesuai nama file CSV kamu
filename = "emg_data4-genggam.csv"

# Baca file CSV
data = pd.read_csv(filename)

# Konversi kolom waktu ke datetime
data['Waktu (HH:MM:SS.ms)'] = pd.to_datetime(data['Waktu (HH:MM:SS.ms)'], format='%H:%M:%S.%f')

# Tambahkan kolom baru: konversi ADC ke tegangan (0 - 3.3V)
data['Tegangan (V)'] = (data['Nilai ADC'] / 4095) * 3.3

# Buat plot
plt.figure(figsize=(10,5))
plt.plot(data['Waktu (HH:MM:SS.ms)'], data['Tegangan (V)'], color='blue', linewidth=1.2)

# Biar tampilannya mirip Serial Plotter
plt.title("Plot Sinyal EMG MyoWare (ESP32)", fontsize=14)
plt.xlabel("Waktu", fontsize=12)
plt.ylabel("Tegangan (Volt)", fontsize=12)
plt.grid(True)
plt.tight_layout()
plt.show()
