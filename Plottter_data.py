import pandas as pd
import matplotlib.pyplot as plt

# Ganti sesuai nama file CSV kamu
filename = "GENGAM_NEW/emg_data3-GenggamA.csv"

# Baca file CSV
data = pd.read_csv(filename)

# Konversi kolom waktu ke datetime biar bisa dipakai di sumbu X
data['Waktu (HH:MM:SS.ms)'] = pd.to_datetime(data['Waktu (HH:MM:SS.ms)'], format='%H:%M:%S.%f')

# Buat plot
plt.figure(figsize=(10,5))
plt.plot(data['Waktu (HH:MM:SS.ms)'], data['Nilai ADC'], color='blue', linewidth=1.2)

# Biar tampilannya mirip Serial Plotter
plt.title("Plot Sinyal EMG (Simulasi MyoWare)", fontsize=14)
plt.xlabel("Waktu", fontsize=12)
plt.ylabel("Nilai ADC", fontsize=12)
plt.grid(True)
plt.tight_layout()
plt.show()
