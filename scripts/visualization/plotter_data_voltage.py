import pandas as pd
import matplotlib.pyplot as plt


def main():
    filename = "emg_data4-genggam.csv"
    
    data = pd.read_csv(filename)
    data['Waktu (HH:MM:SS.ms)'] = pd.to_datetime(
        data['Waktu (HH:MM:SS.ms)'], 
        format='%H:%M:%S.%f'
    )
    
    # Convert ADC to voltage (0-3.3V range for ESP32)
    data['Tegangan (V)'] = (data['Nilai ADC'] / 4095) * 3.3
    
    plt.figure(figsize=(10, 5))
    plt.plot(data['Waktu (HH:MM:SS.ms)'], data['Tegangan (V)'], 
            color='blue', linewidth=1.2)
    plt.title("EMG Signal Plot - MyoWare (ESP32)", fontsize=14)
    plt.xlabel("Time", fontsize=12)
    plt.ylabel("Voltage (V)", fontsize=12)
    plt.grid(True)
    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    main()
