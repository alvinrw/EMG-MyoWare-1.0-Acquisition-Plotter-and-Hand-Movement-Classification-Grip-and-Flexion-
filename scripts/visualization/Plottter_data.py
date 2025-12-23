import pandas as pd
import matplotlib.pyplot as plt


def main():
    filename = "Data_test/RELAKS/emg_data1-relak.csv"
    
    data = pd.read_csv(filename)
    data['Waktu (HH:MM:SS.ms)'] = pd.to_datetime(
        data['Waktu (HH:MM:SS.ms)'], 
        format='%H:%M:%S.%f'
    )
    
    plt.figure(figsize=(10, 5))
    plt.plot(data['Waktu (HH:MM:SS.ms)'], data['Nilai ADC'], 
            color='blue', linewidth=1.2)
    plt.title("EMG Signal Plot (MyoWare)", fontsize=14)
    plt.xlabel("Time", fontsize=12)
    plt.ylabel("ADC Value", fontsize=12)
    plt.grid(True)
    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    main()
