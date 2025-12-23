import serial
import time
import csv


def main():
    port = "COM11"
    baud = 9600
    filename = "emg_data.csv"
    
    ser = serial.Serial(port, baud)
    time.sleep(2)
    
    with open(filename, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["Waktu (HH:MM:SS.ms)", "Nilai ADC"])
        
        print("Recording started...")
        try:
            while True:
                line = ser.readline().decode('utf-8', errors='ignore').strip()
                if line.isdigit():
                    timestamp = time.strftime("%H:%M:%S.") + f"{int((time.time() % 1)*1000):03d}"
                    writer.writerow([timestamp, line])
                    print(f"{timestamp} | {line}")
        except KeyboardInterrupt:
            print("\nRecording stopped.")


if __name__ == "__main__":
    main()
