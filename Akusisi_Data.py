import serial
import time
import csv

port = "COM11"
baud = 9600
filename = "emg_data10-TekukA.csv"

ser = serial.Serial(port, baud)
time.sleep(2)

with open(filename, "w", newline="") as f:
    writer = csv.writer(f)
    writer.writerow(["Waktu (HH:MM:SS.ms)", "Nilai ADC"])
    
    print("Mulai merekam...")
    while True:
        try:
            line = ser.readline().decode('utf-8', errors='ignore').strip()
            if line.isdigit():
                timestamp = time.strftime("%H:%M:%S.") + f"{int((time.time() % 1)*1000):03d}"
                writer.writerow([timestamp, line])
                print(timestamp, line)
        except KeyboardInterrupt:
            print("Selesai merekam.")
            break
