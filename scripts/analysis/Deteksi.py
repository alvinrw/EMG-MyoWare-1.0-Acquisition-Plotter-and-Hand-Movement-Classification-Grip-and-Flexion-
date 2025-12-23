import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os
from collections import Counter
from datetime import datetime


# Configuration
VAR_RELAKS_MAX = 50000
VAR_TEKUK_MAX = 500000
WINDOW_SIZE = 50
MIN_WINDOW_PERCENT = 0.8
OUTPUT_FOLDER = "HASIL"


def extract_var(signal):
    """Calculate variance of signal"""
    if len(signal) < 2:
        return 0
    return np.var(signal)


def classify_emg(var):
    """Classify movement based on variance threshold"""
    if var < VAR_RELAKS_MAX:
        return "RELAKS"
    elif var < VAR_TEKUK_MAX:
        return "TEKUK"
    else:
        return "GENGGAM"


def process_csv():
    """Load CSV, extract VAR, predict, plot results, and save to unique folder"""
    
    print("\n" + "=" * 60)
    print("EMG CLASSIFICATION - 3 HAND MOVEMENTS")
    print("=" * 60)
    
    file_path = input("\nEnter CSV file path: ").strip()
    file_path = file_path.replace('"', '').replace("'", '')
    
    if not file_path or not os.path.exists(file_path):
        print("File not found!")
        return
    
    file_name = os.path.basename(file_path)
    print(f"\nFile: {file_name}")
    
    try:
        df = pd.read_csv(file_path)
        
        if 'Nilai ADC' not in df.columns:
            print("Error: Column 'Nilai ADC' not found!")
            return
        
        # Skip first 10 rows and reset index
        if len(df) > 10:
            df = df.iloc[10:].reset_index(drop=True)
        
        # Convert time column to datetime for plotting
        if 'Waktu (HH:MM:SS.ms)' in df.columns:
            df['Waktu (HH:MM:SS.ms)'] = pd.to_datetime(
                df['Waktu (HH:MM:SS.ms)'], 
                format='%H:%M:%S.%f', 
                errors='coerce'
            )
        else:
            df['Waktu (HH:MM:SS.ms)'] = df.index
            print("Warning: Time column not found, using index as time.")
        
        signal = df['Nilai ADC'].values
        print(f"Total samples: {len(signal)}")
        
    except Exception as e:
        print(f"Error loading or processing data: {e}")
        return
    
    # Process windows
    print(f"\nProcessing (window size: {WINDOW_SIZE})...")
    print("-" * 60)
    
    results = []
    
    for i in range(0, len(signal), WINDOW_SIZE):
        window = signal[i:i + WINDOW_SIZE]
        
        if len(window) < WINDOW_SIZE * MIN_WINDOW_PERCENT:
            continue
        
        var = extract_var(window)
        prediction = classify_emg(var)
        
        results.append({
            'prediction': prediction,
            'var': var
        })
        
        print(f"Window {len(results):2d}: {prediction:8s} | VAR = {var:12,.2f}")
    
    print("-" * 60)
    
    # Summary
    if results:
        print(f"\nTotal windows: {len(results)}")
        
        summary_lines = "Prediction Results:\n"
        results_counts = Counter([r['prediction'] for r in results])
        total_windows = len(results)
        
        print("\nPrediction Results:")
        for pred_type in ['RELAKS', 'TEKUK', 'GENGGAM']:
            count = results_counts.get(pred_type, 0)
            percentage = (count / total_windows) * 100
            print(f"   {pred_type:8s}: {count:3d} ({percentage:5.1f}%)")
            summary_lines += f"   {pred_type:8s}: {count:3d} ({percentage:5.1f}%)\n"
        
        dominant = Counter([r['prediction'] for r in results]).most_common(1)[0][0]
        print(f"\nDominant Prediction: {dominant}")
        
        # VAR statistics
        var_values = [r['var'] for r in results]
        print(f"\nVAR Statistics:")
        print(f"   Min   : {min(var_values):12,.2f}")
        print(f"   Max   : {max(var_values):12,.2f}")
        print(f"   Mean  : {np.mean(var_values):12,.2f}")
        
        # Setup output folder
        base_output_dir = os.path.join(OUTPUT_FOLDER, dominant.lower())
        base_name, _ = os.path.splitext(file_name)
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        run_folder = f"{base_name}_{timestamp}"
        target_dir = os.path.join(base_output_dir, run_folder)
        os.makedirs(target_dir, exist_ok=True)
        
        print(f"\n[OUTPUT] All results saved to: {target_dir}")
        
        # Save prediction summary
        summary_path = os.path.join(target_dir, "Hasil_Prediksi.txt")
        with open(summary_path, 'w') as f:
            f.write(summary_lines)
        print(f"[SAVED] Prediction Summary: {summary_path}")
        
        # Calculate voltage for plotting
        df['Tegangan (V)'] = (df['Nilai ADC'] / 4095) * 3.3
        
        # Plot and save
        print("Displaying and saving signal plots...")
        
        # Plot 1: ADC Value vs Time
        plt.figure(figsize=(12, 5))
        plt.plot(df['Waktu (HH:MM:SS.ms)'], df['Nilai ADC'], 
                color='#2E86AB', linewidth=1.2)
        plt.title("EMG Signal Plot (ADC Value)", fontsize=14)
        plt.xlabel("Time", fontsize=12)
        plt.ylabel("ADC Value (0-4095)", fontsize=12)
        plt.grid(True, linestyle='--', alpha=0.6)
        plt.xticks(rotation=45, ha='right')
        plt.tight_layout()
        
        plot_adc_path = os.path.join(target_dir, f"{base_name}_PLOT_ADC.png")
        plt.savefig(plot_adc_path)
        print(f"[SAVED] ADC Plot: {plot_adc_path}")
        
        # Plot 2: Voltage vs Time
        plt.figure(figsize=(12, 5))
        plt.plot(df['Waktu (HH:MM:SS.ms)'], df['Tegangan (V)'], 
                color='#A23B72', linewidth=1.2)
        plt.title("EMG Signal Plot (Voltage Output)", fontsize=14)
        plt.xlabel("Time", fontsize=12)
        plt.ylabel("Voltage (V)", fontsize=12)
        plt.grid(True, linestyle='--', alpha=0.6)
        plt.xticks(rotation=45, ha='right')
        plt.tight_layout()
        
        plot_voltage_path = os.path.join(target_dir, f"{base_name}_PLOT_VOLTAGE.png")
        plt.savefig(plot_voltage_path)
        print(f"[SAVED] Voltage Plot: {plot_voltage_path}")
        
        plt.show()
        
    else:
        print("\nNo data processed.")
    
    print("\n" + "=" * 60)


def main():
    print("=" * 60)
    print("EMG CLASSIFICATION - 3 HAND MOVEMENTS (TERMINAL MODE)")
    print("=" * 60)
    print(f"Results will be saved to '{OUTPUT_FOLDER}/[MOVEMENT]/[FILENAME_TIME]'")
    print("Press Ctrl+C to exit")
    print("=" * 60)
    
    try:
        while True:
            process_csv()
            print("\n\nReady for next file...")
            input("Press Enter to select another file (or Ctrl+C to exit)...")
            print("\n")
            
    except KeyboardInterrupt:
        print("\n\nProgram stopped.")
    except Exception as e:
        print(f"\nUnexpected error: {e}")


if __name__ == "__main__":
    main()