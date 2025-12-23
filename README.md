# EMG Hand Movement Classification

Single-channel EMG-based hand movement classification system using the MyoWare sensor. This project classifies three movement types: Grip (Genggam), Flexion (Tekuk), and Relaxation (Relaks).

## Overview

The system uses a single MyoWare EMG sensor placed on the Flexor Carpi Radialis (FCR) muscle to capture electrical signals during hand movements. Classification is performed using a simple threshold-based algorithm on the variance feature extracted from the EMG signal.

## Project Structure

```
biomed/
├── Code_arduino/
│   └── sketch_oct27a/
│       └── sketch_oct27a.ino        # ESP32 firmware for data acquisition
│
├── scripts/
│   ├── acquisition/
│   │   └── Akusisi_Data.py          # Real-time data acquisition
│   ├── analysis/
│   │   ├── Cari_fitur.py            # Feature analysis and ranking
│   │   └── Deteksi.py               # Classification model
│   └── visualization/
│       ├── Plottter_data.py         # ADC signal visualization
│       └── plotter_data_voltage.py  # Voltage signal visualization
│
├── Data_test/                        # Test data by movement type
│   ├── GENGGAM/
│   ├── RELAKS/
│   └── TEKUK/
│
├── DETEKSI_FITUR/                    # Training data for feature analysis
│
├── HASIL/                            # Classification output folder
│   ├── genggam/
│   ├── relaks/
│   └── tekuk/
│
├── results/                          # Analysis results
│   ├── feature_ranking.csv
│   └── feature_ranking.png
│
└── README.md
```

## Hardware Requirements

- ESP32 microcontroller
- MyoWare EMG sensor (v1.0)
- USB cable for serial communication

## Software Requirements

Install dependencies using pip:

```bash
pip install pyserial pandas numpy scikit-learn matplotlib scipy
```

## Usage

### 1. Data Acquisition

Connect the MyoWare sensor to ESP32 pin 34 and upload the Arduino sketch. Then run:

```bash
python scripts/acquisition/Akusisi_Data.py
```

This will record EMG data to a CSV file with timestamp and ADC values.

### 2. Feature Analysis

To analyze which features best distinguish between movements:

```bash
python scripts/analysis/Cari_fitur.py
```

The script will prompt for folder paths containing training data for each movement type. Results show that variance (VAR) is the most discriminative feature.

### 3. Classification

To classify new EMG recordings:

```bash
python scripts/analysis/Deteksi.py
```

Enter the path to a CSV file when prompted. The script will:
- Extract variance features from windowed segments
- Classify each window as RELAKS, TEKUK, or GENGGAM
- Generate plots of ADC values and voltage
- Save results to `HASIL/[movement]/[filename_timestamp]/`

### 4. Visualization

To visualize recorded EMG data:

```bash
# Plot ADC values
python scripts/visualization/Plottter_data.py

# Plot voltage values
python scripts/visualization/plotter_data_voltage.py
```

## Classification Method

The system uses a single-feature threshold approach based on signal variance:

| Movement | Variance Range |
|----------|----------------|
| RELAKS   | VAR < 50,000 |
| TEKUK    | 50,000 ≤ VAR < 500,000 |
| GENGGAM  | VAR ≥ 500,000 |

These thresholds may need adjustment based on sensor calibration and individual subjects.

## Feature Analysis Results

Based on combined scoring from Random Forest importance, Mutual Information, and F-ANOVA:

| Rank | Feature | Score |
|------|---------|-------|
| 1 | VAR | 0.6194 |
| 2 | MAV | 0.4968 |
| 3 | RMS | 0.3449 |
| 4 | ZC | 0.3333 |
| 5 | WL | 0.2428 |
| 6 | SSC | 0.2315 |

Results are saved in `results/` folder.

## Output Files

For each classification run, the system generates:
- `Hasil_Prediksi.txt` - Summary of prediction counts and percentages
- `[filename]_PLOT_ADC.png` - ADC value vs time plot
- `[filename]_PLOT_VOLTAGE.png` - Voltage vs time plot

## Notes

- The first 10 samples of each recording are automatically discarded to avoid initialization artifacts
- Window size is set to 50 samples with 80% minimum overlap requirement
- ADC values are converted to voltage using: `V = (ADC / 4095) × 3.3`

## License

This project is for educational and research purposes.
