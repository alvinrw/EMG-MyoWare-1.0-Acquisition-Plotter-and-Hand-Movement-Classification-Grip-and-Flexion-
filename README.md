# Project Myoware 1.0: Grip and Flexion Classification

## 1. Project Description

This project aims to develop a minimalist hand movement classification system, focusing on two main motion classes: **Grip (Genggam)** and **Flexion (Tekuk/Fleksi)**.

The system utilizes one Myoware Electromyography (EMG) sensor strategically placed on the **Flexor Carpi Radialis (FCR)** muscle. The EMG signal data is acquired, features are extracted, and then classified using a threshold-based algorithm for simple yet effective real-time classification.

![Uploading image.png…]()


---

## 2. Project Structure

The project repository is structured as follows:

```
├── BIOMED/
│   ├── Data_Genggam/            # Raw data files for the "Grip" class
│   ├── Data_Tekuk/              # Raw data files for the "Flexion" class
│   └── Data_Irrelevant/         # Miscellaneous or corrupted data (not used for training)
├── Code_arduino/
│   └── Kode.ino                 # Firmware for the microcontroller (Arduino)
├── Akusisi_Data.py              # Script for data acquisition from Myoware
├── Cari_Fitur.py                # Script for feature analysis and selection
├── DeteksiV1.py                 # Initial classification model (Multi-feature approach)
├── DeteksiV2.py                 # Final classification model (Single-feature WAMP approach)
├── Plotter_data.py              # General script for visualizing processed data
├── plotter_data(analog).py      # Script for visualizing raw ADC signal data
├── feature_distribution_comparison.png
├── feature_distribution_comparison_new_features.png
├── feature_importance_analysis.png
├── feature_importance_analysis_new_features.png
└── tempCodeRunnerFile.py
```

---

## 3. Core Code Explanation

### 3.1. Code_arduino/Kode.ino

This is the firmware designed for the microcontroller (e.g., Arduino Uno/Nano).

| Component | Description |
|-----------|-------------|
| **Function** | Reads the analog signal output directly from the Myoware sensor |
| **Process** | Converts the analog voltage into a digital ADC Value (typically 0-1023) |
| **Output** | Sends the raw ADC values continuously to the connected computer via the serial port |

---

### 3.2. Akusisi_Data.py

This file is responsible for real-time data collection from the Myoware sensor.

| Component | Description |
|-----------|-------------|
| **Main Function** | Reads serial data from the port connected to the Arduino (e.g., COM11) |
| **Process** | Parses the incoming digital ADC values and records them |
| **Output** | Saves the EMG data into CSV files (e.g., `emg_data6-tekuk.csv`), including timestamp and the raw ADC Value |
| **Dependencies** | `serial`, `time`, `csv` |

---

### 3.3. Cari_Fitur.py (Feature Analysis)

This script is the core analysis step, designed to determine which statistical features (Time Domain - TD and Frequency Domain - FD) are most effective at distinguishing between Grip and Flexion movements.

#### Extracted Features (per Window):

| Feature | Domain | Description |
|---------|--------|-------------|
| **RMS** | TD | Root Mean Square |
| **MAV** | TD | Mean Absolute Value |
| **VAR** | TD | Variance |
| **ZC** | TD | Zero Crossing |
| **SSC** | TD | Slope Sign Change |
| **WL** | TD | Waveform Length |
| **WAMP** | TD | Willison Amplitude |
| **MNF** | FD | Mean Frequency |
| **MDF** | FD | Median Frequency |

#### Key Feature Importance Results:

Based on the **Combined Score** (aggregating Random Forest Importance, Mutual Information Score, and Cohen's d Separation Score), the **WAMP (Willison Amplitude)** feature is the most significant for this classification task.

| Rank | Feature | Combined Score | RF Importance | Separation Score |
|------|---------|----------------|---------------|------------------|
| 🏆 **1** | **WAMP** | 1.0000 | 0.2034 | 2.1322 |
| 🏆 **2** | **MNF** | 0.7434 | 0.1616 | 1.6064 |
| 🏆 **3** | **SSC** | 0.5969 | 0.0772 | 1.3167 |

---

### 3.4. Plotter_data.py & plotter_data(analog).py

These files are used for data visualization. They load collected data from CSV files and display them in graphical form.

- **Plotter_data.py**: General script for visualizing processed or feature data
- **plotter_data(analog).py**: Specific script for plotting raw data, focusing the Y-axis on Analog Values (ADC) for signal inspection

---

### 3.5. DeteksiV1.py

This is the initial version (V1) of the classification code.

- **Approach**: Uses a broad range of Time-Domain features (RMS, MAV, VAR, ZC, etc.) for movement classification
- **Purpose**: To serve as a baseline for testing the performance of the classification model before moving to a more optimized feature set

---

### 3.6. DeteksiV2.py

This is the **final, optimized classification code** based on the feature analysis results from `Cari_Fitur.py`.

- **Approach**: Utilizes only the single best feature, **WAMP (Willison Amplitude)**, for classification

#### Concept: Threshold Tuning

Tuning is the empirical process of finding the most accurate boundary values (thresholds) to separate different classes.

In this code, thresholds are tuned to distinguish between three states: **FLEXION**, **GRIP**, and **RELAXATION**.

**Tuned Threshold Examples:**

```python
WAMP_THRESHOLD_TEKUK = 100      # If WAMP > this, classified as FLEXION
WAMP_THRESHOLD_GENGGAM = 99     # If WAMP < this, classified as GRIP
RELAKSASI_WAMP_MAX = 50         # WAMP below this is classified as RELAXATION
```

---

## 4. Requirements

To run the Python scripts, you need the following libraries:

```bash
pip install pyserial pandas numpy scikit-learn matplotlib
```

---

## 5. Usage

1. **Upload Arduino Code**: Flash `Code_arduino/Kode.ino` to your Arduino board
2. **Data Acquisition**: Run `Akusisi_Data.py` to collect EMG data
3. **Feature Analysis**: Execute `Cari_Fitur.py` to analyze feature importance
4. **Classification**: Use `DeteksiV2.py` for real-time movement classification

---


**Note**: Ensure your Myoware sensor is properly connected to the Flexor Carpi Radialis muscle for optimal signal acquisition.
