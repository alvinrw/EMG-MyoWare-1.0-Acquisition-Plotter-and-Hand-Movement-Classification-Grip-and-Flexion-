# Project Myoware 1.0: Grip and Flexion Classification ✋💪

## 1. Project Description

This project aims to develop a minimalist, single-channel **Electromyography (EMG)**-based hand movement classification system.  
It focuses on classifying three main movement classes:  
**Grip (Genggam)**, **Flexion (Tekuk/Fleksi)**, and **Relaxation (Relaks)**.

The system utilizes one **Myoware EMG sensor** strategically placed on the **Flexor Carpi Radialis (FCR)** muscle.  
The process involves EMG signal data acquisition, feature extraction (Time Domain), and real-time classification using a **single-feature threshold-based algorithm** for simplicity and effectiveness.

---

## 2. Project Structure

The project repository is organized as follows:



```
├── BIOMED/
│ ├── Code_arduino/ # Microcontroller Firmware (Kode.ino)
│ ├── Data_test/ # Test data organized by movement (GENGGAM, RELAKS, TEKUK)
│ ├── DETEKSI_FITUR/ # Data used for feature analysis and training
│ └── HASIL/ # Output folder for classification results (plots, summaries)
│
├── Akusisi_Data.py # Script for real-time data acquisition from Myoware
├── Cari_fitur.py # Script for feature analysis and selection
├── Deteksi.py # Final classification model (Single-feature VAR approach)
├── feature_ranking.csv # Result of feature ranking from Cari_fitur.py
├── feature_ranking.png # Visualization of feature ranking
└── README.md
```

---
---

## 3. Core Code Explanation

### 3.1. `Code_arduino/Kode.ino` (Firmware)

| Component | Description |
|-----------|-------------|
| **Function** | Reads the analog signal output directly from the Myoware sensor |
| **Process** | Converts the analog voltage into a digital **ADC Value** (typically 0–1023 or 0–4095 depending on ADC resolution) |
| **Output** | Sends the raw ADC values continuously to the connected computer via the serial port |

---

### 3.2. `Akusisi_Data.py` (Data Acquisition)

This script manages real-time EMG data collection from the Myoware sensor via the serial port.

| Component | Description |
|-----------|-------------|
| **Main Function** | Reads serial data from the port connected to the Arduino (e.g., COM11) |
| **Output** | Saves EMG data into CSV files (e.g., `emg_data_tekuk.csv`), including timestamp and raw **ADC Value** |
| **Dependencies** | `pyserial`, `time`, `csv` |

---

### 3.3. `Cari_fitur.py` (Feature Analysis)

This script analyzes the effectiveness of various time-domain statistical features in distinguishing between the movements.

#### 🧩 Key Feature Importance (from `feature_ranking.csv`)

Based on the combined score, the **Variance (VAR)** feature is the most significant for this classification task.

| Rank | Feature | Final Score |
|------|----------|-------------|
| 🏆 **1** | **VAR** | 0.6194 |
| 🥈 2 | MAV | 0.4968 |
| 🥉 3 | RMS | 0.3449 |
| 4 | ZC | 0.3333 |
| 5 | WL | 0.2428 |
| 6 | SSC | 0.2315 |

---

### 3.4. `Deteksi.py` (Final Classification Model)

This is the **final optimized classification code** that utilizes the single best feature — **VAR (Variance)** — for movement detection.

#### 🔍 Classification Concept: Single-Feature Thresholding (VAR)

The script performs windowing on the raw ADC signal, extracts the VAR feature for each window, and classifies the movement based on empirical thresholds.

| State | Feature Range (VAR) |
|:---:|:---:|
| **RELAKS** | VAR < VAR_RELAKS_MAX (≈ 50,000) |
| **TEKUK** | VAR_RELAKS_MAX ≤ VAR < VAR_TEKUK_MAX (≈ 500,000) |
| **GENGGAM** | VAR ≥ VAR_TEKUK_MAX (≈ 500,000) |

#### 🧾 Output Handling

After classification, the script automatically:

1. **Plots:**
   - **ADC Value vs. Time**
   - **Voltage vs. Time** (calculated as `Voltage = (ADC / 4095) × 3.3`)
2. **Saves Results:**
   - Creates a timestamped output folder:  
     `HASIL/[DOMINANT_PREDICTION]/[TIMESTAMP]/`
   - Saves:
     - **Plots** (`*_PLOT_ADC.png`, `*_PLOT_VOLTAGE.png`)
     - **Summary text file** (`Hasil_Prediksi.txt`) containing prediction counts and percentages

---

## 4. Requirements

Install all dependencies before running the scripts:

```bash
pip install pyserial pandas numpy scikit-learn matplotlib
