import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_selection import mutual_info_classif
from scipy import stats
import glob
import os


def extract_emg_features(signal, label, window_size=200):
    """Extract 6 time-domain features: RMS, VAR, MAV, SSC, ZC, WL"""
    features = []
    N_signal = len(signal)
    threshold_ssc = 5

    for i in range(0, N_signal, window_size):
        window = signal[i:i + window_size]
        
        if len(window) < window_size * 0.5:
            continue
        
        # Calculate features
        rms = np.sqrt(np.mean(window**2))
        var = np.var(window)
        mav = np.mean(np.abs(window))
        
        # Slope Sign Changes
        ssc = np.sum(np.logical_and(
            (window[1:-1] > window[0:-2]) & (window[1:-1] > window[2:]),
            np.abs(window[1:-1] - window[0:-2]) > threshold_ssc
        )) + np.sum(np.logical_and(
            (window[1:-1] < window[0:-2]) & (window[1:-1] < window[2:]),
            np.abs(window[1:-1] - window[0:-2]) > threshold_ssc
        ))
        
        # Zero Crossing
        zc = np.sum(np.diff(np.sign(window)) != 0)
        
        # Waveform Length
        wl = np.sum(np.abs(np.diff(window)))
        
        features.append({
            'label': label,
            'RMS': rms,
            'VAR': var,
            'MAV': mav,
            'SSC': ssc,
            'ZC': zc,
            'WL': wl
        })
    
    return pd.DataFrame(features)


def safe_normalize(series):
    """Normalize series safely, avoiding division by zero"""
    min_val = series.min()
    max_val = series.max()
    if max_val == min_val:
        return np.zeros(len(series))
    return (series - min_val) / (max_val - min_val)


def main():
    print("=" * 60)
    print("FEATURE ANALYSIS")
    print("=" * 60)
    
    # Input folder paths
    print("\nEnter folder paths:")
    gesture_folders = {}
    gesture_names = ['GENGGAM', 'RELAKS', 'TEKUK']
    
    for gesture in gesture_names:
        path = input(f"{gesture}: ").strip().replace('"', '').replace("'", '')
        if not os.path.exists(path):
            print(f"Folder {gesture} not found!")
            exit()
        gesture_folders[gesture.lower()] = path
    
    # Load and extract features
    print("\nProcessing data...")
    feature_list = []
    
    for gesture_name, folder_path in gesture_folders.items():
        csv_files = glob.glob(os.path.join(folder_path, '*.csv'))
        
        for file in csv_files:
            try:
                df = pd.read_csv(file)
                if len(df) > 10:
                    df = df.iloc[10:].reset_index(drop=True)
                
                if 'Nilai ADC' not in df.columns:
                    continue
                
                signal = df['Nilai ADC'].values
                features_df = extract_emg_features(signal, gesture_name, window_size=200)
                
                if not features_df.empty:
                    feature_list.append(features_df)
            except Exception:
                continue
    
    if not feature_list:
        print("No valid data found!")
        exit()
    
    # Prepare data
    final_data = pd.concat(feature_list, ignore_index=True)
    X = final_data.drop(['label'], axis=1)
    y = final_data['label']
    
    label_mapping = {'genggam': 0, 'relaks': 1, 'tekuk': 2}
    y_encoded = y.map(label_mapping)
    
    if X.empty or len(y_encoded.dropna()) == 0:
        print("Empty feature data or invalid labels.")
        exit()
    
    # Calculate feature importance using 3 methods
    # Method 1: Random Forest
    rf = RandomForestClassifier(n_estimators=100, random_state=42, max_depth=10)
    rf.fit(X, y_encoded)
    rf_importance = pd.DataFrame({'Feature': X.columns, 'Score': rf.feature_importances_})
    
    # Method 2: Mutual Information
    mi_scores = mutual_info_classif(X, y_encoded, random_state=42)
    mi_importance = pd.DataFrame({'Feature': X.columns, 'Score': mi_scores})
    
    # Method 3: F-ANOVA
    separation_scores = []
    for col in X.columns:
        groups = [X[y == g.lower()][col].values for g in gesture_names if g.lower() in y.unique()]
        
        if len(groups) >= 2:
            f_stat, _ = stats.f_oneway(*groups)
            separation_scores.append(f_stat)
        else:
            separation_scores.append(0.0)
    
    separation_df = pd.DataFrame({'Feature': X.columns, 'Score': separation_scores})
    
    # Combine all methods
    final_ranking = rf_importance.merge(mi_importance, on='Feature', suffixes=('_RF', '_MI'))
    final_ranking = final_ranking.merge(separation_df, on='Feature')
    final_ranking.columns = ['Feature', 'RF_Score', 'MI_Score', 'Sep_Score']
    final_ranking[['RF_Score', 'MI_Score', 'Sep_Score']] = final_ranking[['RF_Score', 'MI_Score', 'Sep_Score']].fillna(0)
    
    # Normalize and calculate final score
    final_ranking['RF_Norm'] = safe_normalize(final_ranking['RF_Score'])
    final_ranking['MI_Norm'] = safe_normalize(final_ranking['MI_Score'])
    final_ranking['Sep_Norm'] = safe_normalize(final_ranking['Sep_Score'])
    
    final_ranking['Final_Score'] = (
        final_ranking['RF_Norm'] + 
        final_ranking['MI_Norm'] + 
        final_ranking['Sep_Norm']
    ) / 3
    
    final_ranking = final_ranking.sort_values('Final_Score', ascending=False).reset_index(drop=True)
    
    # Print results
    print("\n" + "=" * 60)
    print("FEATURE RANKING:")
    print("=" * 60)
    for idx, row in final_ranking.iterrows():
        final_score = row['Final_Score']
        if pd.isna(final_score):
            final_score = 0.0
        bar = "█" * int(final_score * 50)
        print(f"{idx+1}. {row['Feature']:6s} [{bar:<50}] {final_score:.3f}")
    
    print("\n" + "=" * 60)
    print(f"BEST FEATURE: {final_ranking.iloc[0]['Feature']}")
    print("=" * 60)
    
    # Visualization
    fig, ax = plt.subplots(figsize=(10, 6))
    colors = ['#2E86AB', '#A23B72', '#F18F01', '#C73E1D', '#6A994E', '#BC4B51']
    bars = ax.barh(final_ranking['Feature'], final_ranking['Final_Score'], 
                   color=colors[:len(final_ranking)])
    
    ax.set_xlabel('Score', fontsize=12)
    ax.set_title('Feature Ranking for Classification', fontsize=14, fontweight='bold')
    ax.invert_yaxis()
    
    for i, (bar, score) in enumerate(zip(bars, final_ranking['Final_Score'])):
        ax.text(score + 0.01, i, f'{score:.3f}', va='center', fontsize=10)
    
    plt.tight_layout()
    plt.savefig('feature_ranking.png', dpi=300, bbox_inches='tight')
    print("\nSaved: feature_ranking.png")
    
    # Save results
    final_ranking[['Feature', 'Final_Score']].to_csv('feature_ranking.csv', index=False)
    print("Saved: feature_ranking.csv")
    
    plt.show()


if __name__ == "__main__":
    main()