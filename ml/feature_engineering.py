"""
Feature engineering functions for raw sensor signals.
"""
import pandas as pd
import numpy as np
from scipy.stats import kurtosis
from scipy.fft import fft, fftfreq

def compute_rms(signal: np.ndarray) -> float:
    """Computes Root Mean Square of a signal."""
    return np.sqrt(np.mean(signal**2))

def compute_fft_peak(signal: np.ndarray, sample_rate: float = 100.0) -> float:
    """Computes the peak frequency using FFT."""
    N = len(signal)
    if N == 0:
        return 0.0
    yf = fft(signal)
    xf = fftfreq(N, 1 / sample_rate)
    
    # Get positive frequencies
    idx = np.argmax(np.abs(yf[0:N//2]))
    return np.abs(xf[idx])

def extract_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Extracts features from sensor data dataframe.
    Features: RMS vibration, FFT peak vibration, kurtosis vibration,
    rolling mean temperature, pressure deviation from baseline.
    """
    # Assuming df is sorted by time for a given machine
    # In a real scenario, these would be computed over a time window.
    # For this synthetic dataset where each row is a snapshot, we'll apply rolling where applicable 
    # and simulate signal-based features (RMS, FFT, kurtosis) from the snapshot value.
    # To simulate a signal from a single snapshot, we'll just use the snapshot value as a proxy 
    # or create a mock window if we group by machine.
    
    features_df = df.copy()
    features_df = features_df.sort_values(by=['machine_id', 'timestamp'])
    
    # 1. Rolling mean of temperature (window=10)
    features_df['temp_rolling_mean'] = features_df.groupby('machine_id')['temperature'].transform(lambda x: x.rolling(10, min_periods=1).mean())
    
    # 2. Pressure deviation from baseline (assume baseline is 1.2)
    features_df['pressure_dev'] = np.abs(features_df['pressure'] - 1.2)
    
    # 3. Vibration Kurtosis
    # Since we have single point per timestamp, we compute rolling kurtosis
    features_df['vib_kurtosis'] = features_df.groupby('machine_id')['vibration'].transform(lambda x: x.rolling(10, min_periods=3).apply(kurtosis))
    features_df['vib_kurtosis'] = features_df['vib_kurtosis'].fillna(0)
    
    # 4. Vibration RMS
    # Mocking RMS from rolling window
    features_df['vib_rms'] = features_df.groupby('machine_id')['vibration'].transform(lambda x: x.rolling(10, min_periods=1).apply(lambda w: np.sqrt(np.mean(w**2))))
    
    # 5. Vibration FFT peak
    # Mocking FFT peak from rolling window
    def local_fft_peak(w):
        if len(w) < 3: return 0.0
        yf = np.fft.fft(w)
        return np.abs(yf).max()
    
    features_df['vib_fft_peak'] = features_df.groupby('machine_id')['vibration'].transform(lambda x: x.rolling(10, min_periods=1).apply(local_fft_peak))
    
    return features_df

def extract_features_single(reading: dict, recent_history: pd.DataFrame) -> dict:
    """
    Extract features for a single new reading, using recent history for rolling calculations.
    """
    df = pd.concat([recent_history, pd.DataFrame([reading])], ignore_index=True)
    features = extract_features(df)
    return features.iloc[-1].to_dict()
