"""Ekstraksi fitur EEG per jendela waktu. Ringan untuk CPU."""
import numpy as np
from scipy.signal import welch

BANDS = {
    "delta": (0.5, 4),
    "theta": (4, 8),
    "alpha": (8, 13),
    "beta": (13, 30),
    "gamma": (30, 60),
}


def make_windows(n_samples: int, fs: float, win_sec: float, step_sec: float) -> np.ndarray:
    """Indeks awal setiap jendela (dalam sampel)."""
    win, step = int(win_sec * fs), int(step_sec * fs)
    return np.arange(0, n_samples - win + 1, step)


def window_features(x: np.ndarray, fs: float) -> np.ndarray:
    """x: (n_kanal, n_sampel) satu jendela -> vektor fitur 1D."""
    freqs, psd = welch(x, fs=fs, nperseg=min(x.shape[1], int(fs * 2)), axis=-1)
    total = psd[:, (freqs >= 0.5) & (freqs <= 60)].sum(axis=1) + 1e-12
    feats = []
    for lo, hi in BANDS.values():
        band = psd[:, (freqs >= lo) & (freqs < hi)].sum(axis=1)
        feats.append(np.log(band + 1e-12))   # daya absolut (log)
        feats.append(band / total)           # daya relatif
    feats.append(np.log(x.var(axis=1) + 1e-12))
    feats.append(np.log(np.abs(np.diff(x, axis=1)).sum(axis=1) + 1e-12))  # line length
    return np.concatenate(feats).astype(np.float32)


def feature_names(channels: list[str]) -> list[str]:
    names = []
    for b in BANDS:
        names += [f"{c}|{b}_abs" for c in channels]
        names += [f"{c}|{b}_rel" for c in channels]
    names += [f"{c}|var" for c in channels]
    names += [f"{c}|line_length" for c in channels]
    return names


def extract(signal: np.ndarray, fs: float, win_sec: float = 4.0, step_sec: float = 2.0):
    """signal: (n_kanal, n_sampel). Kembalikan (fitur (n_jendela, n_fitur), waktu_mulai_detik)."""
    starts = make_windows(signal.shape[1], fs, win_sec, step_sec)
    win = int(win_sec * fs)
    X = np.stack([window_features(signal[:, s : s + win], fs) for s in starts])
    return X, starts / fs


def label_windows(t_start: np.ndarray, win_sec: float, seizures: list[tuple[float, float]],
                  min_overlap: float = 0.5) -> np.ndarray:
    """Label 1 jika jendela tumpang tindih dengan kejang >= min_overlap dari panjang jendela."""
    y = np.zeros(len(t_start), dtype=np.int8)
    t_end = t_start + win_sec
    for a, b in seizures:
        overlap = np.clip(np.minimum(t_end, b) - np.maximum(t_start, a), 0, None)
        y[overlap >= min_overlap * win_sec] = 1
    return y
