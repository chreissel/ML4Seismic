"""Shared I/O and signal utilities for reproducing the linear-analysis figures.

Loads the raw broadband ``.mat`` (e.g. ``MLdata_L1HAM5_1381528818_4000_matrix_v2.mat``),
selects channels via a configurable name->column map, applies the three bandpass
filters used in the paper (non-causal top-hat, non-causal f^2 roll-off, causal),
and computes the multiple coherence that sets the linear-subtraction floor.

Channel map
-----------
``DEFAULT_CHANNEL_MAP`` matches ``data_prep.py`` (10 channels at columns 0-9).
The figures in the paper use the *six* GS13 degrees of freedom plus the CPS and
GND witnesses, so for full fidelity extend the map to the extra GS13 columns
present in your ``.mat`` (pass ``--channel-map map.json`` to the scripts, or edit
the dict below). A JSON file is simply ``{"L1:...": 0, "L1:...": 1, ...}``.
"""

import json
import numpy as np
from scipy.signal import butter, filtfilt, lfilter, welch, csd

# name -> column index in the raw .mat 'data_matrix' (from data_prep.py).
DEFAULT_CHANNEL_MAP = {
    'L1:ISI-GND_STS_ITMY_X_DQ': 0,
    'L1:ISI-GND_STS_ITMY_Y_DQ': 1,
    'L1:ISI-GND_STS_ITMY_Z_DQ': 2,
    'L1:ISI-HAM5_SCSUM_CPS_X_IN_DQ': 3,
    'L1:ISI-HAM5_SCSUM_CPS_Y_IN_DQ': 4,
    'L1:ISI-HAM5_SCSUM_CPS_Z_IN_DQ': 5,
    'L1:ISI-HAM5_BLND_CPSRX_IN1_DQ': 6,
    'L1:ISI-HAM5_BLND_CPSRY_IN1_DQ': 7,
    'L1:ISI-HAM5_BLND_CPSRZ_IN1_DQ': 8,
    'L1:ISI-HAM5_BLND_GS13X_IN1_DQ': 9,
}

# Short labels for plotting.
SHORT = {
    'L1:ISI-GND_STS_ITMY_X_DQ': 'GND_X', 'L1:ISI-GND_STS_ITMY_Y_DQ': 'GND_Y',
    'L1:ISI-GND_STS_ITMY_Z_DQ': 'GND_Z', 'L1:ISI-HAM5_SCSUM_CPS_X_IN_DQ': 'CPS_X',
    'L1:ISI-HAM5_SCSUM_CPS_Y_IN_DQ': 'CPS_Y', 'L1:ISI-HAM5_SCSUM_CPS_Z_IN_DQ': 'CPS_Z',
    'L1:ISI-HAM5_BLND_CPSRX_IN1_DQ': 'CPS_RX', 'L1:ISI-HAM5_BLND_CPSRY_IN1_DQ': 'CPS_RY',
    'L1:ISI-HAM5_BLND_CPSRZ_IN1_DQ': 'CPS_RZ', 'L1:ISI-HAM5_BLND_GS13X_IN1_DQ': 'GS13_X',
}


def short_label(name):
    return SHORT.get(name, name.split(':')[-1] if ':' in name else name)


def load_channel_map(path=None):
    """Return the channel name->column map (JSON file overrides the default)."""
    if path is None:
        return dict(DEFAULT_CHANNEL_MAP)
    with open(path) as fh:
        return json.load(fh)


def load_mat(path, channels, channel_map, skip_seconds=100, sample_rate=128.0):
    """Load selected channels from the raw .mat.

    Returns ``(data, fs)`` with ``data`` of shape ``(len(channels), N)`` after
    dropping the first ``skip_seconds`` for stability.
    """
    import scipy.io
    mat = scipy.io.loadmat(path)
    matrix = mat['data_matrix']
    start = int(skip_seconds * sample_rate)
    cols = [channel_map[c] for c in channels]
    data = np.asarray(matrix[start:, cols], dtype=np.float64).T  # (C, N)
    return data, float(sample_rate)


def load_npy(path, channels, channel_map):
    """Load selected channels from a preprocessed .npy (already (C_all, N))."""
    arr = np.load(path)
    rows = [channel_map[c] for c in channels]
    return np.asarray(arr[rows], dtype=np.float64)


# --------------------------------------------------------------------------- #
# Bandpass filters: the three variants compared in the paper.
# --------------------------------------------------------------------------- #
def _as2d(x):
    x = np.asarray(x, dtype=np.float64)
    return (x[None, :], True) if x.ndim == 1 else (x, False)


def bandpass(x, fs, f1, f2, kind="causal", order=4):
    """Bandpass ``x`` (1D or (C, N)) with one of three filters.

    kind = 'tophat'  : non-causal brick-wall (FFT) -- ideal, non-realizable bound
    kind = 'f2'      : non-causal, ~f^2 roll-off (1st-order Butterworth, zero-phase)
    kind = 'causal'  : causal Butterworth (realizable, used for real-time control)
    """
    x2, squeeze = _as2d(x)
    if kind == "tophat":
        n = x2.shape[1]
        freqs = np.fft.rfftfreq(n, d=1.0 / fs)
        mask = (freqs >= f1) & (freqs <= f2)
        out = np.fft.irfft(np.fft.rfft(x2, axis=1) * mask, n=n, axis=1)
    elif kind == "f2":
        b, a = butter(1, [f1, f2], btype="band", fs=fs)
        out = filtfilt(b, a, x2, axis=1)            # zero-phase => non-causal
    elif kind == "causal":
        b, a = butter(order, [f1, f2], btype="band", fs=fs)
        out = lfilter(b, a, x2, axis=1)             # causal
    else:
        raise ValueError(f"unknown filter kind: {kind}")
    return out[0] if squeeze else out


FILTER_KINDS = ("tophat", "f2", "causal")
FILTER_LABELS = {"tophat": "top-hat (non-causal)",
                 "f2": "f$^2$ roll-off (non-causal)",
                 "causal": "causal"}


# --------------------------------------------------------------------------- #
# Spectral helpers
# --------------------------------------------------------------------------- #
def asd(x, fs, nperseg=None):
    x = np.asarray(x, dtype=np.float64).reshape(-1)
    if nperseg is None:
        nperseg = min(len(x), 1024)
    f, pxx = welch(x, fs=fs, nperseg=nperseg)
    return f, np.sqrt(pxx)


def multiple_coherence(target, witnesses, fs, nperseg=None, ridge=1e-10):
    """Multiple coherence gamma_M^2(f) of ``target`` against all ``witnesses``.

    gamma_M^2 = S_yx S_xx^{-1} S_xy / S_yy  (real part), in [0, 1]. It is the
    fraction of the target power linearly predictable from the witnesses; the
    best-case linear residual ASD is ``ASD_target * sqrt(1 - gamma_M^2)``.

    Parameters
    ----------
    target : (N,)
    witnesses : (k, N)
    Returns ``(freqs, gamma2)``.
    """
    target = np.asarray(target, dtype=np.float64).reshape(-1)
    witnesses = np.asarray(witnesses, dtype=np.float64)
    k = witnesses.shape[0]
    if nperseg is None:
        nperseg = min(len(target), 1024)

    f, Syy = welch(target, fs=fs, nperseg=nperseg)
    nf = len(f)
    Syx = np.empty((nf, k), dtype=complex)          # S between y and each x
    for j in range(k):
        _, Syx[:, j] = csd(target, witnesses[j], fs=fs, nperseg=nperseg)
    Sxx = np.empty((nf, k, k), dtype=complex)
    for i in range(k):
        _, Sxx[:, i, i] = welch(witnesses[i], fs=fs, nperseg=nperseg)
        for j in range(i + 1, k):
            _, sij = csd(witnesses[i], witnesses[j], fs=fs, nperseg=nperseg)
            Sxx[:, i, j] = sij
            Sxx[:, j, i] = np.conj(sij)

    gamma2 = np.zeros(nf)
    eye = np.eye(k)
    with np.errstate(divide="ignore", invalid="ignore", over="ignore"):
        for m in range(nf):
            if not np.isfinite(Syy[m]) or Syy[m] <= 0:
                continue
            A = Sxx[m] + ridge * np.trace(Sxx[m]).real / k * eye
            try:
                sol = np.linalg.solve(A, Syx[m].conj())
            except np.linalg.LinAlgError:
                continue
            num = np.real(Syx[m] @ sol)
            val = num / Syy[m]
            gamma2[m] = val if np.isfinite(val) else 0.0
    return f, np.clip(gamma2, 0.0, 1.0)
