"""Reproduce the coefficient-evolution figure (paper Fig. 'spike').

Refit the linear regression of the GS13 target on the CPS+GND witnesses every
``--step`` seconds and plot the evolution of the most significant coefficients
with their per-fit statistical (OLS) uncertainty. The fitted parameters fluctuate
well beyond their per-fit error bars, demonstrating genuine non-stationarity of
the linear coupling rather than estimation noise.

Usage
-----
    python -m linear.coefficient_evolution --mat MLdata_..._v2.mat
    python -m linear.coefficient_evolution --npy data/train_1381528818.npy --npy-fs 4
"""
import argparse
import os

import numpy as np
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

import seismic_data as sd

DEFAULT_WITNESS = [
    'L1:ISI-GND_STS_ITMY_X_DQ', 'L1:ISI-GND_STS_ITMY_Y_DQ', 'L1:ISI-GND_STS_ITMY_Z_DQ',
    'L1:ISI-HAM5_SCSUM_CPS_X_IN_DQ', 'L1:ISI-HAM5_SCSUM_CPS_Y_IN_DQ', 'L1:ISI-HAM5_SCSUM_CPS_Z_IN_DQ',
    'L1:ISI-HAM5_BLND_CPSRX_IN1_DQ', 'L1:ISI-HAM5_BLND_CPSRY_IN1_DQ', 'L1:ISI-HAM5_BLND_CPSRZ_IN1_DQ',
]
DEFAULT_TARGET = 'L1:ISI-HAM5_BLND_GS13X_IN1_DQ'


def ols(X, y):
    """OLS fit with statistical uncertainty. Returns ``(beta, se)`` incl intercept."""
    n, p = X.shape
    XtX = X.T @ X
    XtX_inv = np.linalg.pinv(XtX)
    beta = XtX_inv @ (X.T @ y)
    resid = y - X @ beta
    dof = max(n - p, 1)
    sigma2 = float(resid @ resid) / dof
    se = np.sqrt(np.maximum(np.diag(XtX_inv) * sigma2, 0.0))
    return beta, se


def main():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    src = p.add_mutually_exclusive_group(required=True)
    src.add_argument("--mat", help="raw broadband .mat path")
    src.add_argument("--npy", help="preprocessed .npy path")
    p.add_argument("--channel-map", help="JSON name->column map override")
    p.add_argument("--target", default=DEFAULT_TARGET)
    p.add_argument("--witness", nargs="+", default=DEFAULT_WITNESS)
    p.add_argument("--sample-rate", type=float, default=128.0, help="raw .mat rate [Hz]")
    p.add_argument("--npy-fs", type=float, default=4.0, help="rate of the .npy [Hz]")
    p.add_argument("--fmin", type=float, default=0.1)
    p.add_argument("--fmax", type=float, default=0.3)
    p.add_argument("--filter", default="causal", choices=sd.FILTER_KINDS)
    p.add_argument("--step", type=float, default=10.0, help="refit interval [s]")
    p.add_argument("--win", type=float, default=None,
                   help="fit window length [s] (default = step)")
    p.add_argument("--top", type=int, default=5,
                   help="number of most-significant coefficients to plot")
    p.add_argument("--outdir", default="linear/results")
    args = p.parse_args()
    os.makedirs(args.outdir, exist_ok=True)
    win = args.win or args.step

    cmap = sd.load_channel_map(args.channel_map)
    channels = list(args.witness) + [args.target]
    if args.mat:
        raw, fs = sd.load_mat(args.mat, channels, cmap, sample_rate=args.sample_rate)
    else:
        raw, fs = sd.load_npy(args.npy, channels, cmap), args.npy_fs

    # band-limit, then standardise witnesses globally so coefficients are comparable
    filt = sd.bandpass(raw, fs, args.fmin, args.fmax, kind=args.filter)
    W = filt[:-1]
    y = filt[-1]
    mean = W.mean(axis=1, keepdims=True)
    std = W.std(axis=1, keepdims=True) + 1e-12
    Wn = (W - mean) / std

    step = int(args.step * fs)
    wlen = int(win * fs)
    n = raw.shape[1]
    starts = list(range(0, n - wlen + 1, step))
    times = np.array([(s + wlen / 2) / fs for s in starts])

    n_w = W.shape[0]
    betas = np.full((len(starts), n_w), np.nan)   # exclude intercept
    ses = np.full((len(starts), n_w), np.nan)
    for k, s in enumerate(starts):
        Xw = Wn[:, s:s + wlen].T
        X = np.column_stack([Xw, np.ones(Xw.shape[0])])   # + intercept
        b, e = ols(X, y[s:s + wlen])
        betas[k] = b[:-1]
        ses[k] = e[:-1]

    # rank channels by how much they move relative to their error bars
    spread = np.nanstd(betas, axis=0)
    median_se = np.nanmedian(ses, axis=0)
    score = spread / (median_se + 1e-12)
    order = np.argsort(score)[::-1][:min(args.top, n_w)]

    plt.figure(figsize=(10, 5.5))
    for j in order:
        lbl = sd.short_label(args.witness[j])
        line, = plt.plot(times, betas[:, j], marker="o", ms=3, lw=1, label=lbl)
        plt.fill_between(times, betas[:, j] - ses[:, j], betas[:, j] + ses[:, j],
                         color=line.get_color(), alpha=0.2)
    plt.axhline(0, color="k", lw=0.6, alpha=0.5)
    plt.xlabel("Time [s]")
    plt.ylabel("Standardised regression coefficient")
    plt.title(f"Coefficient evolution predicting {sd.short_label(args.target)} "
              f"(refit every {args.step:g}s, {args.filter} filter); bands = $\\pm1\\sigma$")
    plt.legend(fontsize=8, ncol=2)
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    for ext in ("png", "pdf"):
        plt.savefig(os.path.join(args.outdir, f"coefficient_evolution.{ext}"), dpi=150)
    plt.close()
    print(f"Refit {len(starts)} windows. Wrote coefficient_evolution.png/.pdf to {args.outdir}/")


if __name__ == "__main__":
    main()
