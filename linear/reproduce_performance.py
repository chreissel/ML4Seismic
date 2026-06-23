"""Reproduce the linear-performance figure (paper Fig. 'performance').

Three-panel ASD of the GS13 target (blue), the linear-regression prediction from
the CPS+GND witnesses (green), and the residual (orange), for the three filters:
(a) non-causal top-hat, (b) non-causal f^2 roll-off, (c) causal. Each panel also
shows the linear-subtraction floor ``ASD_target * sqrt(1 - gamma_M^2(f))`` set by
the multiple coherence against all witnesses -- the residual bound for any linear
filter.

Usage
-----
    # raw broadband .mat (full fidelity)
    python -m linear.reproduce_performance --mat MLdata_..._v2.mat
    # quick check on the shipped narrowband .npy
    python -m linear.reproduce_performance --npy data/train_1381528818.npy

Set --target / --witness to channel names, or --channel-map map.json to point at
the columns of your .mat (see seismic_data.py).
"""
import argparse
import os

import numpy as np
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

import seismic_data as sd
from .wiener import LinearSubtractor

DEFAULT_WITNESS = [
    'L1:ISI-GND_STS_ITMY_X_DQ', 'L1:ISI-GND_STS_ITMY_Y_DQ', 'L1:ISI-GND_STS_ITMY_Z_DQ',
    'L1:ISI-HAM5_SCSUM_CPS_X_IN_DQ', 'L1:ISI-HAM5_SCSUM_CPS_Y_IN_DQ', 'L1:ISI-HAM5_SCSUM_CPS_Z_IN_DQ',
    'L1:ISI-HAM5_BLND_CPSRX_IN1_DQ', 'L1:ISI-HAM5_BLND_CPSRY_IN1_DQ', 'L1:ISI-HAM5_BLND_CPSRZ_IN1_DQ',
]
DEFAULT_TARGET = 'L1:ISI-HAM5_BLND_GS13X_IN1_DQ'


def main():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    src = p.add_mutually_exclusive_group(required=True)
    src.add_argument("--mat", help="raw broadband .mat path")
    src.add_argument("--npy", help="preprocessed .npy path (already band-limited)")
    p.add_argument("--channel-map", help="JSON name->column map override")
    p.add_argument("--target", default=DEFAULT_TARGET)
    p.add_argument("--witness", nargs="+", default=DEFAULT_WITNESS)
    p.add_argument("--sample-rate", type=float, default=128.0, help="raw .mat rate [Hz]")
    p.add_argument("--npy-fs", type=float, default=4.0, help="rate of the .npy [Hz]")
    p.add_argument("--fmin", type=float, default=0.1, help="band low edge [Hz]")
    p.add_argument("--fmax", type=float, default=0.3, help="band high edge [Hz]")
    p.add_argument("--n-taps", type=int, default=1,
                   help="FIR taps for the regression (1 = instantaneous linear regression)")
    p.add_argument("--alpha", type=float, default=1e-6, help="ridge strength")
    p.add_argument("--nperseg", type=int, default=1024, help="Welch segment length")
    p.add_argument("--outdir", default="linear/results")
    args = p.parse_args()
    os.makedirs(args.outdir, exist_ok=True)

    cmap = sd.load_channel_map(args.channel_map)
    channels = list(args.witness) + [args.target]
    if args.mat:
        raw, fs = sd.load_mat(args.mat, channels, cmap, sample_rate=args.sample_rate)
    else:
        raw = sd.load_npy(args.npy, channels, cmap)
        fs = args.npy_fs
    witnesses_raw = raw[:-1]
    target_raw = raw[-1]
    nperseg = min(args.nperseg, raw.shape[1])

    fig, axes = plt.subplots(1, 3, figsize=(16, 4.6), sharey=True)
    panel = ["(a)", "(b)", "(c)"]
    for ax, kind, tag in zip(axes, sd.FILTER_KINDS, panel):
        # bandpass every channel with this filter, then fit the regression
        W = sd.bandpass(witnesses_raw, fs, args.fmin, args.fmax, kind=kind)
        y = sd.bandpass(target_raw, fs, args.fmin, args.fmax, kind=kind)
        model = LinearSubtractor(n_taps=args.n_taps, alpha=args.alpha,
                                 causal=(kind == "causal"))
        model.fit(W, y)
        pred, resid = model.clean(W, y)

        f, a_t = sd.asd(y, fs, nperseg)
        _, a_p = sd.asd(pred, fs, nperseg)
        _, a_r = sd.asd(resid, fs, nperseg)
        fc, g2 = sd.multiple_coherence(y, W, fs, nperseg)
        floor = np.interp(f, fc, np.sqrt(1.0 - g2)) * a_t

        ax.loglog(f, a_t, color="tab:blue", label="GS13 target")
        ax.loglog(f, a_p, color="tab:green", label="linear prediction")
        ax.loglog(f, a_r, color="tab:orange", label="residual")
        ax.loglog(f, floor, color="k", ls="--", lw=1,
                  label=r"floor $\sqrt{1-\gamma_M^2}$")
        ax.axvspan(args.fmin, args.fmax, color="grey", alpha=0.15)
        ax.set_title(f"{tag} {sd.FILTER_LABELS[kind]}")
        ax.set_xlabel("Frequency [Hz]")
        ax.grid(True, which="both", alpha=0.3)
    axes[0].set_ylabel(r"ASD [arb/$\sqrt{\mathrm{Hz}}$]")
    axes[0].legend(fontsize=8, loc="lower left")
    fig.suptitle(f"Linear subtraction of {sd.short_label(args.target)} "
                 f"({len(args.witness)} CPS+GND witnesses)")
    fig.tight_layout()
    for ext in ("png", "pdf"):
        fig.savefig(os.path.join(args.outdir, f"linear_performance.{ext}"), dpi=150)
    plt.close(fig)
    print(f"Wrote linear_performance.png/.pdf to {args.outdir}/")


if __name__ == "__main__":
    main()
