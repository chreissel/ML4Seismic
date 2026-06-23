"""Run the linear (Wiener / FIR least-squares) noise subtraction baseline.

Fits a multichannel FIR Wiener filter on the training segment to predict the
GS13 target channel from the witness sensors, then evaluates the subtraction on
the held-out test segment. Reproduces the *linear* baseline of arXiv:2511.19682.

Usage
-----
    python -m linear.run_linear                      # defaults
    python -m linear.run_linear --n-taps 128 --alpha 1e-2 --acausal

Outputs (written to ``--outdir``, default ``linear/results/``):
    * metrics.txt            band-limited RMS before/after + reduction factor
    * asd.png                target ASD before vs after subtraction
    * timeseries.png         target / prediction / residual time series
"""

import argparse
import os

import numpy as np
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

from .wiener import LinearSubtractor, asd, band_rms

# Channel ordering of the provided .npy arrays (see ../data_prep.py).
CHANNELS = [
    "GND_STS_ITMY_X", "GND_STS_ITMY_Y", "GND_STS_ITMY_Z",
    "HAM5_CPS_X", "HAM5_CPS_Y", "HAM5_CPS_Z",
    "HAM5_CPSRX", "HAM5_CPSRY", "HAM5_CPSRZ",
    "HAM5_GS13X",
]


def load_split(time, split, data_dir):
    """Load a (n_channels, n_samples) array for the given split."""
    path = os.path.join(data_dir, f"{split}_{time}.npy")
    return np.load(path)


def split_target_witness(data, target_idx, witness_idx=None):
    """Return ``(witnesses (C, N), target (N,))``.

    ``witness_idx`` selects which channels to use as witnesses. If ``None`` all
    channels except the target are used.
    """
    target = data[target_idx]
    if witness_idx is None:
        witnesses = np.delete(data, target_idx, axis=0)
    else:
        witnesses = data[list(witness_idx)]
    return witnesses, target


def main():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--time", type=int, default=1381528818, help="data timestamp tag")
    p.add_argument("--data-dir", default="data", help="directory holding the .npy splits")
    p.add_argument("--target-idx", type=int, default=9, help="index of the target channel")
    p.add_argument("--witness-idx", type=int, nargs="+",
                   default=[0, 1, 2, 3, 4, 5, 6, 7, 8],
                   help="channel indices to use as witnesses (default: all GND and CPS "
                        "directions, i.e. X/Y/Z + RX/RY/RZ). The GS13 target (idx 9) is "
                        "never used as a witness.")
    p.add_argument("--fs", type=float, default=4.0, help="sampling rate [Hz]")
    p.add_argument("--n-taps", type=int, default=64, help="FIR taps per witness channel")
    p.add_argument("--alpha", type=float, default=1e-3, help="ridge regularisation strength")
    p.add_argument("--acausal", action="store_true",
                   help="use a centered (non-causal) filter instead of causal")
    p.add_argument("--fmin", type=float, default=0.1, help="microseismic band low edge [Hz]")
    p.add_argument("--fmax", type=float, default=0.3, help="microseismic band high edge [Hz]")
    p.add_argument("--outdir", default="linear/results", help="output directory")
    args = p.parse_args()

    os.makedirs(args.outdir, exist_ok=True)

    train = load_split(args.time, "train", args.data_dir)
    test = load_split(args.time, "test", args.data_dir)

    Xtr, ytr = split_target_witness(train, args.target_idx, args.witness_idx)
    Xte, yte = split_target_witness(test, args.target_idx, args.witness_idx)
    used = [CHANNELS[i] for i in args.witness_idx]
    print(f"Witness channels: {used}\nTarget channel:  {CHANNELS[args.target_idx]}\n")

    model = LinearSubtractor(n_taps=args.n_taps, alpha=args.alpha,
                             causal=not args.acausal)
    model.fit(Xtr, ytr)

    pred, resid = model.clean(Xte, yte)

    # band-limited RMS in the microseismic band, before vs after
    nperseg = min(len(yte), 256)
    rms_before = band_rms(yte, args.fs, args.fmin, args.fmax, nperseg)
    rms_after = band_rms(resid, args.fs, args.fmin, args.fmax, nperseg)
    reduction = rms_before / rms_after if rms_after > 0 else float("inf")

    # broadband (full available band) for reference
    bb_before = band_rms(yte, args.fs, 0.0, args.fs / 2, nperseg)
    bb_after = band_rms(resid, args.fs, 0.0, args.fs / 2, nperseg)
    bb_reduction = bb_before / bb_after if bb_after > 0 else float("inf")

    lines = [
        "Linear (Wiener/FIR) noise subtraction -- test segment",
        f"  witnesses={used}",
        f"  n_taps={args.n_taps}  alpha={args.alpha}  causal={not args.acausal}",
        f"  microseismic band [{args.fmin}, {args.fmax}] Hz:",
        f"    RMS before = {rms_before:.6g}",
        f"    RMS after  = {rms_after:.6g}",
        f"    reduction factor = {reduction:.3f}x",
        f"  broadband [0, {args.fs/2}] Hz:",
        f"    RMS before = {bb_before:.6g}",
        f"    RMS after  = {bb_after:.6g}",
        f"    reduction factor = {bb_reduction:.3f}x",
    ]
    report = "\n".join(lines)
    print(report)
    with open(os.path.join(args.outdir, "metrics.txt"), "w") as fh:
        fh.write(report + "\n")

    # --- ASD before/after ---
    f, asd_before = asd(yte, args.fs, nperseg)
    _, asd_after = asd(resid, args.fs, nperseg)
    plt.figure(figsize=(7, 4.5))
    plt.loglog(f, asd_before, label="target (before)")
    plt.loglog(f, asd_after, label="residual (after)")
    plt.axvspan(args.fmin, args.fmax, color="grey", alpha=0.2, label="microseismic band")
    plt.xlabel("Frequency [Hz]")
    plt.ylabel("ASD [arb/$\\sqrt{\\mathrm{Hz}}$]")
    plt.title("Linear subtraction: target ASD")
    plt.legend()
    plt.grid(True, which="both", alpha=0.3)
    plt.tight_layout()
    plt.savefig(os.path.join(args.outdir, "asd.png"), dpi=150)
    plt.close()

    # --- time series ---
    t = np.arange(len(yte)) / args.fs
    plt.figure(figsize=(9, 4.5))
    plt.plot(t, yte, label="target", lw=0.8)
    plt.plot(t, pred, label="linear prediction", lw=0.8)
    plt.plot(t, resid, label="residual", lw=0.8)
    plt.xlabel("Time [s]")
    plt.ylabel("Amplitude [arb]")
    plt.title("Linear subtraction: time series (test)")
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(os.path.join(args.outdir, "timeseries.png"), dpi=150)
    plt.close()

    print(f"\nWrote results to {args.outdir}/")


if __name__ == "__main__":
    main()
