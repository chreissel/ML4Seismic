"""Run the coherence analysis of the microseismic sensor array.

Reproduces the coherence study of arXiv:2511.19682:
    1. target-vs-witness coherence spectra,
    2. band-averaged pairwise coherence matrix in the microseismic band,
    3. time dependence of the mean target-vs-witness coherence, comparing the
       broad available band with the 0.1-0.3 Hz microseismic band -- the
       "time dependent cross couplings" highlighted in the paper.

Usage
-----
    python -m coherence.run_coherence
    python -m coherence.run_coherence --split train --n-segments 8

Note on frequency range
-----------------------
The provided .npy data is resampled to 4 Hz and bandpassed to 0.1-0.3 Hz, so the
analysis here covers the available 0.1-2 Hz band. For the paper's full 0.1-100 Hz
comparison, point ``--data-dir`` / ``--fs`` at broadband (e.g. 128 Hz) data and
set ``--broad-fmax`` accordingly.

Outputs (written to ``--outdir``, default ``coherence/results/``):
    * coherence_spectra.png   target-vs-witness coherence vs frequency
    * coherence_matrix.png    band-averaged pairwise coherence heatmap
    * coherence_time.png      mean coherence over time, broad vs microseismic
    * summary.txt             band-averaged coherence values
"""

import argparse
import os

import numpy as np
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

from .coherence import (
    pair_coherence,
    band_average,
    coherence_matrix,
    time_dependent_coherence,
)

# Channel ordering of the provided .npy arrays (see ../data_prep.py).
CHANNELS = [
    "GND_STS_ITMY_X", "GND_STS_ITMY_Y", "GND_STS_ITMY_Z",
    "HAM5_CPS_X", "HAM5_CPS_Y", "HAM5_CPS_Z",
    "HAM5_CPSRX", "HAM5_CPSRY", "HAM5_CPSRZ",
    "HAM5_GS13X",
]


def load_split(time, split, data_dir):
    return np.load(os.path.join(data_dir, f"{split}_{time}.npy"))


def main():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--time", type=int, default=1381528818, help="data timestamp tag")
    p.add_argument("--data-dir", default="data", help="directory holding the .npy splits")
    p.add_argument("--split", default="train", choices=["train", "val", "test"],
                   help="which split to analyse (train has the most samples)")
    p.add_argument("--target-idx", type=int, default=9, help="index of the target channel")
    p.add_argument("--fs", type=float, default=4.0, help="sampling rate [Hz]")
    p.add_argument("--nperseg", type=int, default=256, help="Welch segment length")
    p.add_argument("--micro-fmin", type=float, default=0.1, help="microseismic band low edge [Hz]")
    p.add_argument("--micro-fmax", type=float, default=0.3, help="microseismic band high edge [Hz]")
    p.add_argument("--broad-fmin", type=float, default=0.1, help="broad band low edge [Hz]")
    p.add_argument("--broad-fmax", type=float, default=2.0, help="broad band high edge [Hz]")
    p.add_argument("--n-segments", type=int, default=6,
                   help="number of time segments for the time-dependent analysis")
    p.add_argument("--outdir", default="coherence/results", help="output directory")
    args = p.parse_args()

    os.makedirs(args.outdir, exist_ok=True)

    data = load_split(args.time, args.split, args.data_dir)
    nperseg = min(args.nperseg, data.shape[1])
    tidx = args.target_idx
    micro = (args.micro_fmin, args.micro_fmax)
    broad = (args.broad_fmin, args.broad_fmax)

    # --- 1. target-vs-witness coherence spectra ---
    witness_idx = [c for c in range(data.shape[0]) if c != tidx]
    plt.figure(figsize=(7.5, 5))
    summary_lines = [
        f"Coherence analysis -- split={args.split}, fs={args.fs} Hz",
        f"Target channel: {CHANNELS[tidx]}",
        "",
        f"Band-averaged coherence (target vs witness), "
        f"microseismic [{micro[0]},{micro[1]}] Hz | broad [{broad[0]},{broad[1]}] Hz:",
    ]
    for w in witness_idx:
        f, cxy = pair_coherence(data[tidx], data[w], args.fs, nperseg)
        plt.plot(f, cxy, lw=1.0, label=CHANNELS[w])
        c_micro = band_average(f, cxy, *micro)
        c_broad = band_average(f, cxy, *broad)
        summary_lines.append(
            f"  {CHANNELS[w]:<16s}  micro={c_micro:.3f}  broad={c_broad:.3f}")
    plt.axvspan(micro[0], micro[1], color="grey", alpha=0.2, label="microseismic band")
    plt.xlabel("Frequency [Hz]")
    plt.ylabel("Magnitude-squared coherence")
    plt.title(f"Target ({CHANNELS[tidx]}) vs witness coherence")
    plt.ylim(0, 1)
    plt.legend(fontsize=7, ncol=2)
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(os.path.join(args.outdir, "coherence_spectra.png"), dpi=150)
    plt.close()

    # --- 2. band-averaged pairwise coherence matrix (microseismic band) ---
    M = coherence_matrix(data, args.fs, micro[0], micro[1], nperseg)
    plt.figure(figsize=(7, 6))
    im = plt.imshow(M, vmin=0, vmax=1, cmap="viridis")
    plt.colorbar(im, label="mean coherence (0.1-0.3 Hz)")
    plt.xticks(range(len(CHANNELS)), CHANNELS, rotation=90, fontsize=7)
    plt.yticks(range(len(CHANNELS)), CHANNELS, fontsize=7)
    plt.title("Pairwise coherence in the microseismic band")
    plt.tight_layout()
    plt.savefig(os.path.join(args.outdir, "coherence_matrix.png"), dpi=150)
    plt.close()

    # --- 3. time-dependent coherence (broad vs microseismic) ---
    seg_times, series = time_dependent_coherence(
        data, args.fs, [broad, micro], args.n_segments, tidx, nperseg)
    plt.figure(figsize=(7.5, 4.5))
    plt.plot(seg_times, series[broad], "o-",
             label=f"broad [{broad[0]}-{broad[1]} Hz]")
    plt.plot(seg_times, series[micro], "s-",
             label=f"microseismic [{micro[0]}-{micro[1]} Hz]")
    plt.xlabel("Time [s]")
    plt.ylabel("Mean target-vs-witness coherence")
    plt.title("Time dependence of cross-coupling coherence")
    plt.ylim(0, 1)
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(os.path.join(args.outdir, "coherence_time.png"), dpi=150)
    plt.close()

    summary_lines += [
        "",
        "Time-dependent mean coherence (target vs all witnesses):",
        "  segment_centre[s]    broad      micro",
    ]
    for t, cb, cm in zip(seg_times, series[broad], series[micro]):
        summary_lines.append(f"  {t:>14.1f}    {cb:.3f}      {cm:.3f}")

    report = "\n".join(summary_lines)
    print(report)
    with open(os.path.join(args.outdir, "summary.txt"), "w") as fh:
        fh.write(report + "\n")

    print(f"\nWrote results to {args.outdir}/")


if __name__ == "__main__":
    main()
