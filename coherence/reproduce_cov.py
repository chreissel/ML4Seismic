"""Reproduce the coherence-matrix figure (paper Fig. 'cov').

Three band-averaged pairwise coherence matrices over all channels:
(a) broad band (0.1 -- min(100, Nyquist) Hz) at time t1,
(b) narrow microseismic band (0.1 -- 0.3 Hz) at time t1,
(c) the same narrow band at a different time t2.
The broad window shows the expected simple structure while the narrow window
reveals cross couplings that, crucially, differ between t1 and t2 -- the
time-dependent coupling highlighted in the paper.

Usage
-----
    python -m coherence.reproduce_cov --mat MLdata_..._v2.mat
    python -m coherence.reproduce_cov --npy data/train_1381528818.npy --npy-fs 4

For the full 0.1--100 Hz panel you need the raw broadband .mat; the 4 Hz .npy
only supports up to ~2 Hz.
"""
import argparse
import os

import numpy as np
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

import seismic_data as sd
from .coherence import coherency_matrices


def main():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    src = p.add_mutually_exclusive_group(required=True)
    src.add_argument("--mat", help="raw broadband .mat path")
    src.add_argument("--npy", help="preprocessed .npy path")
    p.add_argument("--channel-map", help="JSON name->column map override")
    p.add_argument("--channels", nargs="+",
                   help="channels to include (default: all in the map, by column)")
    p.add_argument("--sample-rate", type=float, default=128.0, help="raw .mat rate [Hz]")
    p.add_argument("--npy-fs", type=float, default=4.0, help="rate of the .npy [Hz]")
    p.add_argument("--seg-len", type=float, default=20.0, help="segment length [s]")
    p.add_argument("--t1", type=float, default=0.0, help="start of segment 1 [s]")
    p.add_argument("--t2", type=float, default=20.0, help="start of segment 2 [s]")
    p.add_argument("--broad", type=float, nargs=2, default=None,
                   help="broad band [Hz] (default 0.1 .. min(100, Nyquist))")
    p.add_argument("--narrow", type=float, nargs=2, default=[0.1, 0.3],
                   help="narrow band [Hz]")
    p.add_argument("--nperseg", type=int, default=512)
    p.add_argument("--annotate-threshold", type=float, default=0.7,
                   help="annotate cells whose |signed coherency| >= this (0 = all, "
                        ">1 = none)")
    p.add_argument("--outdir", default="coherence/results")
    args = p.parse_args()
    os.makedirs(args.outdir, exist_ok=True)

    cmap = sd.load_channel_map(args.channel_map)
    channels = args.channels or sorted(cmap, key=lambda c: cmap[c])
    if args.mat:
        data, fs = sd.load_mat(args.mat, channels, cmap, sample_rate=args.sample_rate)
    else:
        data, fs = sd.load_npy(args.npy, channels, cmap), args.npy_fs

    broad = tuple(args.broad) if args.broad else (0.1, min(100.0, fs / 2))
    narrow = tuple(args.narrow)
    labels = [sd.short_label(c) for c in channels]

    def segment(t0):
        i0 = int(t0 * fs)
        i1 = i0 + int(args.seg_len * fs)
        return data[:, i0:i1]

    seg1, seg2 = segment(args.t1), segment(args.t2)
    nperseg = min(args.nperseg, seg1.shape[1])

    panels = [
        (seg1, broad, f"(a) {broad[0]:g}-{broad[1]:g} Hz, t={args.t1:g}s"),
        (seg1, narrow, f"(b) {narrow[0]:g}-{narrow[1]:g} Hz, t={args.t1:g}s"),
        (seg2, narrow, f"(c) {narrow[0]:g}-{narrow[1]:g} Hz, t={args.t2:g}s"),
    ]
    n = len(labels)
    fs_annot = max(4, min(8, int(90 / n)))   # shrink the numbers as channels grow
    fig, axes = plt.subplots(1, 3, figsize=(17, 5.4))
    im = None
    for ax, (seg, band, title) in zip(axes, panels):
        coh, signed = coherency_matrices(seg, fs, band[0], band[1], nperseg)
        im = ax.imshow(coh, vmin=0, vmax=1, cmap="viridis")
        # annotate the significant cells with the signed coherency (skip diagonal)
        for i in range(n):
            for j in range(n):
                if i == j:
                    continue
                v = signed[i, j]
                if np.isfinite(v) and abs(v) >= args.annotate_threshold:
                    ax.text(j, i, f"{v:.2f}", ha="center", va="center",
                            fontsize=fs_annot,
                            color="black" if coh[i, j] > 0.6 else "white")
        ax.set_title(title, fontsize=11)
        ax.set_xticks(range(n))
        ax.set_xticklabels(labels, rotation=90, fontsize=7)
        ax.set_yticks(range(n))
        ax.set_yticklabels(labels if ax is axes[0] else [], fontsize=7)
    fig.colorbar(im, ax=axes, fraction=0.025, pad=0.02, label="Average Coherence")
    fig.suptitle("Average pairwise coherence: broad band vs microseismic band at two times")
    for ext in ("png", "pdf"):
        fig.savefig(os.path.join(args.outdir, f"coherence_cov.{ext}"), dpi=150,
                    bbox_inches="tight")
    plt.close(fig)
    print(f"Wrote coherence_cov.png/.pdf to {args.outdir}/")


if __name__ == "__main__":
    main()
