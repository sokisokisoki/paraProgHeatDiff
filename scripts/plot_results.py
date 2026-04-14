#!/usr/bin/env python3
"""
Parse heat diffusion CSV rows (from --csv or scripts/run_strong_weak.sh logs) and write PNG plots.

Dependencies: matplotlib, numpy (see requirements-plot.txt)

Example:
  python3 scripts/plot_results.py scaling_log.csv -o plots/
"""

from __future__ import annotations

import argparse
import csv
from collections import defaultdict
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np


def _to_float(x: str) -> float:
    return float(x)


def _to_int(x: str) -> int:
    return int(x)


def parse_log_text(text: str) -> list[dict[str, Any]]:
    """Collect data rows; skip comments and 'mode' header lines."""
    rows: list[dict[str, Any]] = []
    for raw in text.splitlines():
        line = raw.strip()
        if not line or line.startswith("#"):
            continue
        if line.startswith("=="):
            continue
        try:
            parts = next(csv.reader([line]))
        except StopIteration:
            continue
        if not parts or parts[0] == "mode":
            continue
        mode = parts[0]
        if mode == "serial" and len(parts) >= 7:
            rows.append(
                {
                    "mode": mode,
                    "nx": _to_int(parts[1]),
                    "ny": _to_int(parts[2]),
                    "niters": _to_int(parts[3]),
                    "nprocs": _to_int(parts[4]),
                    "checksum": _to_float(parts[5]),
                    "wall_s": _to_float(parts[6]),
                    "t_comm": 0.0,
                    "t_comp": _to_float(parts[6]),
                    "t_io": 0.0,
                }
            )
        elif mode == "mpi_cpu" and len(parts) >= 13:
            rows.append(
                {
                    "mode": mode,
                    "nx": _to_int(parts[1]),
                    "ny": _to_int(parts[2]),
                    "niters": _to_int(parts[3]),
                    "nprocs": _to_int(parts[4]),
                    "checksum": _to_float(parts[5]),
                    "wall_s": _to_float(parts[6]),
                    "t_comm": _to_float(parts[7]),
                    "t_comp": _to_float(parts[8]),
                    "t_io": _to_float(parts[9]),
                    "cyc_comm": int(parts[10]) if parts[10] else 0,
                    "cyc_comp": int(parts[11]) if parts[11] else 0,
                    "cyc_io": int(parts[12]) if parts[12] else 0,
                }
            )
        elif mode == "mpi_cuda" and len(parts) >= 17:
            rows.append(
                {
                    "mode": mode,
                    "nx": _to_int(parts[1]),
                    "ny": _to_int(parts[2]),
                    "niters": _to_int(parts[3]),
                    "nprocs": _to_int(parts[4]),
                    "checksum": _to_float(parts[5]),
                    "wall_s": _to_float(parts[6]),
                    "t_comm": _to_float(parts[7]),
                    "t_comp": _to_float(parts[8]),
                    "t_halo_d2h": _to_float(parts[9]),
                    "t_halo_h2d": _to_float(parts[10]),
                    "t_io": _to_float(parts[11]),
                    "cyc_comm": int(parts[12]) if parts[12] else 0,
                    "cyc_comp": int(parts[13]) if parts[13] else 0,
                    "cyc_halo_d2h": int(parts[14]) if parts[14] else 0,
                    "cyc_halo_h2d": int(parts[15]) if parts[15] else 0,
                    "cyc_io": int(parts[16]) if parts[16] else 0,
                }
            )
    return rows


def parse_log_files(paths: list[Path]) -> list[dict[str, Any]]:
    chunks: list[str] = []
    for p in paths:
        chunks.append(p.read_text(encoding="utf-8", errors="replace"))
    return parse_log_text("\n".join(chunks))


def group_key_mpi(r: dict[str, Any]) -> tuple[str, int, int, int]:
    return (r["mode"], r["nx"], r["ny"], r["niters"])


def plot_walltime_vs_ranks(rows: list[dict[str, Any]], outdir: Path) -> None:
    """Strong-style: wall clock vs MPI ranks for each (mode, nx, ny, niters) group."""
    mpi = [r for r in rows if r["mode"] in ("mpi_cpu", "mpi_cuda")]
    if not mpi:
        return
    groups: dict[tuple[str, int, int, int], list[dict[str, Any]]] = defaultdict(list)
    for r in mpi:
        groups[group_key_mpi(r)].append(r)

    fig, ax = plt.subplots(figsize=(8, 5))
    markers = {"mpi_cpu": "o-", "mpi_cuda": "s-"}
    for (mode, nx, ny, nit), grp in sorted(groups.items()):
        grp = sorted(grp, key=lambda x: x["nprocs"])
        xs = [r["nprocs"] for r in grp]
        ys = [r["wall_s"] for r in grp]
        label = f"{mode} nx={nx} ny={ny} n={nit}"
        ax.plot(xs, ys, markers.get(mode, "o-"), label=label, linewidth=2, markersize=6)

    ax.set_xlabel("MPI ranks")
    ax.set_ylabel("Wall time (s)")
    ax.set_title("Wall time vs ranks (from log)")
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(outdir / "walltime_vs_ranks.png", dpi=150)
    plt.close(fig)


def plot_speedup(rows: list[dict[str, Any]], outdir: Path) -> None:
    """Speedup vs ideal, using smallest rank count in each group as T_baseline."""
    mpi = [r for r in rows if r["mode"] in ("mpi_cpu", "mpi_cuda")]
    if not mpi:
        return
    groups: dict[tuple[str, int, int, int], list[dict[str, Any]]] = defaultdict(list)
    for r in mpi:
        groups[group_key_mpi(r)].append(r)

    fig, ax = plt.subplots(figsize=(8, 5))
    for (mode, nx, ny, nit), grp in sorted(groups.items()):
        grp = sorted(grp, key=lambda x: x["nprocs"])
        p0 = grp[0]["nprocs"]
        t0 = grp[0]["wall_s"]
        if t0 <= 0:
            continue
        xs = [r["nprocs"] for r in grp]
        ideal = [r["nprocs"] / p0 for r in grp]
        actual = [t0 / r["wall_s"] for r in grp]
        label = f"{mode} nx={nx} ny={ny}"
        ax.plot(xs, actual, "o-", label=f"{label} actual", linewidth=2)
        ax.plot(xs, ideal, "--", alpha=0.45, label=f"{label} ideal")

    ax.set_xlabel("MPI ranks")
    ax.set_ylabel("Speedup")
    ax.set_title("Speedup (baseline = smallest P in group)")
    ax.legend(fontsize=7, ncol=2)
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(outdir / "speedup.png", dpi=150)
    plt.close(fig)


def plot_phase_fraction(rows: list[dict[str, Any]], outdir: Path) -> None:
    """Stacked fraction of time: compute vs MPI vs I/O (and halo copies for CUDA)."""
    mpi = [r for r in rows if r["mode"] in ("mpi_cpu", "mpi_cuda")]
    if not mpi:
        return

    fig, axes = plt.subplots(1, 2, figsize=(11, 5), sharey=True)
    for ax, mode in zip(axes, ("mpi_cpu", "mpi_cuda")):
        sub = [r for r in mpi if r["mode"] == mode]
        if not sub:
            ax.set_visible(False)
            continue
        sub = sorted(sub, key=lambda x: (x["nx"], x["ny"], x["niters"], x["nprocs"]))[-8:]
        labels = [str(r["nprocs"]) for r in sub]
        w = np.array([max(r["wall_s"], 1e-30) for r in sub])
        comm = np.array([r["t_comm"] for r in sub]) / w
        comp = np.array([r["t_comp"] for r in sub]) / w
        io = np.array([r["t_io"] for r in sub]) / w
        if mode == "mpi_cuda":
            d2h = np.array([r.get("t_halo_d2h", 0.0) for r in sub]) / w
            h2d = np.array([r.get("t_halo_h2d", 0.0) for r in sub]) / w
            rest = 1.0 - (comm + comp + io + d2h + h2d)
            rest = np.maximum(rest, 0.0)
            ax.bar(labels, comp, label="compute")
            ax.bar(labels, comm, bottom=comp, label="MPI")
            b = comp + comm
            ax.bar(labels, d2h, bottom=b, label="halo D2H")
            b = b + d2h
            ax.bar(labels, h2d, bottom=b, label="halo H2D")
            b = b + h2d
            ax.bar(labels, io, bottom=b, label="I/O")
            b = b + io
            ax.bar(labels, rest, bottom=b, label="other", color="#dddddd")
        else:
            rest = 1.0 - (comm + comp + io)
            rest = np.maximum(rest, 0.0)
            ax.bar(labels, comp, label="compute")
            ax.bar(labels, comm, bottom=comp, label="MPI")
            b = comp + comm
            ax.bar(labels, io, bottom=b, label="I/O")
            b = b + io
            ax.bar(labels, rest, bottom=b, label="other", color="#dddddd")
        ax.set_xlabel("MPI ranks (last runs in log)")
        ax.set_ylabel("Fraction of wall time")
        ax.set_title(mode)
        ax.legend(fontsize=7, loc="upper right")
        ax.set_ylim(0, 1.05)
        ax.grid(True, axis="y", alpha=0.3)
    fig.suptitle("Phase breakdown (normalized to wall time)")
    fig.tight_layout()
    fig.savefig(outdir / "phase_fraction.png", dpi=150)
    plt.close(fig)


def plot_comm_vs_compute(rows: list[dict[str, Any]], outdir: Path) -> None:
    """Scatter: communication time vs compute time per run."""
    mpi = [r for r in rows if r["mode"] in ("mpi_cpu", "mpi_cuda")]
    if not mpi:
        return
    fig, ax = plt.subplots(figsize=(7, 6))
    for mode, c, m in (("mpi_cpu", "C0", "o"), ("mpi_cuda", "C1", "s")):
        sub = [r for r in mpi if r["mode"] == mode]
        if not sub:
            continue
        xc = [r["t_comm"] for r in sub]
        yc = [r["t_comp"] for r in sub]
        sz = [40 + 15 * r["nprocs"] for r in sub]
        ax.scatter(xc, yc, s=sz, alpha=0.7, c=c, marker=m, edgecolors="k", linewidths=0.3, label=mode)
        for r in sub:
            ax.annotate(
                str(r["nprocs"]),
                (r["t_comm"], r["t_comp"]),
                textcoords="offset points",
                xytext=(3, 3),
                fontsize=7,
            )
    ax.set_xlabel("MPI halo time (s)")
    ax.set_ylabel("Compute time (s)")
    ax.set_title("Communication vs compute (marker size ∝ ranks; label = ranks)")
    ax.legend()
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(outdir / "comm_vs_compute.png", dpi=150)
    plt.close(fig)


def main() -> None:
    ap = argparse.ArgumentParser(description="Plot heat diffusion CSV logs.")
    ap.add_argument("inputs", nargs="+", type=Path, help="Log file(s) containing CSV data rows")
    ap.add_argument("-o", "--outdir", type=Path, default=Path("plots"), help="Output directory for PNGs")
    args = ap.parse_args()

    for p in args.inputs:
        if not p.is_file():
            raise SystemExit(f"Not a file: {p}")

    args.outdir.mkdir(parents=True, exist_ok=True)
    rows = parse_log_files(args.inputs)
    if not rows:
        raise SystemExit("No data rows parsed (expect lines starting with serial, mpi_cpu, or mpi_cuda).")

    plot_walltime_vs_ranks(rows, args.outdir)
    plot_speedup(rows, args.outdir)
    plot_phase_fraction(rows, args.outdir)
    plot_comm_vs_compute(rows, args.outdir)

    print(f"Wrote plots under {args.outdir.resolve()}")


if __name__ == "__main__":
    main()
