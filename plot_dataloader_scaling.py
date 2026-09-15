#!/usr/bin/env python3
"""Plot the dataloader worker/mode scaling benchmark.

Reads either a results CSV written by ``dataloader_scaling_benchmark.py`` or the
raw ``.out`` log it prints, and writes figures that let you read off:

  * throughput vs workers  -- data-only and end-to-end for each loader mode,
    against the storage-free GPU compute ceiling;
  * starvation vs workers  -- fraction of GPU capacity lost to the data path;
  * first-batch latency vs workers -- pipeline start-up cost per mode.

Usage:
    python plot_dataloader_scaling.py INPUT [--out-dir DIR] [--title TAG]

INPUT may be a .csv (preferred; full columns) or a benchmark .out log.
"""

from __future__ import annotations

import argparse
import re
from pathlib import Path

import matplotlib

matplotlib.use("Agg")  # headless / batch-node safe
import matplotlib.pyplot as plt
import pandas as pd

MODE_STYLE = {
    "random": dict(color="#1f77b4", marker="o"),
    "chunked": dict(color="#d62728", marker="s"),
}
_LOG_ROW = re.compile(
    r"(?P<mode>\w+)\s+workers=(?P<workers>\d+)\s*\|\s*data\s+([\d,]+)\s*\|\s*"
    r"e2e\s+([\d,]+)\s*\|\s*starvation\s+([\d.]+)%\s*\|\s*first batch\s+([\d.]+)s"
)
_CEILING = re.compile(r"Compute-only ceiling:\s*([\d,]+)\s*bases/s")


def _num(text):
    return float(str(text).replace(",", ""))


def parse_log(path: Path):
    ceiling = None
    rows = []
    for line in path.read_text().splitlines():
        m_ceiling = _CEILING.search(line)
        if m_ceiling:
            ceiling = _num(m_ceiling.group(1))
        m = _LOG_ROW.search(line)
        if m:
            rows.append({
                "mode": m.group("mode"),
                "num_workers": int(m.group("workers")),
                "data_bases_per_second": _num(m.group(3)),
                "e2e_bases_per_second": _num(m.group(4)),
                "starvation_fraction": _num(m.group(5)) / 100.0,
                "first_batch_seconds": _num(m.group(6)),
                "compute_bases_per_second": None,
            })
    df = pd.DataFrame(rows)
    if ceiling is not None:
        df["compute_bases_per_second"] = ceiling
    return df


def load(path: Path) -> pd.DataFrame:
    if path.suffix.lower() == ".csv":
        return pd.read_csv(path)
    return parse_log(path)


def _order(df):
    return sorted(df["num_workers"].unique())


def plot_throughput(df, ceiling, out_path, title):
    fig, ax = plt.subplots(figsize=(8, 5))
    for mode, sub in df.groupby("mode"):
        sub = sub.sort_values("num_workers")
        style = MODE_STYLE.get(mode, {})
        ax.plot(sub["num_workers"], sub["data_bases_per_second"] / 1e6,
                linestyle="-", label=f"{mode}: data-only", **style)
        ax.plot(sub["num_workers"], sub["e2e_bases_per_second"] / 1e6,
                linestyle="--", marker=style.get("marker"), color=style.get("color"),
                alpha=0.7, label=f"{mode}: end-to-end")
    if ceiling:
        ax.axhline(ceiling / 1e6, color="black", linestyle=":", linewidth=1.5,
                   label=f"compute ceiling ({ceiling/1e6:.2f} M/s)")
    ax.set_xlabel("DataLoader workers")
    ax.set_ylabel("Throughput (M bases/s)")
    ax.set_title(f"Throughput vs workers — {title}")
    ax.set_xticks(_order(df))
    ax.grid(True, alpha=0.3)
    ax.legend(fontsize=8, ncol=2)
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)


def plot_throughput_log(df, ceiling, out_path, title):
    """Same as throughput but log-y, so the compute-bound e2e plateau is legible
    next to the much larger data-only headroom."""
    fig, ax = plt.subplots(figsize=(8, 5))
    for mode, sub in df.groupby("mode"):
        sub = sub.sort_values("num_workers")
        style = MODE_STYLE.get(mode, {})
        ax.plot(sub["num_workers"], sub["data_bases_per_second"] / 1e6,
                linestyle="-", label=f"{mode}: data-only", **style)
        ax.plot(sub["num_workers"], sub["e2e_bases_per_second"] / 1e6,
                linestyle="--", marker=style.get("marker"), color=style.get("color"),
                alpha=0.7, label=f"{mode}: end-to-end")
    if ceiling:
        ax.axhline(ceiling / 1e6, color="black", linestyle=":", linewidth=1.5,
                   label=f"compute ceiling ({ceiling/1e6:.2f} M/s)")
    ax.set_yscale("log")
    ax.set_xlabel("DataLoader workers")
    ax.set_ylabel("Throughput (M bases/s, log scale)")
    ax.set_title(f"Throughput vs workers (log) — {title}")
    ax.set_xticks(_order(df))
    ax.grid(True, which="both", alpha=0.3)
    ax.legend(fontsize=8, ncol=2)
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)


def plot_starvation(df, out_path, title):
    fig, ax = plt.subplots(figsize=(8, 5))
    workers = _order(df)
    modes = list(df["mode"].unique())
    width = 0.8 / max(1, len(modes))
    for i, mode in enumerate(modes):
        sub = df[df["mode"] == mode].sort_values("num_workers").set_index("num_workers")
        vals = [sub.loc[w, "starvation_fraction"] * 100 if w in sub.index else 0 for w in workers]
        xs = [x + (i - (len(modes) - 1) / 2) * width for x in range(len(workers))]
        ax.bar(xs, vals, width=width, label=mode, color=MODE_STYLE.get(mode, {}).get("color"))
    ax.set_xticks(range(len(workers)))
    ax.set_xticklabels(workers)
    ax.set_xlabel("DataLoader workers")
    ax.set_ylabel("GPU starvation (%)")
    ax.set_title(f"GPU starvation vs workers — {title}")
    ax.grid(True, axis="y", alpha=0.3)
    ax.legend()
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)


def plot_first_batch(df, out_path, title):
    fig, ax = plt.subplots(figsize=(8, 5))
    for mode, sub in df.groupby("mode"):
        sub = sub.sort_values("num_workers")
        ax.plot(sub["num_workers"], sub["first_batch_seconds"],
                label=mode, **MODE_STYLE.get(mode, {}))
    ax.set_yscale("log")
    ax.set_xlabel("DataLoader workers")
    ax.set_ylabel("First-batch latency (s, log scale)")
    ax.set_title(f"First-batch (startup) latency vs workers — {title}")
    ax.set_xticks(_order(df))
    ax.grid(True, which="both", alpha=0.3)
    ax.legend()
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("input", type=Path, help="A results .csv or a benchmark .out log")
    parser.add_argument("--out-dir", type=Path, default=Path("dataloader_scaling_plots"))
    parser.add_argument("--title", default=None, help="Title tag for the figures (default: inferred)")
    args = parser.parse_args(argv)

    df = load(args.input)
    if df.empty:
        parser.error(f"No benchmark rows parsed from {args.input}")
    df = df.sort_values(["mode", "num_workers"]).reset_index(drop=True)

    ceiling = None
    if "compute_bases_per_second" in df and df["compute_bases_per_second"].notna().any():
        ceiling = float(df["compute_bases_per_second"].dropna().iloc[0])

    title = args.title
    if title is None:
        tier = df["tier"].dropna().iloc[0] if "tier" in df and df["tier"].notna().any() else ""
        model = df["model"].dropna().iloc[0] if "model" in df and df["model"].notna().any() else ""
        title = " ".join(str(t) for t in (model, tier) if t) or args.input.stem

    args.out_dir.mkdir(parents=True, exist_ok=True)
    plot_throughput(df, ceiling, args.out_dir / "throughput_vs_workers.png", title)
    plot_throughput_log(df, ceiling, args.out_dir / "throughput_vs_workers_log.png", title)
    plot_starvation(df, args.out_dir / "starvation_vs_workers.png", title)
    plot_first_batch(df, args.out_dir / "first_batch_latency_vs_workers.png", title)
    df.to_csv(args.out_dir / "parsed_results.csv", index=False)

    print(f"Parsed {len(df)} rows; ceiling = {ceiling:,.0f} bases/s" if ceiling else f"Parsed {len(df)} rows")
    print(f"Wrote figures + parsed_results.csv to: {args.out_dir}/")
    for name in ("throughput_vs_workers.png", "throughput_vs_workers_log.png",
                 "starvation_vs_workers.png", "first_batch_latency_vs_workers.png"):
        print(f"  - {name}")


if __name__ == "__main__":
    main()
