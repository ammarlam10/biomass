"""
Generate three NeurIPS paper figures for the Bavaria Tree Benchmark dataset.

Figure 1  (fig1_hero.png)
    2×3 panel showing multi-modal inputs (top row) and regression targets /
    valid mask (bottom row) for a single illustrative 128×128 patch.

Figure 2  (fig2_bavaria_map.png)
    Bavaria state map with 6×6 spatial split blocks colour-coded by split
    (train / val / test).

Figure 3  (fig3_histograms.png)
    1×3 histograms illustrating label sparsity and distributional challenge:
    tree_pixel_pct (patch level), per-pixel tree_count, per-pixel mean_height.

Usage (Docker – see make_figures service in docker-compose.yml)
--------------------------------------------------------------
docker compose run --rm make_figures

Or ad-hoc:
    docker run --rm \\
      -v /data/ammar/4g.zarr:/data/ammar/4g.zarr:ro \\
      -v /data/ammar/biomass:/workspace \\
      biomass:latest \\
      bash -c "pip install geopandas --quiet && \\
               python scripts/make_figures.py --out_dir /workspace/artifacts/figures"

CLI flags
---------
--zarr_root   Path to the Zarr dataset root   (default /data/ammar/4g.zarr)
--out_dir     Output directory                (default /workspace/artifacts/figures)
--zarr_idx    Force a specific patch for Figure 1 (default: auto)
--sample_n    Training patches to sample for Figure 3 pixel histograms (default 300)
--seed        RNG seed                        (default 42)
--no_boundary Skip Bavaria boundary download in Figure 2
--figures     Which figure numbers to produce (default: 1 2 3)
"""

from __future__ import annotations

import argparse
import sys
import urllib.request
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.colors as mcolors
import matplotlib.patches as mpatches
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import numpy as np
import pandas as pd
import zarr

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

try:
    from scipy.stats import gaussian_kde
    _HAS_SCIPY = True
except ImportError:
    _HAS_SCIPY = False

# ── publication typography ────────────────────────────────────────────────────
plt.rcParams.update({
    "font.family": "serif",
    "font.size": 9,
    "axes.titlesize": 10,
    "axes.labelsize": 9,
    "legend.fontsize": 8,
    "xtick.labelsize": 8,
    "ytick.labelsize": 8,
    "axes.linewidth": 0.8,
    "xtick.major.width": 0.8,
    "ytick.major.width": 0.8,
    "lines.linewidth": 1.2,
})

# ── constants ─────────────────────────────────────────────────────────────────
SPLIT_COLORS = {"train": "#4575b4", "val": "#1a9850", "test": "#d73027"}
SPLIT_LABELS = {"train": "Train", "val": "Validation", "test": "Test"}

# Official label semantics for tree_species Zarr (uint8 class codes per pixel).
_SPECIES_NAMES: dict[int, str] = {
    0: "Beech",
    1: "Douglas fir",
    2: "Fir",
    3: "Larch",
    4: "Oak",
    5: "Other deciduous",
    6: "Pine",
    7: "Spruce",
    11: "No tree",
}


def _species_label(cls_id: int) -> str:
    """Human-readable legend entry for a dataset species code."""
    return _SPECIES_NAMES.get(int(cls_id), f"Class {cls_id}")

# Natural Earth 1:10 m admin-1 shapefile (free, no auth required)
_NE_URL = (
    "https://naciscdn.org/naturalearth/10m/cultural/"
    "ne_10m_admin_1_states_provinces.zip"
)
_NE_CACHE = Path("/tmp/_neurips_ne_admin1.zip")

# Patch size in metres (128 px × 10 m/px)
_PATCH_M = 1280.0
_HALF_PATCH = _PATCH_M / 2.0


# ─────────────────────────────────────────────────────────────────────────────
# Low-level Zarr readers
# ─────────────────────────────────────────────────────────────────────────────

def _read_s2_rgb(root: Path, zarr_idx: int, season: str = "summer") -> np.ndarray:
    """Return percentile-stretched S2 RGB (R=B4, G=B3, B=B2), shape [H,W,3] float32."""
    store = zarr.open(str(root / "inputs" / f"s2_{season}"), mode="r")
    raw = np.array(store[zarr_idx], dtype=np.float32)   # [H, W, 10]
    rgb = raw[:, :, [2, 1, 0]].copy()                   # R=B4, G=B3, B=B2
    for c in range(3):
        lo, hi = np.nanpercentile(rgb[:, :, c], [2, 98])
        rgb[:, :, c] = np.clip((rgb[:, :, c] - lo) / max(hi - lo, 1e-8), 0.0, 1.0)
    return rgb


def _read_s1_vv(root: Path, zarr_idx: int, season: str = "summer") -> np.ndarray:
    """Return percentile-stretched S1 VV band, shape [H,W] float32 [0,1]."""
    store = zarr.open(str(root / "inputs" / f"s1_{season}"), mode="r")
    raw = np.array(store[zarr_idx], dtype=np.float32)[:, :, 0]  # VV
    lo, hi = np.nanpercentile(raw, [2, 98])
    return np.clip((raw - lo) / max(hi - lo, 1e-8), 0.0, 1.0)


def _read_species(root: Path, zarr_idx: int) -> np.ndarray:
    store = zarr.open(str(root / "inputs" / "tree_species"), mode="r")
    return np.array(store[zarr_idx], dtype=np.uint8)   # [H, W]


def _read_tree_count(root: Path, zarr_idx: int) -> np.ndarray:
    store = zarr.open(str(root / "labels" / "tree_count"), mode="r")
    return np.array(store[zarr_idx], dtype=np.float32) # [H, W]


def _read_mean_height(root: Path, zarr_idx: int) -> np.ndarray:
    store = zarr.open(str(root / "labels" / "mean_height"), mode="r")
    return np.array(store[zarr_idx], dtype=np.float32) # [H, W]


# ─────────────────────────────────────────────────────────────────────────────
# Figure 1 – Hero patch
# ─────────────────────────────────────────────────────────────────────────────

def make_figure1(
    root: Path,
    df: pd.DataFrame,
    out_path: Path,
    zarr_idx: int | None = None,
) -> None:
    print("  Figure 1 – hero patch…")

    if zarr_idx is None:
        # Select a visually representative patch: good S2 quality, typical sparsity
        # (tree_pixel_pct 20–60% shows both rich forest structure AND sparse mask),
        # prioritising the patch with the highest mean_tree_count for visual variety.
        cand = df[
            (df["valid_pixel_pct"] > 0.90)
            & (df["tree_pixel_pct"] >= 0.20)
            & (df["tree_pixel_pct"] <= 0.60)
        ]
        if cand.empty:
            cand = df[df["tree_pixel_pct"] > 0.10]
        zarr_idx = int(cand.loc[cand["mean_tree_count"].idxmax(), "zarr_idx"])
        row = cand.loc[cand["mean_tree_count"].idxmax()]
        print(
            f"    auto-selected zarr_idx={zarr_idx}  "
            f"(tree_pct={row['tree_pixel_pct']:.1%}, "
            f"mean_tc={row['mean_tree_count']:.2f})"
        )

    rgb   = _read_s2_rgb(root, zarr_idx)
    vv    = _read_s1_vv(root, zarr_idx)
    sp    = _read_species(root, zarr_idx)
    tc    = _read_tree_count(root, zarr_idx)
    mh    = _read_mean_height(root, zarr_idx)
    valid = np.isfinite(mh)                 # True on tree pixels

    tc_disp = np.ma.masked_where(~valid, tc)
    mh_disp = np.ma.masked_where(~valid, mh)

    # Discrete species colourmap – one colour per class code present in this patch
    present_ids = sorted(int(v) for v in np.unique(sp))
    n_cls = max(len(present_ids), 1)
    sp_colors = plt.cm.Set1(np.linspace(0, 0.9, n_cls))
    sp_cmap   = mcolors.ListedColormap(sp_colors)
    sp_idx    = np.full_like(sp, -1, dtype=np.int32)
    for k, cls_id in enumerate(present_ids):
        sp_idx[sp == cls_id] = k

    fig, axes = plt.subplots(2, 3, figsize=(14, 5.4), constrained_layout=True)

    # ── Top row: Inputs ───────────────────────────────────────────────────────
    axes[0, 0].imshow(rgb, interpolation="nearest")
    axes[0, 0].set_title("Sentinel-2 RGB  (Summer, B4/B3/B2)")

    axes[0, 1].imshow(vv, cmap="gray", vmin=0, vmax=1, interpolation="nearest")
    axes[0, 1].set_title("Sentinel-1 VV  (Summer)")

    im_sp = axes[0, 2].imshow(
        sp_idx, cmap=sp_cmap,
        vmin=-0.5, vmax=n_cls - 0.5,
        interpolation="nearest",
    )
    axes[0, 2].set_title("Tree Species  (categorical)")
    legend_handles = [
        mpatches.Patch(
            color=sp_cmap(k / max(n_cls - 1, 1)),
            label=_species_label(cls_id),
        )
        for k, cls_id in enumerate(present_ids)
    ]
    axes[0, 2].legend(
        handles=legend_handles, fontsize=6.5, ncol=2,
        loc="upper right", framealpha=0.85,
    )

    # ── Bottom row: Targets & mask ────────────────────────────────────────────
    im_tc = axes[1, 0].imshow(
        tc_disp, cmap="YlOrRd", vmin=1, vmax=8, interpolation="nearest"
    )
    axes[1, 0].set_title("Tree Count  (GT)")
    plt.colorbar(im_tc, ax=axes[1, 0], fraction=0.046, pad=0.04,
                 label="trees / pixel")

    mh_vmax = float(np.nanpercentile(mh[valid], 98)) if valid.any() else 40.0
    im_mh = axes[1, 1].imshow(
        mh_disp, cmap="viridis", vmin=0, vmax=mh_vmax, interpolation="nearest"
    )
    axes[1, 1].set_title("Mean Height  (GT)")
    plt.colorbar(im_mh, ax=axes[1, 1], fraction=0.046, pad=0.04, label="m")

    axes[1, 2].imshow(
        valid.astype(np.uint8), cmap="gray", vmin=0, vmax=1, interpolation="nearest"
    )
    axes[1, 2].set_title("Valid Mask")
    pct_valid = 100.0 * valid.mean()
    axes[1, 2].text(
        0.5, 0.03,
        f"{pct_valid:.1f}% labelled pixels",
        transform=axes[1, 2].transAxes,
        ha="center", va="bottom", fontsize=7.5, color="white",
        bbox=dict(boxstyle="round,pad=0.25", facecolor="black", alpha=0.55),
    )

    for ax in axes.flat:
        ax.set_aspect("auto")
        ax.axis("off")

    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(f"    → {out_path}")


# ─────────────────────────────────────────────────────────────────────────────
# Figure 2 – Bavaria geographic distribution
# ─────────────────────────────────────────────────────────────────────────────

def _download_bavaria(epsg: int = 25832):
    """Return geopandas GeoDataFrame of the Bayern state boundary (EPSG:epsg)."""
    import geopandas as gpd

    if not _NE_CACHE.exists():
        print(f"    Downloading Natural Earth admin-1 → {_NE_CACHE} …")
        urllib.request.urlretrieve(_NE_URL, _NE_CACHE)

    gdf = gpd.read_file(f"zip://{_NE_CACHE}")
    bav = gdf[(gdf["admin"] == "Germany") & (gdf["name"] == "Bayern")].copy()
    if bav.empty:
        bav = gdf[
            (gdf["admin"] == "Germany")
            & gdf["name_en"].str.contains("Bavaria", na=False)
        ].copy()
    if bav.empty:
        raise RuntimeError("Bayern not found in Natural Earth admin-1 data.")
    return bav.to_crs(epsg=epsg)


def _block_grid_lines(df: pd.DataFrame) -> tuple[list[float], list[float]]:
    """
    Compute x-boundaries between adjacent block columns and y-boundaries between
    adjacent block rows, as midpoints of the inter-block gap.
    """
    x_bounds, y_bounds = [], []

    for bc in range(5):
        left  = df[df["block_col"] == bc]
        right = df[df["block_col"] == bc + 1]
        if left.empty or right.empty:
            continue
        x_bounds.append((left["center_x"].max() + right["center_x"].min()) / 2.0)

    for br in range(5):
        below = df[df["block_row"] == br]
        above = df[df["block_row"] == br + 1]
        if below.empty or above.empty:
            continue
        y_bounds.append((below["center_y"].max() + above["center_y"].min()) / 2.0)

    return x_bounds, y_bounds


def make_figure2(
    df: pd.DataFrame,
    out_path: Path,
    no_boundary: bool = False,
) -> None:
    print("  Figure 2 – Bavaria map…")

    x_bounds, y_bounds = _block_grid_lines(df)

    # Axis extent (patch-level padding)
    xmin = df["center_x"].min() - _HALF_PATCH * 4
    xmax = df["center_x"].max() + _HALF_PATCH * 4
    ymin = df["center_y"].min() - _HALF_PATCH * 4
    ymax = df["center_y"].max() + _HALF_PATCH * 4

    fig, ax = plt.subplots(figsize=(7.5, 9))

    # ── Bavaria state boundary ────────────────────────────────────────────────
    if not no_boundary:
        try:
            bav_gdf = _download_bavaria()
            bav_gdf.boundary.plot(ax=ax, color="#111111", linewidth=1.3, zorder=6)
        except Exception as exc:
            print(f"    [WARN] Bavaria boundary skipped: {exc}")

    # ── Patch scatter (rasterised – forms the density silhouette) ─────────────
    for split_name, color in SPLIT_COLORS.items():
        m = df["split"] == split_name
        ax.scatter(
            df.loc[m, "center_x"], df.loc[m, "center_y"],
            c=color, s=2.5, alpha=0.35, linewidths=0,
            rasterized=True, zorder=2,
        )

    # ── Block grid lines ──────────────────────────────────────────────────────
    for xb in x_bounds:
        ax.axvline(xb, color="white", lw=1.0, alpha=0.7, zorder=3)
    for yb in y_bounds:
        ax.axhline(yb, color="white", lw=1.0, alpha=0.7, zorder=3)

    # ── Block split labels ────────────────────────────────────────────────────
    for (br, bc), grp in df.groupby(["block_row", "block_col"]):
        cx = grp["center_x"].mean()
        cy = grp["center_y"].mean()
        split_name = grp["split"].mode().iloc[0]
        color = SPLIT_COLORS[split_name]
        _SPLIT_ABBREV = {"train": "Tr", "val": "Va", "test": "Te"}
        ax.text(
            cx, cy, _SPLIT_ABBREV[split_name],
            ha="center", va="center", fontsize=8, fontweight="bold",
            color=color, alpha=0.80, zorder=4,
        )

    # ── Legend ────────────────────────────────────────────────────────────────
    legend_handles = [
        mpatches.Patch(facecolor=SPLIT_COLORS[s], alpha=0.65, label=SPLIT_LABELS[s])
        for s in ["train", "val", "test"]
    ]
    ax.legend(
        handles=legend_handles, loc="upper right",
        framealpha=0.9, fontsize=8, title="Split", title_fontsize=8,
    )

    ax.set_xlabel("Easting (EPSG:25832, m)")
    ax.set_ylabel("Northing (EPSG:25832, m)")
    ax.set_title("Geographic Distribution & Spatial Split Blocks  (Bavaria)")
    ax.set_aspect("equal")
    ax.set_xlim(xmin, xmax)
    ax.set_ylim(ymin, ymax)
    ax.ticklabel_format(style="sci", axis="both", scilimits=(5, 5), useMathText=True)
    ax.grid(False)

    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(f"    → {out_path}")


# ─────────────────────────────────────────────────────────────────────────────
# Figure 3 – Histograms
# ─────────────────────────────────────────────────────────────────────────────

def _sample_pixel_labels(
    root: Path,
    zarr_idxs: np.ndarray,
    n_max: int = 300,
    seed: int = 42,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Sample up to *n_max* patches; return (tree_count_pixels, mean_height_pixels)
    containing only valid (tree) pixel values.
    """
    rng    = np.random.default_rng(seed)
    sample = rng.choice(zarr_idxs, size=min(n_max, len(zarr_idxs)), replace=False)
    sample = np.sort(sample)

    tc_store = zarr.open(str(root / "labels" / "tree_count"),  mode="r")
    mh_store = zarr.open(str(root / "labels" / "mean_height"), mode="r")

    tc_list, mh_list = [], []
    for zidx in sample:
        tc = np.array(tc_store[int(zidx)], dtype=np.float32).ravel()
        mh = np.array(mh_store[int(zidx)], dtype=np.float32).ravel()
        valid = np.isfinite(mh)
        if valid.any():
            tc_list.append(tc[valid])
            mh_list.append(mh[valid])

    if not tc_list:
        return np.array([], dtype=np.float32), np.array([], dtype=np.float32)
    return np.concatenate(tc_list), np.concatenate(mh_list)


def make_figure3(
    root: Path,
    df: pd.DataFrame,
    out_path: Path,
    sample_n: int = 300,
    seed: int = 42,
) -> None:
    print(f"  Figure 3 – histograms (sampling {sample_n} training patches)…")

    fig, axes = plt.subplots(1, 3, figsize=(12, 3.5), constrained_layout=True)

    # ── Panel 1: tree_pixel_pct (patch level, from parquet) ───────────────────
    ax = axes[0]
    pct = df["tree_pixel_pct"].values
    bins = np.arange(0.0, 1.0 + 0.025, 0.025)
    cnts, edges = np.histogram(pct, bins=bins)

    ax.bar(
        edges[:-1], cnts, width=np.diff(edges),
        color=SPLIT_COLORS["train"], alpha=0.75,
        edgecolor="white", linewidth=0.35, align="edge",
    )
    median_pct = float(np.median(pct))
    ax.axvline(
        median_pct, color="#d73027", lw=1.3, ls="--",
        label=f"Median  {median_pct:.0%}",
    )
    zero_n   = int((pct == 0.0).sum())
    zero_frac = zero_n / len(pct) * 100
    ax.text(
        0.97, 0.97,
        f"{zero_frac:.1f}% patches\nhave zero trees\n(n = {zero_n:,})",
        transform=ax.transAxes, ha="right", va="top", fontsize=7.5,
        bbox=dict(
            boxstyle="round,pad=0.3", facecolor="#fffde7",
            edgecolor="#cccccc", alpha=0.9,
        ),
    )
    ax.set_xlabel("Tree pixel fraction per patch")
    ax.set_ylabel("Number of patches")
    ax.set_title("(a) Label Sparsity")
    ax.set_xlim(0.0, 1.0)
    ax.legend(fontsize=7.5, loc="upper center")

    # ── Sample pixel-level data from training Zarr ────────────────────────────
    train_idxs = df.loc[df["split"] == "train", "zarr_idx"].values
    tc_px, mh_px = _sample_pixel_labels(root, train_idxs, n_max=sample_n, seed=seed)
    print(f"    Collected {len(tc_px):,} tree pixels from {sample_n} sampled patches.")

    # ── Panel 2: per-pixel tree_count (1–8) ───────────────────────────────────
    ax = axes[1]
    if len(tc_px) > 0:
        tc_int  = tc_px.round().astype(int)
        vals_u, cnts_u = np.unique(tc_int[tc_int >= 1], return_counts=True)

        ax.bar(
            vals_u, cnts_u,
            color=SPLIT_COLORS["val"], alpha=0.75,
            edgecolor="white", linewidth=0.5,
        )
        median_tc = float(np.median(tc_px[tc_px >= 1]))
        ax.axvline(
            median_tc, color="#d73027", lw=1.3, ls="--",
            label=f"Median  {median_tc:.1f}",
        )
        ax.set_xticks(vals_u)
        ax.legend(fontsize=7.5)
    ax.set_yscale("log")
    ax.yaxis.set_major_formatter(mticker.LogFormatterSciNotation(base=10, labelOnlyBase=True))
    ax.set_xlabel("Trees per pixel")
    ax.set_ylabel("Number of pixels")
    ax.set_title("(b) Tree Count Distribution  (log scale)")

    # ── Panel 3: per-pixel mean_height ────────────────────────────────────────
    ax = axes[2]
    if len(mh_px) > 0:
        ax.hist(
            mh_px, bins=40,
            color="#f4a582", alpha=0.80,
            edgecolor="white", linewidth=0.35,
            label="observed",
        )
        median_mh = float(np.median(mh_px))
        ax.axvline(
            median_mh, color="#d73027", lw=1.3, ls="--",
            label=f"Median  {median_mh:.1f} m",
        )

        if _HAS_SCIPY:
            kde  = gaussian_kde(mh_px, bw_method=0.15)
            x_k  = np.linspace(mh_px.min(), mh_px.max(), 400)
            ax2  = ax.twinx()
            ax2.plot(x_k, kde(x_k), color="#c51b7d", lw=1.5, label="KDE", zorder=5)
            ax2.set_yticks([])
            ax2.set_ylabel("")
            # Merge legends
            h1, l1 = ax.get_legend_handles_labels()
            h2, l2 = ax2.get_legend_handles_labels()
            ax.legend(h1 + h2, l1 + l2, fontsize=7.5, loc="upper right")
        else:
            ax.legend(fontsize=7.5)

    ax.set_xlabel("Mean tree height (m)")
    ax.set_ylabel("Number of pixels")
    ax.set_title("(c) Mean Height Distribution")

    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(f"    → {out_path}")


# ─────────────────────────────────────────────────────────────────────────────
# Entry point
# ─────────────────────────────────────────────────────────────────────────────

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Generate NeurIPS dataset paper figures.")
    p.add_argument("--zarr_root",   default="/data/ammar/4g.zarr")
    p.add_argument("--out_dir",     default="/workspace/artifacts/figures")
    p.add_argument("--zarr_idx",    type=int, default=None,
                   help="Force a specific zarr_idx for Figure 1 (default: auto-select).")
    p.add_argument("--sample_n",    type=int, default=300,
                   help="Training patches to sample for Figure 3 pixel histograms.")
    p.add_argument("--seed",        type=int, default=42)
    p.add_argument("--no_boundary", action="store_true",
                   help="Skip Bavaria boundary download in Figure 2.")
    p.add_argument("--figures",     nargs="+", type=int, default=[1, 2, 3],
                   metavar="N",
                   help="Figure numbers to produce (default: 1 2 3).")
    return p.parse_args()


def main() -> None:
    args    = parse_args()
    root    = Path(args.zarr_root)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    print("Loading parquet index…")
    df = pd.read_parquet(root / "patch_index_subset.parquet")
    df = df[df["in_bavaria"]].reset_index(drop=True)
    n_tr = (df["split"] == "train").sum()
    n_va = (df["split"] == "val").sum()
    n_te = (df["split"] == "test").sum()
    print(f"  {len(df):,} Bavaria patches  (train={n_tr:,}  val={n_va:,}  test={n_te:,})")

    figs_to_run = args.figures
    if 1 in figs_to_run:
        make_figure1(root, df, out_dir / "fig1_hero.png", zarr_idx=args.zarr_idx)
    if 2 in figs_to_run:
        make_figure2(df, out_dir / "fig2_bavaria_map.png",
                     no_boundary=args.no_boundary)
    if 3 in figs_to_run:
        make_figure3(root, df, out_dir / "fig3_histograms.png",
                     sample_n=args.sample_n, seed=args.seed)

    print(f"\nAll done. Figures saved to {out_dir}/")


if __name__ == "__main__":
    main()
