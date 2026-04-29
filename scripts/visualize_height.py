"""
Visualize predicted vs ground-truth mean height on a few patches.

Uses raw Zarr `labels/mean_height` for GT (preserves NaN outside trees).
Model output channel 1 = mean_height in metres (no transform in default config).

By default samples both **val** and **test** (outputs go to subfolders).
Use `--splits val` for validation only. Each row shows **S2 summer and spring** RGB
(via `--s2-seasons`, default `summer spring`) plus GT, prediction, and error maps.

Usage (Docker):
  docker run --rm --shm-size=8g --workdir /workspace --gpus all \\
    -e CUDA_VISIBLE_DEVICES=2 \\
    -v /work/ammar/sslrp/data/biomass:/data:ro \\
    -v /work/ammar/sslrp/biomass:/workspace \\
    biomass:latest \\
    python scripts/visualize_height.py \\
      --checkpoint /workspace/artifacts/checkpoints/best.pt \\
      --num_patches 6 \\
      --out_dir /workspace/artifacts/vis_height
"""

from __future__ import annotations

import argparse
import hashlib
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch
import zarr

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import src.models  # noqa: F401
from src.data.dataset import BiomassDataset
from src.models.factory import build_model
from src.utils.config import load_config, load_norm_stats

_SPLITS = ("train", "val", "test")


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Visualize mean_height predictions")
    p.add_argument("--config", default="configs/unet_resnet50.yaml")
    p.add_argument(
        "--checkpoint",
        default="/workspace/artifacts/checkpoints/best.pt",
    )
    p.add_argument(
        "--splits",
        nargs="+",
        default=["val", "test"],
        metavar="SPLIT",
        help="One or more of: train val test. Default: val test (few val + few test).",
    )
    p.add_argument("--num_patches", type=int, default=6, help="Patches per split (if multiple splits).")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument(
        "--out_dir",
        default="/workspace/artifacts/vis_height",
        help="Base directory; PNGs go to <out_dir>/<split>/",
    )
    p.add_argument(
        "--indices",
        type=int,
        nargs="*",
        default=None,
        help="Only when a single split is set: row indices in that DataLoader order.",
    )
    p.add_argument(
        "--s2-seasons",
        dest="s2_seasons",
        nargs="+",
        default=["summer", "spring"],
        metavar="SEASON",
        help="S2 seasons for RGB panels (B4/B3/B2), must match zarr keys s2_<season>. Default: summer spring.",
    )
    return p.parse_args()


def _validate_splits(splits: list[str]) -> list[str]:
    out: list[str] = []
    for s in splits:
        if s not in _SPLITS:
            raise SystemExit(f"Invalid split {s!r}, expected one of {_SPLITS}")
        if s not in out:
            out.append(s)
    if not out:
        raise SystemExit("No splits given.")
    return out


def _split_seed(base: int, split: str) -> int:
    """Per-split PRNG so val vs test do not use the same patch indices when seed and num_patches match."""
    h = int(hashlib.md5(split.encode(), usedforsecurity=False).hexdigest()[:8], 16)
    return (base + h) % (2**31)


def s2_season_rgb(root: Path, zarr_idx: int, season: str) -> np.ndarray:
    """
    False-colour RGB from S2: R=B4, G=B3, B=B2 (indices 2,1,0 in zarr channel order).
    Stretched per image with 2nd–98th percentiles.
    """
    s2 = zarr.open(str(root / "inputs" / f"s2_{season}"), mode="r")
    x = np.array(s2[zarr_idx], dtype=np.float32)  # [H, W, 10]
    r, g, b = x[:, :, 2], x[:, :, 1], x[:, :, 0]
    rgb = np.stack([r, g, b], axis=-1)
    lo, hi = np.nanpercentile(rgb, [2, 98])
    rgb = np.clip((rgb - lo) / (hi - lo + 1e-8), 0, 1)
    return rgb


def plot_patches(
    split: str,
    row_ids: list[int],
    out_dir: Path,
    cfg: dict,
    norm_stats: dict,
    ckpt: dict,
    model: torch.nn.Module,
    device: torch.device,
    root: Path,
    s2_seasons: list[str],
) -> None:
    mh_zarr = zarr.open(str(root / "labels" / "mean_height"), mode="r")
    ds = BiomassDataset(
        root=cfg["data"]["root"],
        split=split,
        cfg=cfg,
        norm_stats=norm_stats,
        transform=None,
    )
    n = len(ds)
    for row_i in row_ids:
        if row_i < 0 or row_i >= n:
            print(f"  [skip] row {row_i} out of range [0, {n}) for split={split}")
            continue
        zidx = int(ds.patch_indices[row_i])
        x, y, mask = ds[row_i]
        xb = x.unsqueeze(0).to(device, non_blocking=True)
        with torch.no_grad():
            pred = model(xb).cpu().numpy()[0, 1]  # [H, W] mean_height

        gt = np.array(mh_zarr[zidx], dtype=np.float32)
        m = mask.numpy().astype(bool)
        valid = m & np.isfinite(gt)

        err = np.full_like(pred, np.nan, dtype=np.float64)
        err[valid] = pred[valid] - gt[valid]

        h_stack = np.concatenate([gt[valid], pred[valid]])
        vmax = float(np.nanpercentile(h_stack, 98))
        vmin = float(np.nanpercentile(h_stack, 2))
        if vmax <= vmin:
            vmax = vmin + 1.0

        n_rgb = len(s2_seasons)
        ncols = n_rgb + 4  # RGBs + GT + pred + |err| + signed err
        fig, axes = plt.subplots(1, ncols, figsize=(4 * ncols, 4), constrained_layout=True)
        for j, season in enumerate(s2_seasons):
            rgb = s2_season_rgb(root, zidx, season)
            axes[j].imshow(rgb)
            axes[j].set_title(f"S2 {season} RGB")
            axes[j].axis("off")

        h_gt = np.ma.masked_where(~np.isfinite(gt) | ~m, gt)
        h_pr = np.ma.masked_where(~m, pred)
        j0 = n_rgb

        im1 = axes[j0 + 0].imshow(h_gt, cmap="viridis", vmin=vmin, vmax=vmax)
        axes[j0 + 0].set_title("GT mean_height (m)")
        axes[j0 + 0].axis("off")
        plt.colorbar(im1, ax=axes[j0 + 0], fraction=0.046)

        im2 = axes[j0 + 1].imshow(h_pr, cmap="viridis", vmin=vmin, vmax=vmax)
        axes[j0 + 1].set_title("Pred mean_height (m)")
        axes[j0 + 1].axis("off")
        plt.colorbar(im2, ax=axes[j0 + 1], fraction=0.046)

        e_abs = np.abs(err)
        emax = float(np.nanpercentile(e_abs[valid], 95)) if valid.any() else 1.0
        im3 = axes[j0 + 2].imshow(e_abs, cmap="magma", vmin=0, vmax=max(emax, 0.5))
        axes[j0 + 2].set_title("|error| (m) on valid")
        axes[j0 + 2].axis("off")
        plt.colorbar(im3, ax=axes[j0 + 2], fraction=0.046)

        im4 = axes[j0 + 3].imshow(err, cmap="coolwarm", vmin=-emax, vmax=emax)
        axes[j0 + 3].set_title("error pred−GT (m)")
        axes[j0 + 3].axis("off")
        plt.colorbar(im4, ax=axes[j0 + 3], fraction=0.046)

        rmse = float(np.sqrt(np.nanmean(err[valid] ** 2))) if valid.any() else float("nan")
        mae = float(np.nanmean(np.abs(err[valid]))) if valid.any() else float("nan")
        fig.suptitle(
            f"split={split}  zarr_idx={zidx}  "
            f"valid_pix={100 * valid.mean():.1f}%  RMSE={rmse:.2f}m  MAE={mae:.2f}m  "
            f"epoch_ckpt={ckpt.get('epoch', '?')}"
        )
        out_path = out_dir / f"height_z{zidx:05d}_r{row_i:04d}.png"
        fig.savefig(out_path, dpi=150, bbox_inches="tight")
        plt.close(fig)
        print(f"  saved {out_path}")


def main() -> None:
    args = parse_args()
    splits = _validate_splits([s for s in args.splits])
    out_base = Path(args.out_dir)
    out_base.mkdir(parents=True, exist_ok=True)

    if args.indices is not None and len(splits) > 1:
        print("[WARN] --indices is ignored when multiple --splits are set; use --splits val (only) to pick row indices.")
        args.indices = None

    ckpt = torch.load(args.checkpoint, map_location="cpu")
    cfg = ckpt.get("cfg") or load_config(args.config)
    norm_stats = load_norm_stats(cfg["data"]["norm_stats_path"])
    root = Path(cfg["data"]["root"])

    n_ch = len(norm_stats["mean"])
    model = build_model(cfg, num_input_channels=n_ch)
    model.load_state_dict(ckpt["model_state_dict"])
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device).eval()

    s2s = [s.lower() for s in args.s2_seasons]
    for s in s2s:
        zpath = root / "inputs" / f"s2_{s}"
        if not zpath.exists():
            raise SystemExit(
                f"S2 zarr not found: {zpath}  (check --s2-seasons and data layout)"
            )

    for sp in splits:
        sub = out_base / sp
        sub.mkdir(parents=True, exist_ok=True)
        # Probe length for this split
        ds0 = BiomassDataset(
            root=cfg["data"]["root"], split=sp, cfg=cfg, norm_stats=norm_stats, transform=None
        )
        n = len(ds0)
        if args.indices is not None and len(splits) == 1:
            row_ids = [int(i) for i in args.indices if 0 <= int(i) < n]
        else:
            rng = np.random.default_rng(_split_seed(args.seed, sp))
            k = min(args.num_patches, n)
            row_ids = sorted(rng.choice(n, size=k, replace=False).tolist())
        print(f"split='{sp}'  ({n} patches)  plotting {len(row_ids)}  →  {sub}/  S2_RGB={s2s}")
        plot_patches(
            sp, row_ids, sub, cfg, norm_stats, ckpt, model, device, root, s2_seasons=s2s
        )
    print("Done.")


if __name__ == "__main__":
    main()
