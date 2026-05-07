"""
Offline DOP20 feature extraction — supports 4-GPU parallel mode.

Reads 20 cm RGB patches from train_dop20.zarr, tiles each 6400×6400 image
into a 128×128 grid of 50×50 blocks (one block per Sentinel pixel), runs
every block through a frozen ImageNet ResNet50, and applies PCA to reduce
2048 → D dimensions.  The resulting [D, 128, 128] feature maps are written
to train_dop20_features.zarr (float16, one chunk per patch → safe for
concurrent multi-process writes).

──────────────────────────────────────────────────────────────────────────────
SINGLE GPU  (original behaviour, all 13626 patches on one card)
──────────────────────────────────────────────────────────────────────────────
  docker run --rm -it --gpus all -e CUDA_VISIBLE_DEVICES=1 \\
    -v ~/tree/Bavaria-EO-Benchmark/data/zarr:/data \\
    -v ~/biomass:/workspace -w /workspace --shm-size=8g \\
    biomass:latest \\
    python scripts/extract_dop20_features.py \\
      --zarr-root /data --output /data/train_dop20_features.zarr \\
      --feature-dim 256 --batch-size 512

──────────────────────────────────────────────────────────────────────────────
4-GPU PARALLEL  (≈4× faster, ~3400 patches per card)
──────────────────────────────────────────────────────────────────────────────
Step 1 – fit PCA + create zarr skeleton (one-off, ~30 min on 1 GPU):

  docker run --rm -it --gpus all -e CUDA_VISIBLE_DEVICES=1 \\
    -v ~/tree/Bavaria-EO-Benchmark/data/zarr:/data \\
    -v ~/biomass:/workspace -w /workspace --shm-size=8g \\
    biomass:latest \\
    python scripts/extract_dop20_features.py \\
      --zarr-root /data --output /data/train_dop20_features.zarr \\
      --feature-dim 256 --batch-size 512 --pca-only

Step 2 – launch 4 workers in parallel (run all four at once in separate terminals):

  RANK=0  GPU=1
  RANK=1  GPU=2
  RANK=2  GPU=3
  RANK=3  GPU=4

  docker run --rm -it --gpus all -e CUDA_VISIBLE_DEVICES=<GPU> \\
    -v ~/tree/Bavaria-EO-Benchmark/data/zarr:/data \\
    -v ~/biomass:/workspace -w /workspace --shm-size=8g \\
    biomass:latest \\
    python scripts/extract_dop20_features.py \\
      --zarr-root /data --output /data/train_dop20_features.zarr \\
      --feature-dim 256 --batch-size 512 \\
      --rank <RANK> --world-size 4

Resuming an interrupted shard: re-run the same command; it picks up from the
last written index for that rank (stored in the zarr attrs).
──────────────────────────────────────────────────────────────────────────────
"""

from __future__ import annotations

import argparse
import pickle
import sys
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import zarr
from sklearn.decomposition import PCA
from torchvision import models
from tqdm import tqdm

# ── ImageNet normalisation ────────────────────────────────────────────────────
_IMAGENET_MEAN = [0.485, 0.456, 0.406]
_IMAGENET_STD  = [0.229, 0.224, 0.225]

# ── block geometry ────────────────────────────────────────────────────────────
SENTINEL_SIZE = 128
BLOCK_SIZE    = 50      # 10 m / 0.2 m = 50 DOP20 pixels per Sentinel pixel
DOP20_SIZE    = SENTINEL_SIZE * BLOCK_SIZE   # 6400


# ── ResNet50 backbone ─────────────────────────────────────────────────────────

class ResNet50Backbone(nn.Module):
    """ResNet50 up to global average pooling → 2048-d feature vector."""

    def __init__(self) -> None:
        super().__init__()
        base = models.resnet50(weights=models.ResNet50_Weights.IMAGENET1K_V1)
        self.features = nn.Sequential(
            base.conv1, base.bn1, base.relu, base.maxpool,
            base.layer1, base.layer2, base.layer3, base.layer4,
        )
        self.pool = nn.AdaptiveAvgPool2d((1, 1))
        for p in self.parameters():
            p.requires_grad_(False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.pool(self.features(x)).flatten(1)   # [B, 2048]


# ── helpers ───────────────────────────────────────────────────────────────────

def _tile_patch(rgb: np.ndarray) -> np.ndarray:
    """
    (6400, 6400, 3) uint8  →  (16384, 3, 50, 50) float32, ImageNet-normalised.
    """
    img = rgb.reshape(SENTINEL_SIZE, BLOCK_SIZE, SENTINEL_SIZE, BLOCK_SIZE, 3)
    img = img.transpose(0, 2, 1, 3, 4)                     # (H, W, bH, bW, C)
    img = img.reshape(-1, BLOCK_SIZE, BLOCK_SIZE, 3)        # (16384, bH, bW, C)
    img = img.transpose(0, 3, 1, 2).astype(np.float32) / 255.0  # (16384, C, bH, bW)
    mean = np.array(_IMAGENET_MEAN, dtype=np.float32)[:, None, None]
    std  = np.array(_IMAGENET_STD,  dtype=np.float32)[:, None, None]
    return (img - mean) / std   # (16384, 3, 50, 50)


def _extract_features_for_patch(
    blocks: np.ndarray,
    backbone: ResNet50Backbone,
    device: torch.device,
    batch_size: int,
) -> np.ndarray:
    """(16384, 3, 50, 50) → (16384, 2048) float32 via backbone sub-batches."""
    n = blocks.shape[0]
    feats = np.empty((n, 2048), dtype=np.float32)
    with torch.no_grad():
        for s in range(0, n, batch_size):
            e = min(s + batch_size, n)
            feats[s:e] = backbone(torch.from_numpy(blocks[s:e]).to(device)).cpu().numpy()
    return feats


# ── PCA ───────────────────────────────────────────────────────────────────────

def fit_pca(
    dop20_zarr: zarr.Array,
    backbone: ResNet50Backbone,
    device: torch.device,
    batch_size: int,
    feature_dim: int,
    n_sample_patches: int = 200,
    blocks_per_patch: int = 1000,
    seed: int = 42,
) -> PCA:
    rng = np.random.default_rng(seed)
    patch_indices = rng.choice(
        dop20_zarr.shape[0],
        size=min(n_sample_patches, dop20_zarr.shape[0]),
        replace=False,
    )
    all_samples: list[np.ndarray] = []
    print(f"Fitting PCA on {len(patch_indices)} patches × {blocks_per_patch} blocks …")
    for pidx in tqdm(patch_indices, desc="PCA sampling"):
        rgb    = np.array(dop20_zarr[int(pidx)])
        blocks = _tile_patch(rgb)
        feats  = _extract_features_for_patch(blocks, backbone, device, batch_size)
        chosen = rng.choice(feats.shape[0], size=blocks_per_patch, replace=False)
        all_samples.append(feats[chosen])
    X = np.concatenate(all_samples, axis=0)
    print(f"  Fitting PCA({feature_dim}) on {X.shape[0]:,} vectors …")
    pca = PCA(n_components=feature_dim, whiten=False, random_state=seed)
    pca.fit(X)
    print(f"  Done — explained variance: {pca.explained_variance_ratio_.sum()*100:.1f}%")
    return pca


# ── extraction (single shard) ─────────────────────────────────────────────────

def _progress_file(output_path: Path, rank: int) -> Path:
    """Per-rank plain-text progress file — avoids zarr attrs race conditions."""
    return output_path.parent / f"dop20_progress_rank{rank}.txt"


def _read_progress(output_path: Path, rank: int, start_idx: int) -> int:
    """Return the last successfully written patch index for this rank, or start_idx-1."""
    p = _progress_file(output_path, rank)
    if p.exists():
        try:
            return int(p.read_text().strip())
        except (ValueError, OSError):
            pass
    return start_idx - 1


def _write_progress(output_path: Path, rank: int, idx: int) -> None:
    """Atomically update the per-rank progress file."""
    p = _progress_file(output_path, rank)
    tmp = p.with_suffix(".tmp")
    tmp.write_text(str(idx))
    tmp.replace(p)   # atomic on POSIX


def extract_shard(
    dop20_zarr: zarr.Array,
    out_array: zarr.Array,
    backbone: ResNet50Backbone,
    pca: PCA,
    device: torch.device,
    batch_size: int,
    start_idx: int,
    end_idx: int,
    rank: int,
    output_path: Path,
) -> None:
    """
    Process patches [start_idx, end_idx) and write them to out_array.

    Per-rank resume: progress is tracked in a plain text file
    dop20_progress_rank{rank}.txt alongside the zarr store — one file per
    rank, no shared state, no zarr-attrs race conditions.
    """
    D = pca.n_components_
    last_done  = _read_progress(output_path, rank, start_idx)
    resume_from = last_done + 1

    shard_size = end_idx - start_idx
    if resume_from > start_idx:
        done = resume_from - start_idx
        print(f"[rank {rank}] Resuming: {done}/{shard_size} patches already done "
              f"(next index {resume_from}).")
    else:
        print(f"[rank {rank}] Starting: patches {start_idx}..{end_idx-1} ({shard_size} total).")

    if resume_from >= end_idx:
        print(f"[rank {rank}] Shard already complete.")
        return

    t0 = time.time()
    for i in tqdm(
        range(resume_from, end_idx),
        desc=f"rank{rank}",
        initial=resume_from - start_idx,
        total=shard_size,
    ):
        rgb      = np.array(dop20_zarr[i])
        blocks   = _tile_patch(rgb)
        f2048    = _extract_features_for_patch(blocks, backbone, device, batch_size)
        fd       = pca.transform(f2048).astype(np.float16)
        feat_map = fd.reshape(SENTINEL_SIZE, SENTINEL_SIZE, D).transpose(2, 0, 1)
        out_array[i] = feat_map          # atomic chunk write (one file per patch)
        _write_progress(output_path, rank, i)

    elapsed = time.time() - t0
    n_done = end_idx - resume_from
    print(f"[rank {rank}] Done — {n_done} patches in {elapsed/60:.1f} min "
          f"({elapsed/max(n_done,1):.2f} s/patch).")


# ── CLI ───────────────────────────────────────────────────────────────────────

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Extract frozen ResNet50 DOP20 features → zarr store",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    p.add_argument("--zarr-root", default="/data",
                   help="Directory containing train_dop20.zarr (default: /data)")
    p.add_argument("--output", default=None,
                   help="Output zarr path (default: <zarr-root>/train_dop20_features.zarr)")
    p.add_argument("--feature-dim", type=int, default=256, metavar="D",
                   help="PCA output dimension (default: 256)")
    p.add_argument("--batch-size", type=int, default=512,
                   help="Blocks per GPU forward pass (default: 512, try 768-1024 on A40)")
    p.add_argument("--pca-patches", type=int, default=200,
                   help="Patches sampled for PCA fitting (default: 200)")
    p.add_argument("--pca-blocks-per-patch", type=int, default=1000,
                   help="Random blocks per patch for PCA (default: 1000)")
    p.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu",
                   help="Compute device (default: cuda if available)")
    p.add_argument("--seed", type=int, default=42,
                   help="Random seed for PCA sampling (default: 42)")
    p.add_argument("--no-resume", action="store_true",
                   help="Delete existing output and restart from scratch")

    # ── parallelism ──────────────────────────────────────────────────────────
    p.add_argument("--pca-only", action="store_true",
                   help="Fit PCA + create zarr skeleton only, then exit. "
                        "Run this once before launching parallel workers.")
    p.add_argument("--rank", type=int, default=0,
                   help="Index of this worker (0-based). Default 0.")
    p.add_argument("--world-size", type=int, default=1,
                   help="Total number of parallel workers. Default 1 (single process).")
    return p.parse_args()


def main() -> None:
    args = parse_args()
    zarr_root   = Path(args.zarr_root)
    output_path = Path(args.output) if args.output else zarr_root / "train_dop20_features.zarr"
    D           = args.feature_dim
    device      = torch.device(args.device)

    if args.rank >= args.world_size:
        raise ValueError(f"--rank {args.rank} must be < --world-size {args.world_size}")

    print(f"Device      : {device}")
    print(f"DOP20 src   : {zarr_root / 'train_dop20.zarr'}")
    print(f"Output      : {output_path}")
    print(f"Feature D   : {D}")
    if args.world_size > 1 or args.pca_only:
        print(f"Mode        : {'pca-only' if args.pca_only else f'rank {args.rank}/{args.world_size}'}")

    # ── open DOP20 source ─────────────────────────────────────────────────────
    dop20_store = zarr.open(str(zarr_root / "train_dop20.zarr"), mode="r")
    dop20_rgb: zarr.Array = dop20_store["dop20/patches/rgb"]
    n_total = dop20_rgb.shape[0]
    print(f"Total patches: {n_total}")

    # ── backbone ──────────────────────────────────────────────────────────────
    print("Loading ResNet50 (ImageNet weights) …")
    backbone = ResNet50Backbone().to(device).eval()

    # ── PCA: fit or load ──────────────────────────────────────────────────────
    pca_path = output_path.parent / f"dop20_pca_{D}.pkl"

    if pca_path.exists() and not args.no_resume:
        print(f"Loading existing PCA from {pca_path} …")
        with open(pca_path, "rb") as fh:
            pca: PCA = pickle.load(fh)
        print(f"  Loaded — n_components={pca.n_components_}")
    elif args.rank == 0 or args.pca_only:
        # Only rank 0 (or pca-only mode) fits the PCA so all ranks use the same projection
        pca = fit_pca(
            dop20_rgb, backbone, device,
            batch_size=args.batch_size,
            feature_dim=D,
            n_sample_patches=args.pca_patches,
            blocks_per_patch=args.pca_blocks_per_patch,
            seed=args.seed,
        )
        pca_path.parent.mkdir(parents=True, exist_ok=True)
        with open(pca_path, "wb") as fh:
            pickle.dump(pca, fh)
        print(f"PCA saved → {pca_path}")
    else:
        # Non-zero rank: PCA must have been fitted already by rank 0 / pca-only step
        raise FileNotFoundError(
            f"PCA file not found: {pca_path}\n"
            "Run rank 0 first (or use --pca-only) to fit PCA before launching other workers."
        )

    # ── output zarr: create skeleton (idempotent, safe for concurrent opens) ──
    if args.no_resume and output_path.exists():
        import shutil; shutil.rmtree(output_path)

    out_store = zarr.open(str(output_path), mode="a")
    if "features" not in out_store:
        compressor = zarr.Blosc(cname="zstd", clevel=3, shuffle=zarr.Blosc.BITSHUFFLE)
        out_store.require_dataset(
            "features",
            shape=(n_total, D, SENTINEL_SIZE, SENTINEL_SIZE),
            chunks=(1, D, SENTINEL_SIZE, SENTINEL_SIZE),
            dtype="float16",
            compressor=compressor,
            fill_value=0,
            exact=True,
        )
        out_store["features"].attrs.update({
            "description": "Frozen ResNet50 (ImageNet) DOP20 features, PCA-projected",
            "feature_dim": D,
            "block_size_px": BLOCK_SIZE,
            "sentinel_size": SENTINEL_SIZE,
            "pca_pkl": str(pca_path),
        })
        print(f"Created zarr array: shape={out_store['features'].shape}")
    else:
        print("Zarr array already exists (append / resume mode).")

    # ── exit early if pca-only ────────────────────────────────────────────────
    if args.pca_only:
        print("\n--pca-only done. Now launch 4 parallel workers (see script docstring).")
        return

    # ── compute shard boundaries ──────────────────────────────────────────────
    ws   = args.world_size
    rank = args.rank
    # Distribute patches as evenly as possible across ranks
    base, rem = divmod(n_total, ws)
    start_idx = rank * base + min(rank, rem)
    end_idx   = start_idx + base + (1 if rank < rem else 0)

    out_array: zarr.Array = out_store["features"]

    # ── extract ───────────────────────────────────────────────────────────────
    extract_shard(
        dop20_zarr=dop20_rgb,
        out_array=out_array,
        backbone=backbone,
        pca=pca,
        device=device,
        batch_size=args.batch_size,
        start_idx=start_idx,
        end_idx=end_idx,
        rank=rank,
        output_path=output_path,
    )

    # ── storage summary (rank 0 only to avoid races) ──────────────────────────
    if rank == 0:
        try:
            print(f"\nZarr on disk: {out_array.nbytes_stored / 1e9:.1f} GB  "
                  f"(uncompressed {out_array.nbytes / 1e9:.1f} GB)")
        except Exception:
            pass


if __name__ == "__main__":
    main()
