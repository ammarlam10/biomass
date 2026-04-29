"""
Launch three independent training runs in parallel, each on 2 GPUs.

Default layout (6 GPUs total):
  Run 0 – seed=42   CUDA_VISIBLE_DEVICES=0,1  master_port=29500
  Run 1 – seed=123  CUDA_VISIBLE_DEVICES=2,3  master_port=29502
  Run 2 – seed=456  CUDA_VISIBLE_DEVICES=4,5  master_port=29504

Each run outputs to:
  <artifacts-base>/seed_<seed>/checkpoints/
  <artifacts-base>/seed_<seed>/runs/         (TensorBoard)
  <artifacts-base>/seed_<seed>/metrics.csv
  <artifacts-base>/seed_<seed>/train.log     (captured stdout + stderr)

Parallel runs multiply DataLoader workers across processes; Docker's default
/dev/shm (~64 MiB) is often exhausted (OSError errno 28 on SemLock). Mitigations:
  - Pass a large shm to Docker:  --shm-size=16g  (or higher)
  - Lower workers:  --num-workers 2  (default below is conservative)

Usage (inside container):
  python scripts/run_ensemble.py
  python scripts/run_ensemble.py --seeds 1 2 3
  python scripts/run_ensemble.py --gpu-pairs 0,1 2,3 4,5 --artifacts-base /workspace/artifacts/ensemble
"""

from __future__ import annotations

import argparse
import os
import subprocess
import sys
import threading
import time
from pathlib import Path
from typing import List, NamedTuple, Optional, Tuple

REPO_ROOT = Path(__file__).resolve().parents[1]

DEFAULT_SEEDS: List[int] = [42, 123, 456]
DEFAULT_GPU_PAIRS: List[Tuple[int, int]] = [(0, 1), (2, 3), (4, 5)]
DEFAULT_MASTER_PORTS: List[int] = [29500, 29502, 29504]


# ── data classes ──────────────────────────────────────────────────────────────

class RunSpec(NamedTuple):
    seed: int
    gpus: Tuple[int, int]
    master_port: int
    run_dir: Path
    log_path: Path


class RunResult(NamedTuple):
    spec: RunSpec
    returncode: int
    elapsed_sec: float


# ── argument parsing ───────────────────────────────────────────────────────────

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Run training 3× in parallel with different seeds (2 GPUs each)"
    )
    p.add_argument(
        "--config",
        default="configs/default.yaml",
        help="Config file passed to train.py (default: configs/default.yaml)",
    )
    p.add_argument(
        "--seeds",
        nargs=3,
        type=int,
        default=DEFAULT_SEEDS,
        metavar="SEED",
        help=f"Three integer seeds (default: {DEFAULT_SEEDS})",
    )
    p.add_argument(
        "--gpu-pairs",
        nargs=3,
        dest="gpu_pairs",
        metavar="A,B",
        default=None,
        help='Three GPU pairs as "A,B" strings (default: "0,1" "2,3" "4,5")',
    )
    p.add_argument(
        "--master-ports",
        nargs=3,
        dest="master_ports",
        type=int,
        default=DEFAULT_MASTER_PORTS,
        metavar="PORT",
        help=f"torchrun master ports for each run (default: {DEFAULT_MASTER_PORTS})",
    )
    p.add_argument(
        "--artifacts-base",
        dest="artifacts_base",
        default="/workspace/artifacts",
        help="Base artifacts directory; each run writes to <base>/seed_<seed>/",
    )
    p.add_argument(
        "--num-workers",
        dest="num_workers",
        type=int,
        default=2,
        help=(
            "DataLoader workers per rank (passed to train.py). "
            "Default 2 keeps /dev/shm usage low when 3×2-GPU jobs run in one container. "
            "Raise (e.g. 8) if Docker has a large --shm-size."
        ),
    )
    return p.parse_args()


# ── output streaming ──────────────────────────────────────────────────────────

def _stream(proc: subprocess.Popen, label: str, log_path: Path) -> None:
    """Read proc stdout (combined with stderr) line-by-line; write to log and console."""
    with open(log_path, "w", buffering=1) as fh:
        for line in proc.stdout:  # type: ignore[union-attr]
            fh.write(line)
            sys.stdout.write(f"[{label}] {line}")
            sys.stdout.flush()


# ── run launcher ──────────────────────────────────────────────────────────────

def launch(
    spec: RunSpec,
    config: str,
    num_workers: int,
) -> Tuple[subprocess.Popen, threading.Thread]:
    spec.run_dir.mkdir(parents=True, exist_ok=True)

    env = os.environ.copy()
    env["CUDA_VISIBLE_DEVICES"] = f"{spec.gpus[0]},{spec.gpus[1]}"

    cmd = [
        "torchrun",
        "--standalone",
        "--nproc_per_node=2",
        f"--master_port={spec.master_port}",
        "scripts/train.py",
        "--config", config,
        "--ddp",
        "--seed", str(spec.seed),
        "--run-dir", str(spec.run_dir),
        "--num-workers", str(num_workers),
    ]

    proc = subprocess.Popen(
        cmd,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        env=env,
        cwd=str(REPO_ROOT),
    )

    label = f"seed={spec.seed} gpu={spec.gpus[0]},{spec.gpus[1]}"
    thread = threading.Thread(
        target=_stream, args=(proc, label, spec.log_path), daemon=True
    )
    thread.start()
    return proc, thread


# ── main ──────────────────────────────────────────────────────────────────────

def main() -> None:
    args = parse_args()

    # Parse GPU pairs
    if args.gpu_pairs is not None:
        try:
            gpu_pairs: List[Tuple[int, int]] = [
                tuple(int(g) for g in pair.split(","))  # type: ignore[misc]
                for pair in args.gpu_pairs
            ]
        except ValueError:
            sys.exit("--gpu-pairs: expected format is A,B  e.g.  0,1  2,3  4,5")
    else:
        gpu_pairs = DEFAULT_GPU_PAIRS

    seeds: List[int] = args.seeds
    ports: List[int] = args.master_ports
    base = Path(args.artifacts_base)

    specs = [
        RunSpec(
            seed=seed,
            gpus=gpus,
            master_port=port,
            run_dir=base / f"seed_{seed}",
            log_path=base / f"seed_{seed}" / "train.log",
        )
        for seed, gpus, port in zip(seeds, gpu_pairs, ports)
    ]

    # ── banner ────────────────────────────────────────────────────────────────
    sep = "─" * 62
    print(sep)
    print("  Ensemble training   3 seeds × 2 GPUs")
    print(sep)
    for i, s in enumerate(specs):
        print(
            f"  Run {i + 1}:  seed={s.seed:<6d}  "
            f"GPUs={s.gpus[0]},{s.gpus[1]}  port={s.master_port}"
        )
        print(f"         artifacts → {s.run_dir}")
    print(sep)
    print(f"  DataLoader num_workers (per rank): {args.num_workers}")
    print(sep)
    print()

    # ── launch all three ──────────────────────────────────────────────────────
    active: List[Tuple[RunSpec, subprocess.Popen, threading.Thread, float]] = []
    for spec in specs:
        proc, thread = launch(spec, args.config, args.num_workers)
        active.append((spec, proc, thread, time.time()))
        print(f"  Launched seed={spec.seed}  PID={proc.pid}  log → {spec.log_path}")
    print()

    # ── wait and collect results ──────────────────────────────────────────────
    results: List[RunResult] = []
    for spec, proc, thread, t0 in active:
        proc.wait()
        thread.join()
        elapsed = time.time() - t0
        results.append(RunResult(spec=spec, returncode=proc.returncode, elapsed_sec=elapsed))

    # ── summary ───────────────────────────────────────────────────────────────
    print()
    print(sep)
    print("  Ensemble training – results")
    print(sep)
    all_ok = True
    for r in results:
        status = "PASS" if r.returncode == 0 else f"FAIL (exit {r.returncode})"
        mins = r.elapsed_sec / 60
        print(
            f"  seed={r.spec.seed:<6d}  GPUs={r.spec.gpus[0]},{r.spec.gpus[1]}  "
            f"{status:<18s}  {mins:.1f} min"
        )
        print(f"           log  → {r.spec.log_path}")
        ckpt = r.spec.run_dir / "checkpoints" / "best.pt"
        if ckpt.exists():
            print(f"           best → {ckpt}")
        if r.returncode != 0:
            all_ok = False
    print(sep)

    if not all_ok:
        failed = [str(r.spec.seed) for r in results if r.returncode != 0]
        print(f"\n  {len(failed)} run(s) failed (seeds: {', '.join(failed)}).")
        print("  Check the log files above for details.")

    sys.exit(0 if all_ok else 1)


if __name__ == "__main__":
    main()
