"""
Pixel-wise regression metrics computed only on valid (masked) pixels.

Target names: tree_count (index 0), mean_height (index 1).

Two usage modes:

1. Batch-at-a-time streaming (preferred for large val sets – no memory spike):
       rm = RunningMetrics(inv_transforms)
       for pred, target, mask in loader:
           rm.update(pred, target, mask)
       metrics = rm.compute()

2. Full-tensor (legacy – kept for backward compat):
       metrics = compute_masked_metrics(pred_all, target_all, mask_all, inv_t)

All metrics are computed in the transformed space (same as training loss)
unless inverse_transform is provided.
"""

from __future__ import annotations

import math
from typing import Callable, Dict, List, Optional

import torch


TARGET_NAMES: List[str] = ["tree_count", "mean_height"]

# Upper clamp in log1p-space before expm1 when mapping to original units for metrics.
# Prevents overflow from unconstrained regression tails / fp16 noise (expm1 grows fast).
# Values are generous vs. dataset max (count≈8, height≈54 m).
_LOG1P_INV_MAX: tuple[float, float] = (math.log1p(64.0), math.log1p(128.0))


def _inverse_log1p_stable(log_max: float) -> Callable[[torch.Tensor], torch.Tensor]:
    def fn(x: torch.Tensor) -> torch.Tensor:
        return torch.expm1(x.clamp(min=-0.999999, max=log_max))

    return fn


class RunningMetrics:
    """
    Online accumulator for masked regression metrics.

    Tracks per-target sum-of-squared-errors, sum-of-abs-errors, sum of
    targets (for R²), and pixel count — all in float64 on CPU.
    Memory cost: O(n_targets) rather than O(n_patches × H × W).

    Usage:
        rm = RunningMetrics(inv_transforms)
        for pred, target, mask in val_batches:
            rm.update(pred.cpu(), target.cpu(), mask.cpu())
        metrics = rm.compute()
        rm.reset()
    """

    def __init__(self, inverse_transforms: Optional[List[Optional[Callable]]] = None) -> None:
        self._inv_transforms = inverse_transforms or [None] * len(TARGET_NAMES)
        self.reset()

    def reset(self) -> None:
        n = len(TARGET_NAMES)
        # transformed-space accumulators
        self._sse = [0.0] * n        # sum of squared errors
        self._sae = [0.0] * n        # sum of absolute errors
        self._sum_t = [0.0] * n      # sum of targets (for R²)
        self._sum_t2 = [0.0] * n     # sum of target² (for R²)
        self._cnt = [0] * n          # valid pixel count
        # original-space (after inverse transform) – full set for RMSE, MAE, R²
        self._sse_orig = [0.0] * n
        self._sae_orig = [0.0] * n
        self._sum_t_orig = [0.0] * n
        self._sum_t2_orig = [0.0] * n
        self._cnt_orig = [0] * n

    def update(
        self,
        pred: torch.Tensor,
        target: torch.Tensor,
        mask: torch.Tensor,
    ) -> None:
        """
        Accumulate one batch.

        Args:
            pred   : [B, 2, H, W]
            target : [B, 2, H, W]
            mask   : [B, H, W] bool
        """
        with torch.no_grad():
            for i in range(len(TARGET_NAMES)):
                p = pred[:, i][mask].double()
                t = target[:, i][mask].double()
                n = p.numel()
                if n == 0:
                    continue
                residuals = p - t
                self._sse[i] += (residuals ** 2).sum().item()
                self._sae[i] += residuals.abs().sum().item()
                self._sum_t[i] += t.sum().item()
                self._sum_t2[i] += (t ** 2).sum().item()
                self._cnt[i] += n

                if self._inv_transforms[i] is not None:
                    p_orig = self._inv_transforms[i](p)
                    t_orig = self._inv_transforms[i](t)
                    res_orig = p_orig - t_orig
                    self._sse_orig[i] += (res_orig ** 2).sum().item()
                    self._sae_orig[i] += res_orig.abs().sum().item()
                    self._sum_t_orig[i] += t_orig.sum().item()
                    self._sum_t2_orig[i] += (t_orig ** 2).sum().item()
                    self._cnt_orig[i] += n

    def compute(self) -> Dict[str, float]:
        results: Dict[str, float] = {}
        for i, name in enumerate(TARGET_NAMES):
            n = self._cnt[i]
            if n == 0:
                results[f"rmse_{name}"] = float("nan")
                results[f"mae_{name}"] = float("nan")
                results[f"r2_{name}"] = float("nan")
                continue

            mse = self._sse[i] / n
            results[f"rmse_{name}"] = math.sqrt(max(mse, 0.0))
            results[f"mae_{name}"] = self._sae[i] / n

            mean_t = self._sum_t[i] / n
            ss_tot = self._sum_t2[i] - n * mean_t ** 2
            results[f"r2_{name}"] = 1.0 - self._sse[i] / (ss_tot + 1e-8)

            if self._cnt_orig[i] > 0:
                n_o = self._cnt_orig[i]
                results[f"rmse_{name}_orig"] = math.sqrt(
                    max(self._sse_orig[i] / n_o, 0.0)
                )
                results[f"mae_{name}_orig"] = self._sae_orig[i] / n_o
                mean_t_orig = self._sum_t_orig[i] / n_o
                ss_tot_orig = self._sum_t2_orig[i] - n_o * mean_t_orig ** 2
                results[f"r2_{name}_orig"] = 1.0 - self._sse_orig[i] / (ss_tot_orig + 1e-8)

        return results


def compute_masked_metrics(
    pred: torch.Tensor,
    target: torch.Tensor,
    mask: torch.Tensor,
    inverse_transforms: Optional[List[Optional[Callable]]] = None,
) -> Dict[str, float]:
    """
    Args:
        pred               : [B, 2, H, W]  or accumulated stack
        target             : [B, 2, H, W]
        mask               : [B, H, W] bool
        inverse_transforms : list of callables (one per target) or None;
                             used to compute metrics in original scale

    Returns:
        dict with keys  rmse_{name}, mae_{name}, r2_{name}
        and optionally  rmse_{name}_orig, mae_{name}_orig, r2_{name}_orig
    """
    results: Dict[str, float] = {}

    with torch.no_grad():
        for i, name in enumerate(TARGET_NAMES):
            p = pred[:, i][mask]    # [N_valid]
            t = target[:, i][mask]  # [N_valid]

            results.update(_single_target_metrics(p, t, name))

            if inverse_transforms is not None and inverse_transforms[i] is not None:
                p_orig = inverse_transforms[i](p)
                t_orig = inverse_transforms[i](t)
                results.update(_single_target_metrics(p_orig, t_orig, f"{name}_orig"))

    return results


def _single_target_metrics(p: torch.Tensor, t: torch.Tensor, name: str) -> Dict[str, float]:
    if p.numel() == 0:
        return {
            f"rmse_{name}": float("nan"),
            f"mae_{name}": float("nan"),
            f"r2_{name}": float("nan"),
        }

    residuals = p - t
    mse = torch.mean(residuals ** 2).item()
    rmse = math.sqrt(max(mse, 0.0))
    mae = torch.mean(torch.abs(residuals)).item()

    ss_res = torch.sum(residuals ** 2).item()
    ss_tot = torch.sum((t - t.mean()) ** 2).item()
    r2 = 1.0 - ss_res / (ss_tot + 1e-8)

    return {
        f"rmse_{name}": rmse,
        f"mae_{name}": mae,
        f"r2_{name}": r2,
    }


def get_inverse_transforms(cfg: dict) -> List[Optional[Callable]]:
    """
    Build inverse-transform callables matching target_transform config.
    Order: [tree_count, mean_height]

    log1p inverses clamp before expm1 so RMSE/MAE in original units stay finite when
    predictions leave the physical range (common under AMP or heavy tails).
    """
    tt = cfg.get("data", {}).get("target_transform", {})
    result: List[Optional[Callable[[torch.Tensor], torch.Tensor]]] = []
    for i, name in enumerate(TARGET_NAMES):
        mode = tt.get(name, "none")
        if mode == "log1p":
            result.append(_inverse_log1p_stable(_LOG1P_INV_MAX[i]))
        else:
            result.append(None)
    return result
