"""Moments accountant for Rényi Differential Privacy (RDP) composition.

Implements the moments accountant (Abadi et al. 2016) for tight composition
accounting in the Rényi DP framework. Provides:
- Rényi divergence computation for the Gaussian mechanism
- Moments accumulation under sequential composition
- Tight conversion from (α, ε̃)-RDP to (ε, δ)-DP
- Optimal α selection for a given target δ
- Subsampled RDP accounting (Poisson subsampling)

References:
  Abadi et al. (2016), "Deep Learning with Differential Privacy"
  Mironov (2017), "Rényi Differential Privacy of the Gaussian Mechanism"
  Wang et al. (2019), "Subsampled Rényi Differential Privacy"
  Balle et al. (2020), "Hypothesis Testing Interpretations and Renyi DP"
"""

import asyncio
import math
from typing import Any

import numpy as np

from aumos_common.observability import get_logger

logger = get_logger(__name__)

# Default set of Rényi orders to evaluate during optimal α search
DEFAULT_ALPHA_GRID: list[float] = [
    1.25, 1.5, 1.75, 2.0, 2.5, 3.0, 4.0, 5.0, 6.0, 8.0,
    10.0, 12.0, 16.0, 20.0, 24.0, 32.0, 40.0, 48.0, 64.0,
    80.0, 100.0, 128.0, 256.0, 512.0,
]


class MomentAccountant:
    """Rényi DP moments accountant for tight privacy loss composition.

    Tracks the Rényi divergence (ε̃(α)) accumulated across multiple mechanism
    applications and converts the aggregate RDP bound to an (ε, δ)-DP guarantee
    at the end of the computation.

    Key advantage over basic sequential composition (ε_total = Σεᵢ):
    RDP composition often yields significantly tighter bounds, especially
    when many small-ε operations are composed (e.g., DP-SGD training).

    Attributes:
        _alpha_orders: Rényi orders at which moments are tracked.
        _rdp_moments: Accumulated RDP ε̃(α) at each order, per tenant.
        _step_count: Number of mechanism applications recorded per tenant.
    """

    def __init__(
        self,
        alpha_orders: list[float] | None = None,
    ) -> None:
        """Initialize the moments accountant.

        Args:
            alpha_orders: Rényi orders to track. If None, uses DEFAULT_ALPHA_GRID.
                          Must all be > 1.0. More orders = better optimization
                          at cost of more computation.
        """
        orders = alpha_orders if alpha_orders is not None else DEFAULT_ALPHA_GRID
        if any(a <= 1.0 for a in orders):
            raise ValueError("All Rényi orders must be > 1.0 (required for finite RDP)")
        self._alpha_orders: list[float] = sorted(orders)

        # Per-tenant accumulated RDP moments: tenant_id_str -> array of ε̃(αᵢ) values
        self._rdp_moments: dict[str, np.ndarray] = {}
        self._step_count: dict[str, int] = {}

    def _tenant_key(self, tenant_id: str) -> str:
        return tenant_id

    def initialize_tenant(self, tenant_id: str) -> None:
        """Initialize RDP tracking for a new tenant.

        Args:
            tenant_id: Tenant identifier string (typically str(UUID)).
        """
        key = self._tenant_key(tenant_id)
        if key not in self._rdp_moments:
            self._rdp_moments[key] = np.zeros(len(self._alpha_orders), dtype=np.float64)
            self._step_count[key] = 0
            logger.debug("Moment accountant initialized for tenant", tenant_id=tenant_id)

    def _gaussian_rdp(self, sigma: float, alpha: float) -> float:
        """Compute RDP ε̃(α) for the Gaussian mechanism.

        For the Gaussian mechanism with sensitivity 1 and noise std σ:
            ε̃(α) = α / (2σ²)

        For general sensitivity Δ:
            ε̃(α) = α · Δ² / (2σ²)

        Here we normalize to sensitivity=1 (caller scales by Δ²/σ² ratio).

        Args:
            sigma: Noise standard deviation (for sensitivity 1).
            alpha: Rényi order.

        Returns:
            RDP privacy loss ε̃(α) for the Gaussian mechanism.
        """
        if sigma <= 0:
            raise ValueError(f"sigma must be > 0, got {sigma}")
        if alpha <= 1.0:
            raise ValueError(f"alpha must be > 1.0 for RDP, got {alpha}")
        # Mironov (2017), Proposition 3
        return alpha / (2.0 * sigma * sigma)

    def _gaussian_rdp_with_sensitivity(
        self,
        sensitivity: float,
        sigma: float,
        alpha: float,
    ) -> float:
        """Compute RDP ε̃(α) for Gaussian mechanism with arbitrary sensitivity.

        Args:
            sensitivity: L2 sensitivity of the query.
            sigma: Noise standard deviation.
            alpha: Rényi order.

        Returns:
            RDP privacy loss ε̃(α).
        """
        # Scale by (Δ/σ)² * α/2
        ratio = (sensitivity / sigma) ** 2
        return alpha * ratio / 2.0

    def _subsampled_gaussian_rdp(
        self,
        sampling_rate: float,
        sigma: float,
        alpha: float,
    ) -> float:
        """Compute RDP ε̃(α) for Poisson-subsampled Gaussian mechanism.

        Uses the amplification by Poisson subsampling result.
        For small subsampling rate q and moderate alpha, this is approximately:
            ε̃_subsampled(α) ≈ log(1 + q²·(e^{ε̃_full(α)} - 1))

        For more precise bounds at small q (Wang et al. 2019):
            ε̃_subsampled(α) ≤ (1/(α-1)) · log(
                (1-q)^α · e^{(α-1)·ε̃_full(α)/(α)}
                + α·q·(1-q)^{α-1} · e^{(α-1)·ε̃_full(2α-1)/(2α-1)}
                + ...
            )

        We use the simplified amplification formula which is tight for small q.

        Args:
            sampling_rate: Poisson sampling probability q ∈ (0, 1].
            sigma: Noise standard deviation.
            alpha: Rényi order.

        Returns:
            Amplified RDP privacy loss ε̃_subsampled(α).
        """
        if not (0 < sampling_rate <= 1.0):
            raise ValueError(f"sampling_rate must be in (0, 1], got {sampling_rate}")

        if sampling_rate == 1.0:
            return self._gaussian_rdp(sigma, alpha)

        # Full mechanism RDP at order alpha
        rdp_full = self._gaussian_rdp(sigma, alpha)

        # Subsampling amplification (simplified upper bound)
        # ε̃_sub(α) ≤ min(ε̃_full(α), log(1 + q(e^{ε̃_full(α)} - 1)) / (α-1) * (α-1) + log(...))
        # Practical approximation: ε̃_sub(α) ≈ q² * α * (α-1) / (2σ²)
        # This is the standard privacy amplification by subsampling bound for small q
        if sampling_rate < 0.01 and alpha > 1:
            # Use tighter second-order approximation for small q
            amplified = (sampling_rate ** 2) * rdp_full * alpha
            return min(rdp_full, amplified)

        # General case: log(1 + q²*(exp(rdp_full*(α-1))-1)) / (α-1)
        # Numerically stable computation
        inner = (sampling_rate ** 2) * (math.expm1(rdp_full * (alpha - 1.0)))
        if inner > 700:  # prevent overflow
            return rdp_full
        amplified = math.log1p(inner) / (alpha - 1.0)
        return min(rdp_full, amplified)

    def accumulate_gaussian(
        self,
        tenant_id: str,
        sigma: float,
        sensitivity: float = 1.0,
        num_steps: int = 1,
    ) -> None:
        """Accumulate RDP moments for Gaussian mechanism applications.

        Adds the RDP privacy loss for `num_steps` applications of the
        Gaussian mechanism with given sigma and sensitivity.

        Args:
            tenant_id: Tenant accumulating budget.
            sigma: Gaussian noise standard deviation.
            sensitivity: L2 sensitivity of the query (default 1.0).
            num_steps: Number of times the mechanism is applied (for batched accounting).

        Raises:
            RuntimeError: If the tenant is not initialized.
        """
        key = self._tenant_key(tenant_id)
        if key not in self._rdp_moments:
            raise RuntimeError(f"Tenant {tenant_id} not initialized. Call initialize_tenant() first.")

        for i, alpha in enumerate(self._alpha_orders):
            rdp_per_step = self._gaussian_rdp_with_sensitivity(sensitivity, sigma, alpha)
            self._rdp_moments[key][i] += rdp_per_step * num_steps

        self._step_count[key] += num_steps

        logger.debug(
            "Gaussian RDP moments accumulated",
            tenant_id=tenant_id,
            sigma=sigma,
            sensitivity=sensitivity,
            num_steps=num_steps,
            total_steps=self._step_count[key],
        )

    def accumulate_subsampled_gaussian(
        self,
        tenant_id: str,
        sigma: float,
        sampling_rate: float,
        num_steps: int = 1,
    ) -> None:
        """Accumulate RDP moments for Poisson-subsampled Gaussian mechanism.

        Args:
            tenant_id: Tenant accumulating budget.
            sigma: Gaussian noise standard deviation.
            sampling_rate: Poisson subsampling rate q ∈ (0, 1].
            num_steps: Number of mechanism applications.

        Raises:
            RuntimeError: If the tenant is not initialized.
        """
        key = self._tenant_key(tenant_id)
        if key not in self._rdp_moments:
            raise RuntimeError(f"Tenant {tenant_id} not initialized.")

        for i, alpha in enumerate(self._alpha_orders):
            rdp_per_step = self._subsampled_gaussian_rdp(sampling_rate, sigma, alpha)
            self._rdp_moments[key][i] += rdp_per_step * num_steps

        self._step_count[key] += num_steps

        logger.debug(
            "Subsampled Gaussian RDP moments accumulated",
            tenant_id=tenant_id,
            sigma=sigma,
            sampling_rate=sampling_rate,
            num_steps=num_steps,
        )

    def rdp_to_dp(
        self,
        tenant_id: str,
        target_delta: float,
    ) -> tuple[float, float]:
        """Convert accumulated RDP moments to (ε, δ)-DP guarantee.

        Applies the optimal conversion from Balle et al. (2020):
            ε(α) = ε̃(α) + log(1 - 1/α) - (log(δ) + log(α)) / (α - 1)

        and selects the α that minimizes the resulting ε.

        Args:
            tenant_id: Tenant to convert.
            target_delta: Target failure probability δ ∈ (0, 1).

        Returns:
            Tuple of (epsilon, delta) representing the (ε, δ)-DP guarantee.

        Raises:
            RuntimeError: If the tenant is not initialized.
            ValueError: If target_delta is out of range.
        """
        if not (0 < target_delta < 1):
            raise ValueError(f"target_delta must be in (0, 1), got {target_delta}")

        key = self._tenant_key(tenant_id)
        if key not in self._rdp_moments:
            raise RuntimeError(f"Tenant {tenant_id} not initialized.")

        moments = self._rdp_moments[key]
        log_delta = math.log(target_delta)

        best_epsilon = float("inf")
        for i, alpha in enumerate(self._alpha_orders):
            rdp_eps = moments[i]
            if rdp_eps == 0.0:
                continue
            # Mironov (2017) Proposition 3 conversion:
            # ε = ε̃(α) + log(α-1)/α - (log(δ·(α-1)) + 1) / (α-1)
            # Simplified: ε = ε̃(α) - (log(δ) + log(α)) / (α-1) + log((α-1)/α)
            try:
                converted = (
                    rdp_eps
                    - (log_delta + math.log(alpha)) / (alpha - 1.0)
                    + math.log((alpha - 1.0) / alpha)
                )
            except (ValueError, OverflowError):
                continue

            if converted < best_epsilon:
                best_epsilon = converted

        if best_epsilon == float("inf"):
            # No steps accumulated — zero privacy loss
            best_epsilon = 0.0

        # Epsilon must be non-negative
        best_epsilon = max(0.0, best_epsilon)

        logger.debug(
            "RDP converted to (ε,δ)-DP",
            tenant_id=tenant_id,
            epsilon=best_epsilon,
            delta=target_delta,
            total_steps=self._step_count.get(key, 0),
        )

        return (best_epsilon, target_delta)

    def get_optimal_alpha(
        self,
        tenant_id: str,
        target_delta: float,
    ) -> float:
        """Find the Rényi order α that minimizes the resulting (ε, δ)-DP bound.

        Args:
            tenant_id: Tenant to optimize for.
            target_delta: Target failure probability δ.

        Returns:
            Optimal α value from the alpha grid.

        Raises:
            RuntimeError: If the tenant is not initialized.
        """
        key = self._tenant_key(tenant_id)
        if key not in self._rdp_moments:
            raise RuntimeError(f"Tenant {tenant_id} not initialized.")

        moments = self._rdp_moments[key]
        log_delta = math.log(target_delta)
        best_alpha = self._alpha_orders[0]
        best_epsilon = float("inf")

        for i, alpha in enumerate(self._alpha_orders):
            rdp_eps = moments[i]
            if rdp_eps == 0.0:
                continue
            try:
                converted = (
                    rdp_eps
                    - (log_delta + math.log(alpha)) / (alpha - 1.0)
                    + math.log((alpha - 1.0) / alpha)
                )
            except (ValueError, OverflowError):
                continue

            if converted < best_epsilon:
                best_epsilon = converted
                best_alpha = alpha

        return best_alpha

    async def async_rdp_to_dp(
        self,
        tenant_id: str,
        target_delta: float,
    ) -> tuple[float, float]:
        """Async wrapper for rdp_to_dp, running in thread executor for CPU-bound work.

        Args:
            tenant_id: Tenant to convert.
            target_delta: Target failure probability δ.

        Returns:
            Tuple of (epsilon, delta).
        """
        loop = asyncio.get_running_loop()
        return await loop.run_in_executor(
            None,
            self.rdp_to_dp,
            tenant_id,
            target_delta,
        )

    def get_moments_summary(self, tenant_id: str) -> dict[str, Any]:
        """Get the current accumulated RDP moments for a tenant.

        Args:
            tenant_id: Tenant to summarize.

        Returns:
            Dictionary with alpha orders, accumulated RDP values, and step count.
        """
        key = self._tenant_key(tenant_id)
        if key not in self._rdp_moments:
            return {"error": f"Tenant {tenant_id} not initialized"}

        moments = self._rdp_moments[key]
        return {
            "tenant_id": tenant_id,
            "total_steps": self._step_count.get(key, 0),
            "alpha_orders": self._alpha_orders,
            "rdp_moments": moments.tolist(),
            "num_orders": len(self._alpha_orders),
        }

    def reset_tenant(self, tenant_id: str) -> None:
        """Reset all accumulated moments for a tenant (e.g., on budget renewal).

        Args:
            tenant_id: Tenant to reset.
        """
        key = self._tenant_key(tenant_id)
        if key in self._rdp_moments:
            self._rdp_moments[key] = np.zeros(len(self._alpha_orders), dtype=np.float64)
            self._step_count[key] = 0
            logger.info("Moment accountant reset for tenant", tenant_id=tenant_id)
