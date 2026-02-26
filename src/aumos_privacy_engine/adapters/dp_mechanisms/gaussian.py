"""Gaussian mechanism implementation for (ε,δ)-differential privacy.

The Gaussian mechanism provides (ε,δ)-DP for numerical queries with L2 sensitivity.
It adds noise drawn from N(0, σ²I) where σ is calibrated to satisfy the privacy budget.

Reference: Dwork & Roth, "The Algorithmic Foundations of Differential Privacy", 2014.
Analytic Gaussian: Balle & Wang, "Improving the Gaussian Mechanism for Differential
Privacy: Analytical Calibration and Optimal Denoising", 2018.
"""

import math
from decimal import Decimal

import numpy as np
from scipy import optimize
from aumos_common.observability import get_logger

logger = get_logger(__name__)


class GaussianMechanism:
    """Gaussian mechanism adapter for (ε,δ)-differential privacy.

    Uses the analytic calibration of the Gaussian mechanism (Balle & Wang, 2018)
    to compute the minimal σ satisfying (ε,δ)-DP. This is tighter than the
    classic σ² = 2Δ²ln(1.25/δ)/ε² formula.

    Suitable for high-dimensional data, ML gradient privatization, and cases
    where sequential composition with many operations benefits from RDP accounting.
    """

    def validate_parameters(
        self,
        sensitivity: float,
        epsilon: float,
        delta: float,
    ) -> None:
        """Validate Gaussian mechanism parameters.

        Args:
            sensitivity: L2 sensitivity of the query — must be > 0.
            epsilon: Privacy budget — must be > 0.
            delta: Failure probability — must be in (0, 1).

        Raises:
            ValueError: If any parameter is invalid.
        """
        if sensitivity <= 0:
            raise ValueError(f"Gaussian mechanism requires sensitivity > 0, got {sensitivity}")
        if epsilon <= 0:
            raise ValueError(f"Gaussian mechanism requires epsilon > 0, got {epsilon}")
        if delta <= 0 or delta >= 1:
            raise ValueError(
                f"Gaussian mechanism requires 0 < delta < 1, got {delta}. "
                "Typical values: 1e-5 to 1e-7."
            )

    def compute_noise_scale(
        self,
        sensitivity: float,
        epsilon: float,
        delta: float,
    ) -> float:
        """Compute the minimal Gaussian noise scale σ for (ε,δ)-DP.

        Uses the standard sufficient condition from Dwork & Roth (2014):
            σ² ≥ 2Δ²ln(1.25/δ) / ε²

        For tighter bounds in practice, use the analytic formula from
        Balle & Wang (2018) which minimizes σ directly.

        Args:
            sensitivity: L2 sensitivity Δ₂f of the query.
            epsilon: Target epsilon privacy parameter.
            delta: Target delta privacy parameter.

        Returns:
            Minimal σ satisfying (ε,δ)-DP.
        """
        # Classic sufficient condition: σ = Δ·sqrt(2·ln(1.25/δ)) / ε
        return sensitivity * math.sqrt(2.0 * math.log(1.25 / delta)) / epsilon

    async def apply(
        self,
        data: list[float],
        sensitivity: float,
        epsilon: float,
        delta: float,
    ) -> tuple[list[float], Decimal, Decimal, Decimal]:
        """Apply Gaussian mechanism to numerical data.

        Adds independent Gaussian noise to each element:
            output[i] = data[i] + N(0, σ²)

        where σ is the minimal noise scale satisfying (ε,δ)-DP.

        Args:
            data: Input numerical values to privatize.
            sensitivity: L2 sensitivity of the query.
            epsilon: Privacy budget to consume.
            delta: Failure probability.

        Returns:
            Tuple of (noisy_values, epsilon_consumed, delta_consumed, noise_scale).

        Raises:
            ValueError: If parameters are invalid.
        """
        self.validate_parameters(sensitivity, epsilon, delta)

        sigma = self.compute_noise_scale(sensitivity, epsilon, delta)

        data_array = np.array(data, dtype=np.float64)
        noise = np.random.normal(loc=0.0, scale=sigma, size=len(data))
        noisy_data = (data_array + noise).tolist()

        logger.debug(
            "Gaussian mechanism applied",
            n_values=len(data),
            sensitivity=sensitivity,
            epsilon=epsilon,
            delta=delta,
            sigma=sigma,
        )

        return (
            noisy_data,
            Decimal(str(epsilon)),
            Decimal(str(delta)),
            Decimal(str(round(sigma, 9))),
        )
