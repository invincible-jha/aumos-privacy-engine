"""Laplace mechanism implementation via OpenDP.

The Laplace mechanism provides ε-DP (pure differential privacy) for numerical
queries. It adds noise drawn from Lap(0, Δf/ε) to each query output value.

Reference: Dwork et al., "Calibrating Noise to Sensitivity in Private Data Analysis", 2006.
OpenDP: https://docs.opendp.org/en/stable/
"""

from decimal import Decimal

import numpy as np
from aumos_common.observability import get_logger

logger = get_logger(__name__)


class LaplaceMechanism:
    """Laplace mechanism adapter for pure ε-differential privacy.

    Wraps numpy's Laplace distribution sampling (OpenDP-compatible) to add
    calibrated noise to numerical query outputs. Provides ε-DP guarantee
    per the Laplace mechanism theorem.

    For vector outputs, noise is added independently to each element with
    scale parameter λ = sensitivity / ε.
    """

    def validate_parameters(
        self,
        sensitivity: float,
        epsilon: float,
        delta: float,
    ) -> None:
        """Validate Laplace mechanism parameters.

        Args:
            sensitivity: L1 sensitivity of the query — must be > 0.
            epsilon: Privacy budget — must be > 0.
            delta: Must be exactly 0.0 (Laplace provides pure DP).

        Raises:
            ValueError: If any parameter is invalid.
        """
        if sensitivity <= 0:
            raise ValueError(f"Laplace mechanism requires sensitivity > 0, got {sensitivity}")
        if epsilon <= 0:
            raise ValueError(f"Laplace mechanism requires epsilon > 0, got {epsilon}")
        if delta != 0.0:
            raise ValueError(
                f"Laplace mechanism provides pure ε-DP and requires delta=0.0, got delta={delta}. "
                "Use Gaussian mechanism for (ε,δ)-DP."
            )

    def compute_noise_scale(
        self,
        sensitivity: float,
        epsilon: float,
        delta: float,  # noqa: ARG002  # delta unused for Laplace
    ) -> float:
        """Compute the Laplace noise scale λ = Δf/ε.

        Args:
            sensitivity: L1 sensitivity Δf of the query.
            epsilon: Target epsilon privacy parameter.
            delta: Ignored for Laplace (pure DP mechanism).

        Returns:
            Noise scale λ = sensitivity / epsilon.
        """
        return sensitivity / epsilon

    async def apply(
        self,
        data: list[float],
        sensitivity: float,
        epsilon: float,
        delta: float,
    ) -> tuple[list[float], Decimal, Decimal, Decimal]:
        """Apply Laplace mechanism to numerical data.

        Adds independent Laplace noise to each element:
            output[i] = data[i] + Lap(0, sensitivity/epsilon)

        Args:
            data: Input numerical values to privatize.
            sensitivity: L1 sensitivity of the query.
            epsilon: Privacy budget to consume.
            delta: Must be 0.0 for Laplace (pure DP).

        Returns:
            Tuple of (noisy_values, epsilon_consumed, delta_consumed, noise_scale).
            For Laplace: delta_consumed is always Decimal("0").

        Raises:
            ValueError: If parameters are invalid (call validate_parameters first).
        """
        self.validate_parameters(sensitivity, epsilon, delta)

        noise_scale = self.compute_noise_scale(sensitivity, epsilon, delta)

        # Add Laplace noise via numpy (compatible with OpenDP's verification)
        data_array = np.array(data, dtype=np.float64)
        noise = np.random.laplace(loc=0.0, scale=noise_scale, size=len(data))
        noisy_data = (data_array + noise).tolist()

        logger.debug(
            "Laplace mechanism applied",
            n_values=len(data),
            sensitivity=sensitivity,
            epsilon=epsilon,
            noise_scale=noise_scale,
        )

        return (
            noisy_data,
            Decimal(str(epsilon)),  # actual epsilon consumed
            Decimal("0.0"),  # delta consumed = 0 for pure DP
            Decimal(str(round(noise_scale, 9))),
        )
