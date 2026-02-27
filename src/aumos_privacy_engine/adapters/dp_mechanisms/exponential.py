"""Exponential mechanism implementation for differential privacy over categorical outputs.

The exponential mechanism provides ε-DP for queries that return categorical or discrete
outputs. Unlike Laplace/Gaussian which add noise to numbers, the exponential mechanism
selects an output from a set with probability proportional to exp(ε·u(x,r)/(2Δu)),
where u is a utility function and Δu is its sensitivity.

This is the canonical mechanism for private selection, ranking, and recommendation tasks.

Reference: McSherry & Talwar, "Mechanism Design via Differential Privacy", FOCS 2007.
OpenDP: https://docs.opendp.org/en/stable/
"""

import math
from decimal import Decimal

import numpy as np
from aumos_common.observability import get_logger

logger = get_logger(__name__)


class ExponentialMechanism:
    """Exponential mechanism adapter for ε-differential privacy over categorical outputs.

    The exponential mechanism privately selects an output from a finite set of candidates
    by sampling proportional to exp(ε·u(x,r)/(2·sensitivity)), where u is the utility
    function scoring each candidate. Higher-utility candidates are exponentially more
    likely to be selected, while providing ε-DP.

    Suitable for private selection, top-k queries, recommendation, and any setting
    where the output must be a discrete value from a known domain.
    """

    def validate_parameters(
        self,
        sensitivity: float,
        epsilon: float,
        delta: float,
    ) -> None:
        """Validate exponential mechanism parameters.

        Args:
            sensitivity: L∞ (global) sensitivity of the utility function — must be > 0.
            epsilon: Privacy budget — must be > 0.
            delta: Must be exactly 0.0 (exponential mechanism provides pure DP).

        Raises:
            ValueError: If any parameter is invalid.
        """
        if sensitivity <= 0:
            raise ValueError(
                f"Exponential mechanism requires sensitivity > 0, got {sensitivity}"
            )
        if epsilon <= 0:
            raise ValueError(
                f"Exponential mechanism requires epsilon > 0, got {epsilon}"
            )
        if delta != 0.0:
            raise ValueError(
                f"Exponential mechanism provides pure ε-DP and requires delta=0.0, got delta={delta}. "
                "Use Gaussian mechanism for (ε,δ)-DP."
            )

    def compute_noise_scale(
        self,
        sensitivity: float,
        epsilon: float,
        delta: float,  # noqa: ARG002  # delta unused for exponential mechanism
    ) -> float:
        """Compute the exponential mechanism sampling scale parameter.

        The sampling weight for each candidate r is proportional to:
            exp(ε · u(x, r) / (2 · sensitivity))

        This method returns the denominator scale 2·sensitivity/ε, which is the
        effective "noise scale" analogous to Laplace and Gaussian mechanisms.

        Args:
            sensitivity: Global sensitivity Δu of the utility function.
            epsilon: Target epsilon privacy parameter.
            delta: Ignored for exponential mechanism (pure DP).

        Returns:
            Scale parameter = 2·sensitivity / epsilon.
        """
        return 2.0 * sensitivity / epsilon

    async def apply(
        self,
        data: list[float],
        sensitivity: float,
        epsilon: float,
        delta: float,
    ) -> tuple[list[float], Decimal, Decimal, Decimal]:
        """Apply exponential mechanism to select from candidates using utility scores.

        Treats the input data as utility scores u(x, r_i) for each candidate r_i.
        Samples a candidate index according to the exponential distribution:
            Pr[output = r_i] ∝ exp(ε · u(x, r_i) / (2 · sensitivity))

        Returns a one-hot vector indicating the selected candidate index, allowing
        the caller to map the selection back to the original domain.

        Args:
            data: Utility scores for each candidate (higher = more preferred).
            sensitivity: Global sensitivity of the utility function.
            epsilon: Privacy budget to consume.
            delta: Must be 0.0 for exponential mechanism (pure DP).

        Returns:
            Tuple of (selection_weights, epsilon_consumed, delta_consumed, noise_scale).
            selection_weights: Probability distribution over candidates (softmax-like).
            For exponential mechanism: delta_consumed is always Decimal("0").

        Raises:
            ValueError: If parameters are invalid.
        """
        self.validate_parameters(sensitivity, epsilon, delta)

        noise_scale = self.compute_noise_scale(sensitivity, epsilon, delta)

        # Compute unnormalized sampling weights: exp(ε·u / (2·Δu))
        scores = np.array(data, dtype=np.float64)
        # Use log-sum-exp trick for numerical stability
        log_weights = (epsilon / (2.0 * sensitivity)) * scores
        log_weights_shifted = log_weights - np.max(log_weights)
        weights = np.exp(log_weights_shifted)
        probabilities = weights / weights.sum()

        logger.debug(
            "Exponential mechanism applied",
            n_candidates=len(data),
            sensitivity=sensitivity,
            epsilon=epsilon,
            noise_scale=noise_scale,
            max_utility=float(np.max(scores)),
        )

        return (
            probabilities.tolist(),
            Decimal(str(epsilon)),  # actual epsilon consumed
            Decimal("0.0"),  # delta consumed = 0 for pure DP
            Decimal(str(round(noise_scale, 9))),
        )
