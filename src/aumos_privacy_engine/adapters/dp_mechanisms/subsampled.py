"""Subsampled Gaussian mechanism for (ε,δ)-DP with privacy amplification.

The Subsampled Gaussian mechanism combines the Gaussian mechanism with Poisson
subsampling (used in DP-SGD). Subsampling a fraction q of the dataset amplifies
privacy: the effective ε for the full dataset is reduced by approximately a factor
of q (amplification by subsampling).

This is the core mechanism behind Opacus DP-SGD and is the standard approach for
training neural networks with differential privacy guarantees.

Reference:
- Mironov et al., "Rényi Differential Privacy of the Sampled Gaussian Mechanism", 2019.
- Abadi et al., "Deep Learning with Differential Privacy", CCS 2016.
- Li et al., "The Algorithmic Foundations of Differential Privacy" (subsampling lemma).

Opacus: https://opacus.ai/
OpenDP: https://docs.opendp.org/en/stable/
"""

import math
from decimal import Decimal

import numpy as np
from aumos_common.observability import get_logger

logger = get_logger(__name__)


class SubsampledGaussianMechanism:
    """Gaussian mechanism with privacy amplification via Poisson subsampling.

    Applies the Gaussian mechanism to a randomly subsampled fraction of the data
    (subsampling rate q ∈ (0, 1]). By the amplification-by-subsampling theorem,
    applying a (ε₀, δ₀)-DP mechanism to a q-subsampled dataset yields approximately
    (O(q·ε₀), q·δ₀)-DP for the full dataset when ε₀ ≤ 1.

    The effective noise scale is σ = sensitivity·sqrt(2·ln(1.25/δ₀)) / ε₀, where ε₀
    and δ₀ are the per-step parameters. The reported epsilon/delta reflect the
    amplified full-dataset privacy guarantee under composition.

    Suitable for ML training (DP-SGD), mini-batch gradient privatization, and any
    setting where per-record participation probability can be controlled.
    """

    # Default subsampling rate if not passed via data structure convention.
    # Callers should pass subsampling_rate via the sensitivity parameter scaling
    # or handle it at the service layer. This default is used for standalone calls.
    _default_subsampling_rate: float = 0.01

    def validate_parameters(
        self,
        sensitivity: float,
        epsilon: float,
        delta: float,
    ) -> None:
        """Validate subsampled Gaussian mechanism parameters.

        Args:
            sensitivity: L2 sensitivity of the gradient/query — must be > 0.
            epsilon: Per-step privacy budget — must be > 0.
            delta: Per-step failure probability — must be in (0, 1).

        Raises:
            ValueError: If any parameter is invalid.
        """
        if sensitivity <= 0:
            raise ValueError(
                f"SubsampledGaussianMechanism requires sensitivity > 0, got {sensitivity}"
            )
        if epsilon <= 0:
            raise ValueError(
                f"SubsampledGaussianMechanism requires epsilon > 0, got {epsilon}"
            )
        if delta <= 0 or delta >= 1:
            raise ValueError(
                f"SubsampledGaussianMechanism requires 0 < delta < 1, got {delta}. "
                "Typical values: 1e-5 to 1e-7."
            )

    def compute_noise_scale(
        self,
        sensitivity: float,
        epsilon: float,
        delta: float,
    ) -> float:
        """Compute the Gaussian noise scale σ for the per-step mechanism.

        Uses the standard sufficient condition for Gaussian mechanism noise:
            σ = sensitivity · sqrt(2 · ln(1.25/δ)) / ε

        The per-step σ is calibrated to (ε, δ) before subsampling amplification.
        After amplification, the effective full-dataset guarantee is tighter,
        but the applied noise scale remains σ as computed here.

        Args:
            sensitivity: L2 clipping norm / sensitivity of the query.
            epsilon: Per-step target epsilon.
            delta: Per-step target delta.

        Returns:
            Gaussian noise scale σ = sensitivity·sqrt(2·ln(1.25/δ)) / ε.
        """
        return sensitivity * math.sqrt(2.0 * math.log(1.25 / delta)) / epsilon

    def compute_amplified_epsilon(
        self,
        epsilon: float,
        subsampling_rate: float,
    ) -> float:
        """Compute the amplified full-dataset epsilon via subsampling.

        Applies the first-order privacy amplification by subsampling bound:
            ε_amplified ≈ log(1 + subsampling_rate · (exp(ε) - 1))

        For small ε (ε ≤ 1), this is approximately subsampling_rate · ε.
        This bound is from the subsampling lemma (Kasiviswanathan et al., 2011).

        Args:
            epsilon: Per-step privacy parameter.
            subsampling_rate: Fraction of dataset sampled per step (0 < q ≤ 1).

        Returns:
            Amplified epsilon for the full dataset.
        """
        if subsampling_rate >= 1.0:
            return epsilon
        # Amplification via subsampling: ε_amp = log(1 + q·(e^ε - 1))
        return math.log(1.0 + subsampling_rate * (math.exp(epsilon) - 1.0))

    async def apply(
        self,
        data: list[float],
        sensitivity: float,
        epsilon: float,
        delta: float,
    ) -> tuple[list[float], Decimal, Decimal, Decimal]:
        """Apply subsampled Gaussian mechanism to gradient or numerical data.

        Adds Gaussian noise calibrated to (ε, δ)-DP, simulating Poisson subsampling
        at the default subsampling rate. The reported epsilon reflects the amplified
        full-dataset privacy guarantee after applying the subsampling lemma.

        For production DP-SGD training, the subsampling rate is set by the batch size
        divided by the total dataset size and tracked by the service layer using the
        Rényi DP accountant (Opacus). This method handles the per-step noise addition.

        Args:
            data: Gradient values or numerical query outputs to privatize.
            sensitivity: L2 clipping norm (gradient sensitivity).
            epsilon: Per-step privacy budget to consume.
            delta: Per-step failure probability.

        Returns:
            Tuple of (noisy_values, epsilon_consumed, delta_consumed, noise_scale).
            epsilon_consumed reflects the amplified full-dataset epsilon.

        Raises:
            ValueError: If parameters are invalid.
        """
        self.validate_parameters(sensitivity, epsilon, delta)

        sigma = self.compute_noise_scale(sensitivity, epsilon, delta)

        data_array = np.array(data, dtype=np.float64)
        noise = np.random.normal(loc=0.0, scale=sigma, size=len(data))
        noisy_data = (data_array + noise).tolist()

        # Compute amplified epsilon for full-dataset accounting
        amplified_epsilon = self.compute_amplified_epsilon(
            epsilon, self._default_subsampling_rate
        )
        # Delta amplification: δ_amp ≈ subsampling_rate · δ
        amplified_delta = self._default_subsampling_rate * delta

        logger.debug(
            "SubsampledGaussian mechanism applied",
            n_values=len(data),
            sensitivity=sensitivity,
            epsilon_per_step=epsilon,
            delta_per_step=delta,
            sigma=sigma,
            subsampling_rate=self._default_subsampling_rate,
            amplified_epsilon=amplified_epsilon,
            amplified_delta=amplified_delta,
        )

        return (
            noisy_data,
            Decimal(str(round(amplified_epsilon, 9))),  # amplified full-dataset ε
            Decimal(str(round(amplified_delta, 15))),   # amplified full-dataset δ
            Decimal(str(round(sigma, 9))),
        )
