"""Privacy amplification via subsampling and shuffling.

Computes amplified epsilon values when DP mechanisms are applied to
subsampled data rather than the full dataset. Amplification allows tighter
privacy guarantees without increasing noise.

Amplification by Subsampling: Kasiviswanathan et al. (2008)
Amplification by Shuffling (PRISM): Erlingsson et al. (2019), Balle et al. (2019)
Fixed-size subsampling: Balle et al. (2020)

The key insight: running an ε-DP mechanism on a q-fraction subsample of
a dataset gives roughly O(qε)-DP on the full dataset when q << 1.
"""

import asyncio
import hashlib
import json
import math
from datetime import UTC, datetime
from typing import Any
from uuid import UUID, uuid4

from aumos_common.observability import get_logger

logger = get_logger(__name__)


class PrivacyAmplifier:
    """Privacy amplification computations for subsampling and shuffling.

    Provides amplified epsilon bounds for:
    - Poisson subsampling (random inclusion with probability q)
    - Fixed-size subsampling (sample exactly m records from N)
    - Shuffling amplification (PRISM model for federated learning)
    - Composition tracking of amplified mechanisms

    All epsilon computations are conservative upper bounds on the true amplified
    privacy loss. The caller should use the most applicable amplification model.
    """

    def amplify_epsilon_poisson(
        self,
        epsilon: float,
        sampling_rate: float,
    ) -> float:
        """Compute amplified epsilon for Poisson subsampling.

        For a q-fraction Poisson subsample, the amplified epsilon satisfies:
            ε_amplified = log(1 + q * (e^ε - 1))

        This is the tight amplification bound from Kasiviswanathan et al. (2008).
        For small ε, this approximates to q*ε.

        Args:
            epsilon: Original epsilon of the mechanism applied to the subsample.
            sampling_rate: Poisson sampling probability q ∈ (0, 1].

        Returns:
            Amplified epsilon for the full dataset.

        Raises:
            ValueError: If parameters are out of valid range.
        """
        if epsilon <= 0:
            raise ValueError(f"epsilon must be > 0, got {epsilon}")
        if not (0 < sampling_rate <= 1.0):
            raise ValueError(f"sampling_rate must be in (0, 1], got {sampling_rate}")

        if sampling_rate == 1.0:
            return epsilon

        # Tight amplification: log(1 + q*(exp(ε) - 1))
        amplified = math.log1p(sampling_rate * math.expm1(epsilon))

        logger.debug(
            "Poisson amplification computed",
            original_epsilon=epsilon,
            sampling_rate=sampling_rate,
            amplified_epsilon=amplified,
            amplification_ratio=round(amplified / epsilon, 4),
        )

        return amplified

    def amplify_epsilon_fixed_size(
        self,
        epsilon: float,
        sample_size: int,
        dataset_size: int,
    ) -> float:
        """Compute amplified epsilon for fixed-size (without replacement) subsampling.

        For fixed-size sampling of m from N records, the amplified epsilon is:
            ε_amplified = log(1 + (m/N) * (e^ε - 1))

        This uses the same tight bound as Poisson with effective rate q = m/N.
        The true bound for without-replacement sampling is slightly better, but
        the Poisson approximation is conservative and widely used.

        Args:
            epsilon: Original epsilon of the mechanism.
            sample_size: Number of records in the sample (m).
            dataset_size: Total number of records in the full dataset (N).

        Returns:
            Amplified epsilon.

        Raises:
            ValueError: If sample_size > dataset_size or sizes are invalid.
        """
        if sample_size <= 0 or dataset_size <= 0:
            raise ValueError(f"sample_size and dataset_size must be > 0")
        if sample_size > dataset_size:
            raise ValueError(
                f"sample_size ({sample_size}) must be ≤ dataset_size ({dataset_size})"
            )

        sampling_rate = sample_size / dataset_size
        return self.amplify_epsilon_poisson(epsilon, sampling_rate)

    def amplify_epsilon_shuffling(
        self,
        epsilon_local: float,
        num_users: int,
        delta: float,
    ) -> tuple[float, float]:
        """Compute amplified (ε, δ) for the shuffling (PRISM) model.

        In the shuffle model, n users each apply a local ε_L-DP mechanism
        and then shuffle their reports through a trusted shuffler before
        aggregation. The shuffling provides central-model-equivalent privacy:

        Balle et al. (2019) Theorem 3.1 (simplified):
            ε_central ≈ O(e^{ε_L/2} * sqrt(log(1/δ) / n))

        This is an approximation. The exact bound involves more complex analysis.

        Args:
            epsilon_local: Local epsilon (each user's local DP guarantee).
            num_users: Number of participating users n.
            delta: Target failure probability for the central model guarantee.

        Returns:
            Tuple of (central_epsilon, delta) representing the amplified guarantee.

        Raises:
            ValueError: If parameters are invalid.
        """
        if epsilon_local <= 0:
            raise ValueError(f"epsilon_local must be > 0, got {epsilon_local}")
        if num_users < 2:
            raise ValueError(f"Shuffling requires at least 2 users, got {num_users}")
        if not (0 < delta < 1):
            raise ValueError(f"delta must be in (0, 1), got {delta}")

        # Balle et al. (2019) central epsilon bound
        # ε_central ≈ (e^{ε_L} - 1) * sqrt(2 * log(1/δ) / (e^{ε_L} + n - 1)) + ε_L/n
        exp_eps = math.exp(epsilon_local)
        log_inv_delta = math.log(1.0 / delta)

        # Main amplification term
        numerator = 2.0 * log_inv_delta
        denominator = exp_eps + num_users - 1
        amplification = (exp_eps - 1.0) * math.sqrt(numerator / denominator)

        # Correction term for finite n
        correction = epsilon_local / num_users

        central_epsilon = amplification + correction

        logger.debug(
            "Shuffle amplification computed",
            epsilon_local=epsilon_local,
            num_users=num_users,
            delta=delta,
            central_epsilon=central_epsilon,
        )

        return (central_epsilon, delta)

    def compute_amplified_composition(
        self,
        mechanisms: list[dict[str, float]],
        composition: str = "sequential",
    ) -> dict[str, float]:
        """Compute composed epsilon after amplification for a sequence of mechanisms.

        Each mechanism in the list specifies its amplified epsilon and delta.
        Sequential composition sums epsilons; parallel takes the max.

        Args:
            mechanisms: List of dicts with keys 'epsilon' and 'delta', each representing
                        an already-amplified mechanism's privacy parameters.
            composition: 'sequential' or 'parallel'.

        Returns:
            Dictionary with total_epsilon, total_delta, and composition_type.
        """
        if not mechanisms:
            return {"total_epsilon": 0.0, "total_delta": 0.0, "composition_type": composition}

        epsilons = [m["epsilon"] for m in mechanisms]
        deltas = [m["delta"] for m in mechanisms]

        if composition == "sequential":
            total_epsilon = sum(epsilons)
            total_delta = sum(deltas)
        elif composition == "parallel":
            total_epsilon = max(epsilons)
            total_delta = max(deltas)
        else:
            raise ValueError(f"Unknown composition type '{composition}'. Use 'sequential' or 'parallel'.")

        return {
            "total_epsilon": total_epsilon,
            "total_delta": total_delta,
            "num_mechanisms": len(mechanisms),
            "composition_type": composition,
        }

    def optimal_batch_size_for_epsilon(
        self,
        target_epsilon: float,
        mechanism_epsilon: float,
        dataset_size: int,
        num_epochs: int = 1,
        delta: float = 1e-5,
    ) -> dict[str, Any]:
        """Recommend optimal batch size to achieve a target epsilon via amplification.

        Given a target central epsilon and a per-step mechanism epsilon, finds
        the batch size (and thus sampling rate) such that after `num_epochs` steps
        of Poisson-subsampled mechanism applications, the total privacy cost
        (via sequential composition of amplified steps) is within target_epsilon.

        Args:
            target_epsilon: Desired total epsilon budget.
            mechanism_epsilon: Per-step mechanism epsilon (before amplification).
            dataset_size: Total number of records N.
            num_epochs: Number of full passes over the dataset.
            delta: Target delta for the final guarantee.

        Returns:
            Dictionary with recommended_batch_size, sampling_rate,
            steps_per_epoch, amplified_per_step_epsilon, and total_epsilon_estimate.

        Raises:
            ValueError: If no valid batch size achieves the target.
        """
        if target_epsilon <= 0:
            raise ValueError(f"target_epsilon must be > 0, got {target_epsilon}")
        if mechanism_epsilon <= 0:
            raise ValueError(f"mechanism_epsilon must be > 0, got {mechanism_epsilon}")
        if dataset_size <= 0:
            raise ValueError(f"dataset_size must be > 0, got {dataset_size}")

        best_result: dict[str, Any] | None = None

        # Search batch sizes from 1 to dataset_size/2
        candidate_sizes = [
            int(dataset_size * q)
            for q in [0.001, 0.002, 0.005, 0.01, 0.02, 0.05, 0.1, 0.2, 0.5]
            if int(dataset_size * q) >= 1
        ]
        candidate_sizes = sorted(set(candidate_sizes))

        for batch_size in candidate_sizes:
            sampling_rate = batch_size / dataset_size
            steps_per_epoch = max(1, dataset_size // batch_size)
            total_steps = steps_per_epoch * num_epochs

            # Amplify per-step epsilon
            amplified_per_step = self.amplify_epsilon_poisson(mechanism_epsilon, sampling_rate)

            # Sequential composition over all steps
            total_epsilon = amplified_per_step * total_steps

            if total_epsilon <= target_epsilon:
                best_result = {
                    "recommended_batch_size": batch_size,
                    "sampling_rate": round(sampling_rate, 6),
                    "steps_per_epoch": steps_per_epoch,
                    "total_steps": total_steps,
                    "amplified_per_step_epsilon": round(amplified_per_step, 8),
                    "total_epsilon_estimate": round(total_epsilon, 6),
                    "target_epsilon": target_epsilon,
                    "achieves_target": True,
                }
                # Return first batch size that achieves target (smallest budget use with reasonable size)
                break

        if best_result is None:
            # Return the best we can do (largest batch size to minimize privacy cost)
            batch_size = candidate_sizes[-1] if candidate_sizes else 1
            sampling_rate = batch_size / dataset_size
            steps_per_epoch = max(1, dataset_size // batch_size)
            total_steps = steps_per_epoch * num_epochs
            amplified_per_step = self.amplify_epsilon_poisson(mechanism_epsilon, sampling_rate)
            total_epsilon = amplified_per_step * total_steps

            best_result = {
                "recommended_batch_size": batch_size,
                "sampling_rate": round(sampling_rate, 6),
                "steps_per_epoch": steps_per_epoch,
                "total_steps": total_steps,
                "amplified_per_step_epsilon": round(amplified_per_step, 8),
                "total_epsilon_estimate": round(total_epsilon, 6),
                "target_epsilon": target_epsilon,
                "achieves_target": False,
                "warning": (
                    f"No batch size achieves target_epsilon={target_epsilon}. "
                    f"Best achievable: {round(total_epsilon, 6)}. "
                    "Consider increasing mechanism noise or reducing num_epochs."
                ),
            }

        logger.info(
            "Optimal batch size recommendation computed",
            target_epsilon=target_epsilon,
            recommended_batch_size=best_result["recommended_batch_size"],
            total_epsilon=best_result["total_epsilon_estimate"],
            achieves_target=best_result["achieves_target"],
        )

        return best_result

    def generate_amplification_certificate(
        self,
        mechanism: str,
        original_epsilon: float,
        amplification_type: str,
        amplification_params: dict[str, Any],
        amplified_epsilon: float,
        delta: float,
        tenant_id: UUID | None = None,
    ) -> dict[str, Any]:
        """Generate a cryptographically-identified amplification proof certificate.

        Args:
            mechanism: DP mechanism name (gaussian, laplace, etc.).
            original_epsilon: Original epsilon before amplification.
            amplification_type: Type of amplification ('poisson', 'fixed_size', 'shuffle').
            amplification_params: Parameters used in the amplification (e.g., sampling_rate).
            amplified_epsilon: Resulting amplified epsilon.
            delta: Delta for the amplified guarantee.
            tenant_id: Optional tenant UUID to associate with the certificate.

        Returns:
            Certificate dictionary with verification_hash.
        """
        payload: dict[str, Any] = {
            "certificate_id": str(uuid4()),
            "issued_at": datetime.now(UTC).isoformat(),
            "tenant_id": str(tenant_id) if tenant_id else None,
            "mechanism": mechanism,
            "original_epsilon": original_epsilon,
            "amplification_type": amplification_type,
            "amplification_params": amplification_params,
            "amplified_epsilon": amplified_epsilon,
            "delta": delta,
            "amplification_ratio": (
                round(amplified_epsilon / original_epsilon, 6) if original_epsilon > 0 else None
            ),
            "certificate_version": "1.0",
            "theorem_reference": self._get_theorem_reference(amplification_type),
        }

        hash_content = json.dumps(
            {
                "mechanism": mechanism,
                "original_epsilon": original_epsilon,
                "amplification_type": amplification_type,
                "amplification_params": amplification_params,
                "amplified_epsilon": amplified_epsilon,
                "delta": delta,
            },
            sort_keys=True,
        )
        payload["verification_hash"] = hashlib.sha256(hash_content.encode()).hexdigest()

        logger.info(
            "Amplification certificate generated",
            mechanism=mechanism,
            amplification_type=amplification_type,
            amplified_epsilon=amplified_epsilon,
            certificate_id=payload["certificate_id"],
        )

        return payload

    def _get_theorem_reference(self, amplification_type: str) -> str:
        """Return the theorem reference for the given amplification type.

        Args:
            amplification_type: One of 'poisson', 'fixed_size', 'shuffle'.

        Returns:
            Citation string for the applicable theorem.
        """
        references = {
            "poisson": (
                "Kasiviswanathan et al. (2008), Theorem 9. "
                "ε_amplified = log(1 + q*(exp(ε) - 1)) for Poisson subsampling rate q."
            ),
            "fixed_size": (
                "Balle et al. (2020). Fixed-size subsampling amplification. "
                "ε_amplified = log(1 + (m/N)*(exp(ε) - 1)) for m-of-N sampling."
            ),
            "shuffle": (
                "Erlingsson et al. (2019), Balle et al. (2019). "
                "Amplification by shuffling in the shuffle model of DP."
            ),
        }
        return references.get(amplification_type, "Unknown amplification theorem.")

    async def async_amplify_poisson(
        self,
        epsilon: float,
        sampling_rate: float,
    ) -> float:
        """Async wrapper for Poisson amplification (runs synchronously, exposed async for uniformity).

        Args:
            epsilon: Original epsilon.
            sampling_rate: Sampling probability.

        Returns:
            Amplified epsilon.
        """
        loop = asyncio.get_running_loop()
        return await loop.run_in_executor(
            None,
            self.amplify_epsilon_poisson,
            epsilon,
            sampling_rate,
        )
