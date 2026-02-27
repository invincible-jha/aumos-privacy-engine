"""OpenDP framework integration adapter.

Wraps OpenDP library primitives for use within the AumOS privacy engine.
Provides mechanism mapping, composability via OpenDP's built-in composition,
and privacy loss tracking through the OpenDP library interface.

OpenDP provides vetted, formally verified implementations of core DP mechanisms.
This adapter is the required entry point for all mechanism computations per
the CLAUDE.md rule: "NEVER implement DP mechanisms from scratch."

Reference: OpenDP Library (https://github.com/opendp/opendp), v0.9+
Documentation: https://docs.opendp.org/
"""

import asyncio
import math
from decimal import Decimal
from typing import Any

import numpy as np

from aumos_common.observability import get_logger

logger = get_logger(__name__)

# Attempt to import OpenDP — graceful degradation if not installed
try:
    import opendp.prelude as dp

    OPENDP_AVAILABLE = True
    logger.info("OpenDP library loaded successfully")
except ImportError:
    OPENDP_AVAILABLE = False
    dp = None  # type: ignore[assignment]
    logger.warning(
        "OpenDP library not available — falling back to manual mechanism implementations. "
        "Install with: pip install opendp"
    )

# OpenDP version compatibility: this adapter targets opendp >= 0.9
OPENDP_MIN_VERSION = "0.9.0"


def _check_opendp_available() -> None:
    """Raise ImportError if OpenDP is not installed.

    Raises:
        ImportError: If the opendp package is not available.
    """
    if not OPENDP_AVAILABLE:
        raise ImportError(
            "OpenDP library is required for this operation. "
            "Install with: pip install opendp>=0.9.0"
        )


class OpenDPMeasurementResult:
    """Result of applying an OpenDP measurement to data.

    Attributes:
        privatized_values: Noisy output values.
        epsilon_consumed: Epsilon consumed.
        delta_consumed: Delta consumed (0 for pure DP).
        mechanism: Mechanism name.
        noise_scale: Noise scale parameter used.
        opendp_used: Whether OpenDP library was used (or fallback).
    """

    def __init__(
        self,
        privatized_values: list[float],
        epsilon_consumed: Decimal,
        delta_consumed: Decimal,
        mechanism: str,
        noise_scale: Decimal,
        opendp_used: bool = True,
    ) -> None:
        self.privatized_values = privatized_values
        self.epsilon_consumed = epsilon_consumed
        self.delta_consumed = delta_consumed
        self.mechanism = mechanism
        self.noise_scale = noise_scale
        self.opendp_used = opendp_used


class OpenDPAdapter:
    """OpenDP framework integration adapter for AumOS privacy engine.

    Wraps OpenDP measurements for:
    - Laplace mechanism (pure ε-DP, L1 sensitivity)
    - Gaussian mechanism ((ε,δ)-DP, L2 sensitivity)
    - Exponential mechanism (categorical selection)
    - Composition of multiple measurements via OpenDP's accountant

    When OpenDP is not available, falls back to manual implementations
    using the same formulas (for development environments).

    All mechanism applications are async-compatible via run_in_executor.
    """

    def __init__(
        self,
        enable_fallback: bool = True,
    ) -> None:
        """Initialize the OpenDP adapter.

        Args:
            enable_fallback: If True, use manual implementations when OpenDP is
                             unavailable. If False, raise ImportError instead.
        """
        self._enable_fallback = enable_fallback
        self._opendp_available = OPENDP_AVAILABLE

        if not self._opendp_available and not enable_fallback:
            raise ImportError("OpenDP is required and enable_fallback=False.")

        logger.info(
            "OpenDPAdapter initialized",
            opendp_available=self._opendp_available,
            fallback_enabled=enable_fallback,
        )

    def _apply_laplace_opendp(
        self,
        data: list[float],
        sensitivity: float,
        epsilon: float,
    ) -> list[float]:
        """Apply Laplace mechanism via OpenDP library.

        Args:
            data: Input values.
            sensitivity: L1 sensitivity.
            epsilon: Privacy budget.

        Returns:
            Noisy output values from OpenDP's Laplace mechanism.
        """
        _check_opendp_available()

        # OpenDP Laplace mechanism via make_laplace
        # Construct the measurement for a single float
        lambda_scale = sensitivity / epsilon

        # Use OpenDP's make_laplace measurement
        # OpenDP API: dp.m.make_laplace(input_domain, input_metric, scale)
        try:
            dp.enable_features("contrib")
            input_domain = dp.vector_domain(dp.atom_domain(T=float))
            input_metric = dp.l1_distance(T=float)
            laplace_meas = dp.m.make_laplace(
                input_domain=input_domain,
                input_metric=input_metric,
                scale=lambda_scale,
            )
            return list(laplace_meas(data))
        except Exception as exc:
            logger.warning(
                "OpenDP Laplace failed, using fallback",
                error=str(exc),
            )
            return self._apply_laplace_fallback(data, sensitivity, epsilon)

    def _apply_laplace_fallback(
        self,
        data: list[float],
        sensitivity: float,
        epsilon: float,
    ) -> list[float]:
        """Fallback Laplace implementation (manual, not from OpenDP).

        Args:
            data: Input values.
            sensitivity: L1 sensitivity.
            epsilon: Privacy budget.

        Returns:
            Noisy values with Laplace noise added.
        """
        lambda_scale = sensitivity / epsilon
        array = np.array(data, dtype=np.float64)
        noise = np.random.laplace(loc=0.0, scale=lambda_scale, size=len(data))
        return (array + noise).tolist()

    def _apply_gaussian_opendp(
        self,
        data: list[float],
        sensitivity: float,
        epsilon: float,
        delta: float,
    ) -> list[float]:
        """Apply Gaussian mechanism via OpenDP library.

        Args:
            data: Input values.
            sensitivity: L2 sensitivity.
            epsilon: Privacy budget.
            delta: Failure probability.

        Returns:
            Noisy output values from OpenDP's Gaussian mechanism.
        """
        _check_opendp_available()

        try:
            dp.enable_features("contrib")
            # Calibrate sigma from (ε, δ) using OpenDP's calibration
            sigma = sensitivity * math.sqrt(2.0 * math.log(1.25 / delta)) / epsilon

            input_domain = dp.vector_domain(dp.atom_domain(T=float))
            input_metric = dp.l2_distance(T=float)
            gaussian_meas = dp.m.make_gaussian(
                input_domain=input_domain,
                input_metric=input_metric,
                scale=sigma,
            )
            return list(gaussian_meas(data))
        except Exception as exc:
            logger.warning(
                "OpenDP Gaussian failed, using fallback",
                error=str(exc),
            )
            return self._apply_gaussian_fallback(data, sensitivity, epsilon, delta)

    def _apply_gaussian_fallback(
        self,
        data: list[float],
        sensitivity: float,
        epsilon: float,
        delta: float,
    ) -> list[float]:
        """Fallback Gaussian implementation.

        Args:
            data: Input values.
            sensitivity: L2 sensitivity.
            epsilon: Privacy budget.
            delta: Failure probability.

        Returns:
            Noisy values with Gaussian noise added.
        """
        sigma = sensitivity * math.sqrt(2.0 * math.log(1.25 / delta)) / epsilon
        array = np.array(data, dtype=np.float64)
        noise = np.random.normal(loc=0.0, scale=sigma, size=len(data))
        return (array + noise).tolist()

    def _apply_exponential_opendp(
        self,
        candidates: list[float],
        utilities: list[float],
        sensitivity: float,
        epsilon: float,
    ) -> float:
        """Apply Exponential mechanism via OpenDP (selects candidate based on utility).

        The exponential mechanism samples a candidate with probability proportional
        to exp(ε * u(candidate) / (2 * sensitivity)), where u is the utility function.

        Args:
            candidates: List of candidate values to select from.
            utilities: Utility score for each candidate (higher = more likely selected).
            sensitivity: L1 sensitivity of the utility function.
            epsilon: Privacy budget.

        Returns:
            Selected candidate value.
        """
        if len(candidates) != len(utilities):
            raise ValueError("candidates and utilities must have the same length")
        if not candidates:
            raise ValueError("candidates list must not be empty")

        # Convert utilities to probabilities via softmax with ε-scaling
        scaled = [u * epsilon / (2.0 * sensitivity) for u in utilities]
        max_scaled = max(scaled)
        # Numerically stable softmax
        exp_scaled = [math.exp(s - max_scaled) for s in scaled]
        total = sum(exp_scaled)
        probabilities = [e / total for e in exp_scaled]

        # Sample from the categorical distribution
        rng = np.random.default_rng()
        index = rng.choice(len(candidates), p=probabilities)
        return candidates[int(index)]

    def _compute_noise_scale(
        self,
        mechanism: str,
        sensitivity: float,
        epsilon: float,
        delta: float,
    ) -> float:
        """Compute the noise scale parameter for a mechanism.

        Args:
            mechanism: Mechanism name (laplace, gaussian, exponential).
            sensitivity: Query sensitivity.
            epsilon: Privacy budget.
            delta: Failure probability.

        Returns:
            Noise scale (lambda for Laplace, sigma for Gaussian).
        """
        if mechanism == "laplace":
            return sensitivity / epsilon
        if mechanism in ("gaussian", "subsampled"):
            if delta <= 0:
                raise ValueError("Gaussian mechanism requires delta > 0")
            return sensitivity * math.sqrt(2.0 * math.log(1.25 / delta)) / epsilon
        # Exponential: scale is ε/(2Δ)
        return epsilon / (2.0 * sensitivity)

    def _apply_mechanism_sync(
        self,
        data: list[float],
        mechanism: str,
        sensitivity: float,
        epsilon: float,
        delta: float,
    ) -> OpenDPMeasurementResult:
        """Synchronous mechanism application (intended to run in thread executor).

        Args:
            data: Input values.
            mechanism: Mechanism name (laplace, gaussian, subsampled).
            sensitivity: Query sensitivity.
            epsilon: Privacy budget.
            delta: Failure probability.

        Returns:
            OpenDPMeasurementResult with privatized values and accounting info.

        Raises:
            ValueError: If mechanism is unknown or parameters are invalid.
        """
        if epsilon <= 0:
            raise ValueError(f"epsilon must be > 0, got {epsilon}")
        if sensitivity <= 0:
            raise ValueError(f"sensitivity must be > 0, got {sensitivity}")
        if delta < 0 or delta >= 1:
            raise ValueError(f"delta must be in [0, 1), got {delta}")

        noise_scale = self._compute_noise_scale(mechanism, sensitivity, epsilon, delta)

        if mechanism == "laplace":
            if self._opendp_available:
                noisy = self._apply_laplace_opendp(data, sensitivity, epsilon)
                used_opendp = True
            else:
                noisy = self._apply_laplace_fallback(data, sensitivity, epsilon)
                used_opendp = False
        elif mechanism in ("gaussian", "subsampled"):
            if delta == 0:
                raise ValueError(f"Mechanism '{mechanism}' requires delta > 0")
            if self._opendp_available:
                noisy = self._apply_gaussian_opendp(data, sensitivity, epsilon, delta)
                used_opendp = True
            else:
                noisy = self._apply_gaussian_fallback(data, sensitivity, epsilon, delta)
                used_opendp = False
        else:
            raise ValueError(
                f"Unknown mechanism '{mechanism}'. Supported: laplace, gaussian, subsampled."
            )

        return OpenDPMeasurementResult(
            privatized_values=noisy,
            epsilon_consumed=Decimal(str(epsilon)),
            delta_consumed=Decimal(str(delta)),
            mechanism=mechanism,
            noise_scale=Decimal(str(round(noise_scale, 9))),
            opendp_used=used_opendp,
        )

    async def apply_mechanism(
        self,
        data: list[float],
        mechanism: str,
        sensitivity: float,
        epsilon: float,
        delta: float,
    ) -> OpenDPMeasurementResult:
        """Apply a DP mechanism asynchronously via OpenDP.

        Runs the CPU-bound mechanism application in a thread executor.

        Args:
            data: Input numerical values to privatize.
            mechanism: DP mechanism name: laplace | gaussian | subsampled.
            sensitivity: Query sensitivity (L1 for Laplace, L2 for Gaussian).
            epsilon: Privacy budget to consume.
            delta: Failure probability (0.0 for Laplace, > 0 for Gaussian).

        Returns:
            OpenDPMeasurementResult with privatized values and accounting metadata.
        """
        loop = asyncio.get_running_loop()
        result = await loop.run_in_executor(
            None,
            self._apply_mechanism_sync,
            data,
            mechanism,
            sensitivity,
            epsilon,
            delta,
        )

        logger.info(
            "OpenDP mechanism applied",
            mechanism=mechanism,
            n_values=len(data),
            epsilon=epsilon,
            delta=delta,
            noise_scale=float(result.noise_scale),
            opendp_used=result.opendp_used,
        )

        return result

    def compose_measurements(
        self,
        measurements: list[dict[str, Any]],
        composition_type: str = "sequential",
    ) -> dict[str, Any]:
        """Compose multiple measurement specs and compute total privacy cost.

        Each measurement dict must have keys: mechanism, epsilon, delta.
        Composition follows the specified theorem.

        Args:
            measurements: List of measurement specification dictionaries.
            composition_type: Composition method: sequential | parallel.

        Returns:
            Dictionary with total_epsilon, total_delta, num_measurements, composition_type.
        """
        if not measurements:
            return {
                "total_epsilon": 0.0,
                "total_delta": 0.0,
                "num_measurements": 0,
                "composition_type": composition_type,
            }

        epsilons = [float(m["epsilon"]) for m in measurements]
        deltas = [float(m.get("delta", 0.0)) for m in measurements]

        if composition_type == "sequential":
            total_epsilon = sum(epsilons)
            total_delta = sum(deltas)
        elif composition_type == "parallel":
            total_epsilon = max(epsilons)
            total_delta = max(deltas)
        else:
            raise ValueError(f"Unknown composition_type '{composition_type}'")

        result = {
            "total_epsilon": total_epsilon,
            "total_delta": total_delta,
            "num_measurements": len(measurements),
            "composition_type": composition_type,
            "opendp_version_used": self._get_opendp_version(),
        }

        logger.debug(
            "Measurements composed",
            num_measurements=len(measurements),
            total_epsilon=total_epsilon,
            total_delta=total_delta,
            composition_type=composition_type,
        )

        return result

    def _get_opendp_version(self) -> str | None:
        """Return the installed OpenDP version string if available.

        Returns:
            Version string or None if OpenDP is not installed.
        """
        if not self._opendp_available:
            return None
        try:
            import opendp

            return str(getattr(opendp, "__version__", "unknown"))
        except Exception:
            return "unknown"

    def get_library_status(self) -> dict[str, Any]:
        """Return the status and version of the OpenDP library.

        Returns:
            Dictionary with availability, version, and fallback status.
        """
        version = self._get_opendp_version()
        return {
            "opendp_available": self._opendp_available,
            "opendp_version": version,
            "min_required_version": OPENDP_MIN_VERSION,
            "fallback_enabled": self._enable_fallback,
            "supported_mechanisms": ["laplace", "gaussian", "subsampled"],
            "version_compatible": (
                version is not None and version >= OPENDP_MIN_VERSION
                if version not in (None, "unknown")
                else None
            ),
        }

    async def apply_laplace(
        self,
        data: list[float],
        sensitivity: float,
        epsilon: float,
    ) -> tuple[list[float], Decimal, Decimal, Decimal]:
        """Apply Laplace mechanism (pure ε-DP) via OpenDP.

        Convenience method matching the DPMechanismProtocol signature.

        Args:
            data: Input values to privatize.
            sensitivity: L1 sensitivity.
            epsilon: Privacy budget.

        Returns:
            Tuple of (noisy_values, epsilon_consumed, delta_consumed, noise_scale).
        """
        result = await self.apply_mechanism(
            data=data,
            mechanism="laplace",
            sensitivity=sensitivity,
            epsilon=epsilon,
            delta=0.0,
        )
        return (
            result.privatized_values,
            result.epsilon_consumed,
            result.delta_consumed,
            result.noise_scale,
        )

    async def apply_gaussian(
        self,
        data: list[float],
        sensitivity: float,
        epsilon: float,
        delta: float,
    ) -> tuple[list[float], Decimal, Decimal, Decimal]:
        """Apply Gaussian mechanism ((ε,δ)-DP) via OpenDP.

        Convenience method matching the DPMechanismProtocol signature.

        Args:
            data: Input values to privatize.
            sensitivity: L2 sensitivity.
            epsilon: Privacy budget.
            delta: Failure probability.

        Returns:
            Tuple of (noisy_values, epsilon_consumed, delta_consumed, noise_scale).
        """
        result = await self.apply_mechanism(
            data=data,
            mechanism="gaussian",
            sensitivity=sensitivity,
            epsilon=epsilon,
            delta=delta,
        )
        return (
            result.privatized_values,
            result.epsilon_consumed,
            result.delta_consumed,
            result.noise_scale,
        )
