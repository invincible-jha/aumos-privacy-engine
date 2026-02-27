"""Sensitivity analyzer for automatic clipping bound recommendation.

Computes global and local sensitivity for numerical queries, recommends
automatic clipping bounds, and profiles per-column sensitivities in tabular data.

Sensitivity definitions:
  Global sensitivity Δf = max_{D,D'} |f(D) - f(D')| over all adjacent dataset pairs.
  Local sensitivity LS_f(D) = max_{D'} |f(D) - f(D')| for a fixed database D.
  Smooth sensitivity: smoothed local sensitivity that can be used with calibrated noise.

References:
  Nissim et al. (2007), "Smooth Sensitivity and Sampling in Private Data Analysis"
  Dwork & Roth (2014), Chapter 3 — Sensitivity
"""

import asyncio
import math
from typing import Any

import numpy as np
from scipy import stats

from aumos_common.observability import get_logger

logger = get_logger(__name__)

# Query type sensitivity constants (L1 sensitivity for standard aggregate queries)
QUERY_SENSITIVITY_MAP: dict[str, dict[str, float]] = {
    "count": {"l1": 1.0, "l2": 1.0},
    "sum": {"l1": None, "l2": None},   # None = data-dependent
    "mean": {"l1": None, "l2": None},  # None = data-dependent
    "median": {"l1": None, "l2": None},
    "variance": {"l1": None, "l2": None},
    "max": {"l1": None, "l2": None},
    "min": {"l1": None, "l2": None},
    "histogram": {"l1": 2.0, "l2": math.sqrt(2)},  # Per bin, L1=2, L2=sqrt(2)
    "gradient": {"l1": None, "l2": None},  # Per-sample gradient for DP-SGD
}

# Default clipping percentile when no explicit bounds are provided
DEFAULT_CLIP_PERCENTILE: float = 99.0


class ColumnSensitivityProfile:
    """Sensitivity profile for a single column in a tabular dataset.

    Attributes:
        column_name: Name of the analyzed column.
        data_min: Minimum observed value.
        data_max: Maximum observed value.
        data_std: Standard deviation of observed values.
        data_mean: Mean of observed values.
        global_sensitivity_l1: Estimated L1 global sensitivity.
        global_sensitivity_l2: Estimated L2 global sensitivity.
        recommended_clip_bound: Recommended clipping threshold.
        outlier_count: Number of values beyond the recommended clip bound.
        outlier_fraction: Fraction of values beyond the clip bound.
    """

    def __init__(
        self,
        column_name: str,
        data: list[float],
        clip_percentile: float = DEFAULT_CLIP_PERCENTILE,
    ) -> None:
        """Compute sensitivity profile from column data.

        Args:
            column_name: Name of this column.
            data: Numerical values in the column.
            clip_percentile: Percentile used to determine the clip bound.
        """
        if not data:
            raise ValueError(f"Column '{column_name}' has no data")

        array = np.array(data, dtype=np.float64)
        self.column_name = column_name
        self.data_min = float(np.min(array))
        self.data_max = float(np.max(array))
        self.data_std = float(np.std(array))
        self.data_mean = float(np.mean(array))

        # Global sensitivity for sum query: max observed value range
        data_range = self.data_max - self.data_min
        self.global_sensitivity_l1 = float(data_range)
        self.global_sensitivity_l2 = float(data_range)  # Single-column L2 = L1

        # Recommended clip bound at the configured percentile
        self.recommended_clip_bound = float(np.percentile(np.abs(array), clip_percentile))

        outlier_mask = np.abs(array) > self.recommended_clip_bound
        self.outlier_count = int(np.sum(outlier_mask))
        self.outlier_fraction = float(self.outlier_count / len(array))

    def to_dict(self) -> dict[str, Any]:
        """Serialize the profile to a JSON-compatible dictionary.

        Returns:
            Dictionary with all profile fields.
        """
        return {
            "column_name": self.column_name,
            "data_min": self.data_min,
            "data_max": self.data_max,
            "data_std": self.data_std,
            "data_mean": self.data_mean,
            "global_sensitivity_l1": self.global_sensitivity_l1,
            "global_sensitivity_l2": self.global_sensitivity_l2,
            "recommended_clip_bound": self.recommended_clip_bound,
            "outlier_count": self.outlier_count,
            "outlier_fraction": round(self.outlier_fraction, 6),
        }


class SensitivityAnalyzer:
    """Automatic sensitivity estimation and clipping bound recommendation.

    Analyzes datasets to estimate query sensitivity and recommend clipping
    thresholds that minimize privacy-utility tradeoff. Provides:
    - Global sensitivity for standard aggregate queries
    - Local sensitivity for a specific database instance
    - Smooth sensitivity approximation using beta-smooth sensitivity
    - Per-column sensitivity profiles for tabular datasets

    All compute-intensive analysis runs in a thread executor to avoid
    blocking the async event loop.
    """

    def __init__(
        self,
        clip_percentile: float = DEFAULT_CLIP_PERCENTILE,
        smooth_sensitivity_beta: float = 0.1,
    ) -> None:
        """Initialize the sensitivity analyzer.

        Args:
            clip_percentile: Percentile (0-100) at which to set the clip bound.
                             Higher values preserve more data at the cost of higher sensitivity.
            smooth_sensitivity_beta: Beta parameter for smooth sensitivity (0 < β < ε).
                                     Smaller β → tighter smoothing but more noise.
        """
        if not (0 < clip_percentile <= 100):
            raise ValueError(f"clip_percentile must be in (0, 100], got {clip_percentile}")
        if smooth_sensitivity_beta <= 0:
            raise ValueError(f"smooth_sensitivity_beta must be > 0, got {smooth_sensitivity_beta}")

        self._clip_percentile = clip_percentile
        self._smooth_beta = smooth_sensitivity_beta

    def estimate_global_sensitivity(
        self,
        query_type: str,
        data_bound: float | None = None,
        num_dimensions: int = 1,
    ) -> dict[str, float]:
        """Estimate global sensitivity for a named query type.

        For queries with data-dependent sensitivity (sum, mean, etc.),
        a data_bound must be provided representing the maximum value range.

        Args:
            query_type: Query type key (count, sum, mean, median, histogram, gradient).
            data_bound: Maximum absolute value of any individual record (required for
                        sum/mean/variance queries). Ignored for count/histogram.
            num_dimensions: Number of dimensions (for gradient or multi-column queries).

        Returns:
            Dictionary with l1_sensitivity and l2_sensitivity.

        Raises:
            ValueError: If query_type requires data_bound but none is provided.
        """
        known = QUERY_SENSITIVITY_MAP.get(query_type)
        if known is None:
            raise ValueError(
                f"Unknown query type '{query_type}'. Known types: {list(QUERY_SENSITIVITY_MAP.keys())}"
            )

        l1_sens = known["l1"]
        l2_sens = known["l2"]

        if l1_sens is None:
            # Data-dependent sensitivity
            if data_bound is None:
                raise ValueError(
                    f"Query type '{query_type}' requires data_bound (max absolute value per record)"
                )
            if query_type == "sum":
                l1_sens = float(data_bound)
                l2_sens = float(data_bound)
            elif query_type == "mean":
                # Mean of N records: sensitivity = data_bound / N is ideal, but we don't know N
                # Conservative bound: sensitivity = 2 * data_bound (add/remove one record)
                l1_sens = 2.0 * float(data_bound)
                l2_sens = 2.0 * float(data_bound)
            elif query_type == "variance":
                # Sensitivity of empirical variance bounded by data_bound²
                l1_sens = float(data_bound) ** 2
                l2_sens = float(data_bound) ** 2
            elif query_type in ("median", "max", "min"):
                l1_sens = float(data_bound)
                l2_sens = float(data_bound)
            elif query_type == "gradient":
                # L2 sensitivity of gradient: bounded by clipping norm C
                l2_sens = float(data_bound)
                l1_sens = float(data_bound) * math.sqrt(num_dimensions)

        # Multi-dimensional scaling
        if num_dimensions > 1 and query_type not in ("gradient", "histogram"):
            l1_sens = float(l1_sens) * num_dimensions
            l2_sens = float(l2_sens) * math.sqrt(num_dimensions)

        result = {
            "l1_sensitivity": float(l1_sens),
            "l2_sensitivity": float(l2_sens),
        }

        logger.debug(
            "Global sensitivity estimated",
            query_type=query_type,
            data_bound=data_bound,
            num_dimensions=num_dimensions,
            l1=result["l1_sensitivity"],
            l2=result["l2_sensitivity"],
        )

        return result

    def _compute_local_sensitivity_sync(
        self,
        data: list[float],
        query_type: str,
        clip_bound: float | None = None,
    ) -> float:
        """Synchronous computation of local sensitivity (to run in executor).

        Local sensitivity LS_f(D) = max_{D'} |f(D) - f(D')|, where D' differs
        from D by one record. This is computed empirically by simulating adding
        or removing one record at a time.

        Args:
            data: Observed dataset values.
            query_type: Query type (sum, mean, count, etc.).
            clip_bound: If provided, values are clipped to [-clip_bound, clip_bound]
                        before computing sensitivity.

        Returns:
            Empirical local sensitivity estimate.
        """
        array = np.array(data, dtype=np.float64)
        if clip_bound is not None:
            array = np.clip(array, -clip_bound, clip_bound)

        n = len(array)
        if n == 0:
            return 0.0

        if query_type == "count":
            return 1.0

        if query_type == "sum":
            # LS = max value (adding the max or removing 0 creates the most change)
            return float(np.max(np.abs(array)))

        if query_type == "mean":
            # LS of mean: remove the most extreme record
            current_mean = float(np.mean(array))
            max_deviation = 0.0
            for i in range(n):
                without_i = np.delete(array, i)
                mean_without_i = float(np.mean(without_i)) if len(without_i) > 0 else 0.0
                deviation = abs(current_mean - mean_without_i)
                if deviation > max_deviation:
                    max_deviation = deviation
            return max_deviation

        if query_type == "variance":
            current_var = float(np.var(array))
            max_deviation = 0.0
            for i in range(min(n, 50)):  # sample for efficiency on large datasets
                without_i = np.delete(array, i)
                var_without_i = float(np.var(without_i)) if len(without_i) > 0 else 0.0
                deviation = abs(current_var - var_without_i)
                if deviation > max_deviation:
                    max_deviation = deviation
            return max_deviation

        # Default fallback: L2 norm of the data range
        return float(np.max(array) - np.min(array))

    async def compute_local_sensitivity(
        self,
        data: list[float],
        query_type: str,
        clip_bound: float | None = None,
    ) -> float:
        """Compute empirical local sensitivity for a given dataset.

        Runs in a thread executor to avoid blocking the event loop.

        Args:
            data: Observed dataset values.
            query_type: Query type (sum, mean, count, variance).
            clip_bound: Optional clipping bound to apply before analysis.

        Returns:
            Empirical local sensitivity estimate.
        """
        loop = asyncio.get_running_loop()
        result = await loop.run_in_executor(
            None,
            self._compute_local_sensitivity_sync,
            data,
            query_type,
            clip_bound,
        )
        logger.debug(
            "Local sensitivity computed",
            query_type=query_type,
            n_records=len(data),
            local_sensitivity=result,
        )
        return result

    def _smooth_sensitivity_sync(
        self,
        data: list[float],
        query_type: str,
        beta: float | None = None,
    ) -> float:
        """Synchronous smooth sensitivity approximation.

        Computes the beta-smooth sensitivity:
            S*_f(D) = max_{D'} LS_f(D') * e^{-β * d(D, D')}

        where d(D, D') is the Hamming distance between D and D'.
        This is approximated by evaluating local sensitivity for
        neighborhoods of increasing radius.

        Args:
            data: Observed dataset values.
            query_type: Query type.
            beta: Smoothing parameter. Uses instance default if None.

        Returns:
            Approximate beta-smooth sensitivity.
        """
        effective_beta = beta if beta is not None else self._smooth_beta
        array = np.array(data, dtype=np.float64)
        n = len(array)

        # Compute LS at radius 0 (current dataset)
        ls_at_0 = self._compute_local_sensitivity_sync(data, query_type)

        # Estimate LS at radius k by perturbing the dataset
        smooth_sens = ls_at_0
        for radius in range(1, min(n, 10)):
            # Remove the `radius` most extreme values as a proxy for distance-k neighbor
            sorted_indices = np.argsort(np.abs(array))[::-1]
            reduced = np.delete(array, sorted_indices[:radius])
            ls_k = self._compute_local_sensitivity_sync(reduced.tolist(), query_type)
            candidate = ls_k * math.exp(-effective_beta * radius)
            smooth_sens = max(smooth_sens, candidate)

        return smooth_sens

    async def compute_smooth_sensitivity(
        self,
        data: list[float],
        query_type: str,
        beta: float | None = None,
    ) -> float:
        """Compute approximate beta-smooth sensitivity for a dataset.

        Args:
            data: Observed dataset values.
            query_type: Query type.
            beta: Smoothing parameter (default: instance smooth_sensitivity_beta).

        Returns:
            Approximate beta-smooth sensitivity.
        """
        loop = asyncio.get_running_loop()
        result = await loop.run_in_executor(
            None,
            self._smooth_sensitivity_sync,
            data,
            query_type,
            beta,
        )
        logger.debug(
            "Smooth sensitivity computed",
            query_type=query_type,
            n_records=len(data),
            smooth_sensitivity=result,
        )
        return result

    def recommend_clip_bound(
        self,
        data: list[float],
        percentile: float | None = None,
    ) -> dict[str, float]:
        """Recommend a data-driven clipping bound.

        Uses percentile-based analysis and outlier detection (IQR method)
        to recommend a clipping bound that balances utility and sensitivity.

        Args:
            data: Observed values to analyze.
            percentile: Percentile for clip bound (0-100). Defaults to instance setting.

        Returns:
            Dictionary with recommended_clip_bound, percentile_bound, iqr_bound,
            and expected_utility_loss (fraction of values that will be clipped).
        """
        if not data:
            raise ValueError("Cannot recommend clip bound for empty dataset")

        effective_pct = percentile if percentile is not None else self._clip_percentile
        array = np.abs(np.array(data, dtype=np.float64))

        # Percentile-based bound
        pct_bound = float(np.percentile(array, effective_pct))

        # IQR-based bound (Tukey's method: Q3 + 1.5 * IQR)
        q1 = float(np.percentile(array, 25))
        q3 = float(np.percentile(array, 75))
        iqr = q3 - q1
        iqr_bound = q3 + 1.5 * iqr

        # Recommended bound: more conservative of the two
        recommended = min(pct_bound, iqr_bound) if iqr_bound > 0 else pct_bound

        # Utility loss: fraction of values that exceed the clip bound
        clipped_fraction = float(np.mean(np.abs(array) > recommended))

        result = {
            "recommended_clip_bound": float(recommended),
            "percentile_bound": float(pct_bound),
            "iqr_bound": float(iqr_bound),
            "expected_clip_fraction": round(clipped_fraction, 4),
        }

        logger.debug(
            "Clip bound recommendation computed",
            n_values=len(data),
            recommended=result["recommended_clip_bound"],
            clip_fraction=result["expected_clip_fraction"],
        )

        return result

    async def profile_tabular_columns(
        self,
        column_data: dict[str, list[float]],
        clip_percentile: float | None = None,
    ) -> dict[str, Any]:
        """Generate per-column sensitivity profiles for a tabular dataset.

        Args:
            column_data: Dictionary mapping column names to their numeric values.
            clip_percentile: Clip percentile override (uses instance default if None).

        Returns:
            Dictionary with per-column profiles, dataset-level aggregate sensitivity,
            and recommended clipping bounds.
        """
        effective_pct = clip_percentile if clip_percentile is not None else self._clip_percentile

        def _profile_all_columns() -> dict[str, Any]:
            profiles = {}
            for col_name, values in column_data.items():
                if not values:
                    logger.warning("Empty column skipped in sensitivity profile", column=col_name)
                    continue
                try:
                    profile = ColumnSensitivityProfile(col_name, values, effective_pct)
                    profiles[col_name] = profile.to_dict()
                except Exception as exc:
                    logger.error(
                        "Failed to profile column",
                        column=col_name,
                        error=str(exc),
                    )

            # Dataset-level aggregate: max sensitivity across columns (L1 for sequential queries)
            if profiles:
                all_l1 = [p["global_sensitivity_l1"] for p in profiles.values()]
                all_l2 = [p["global_sensitivity_l2"] for p in profiles.values()]
                max_clip = max((p["recommended_clip_bound"] for p in profiles.values()), default=1.0)
            else:
                all_l1 = [0.0]
                all_l2 = [0.0]
                max_clip = 1.0

            return {
                "columns": profiles,
                "num_columns": len(profiles),
                "aggregate_l1_sensitivity": float(max(all_l1)),
                "aggregate_l2_sensitivity": float(max(all_l2)),
                "recommended_global_clip_bound": float(max_clip),
                "clip_percentile_used": effective_pct,
            }

        loop = asyncio.get_running_loop()
        result = await loop.run_in_executor(None, _profile_all_columns)

        logger.info(
            "Tabular sensitivity profiling complete",
            num_columns=result["num_columns"],
            aggregate_l1=result["aggregate_l1_sensitivity"],
            recommended_clip=result["recommended_global_clip_bound"],
        )

        return result

    def generate_sensitivity_report(
        self,
        query_type: str,
        data: list[float],
        local_sensitivity: float,
        smooth_sensitivity: float,
        global_sensitivity: float,
        recommended_clip: float,
    ) -> dict[str, Any]:
        """Generate a structured sensitivity analysis report.

        Args:
            query_type: Type of query analyzed.
            data: The dataset analyzed (used for descriptive statistics only).
            local_sensitivity: Computed local sensitivity.
            smooth_sensitivity: Computed smooth sensitivity.
            global_sensitivity: Global sensitivity (from query type definition).
            recommended_clip: Recommended clipping bound.

        Returns:
            Comprehensive sensitivity report dictionary.
        """
        array = np.array(data, dtype=np.float64)
        return {
            "query_type": query_type,
            "dataset_statistics": {
                "n_records": len(data),
                "mean": float(np.mean(array)),
                "std": float(np.std(array)),
                "min": float(np.min(array)),
                "max": float(np.max(array)),
                "p99": float(np.percentile(np.abs(array), 99)),
            },
            "sensitivity_estimates": {
                "global_sensitivity": round(global_sensitivity, 6),
                "local_sensitivity": round(local_sensitivity, 6),
                "smooth_sensitivity": round(smooth_sensitivity, 6),
                "recommended_clip_bound": round(recommended_clip, 6),
            },
            "mechanism_recommendations": {
                "laplace": {
                    "recommended_noise_scale": round(global_sensitivity, 6),
                    "mechanism_type": "pure_dp",
                },
                "gaussian": {
                    "recommended_sensitivity": round(smooth_sensitivity, 6),
                    "mechanism_type": "approximate_dp",
                },
            },
            "utility_impact": {
                "clip_fraction": round(float(np.mean(np.abs(array) > recommended_clip)), 4),
                "expected_l2_error_from_clipping": round(
                    float(np.sqrt(np.mean(np.maximum(np.abs(array) - recommended_clip, 0) ** 2))),
                    6,
                ),
            },
        }
