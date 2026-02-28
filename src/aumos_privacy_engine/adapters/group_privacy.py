"""Group differential privacy extension (GAP-99).

Group DP extends standard record-level DP to treat all records sharing
a group_key as a single privacy unit. This is required when records
are correlated (e.g., multiple visits from the same patient) and
removing one record could still expose information about the group.

Group DP is achieved by multiplying the sensitivity by max_group_size
before applying the standard DP mechanism.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Literal

from aumos_common.observability import get_logger

logger = get_logger(__name__)


@dataclass
class GroupPrivacyConfig:
    """Configuration for group differential privacy.

    Attributes:
        group_key_column: Column name identifying each privacy group.
        max_group_size: Upper bound on records per group (for sensitivity).
        composition_type: How to compose across multiple group DP operations.
    """

    group_key_column: str
    max_group_size: int
    composition_type: Literal["sequential", "parallel"] = "sequential"


class GroupPrivacyMode:
    """Wraps standard DP mechanisms with group-adjusted sensitivity.

    For group DP with max_group_size k:
    - L1 sensitivity is multiplied by k (Laplace/Exponential)
    - L2 sensitivity is multiplied by k (Gaussian/Subsampled)
    - Epsilon and delta requirements increase by factor k vs individual DP

    Args:
        config: Group privacy configuration.
    """

    def __init__(self, config: GroupPrivacyConfig) -> None:
        """Initialize group privacy mode.

        Args:
            config: Group DP configuration.
        """
        if config.max_group_size <= 0:
            raise ValueError(
                f"max_group_size must be positive, got {config.max_group_size}"
            )
        self._config = config

    def adjusted_sensitivity(self, per_record_sensitivity: float) -> float:
        """Compute group-adjusted sensitivity.

        For a query with per-record sensitivity s, the group sensitivity
        is s * max_group_size. This ensures that removing all records of
        one group changes the output by at most s * max_group_size.

        Args:
            per_record_sensitivity: L1 or L2 sensitivity per individual record.

        Returns:
            Group-level sensitivity = per_record_sensitivity * max_group_size.
        """
        if per_record_sensitivity <= 0:
            raise ValueError(
                f"per_record_sensitivity must be positive, got {per_record_sensitivity}"
            )
        group_sensitivity = per_record_sensitivity * self._config.max_group_size
        logger.debug(
            "group_sensitivity_computed",
            per_record=per_record_sensitivity,
            group_size=self._config.max_group_size,
            group_sensitivity=group_sensitivity,
        )
        return group_sensitivity

    def adjusted_epsilon(
        self,
        epsilon: float,
        mechanism: str,
    ) -> float:
        """Compute the effective per-record epsilon after group DP adjustment.

        Group DP with group size k requires epsilon_group such that the
        overall group sensitivity is handled by the mechanism. Since we
        multiply sensitivity by k, the noise scale automatically handles
        group DP — no epsilon adjustment is needed when using adjusted sensitivity.

        Args:
            epsilon: The epsilon to apply with group-adjusted sensitivity.
            mechanism: The DP mechanism to be used.

        Returns:
            The epsilon value (unchanged — sensitivity adjustment handles group DP).
        """
        # When using group-adjusted sensitivity, the epsilon value stays the same.
        # The increased sensitivity means more noise is added, achieving group DP.
        return epsilon

    def compute_noise_scale_laplace(
        self,
        per_record_sensitivity: float,
        epsilon: float,
    ) -> float:
        """Compute Laplace noise scale for group DP.

        λ = group_sensitivity / epsilon

        Args:
            per_record_sensitivity: Per-record L1 sensitivity.
            epsilon: Privacy budget.

        Returns:
            Laplace noise scale λ.
        """
        group_sens = self.adjusted_sensitivity(per_record_sensitivity)
        return group_sens / epsilon

    def compute_noise_scale_gaussian(
        self,
        per_record_sensitivity: float,
        epsilon: float,
        delta: float,
    ) -> float:
        """Compute Gaussian noise scale for group DP.

        σ = sqrt(2 * ln(1.25/δ)) * group_sensitivity / epsilon

        Args:
            per_record_sensitivity: Per-record L2 sensitivity.
            epsilon: Privacy budget.
            delta: Failure probability (must be > 0).

        Returns:
            Gaussian noise scale σ.
        """
        if delta <= 0:
            raise ValueError(
                f"Gaussian mechanism requires delta > 0, got {delta}"
            )
        group_sens = self.adjusted_sensitivity(per_record_sensitivity)
        return math.sqrt(2 * math.log(1.25 / delta)) * group_sens / epsilon

    @property
    def group_key_column(self) -> str:
        """Name of the group key column.

        Returns:
            The column name identifying privacy groups.
        """
        return self._config.group_key_column

    @property
    def max_group_size(self) -> int:
        """Maximum number of records per group.

        Returns:
            The upper bound on group size.
        """
        return self._config.max_group_size
