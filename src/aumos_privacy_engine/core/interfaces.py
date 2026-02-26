"""Protocol interfaces for the privacy engine.

Defines the contracts that all adapters and services must implement.
Using Protocol (structural subtyping) for loose coupling — no inheritance required.
"""

from decimal import Decimal
from typing import Protocol, runtime_checkable
from uuid import UUID

from aumos_privacy_engine.api.schemas import (
    BudgetAllocateRequest,
    BudgetResponse,
    CompositionPlanRequest,
    CompositionPlanResponse,
    LossVisualizationResponse,
    MechanismApplyRequest,
    MechanismApplyResponse,
    ProofResponse,
)
from aumos_privacy_engine.core.models import CompositionPlan, PrivacyBudget, PrivacyOperation


@runtime_checkable
class DPMechanismProtocol(Protocol):
    """Contract for differential privacy mechanism adapters.

    Each mechanism adapter (Laplace, Gaussian, Exponential, Subsampled) must
    implement this protocol. Mechanisms are stateless and deterministic given
    the same random seed.
    """

    async def apply(
        self,
        data: list[float],
        sensitivity: float,
        epsilon: float,
        delta: float,
    ) -> tuple[list[float], Decimal, Decimal, Decimal]:
        """Apply the DP mechanism to numerical data.

        Args:
            data: Input numerical values to privatize.
            sensitivity: L1 or L2 sensitivity of the query.
            epsilon: Privacy budget to spend for this operation.
            delta: Failure probability (0.0 for pure DP mechanisms).

        Returns:
            Tuple of (noisy_values, actual_epsilon_consumed, actual_delta_consumed, noise_scale).
            The actual consumed values may differ slightly from requested due to
            mechanism parameter rounding.
        """
        ...

    def validate_parameters(
        self,
        sensitivity: float,
        epsilon: float,
        delta: float,
    ) -> None:
        """Validate mechanism parameters before applying.

        Args:
            sensitivity: Query sensitivity — must be > 0.
            epsilon: Privacy budget — must be > 0.
            delta: Failure probability — must be in [0, 1).

        Raises:
            ValueError: If any parameter is out of valid range.
        """
        ...

    def compute_noise_scale(
        self,
        sensitivity: float,
        epsilon: float,
        delta: float,
    ) -> float:
        """Compute the noise scale parameter for this mechanism.

        Args:
            sensitivity: Query sensitivity.
            epsilon: Target epsilon.
            delta: Target delta.

        Returns:
            Noise scale (lambda for Laplace, sigma for Gaussian).
        """
        ...


@runtime_checkable
class BudgetManagerProtocol(Protocol):
    """Contract for privacy budget management operations.

    Budget managers handle allocation, consumption tracking, and renewal
    of tenant-specific privacy budgets. All operations must be atomic
    to prevent race conditions in concurrent synthesis jobs.
    """

    async def get_active_budget(self, tenant_id: UUID) -> PrivacyBudget | None:
        """Retrieve the currently active budget for a tenant.

        Args:
            tenant_id: The tenant whose budget to retrieve.

        Returns:
            Active PrivacyBudget if one exists, None otherwise.
        """
        ...

    async def allocate_budget(
        self,
        tenant_id: UUID,
        request: BudgetAllocateRequest,
    ) -> BudgetResponse:
        """Allocate (reserve) epsilon/delta from the tenant's budget for an operation.

        This is an atomic check-and-reserve operation. It fails if:
        - No active budget exists for the tenant
        - Requested epsilon exceeds remaining budget
        - Requested delta exceeds remaining delta budget

        Args:
            tenant_id: The tenant requesting budget allocation.
            request: Allocation request with epsilon, delta, and operation metadata.

        Returns:
            BudgetResponse with allocation confirmation and remaining budget.

        Raises:
            BudgetExhaustedError: If insufficient budget remains.
            NotFoundError: If no active budget exists for the tenant.
        """
        ...

    async def get_budget_utilization(self, tenant_id: UUID) -> BudgetResponse:
        """Get current budget utilization for a tenant.

        Args:
            tenant_id: The tenant whose utilization to retrieve.

        Returns:
            BudgetResponse with current usage statistics.

        Raises:
            NotFoundError: If no budget exists for the tenant.
        """
        ...

    async def renew_budget(self, tenant_id: UUID) -> PrivacyBudget:
        """Create a new budget period for the tenant (auto-renewal).

        Args:
            tenant_id: The tenant whose budget to renew.

        Returns:
            The newly created PrivacyBudget for the new period.
        """
        ...


@runtime_checkable
class ProofGeneratorProtocol(Protocol):
    """Contract for formal mathematical proof generation.

    Proof generators create verifiable mathematical derivations showing that
    a given mechanism application satisfies (ε,δ)-DP. Used for regulatory audit.
    """

    def generate_mechanism_proof(
        self,
        mechanism: str,
        epsilon: float,
        delta: float,
        sensitivity: float,
        noise_scale: float,
        composition_type: str,
    ) -> dict:  # type: ignore[type-arg]
        """Generate a formal proof for a single mechanism application.

        Args:
            mechanism: DP mechanism used (laplace/gaussian/exponential/subsampled).
            epsilon: Epsilon privacy parameter claimed.
            delta: Delta privacy parameter claimed (0 for pure DP).
            sensitivity: Query sensitivity.
            noise_scale: Actual noise scale applied.
            composition_type: How this operation composes with others.

        Returns:
            Dictionary containing:
                - latex: LaTeX mathematical derivation string
                - json_tree: Structured proof tree for programmatic verification
                - theorem: Name of the DP theorem being applied
                - verification_hash: SHA-256 hash of the proof for tamper detection
        """
        ...

    def verify_proof_chain(
        self,
        operations: list[PrivacyOperation],
        composition_type: str,
    ) -> bool:
        """Verify that a chain of operations satisfies the claimed composition bound.

        Args:
            operations: List of PrivacyOperation records to verify.
            composition_type: Composition theorem to verify against.

        Returns:
            True if the proof chain is valid and consistent.
        """
        ...


@runtime_checkable
class CompositionEngineProtocol(Protocol):
    """Contract for DP composition theorem computations.

    Implements the three main composition approaches:
    - Sequential: for operations on the same dataset
    - Parallel: for operations on disjoint dataset partitions
    - Advanced (Rényi DP): for tighter composition bounds
    """

    def sequential_compose(
        self,
        epsilons: list[float],
        deltas: list[float],
    ) -> tuple[float, float]:
        """Compute sequential composition bound.

        For k mechanisms applied to the same dataset:
        ε_total = Σεᵢ, δ_total = Σδᵢ

        Args:
            epsilons: List of per-mechanism epsilon values.
            deltas: List of per-mechanism delta values.

        Returns:
            Tuple of (total_epsilon, total_delta).
        """
        ...

    def parallel_compose(
        self,
        epsilons: list[float],
        deltas: list[float],
    ) -> tuple[float, float]:
        """Compute parallel composition bound.

        For k mechanisms applied to disjoint data partitions:
        ε_total = max(εᵢ), δ_total = max(δᵢ)

        Args:
            epsilons: List of per-mechanism epsilon values.
            deltas: List of per-mechanism delta values.

        Returns:
            Tuple of (total_epsilon, total_delta).
        """
        ...

    def advanced_compose_rdp(
        self,
        epsilons: list[float],
        deltas: list[float],
        alpha: float,
        target_delta: float,
    ) -> tuple[float, float]:
        """Compute advanced composition bound via Rényi DP.

        Converts each (ε,δ)-DP mechanism to RDP, accumulates RDP moments,
        then converts back to (ε,δ)-DP. Provides tighter bounds than sequential.

        Args:
            epsilons: List of per-mechanism epsilon values.
            deltas: List of per-mechanism delta values.
            alpha: Rényi divergence order (typically 10-100).
            target_delta: Target delta for the final (ε,δ) conversion.

        Returns:
            Tuple of (total_epsilon, total_delta) with tighter bounds.
        """
        ...


@runtime_checkable
class LossVisualizerProtocol(Protocol):
    """Contract for privacy loss curve visualization.

    Generates matplotlib plots showing how privacy budget is consumed over time
    and how privacy loss grows under sequential composition.
    """

    async def generate_loss_curve(
        self,
        tenant_id: UUID,
        output_format: str,
    ) -> LossVisualizationResponse:
        """Generate a privacy loss curve for a tenant's budget history.

        Shows cumulative epsilon consumption over time, remaining budget,
        and projected exhaustion date.

        Args:
            tenant_id: The tenant whose budget history to visualize.
            output_format: Output format: png | svg | json.

        Returns:
            LossVisualizationResponse with base64-encoded image or JSON data.
        """
        ...

    async def generate_budget_utilization_chart(
        self,
        tenant_id: UUID,
        output_format: str,
    ) -> LossVisualizationResponse:
        """Generate a budget utilization breakdown chart by source engine.

        Shows how much epsilon each synthesis engine (tabular, text, image, etc.)
        has consumed from the tenant's budget.

        Args:
            tenant_id: The tenant to analyze.
            output_format: Output format: png | svg | json.

        Returns:
            LossVisualizationResponse with chart data.
        """
        ...
