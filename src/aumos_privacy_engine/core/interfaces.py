"""Protocol interfaces for the privacy engine.

Defines the contracts that all adapters and services must implement.
Using Protocol (structural subtyping) for loose coupling — no inheritance required.

Protocols defined here:
  DPMechanismProtocol        — DP mechanism adapters (Laplace, Gaussian, etc.)
  BudgetManagerProtocol      — Per-tenant budget allocation and tracking
  ProofGeneratorProtocol     — Formal proof generation and verification
  CompositionEngineProtocol  — Sequential/parallel/RDP composition theorems
  LossVisualizerProtocol     — Privacy loss curve visualization

  EpsilonAccountantProtocol  — ε/δ budget ledger with audit certificates
  MomentAccountantProtocol   — Rényi DP moments accumulation and conversion
  SensitivityAnalyzerProtocol — Automatic sensitivity estimation and clip bounds
  PrivacyAmplifierProtocol   — Subsampling/shuffling amplification
  FormalProverProtocol        — DP verification certificates and proof chains
  AuditReporterProtocol      — PIA report generation
  OpenDPAdapterProtocol      — OpenDP framework integration
"""

from decimal import Decimal
from typing import Any, Protocol, runtime_checkable
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


# ---------------------------------------------------------------------------
# New Protocols added in adapter expansion (epsilon_accountant, moment_accountant,
# sensitivity_analyzer, privacy_amplifier, formal_prover, audit_reporter, opendp_adapter)
# ---------------------------------------------------------------------------


@runtime_checkable
class EpsilonAccountantProtocol(Protocol):
    """Contract for per-tenant ε/δ budget ledger with audit certificate generation.

    Maintains an append-only in-process ledger of budget consumption events.
    The caller is responsible for persisting events to durable storage.
    """

    def initialize_tenant(
        self,
        tenant_id: UUID,
        sensitivity_tier: str,
    ) -> None:
        """Initialize budget tracking for a new tenant.

        Args:
            tenant_id: Tenant to initialize.
            sensitivity_tier: Data sensitivity tier (public/internal/confidential/restricted/healthcare).
        """
        ...

    def record_consumption(
        self,
        tenant_id: UUID,
        epsilon_consumed: float,
        delta_consumed: float,
        query_label: str,
        source_engine: str,
        sensitivity_tier: str,
        operation_id: UUID | None,
    ) -> Any:
        """Record a privacy budget consumption event for a tenant.

        Args:
            tenant_id: Tenant consuming budget.
            epsilon_consumed: Epsilon consumed by this operation.
            delta_consumed: Delta consumed by this operation.
            query_label: Human-readable label for the query.
            source_engine: Synthesis engine triggering the consumption.
            sensitivity_tier: Data sensitivity tier.
            operation_id: Optional UUID linking to the PrivacyOperation ORM record.

        Returns:
            The created BudgetRecord.
        """
        ...

    def get_remaining_budget(self, tenant_id: UUID) -> tuple[Decimal, Decimal]:
        """Get remaining epsilon and delta budget for a tenant.

        Args:
            tenant_id: Tenant to query.

        Returns:
            Tuple of (remaining_epsilon, remaining_delta).
        """
        ...

    def get_usage_analytics(self, tenant_id: UUID) -> dict[str, Any]:
        """Get historical usage analytics including per-engine breakdown and projections.

        Args:
            tenant_id: Tenant to analyze.

        Returns:
            Analytics dictionary with usage breakdown and projected exhaustion date.
        """
        ...

    def generate_budget_certificate(
        self,
        tenant_id: UUID,
        certifier_id: str,
    ) -> dict[str, Any]:
        """Generate a cryptographically-identified audit certificate for budget usage.

        Args:
            tenant_id: Tenant to certify.
            certifier_id: Identifier of the certifying system.

        Returns:
            Certificate dictionary with verification_hash.
        """
        ...

    def apply_budget_policy(
        self,
        tenant_id: UUID,
        requested_epsilon: float,
        sensitivity_tier: str,
    ) -> bool:
        """Check whether a requested epsilon consumption is within policy limits.

        Args:
            tenant_id: Tenant requesting consumption.
            requested_epsilon: Epsilon the operation wants to consume.
            sensitivity_tier: Data sensitivity tier.

        Returns:
            True if within policy, False if budget is exhausted.
        """
        ...


@runtime_checkable
class MomentAccountantProtocol(Protocol):
    """Contract for Rényi DP moments accountant.

    Tracks accumulated Rényi divergence per tenant and converts to (ε, δ)-DP
    at the end of a computation for a tight composition bound.
    """

    def initialize_tenant(self, tenant_id: str) -> None:
        """Initialize RDP moment tracking for a new tenant.

        Args:
            tenant_id: Tenant identifier string.
        """
        ...

    def accumulate_gaussian(
        self,
        tenant_id: str,
        sigma: float,
        sensitivity: float,
        num_steps: int,
    ) -> None:
        """Accumulate RDP moments for Gaussian mechanism applications.

        Args:
            tenant_id: Tenant accumulating budget.
            sigma: Gaussian noise standard deviation.
            sensitivity: L2 sensitivity of the query.
            num_steps: Number of mechanism applications.
        """
        ...

    def accumulate_subsampled_gaussian(
        self,
        tenant_id: str,
        sigma: float,
        sampling_rate: float,
        num_steps: int,
    ) -> None:
        """Accumulate RDP moments for Poisson-subsampled Gaussian mechanism.

        Args:
            tenant_id: Tenant accumulating budget.
            sigma: Gaussian noise standard deviation.
            sampling_rate: Poisson subsampling rate q ∈ (0, 1].
            num_steps: Number of mechanism applications.
        """
        ...

    def rdp_to_dp(
        self,
        tenant_id: str,
        target_delta: float,
    ) -> tuple[float, float]:
        """Convert accumulated RDP moments to (ε, δ)-DP guarantee.

        Args:
            tenant_id: Tenant to convert.
            target_delta: Target failure probability δ ∈ (0, 1).

        Returns:
            Tuple of (epsilon, delta) representing the tightest (ε, δ)-DP guarantee.
        """
        ...

    def reset_tenant(self, tenant_id: str) -> None:
        """Reset all accumulated moments for a tenant (e.g., on budget renewal).

        Args:
            tenant_id: Tenant to reset.
        """
        ...


@runtime_checkable
class SensitivityAnalyzerProtocol(Protocol):
    """Contract for automatic sensitivity estimation and clip bound recommendation."""

    def estimate_global_sensitivity(
        self,
        query_type: str,
        data_bound: float | None,
        num_dimensions: int,
    ) -> dict[str, float]:
        """Estimate global sensitivity for a named query type.

        Args:
            query_type: Query type (count/sum/mean/median/histogram/gradient).
            data_bound: Maximum absolute value per record (required for data-dependent queries).
            num_dimensions: Number of dimensions (for multi-column or gradient queries).

        Returns:
            Dictionary with l1_sensitivity and l2_sensitivity.
        """
        ...

    async def compute_local_sensitivity(
        self,
        data: list[float],
        query_type: str,
        clip_bound: float | None,
    ) -> float:
        """Compute empirical local sensitivity for a given dataset.

        Args:
            data: Observed dataset values.
            query_type: Query type.
            clip_bound: Optional clipping bound to apply before analysis.

        Returns:
            Empirical local sensitivity estimate.
        """
        ...

    def recommend_clip_bound(
        self,
        data: list[float],
        percentile: float | None,
    ) -> dict[str, float]:
        """Recommend a data-driven clipping bound.

        Args:
            data: Observed values to analyze.
            percentile: Percentile for clip bound (0-100).

        Returns:
            Dictionary with recommended_clip_bound and supporting statistics.
        """
        ...

    async def profile_tabular_columns(
        self,
        column_data: dict[str, list[float]],
        clip_percentile: float | None,
    ) -> dict[str, Any]:
        """Generate per-column sensitivity profiles for a tabular dataset.

        Args:
            column_data: Dictionary mapping column names to their numeric values.
            clip_percentile: Clip percentile override.

        Returns:
            Dictionary with per-column profiles and dataset-level aggregate sensitivity.
        """
        ...


@runtime_checkable
class PrivacyAmplifierProtocol(Protocol):
    """Contract for privacy amplification via subsampling and shuffling."""

    def amplify_epsilon_poisson(
        self,
        epsilon: float,
        sampling_rate: float,
    ) -> float:
        """Compute amplified epsilon for Poisson subsampling.

        Args:
            epsilon: Original epsilon of the mechanism.
            sampling_rate: Poisson sampling probability q ∈ (0, 1].

        Returns:
            Amplified epsilon for the full dataset.
        """
        ...

    def amplify_epsilon_fixed_size(
        self,
        epsilon: float,
        sample_size: int,
        dataset_size: int,
    ) -> float:
        """Compute amplified epsilon for fixed-size subsampling.

        Args:
            epsilon: Original epsilon of the mechanism.
            sample_size: Number of records in the sample.
            dataset_size: Total number of records in the full dataset.

        Returns:
            Amplified epsilon.
        """
        ...

    def amplify_epsilon_shuffling(
        self,
        epsilon_local: float,
        num_users: int,
        delta: float,
    ) -> tuple[float, float]:
        """Compute amplified (ε, δ) for the shuffle model.

        Args:
            epsilon_local: Local epsilon (each user's local DP guarantee).
            num_users: Number of participating users.
            delta: Target failure probability for the central model.

        Returns:
            Tuple of (central_epsilon, delta).
        """
        ...

    def generate_amplification_certificate(
        self,
        mechanism: str,
        original_epsilon: float,
        amplification_type: str,
        amplification_params: dict[str, Any],
        amplified_epsilon: float,
        delta: float,
        tenant_id: UUID | None,
    ) -> dict[str, Any]:
        """Generate a cryptographically-identified amplification proof certificate.

        Args:
            mechanism: DP mechanism name.
            original_epsilon: Original epsilon before amplification.
            amplification_type: Type of amplification (poisson/fixed_size/shuffle).
            amplification_params: Parameters used in amplification.
            amplified_epsilon: Resulting amplified epsilon.
            delta: Delta for the amplified guarantee.
            tenant_id: Optional tenant UUID.

        Returns:
            Certificate dictionary with verification_hash.
        """
        ...


@runtime_checkable
class FormalProverProtocol(Protocol):
    """Contract for formal DP proof generation and chain verification."""

    def generate_mechanism_proof(
        self,
        mechanism: str,
        epsilon: float,
        delta: float,
        sensitivity: float,
        noise_scale: float,
        composition_type: str,
    ) -> dict[str, Any]:
        """Generate a formal proof for a single mechanism application.

        Args:
            mechanism: DP mechanism used.
            epsilon: Epsilon privacy parameter claimed.
            delta: Delta privacy parameter claimed.
            sensitivity: Query sensitivity.
            noise_scale: Actual noise scale applied.
            composition_type: How this operation composes with others.

        Returns:
            Dictionary with json_tree, theorem, verification_hash, latex, and compliance.
        """
        ...

    def verify_proof_chain(
        self,
        operations: list[PrivacyOperation],
        composition_type: str,
    ) -> bool:
        """Verify that a chain of operations is internally consistent.

        Args:
            operations: List of PrivacyOperation records to verify.
            composition_type: Composition theorem to verify against.

        Returns:
            True if the proof chain is valid.
        """
        ...

    def generate_chain_certificate(
        self,
        operations: list[PrivacyOperation],
        composition_type: str,
        tenant_id: UUID,
        job_id: UUID,
    ) -> dict[str, Any]:
        """Generate a certificate covering a complete chain of DP operations.

        Args:
            operations: Ordered list of PrivacyOperation records.
            composition_type: Composition theorem applied.
            tenant_id: Owning tenant.
            job_id: Synthesis job ID.

        Returns:
            Composite certificate dictionary.
        """
        ...


@runtime_checkable
class AuditReporterProtocol(Protocol):
    """Contract for Privacy Impact Assessment (PIA) report generation."""

    def compute_risk_score(
        self,
        total_epsilon: float,
        num_operations: int,
        source_engines: list[str],
        is_healthcare: bool,
    ) -> dict[str, Any]:
        """Compute a normalized privacy risk score.

        Args:
            total_epsilon: Total epsilon consumed.
            num_operations: Number of DP operations performed.
            source_engines: List of synthesis engines that consumed budget.
            is_healthcare: Whether this involves healthcare (PHI) data.

        Returns:
            Dictionary with risk_score (0.0-1.0), risk_level, and contributing factors.
        """
        ...

    def generate_pia_report(
        self,
        tenant_id: UUID,
        budget: PrivacyBudget,
        operations: list[PrivacyOperation],
        dataset_name: str,
        stakeholder_view: str,
        regulations: list[str] | None,
    ) -> dict[str, Any]:
        """Generate a full Privacy Impact Assessment report.

        Args:
            tenant_id: Tenant being audited.
            budget: Current PrivacyBudget record.
            operations: All PrivacyOperation records for the period.
            dataset_name: Human-readable name of the dataset.
            stakeholder_view: 'technical', 'executive', or 'regulatory'.
            regulations: Regulations to include (GDPR, CCPA, HIPAA).

        Returns:
            Full PIA report as a JSON-serializable dictionary.
        """
        ...

    def generate_json_report(
        self,
        tenant_id: UUID,
        budget: PrivacyBudget,
        operations: list[PrivacyOperation],
        dataset_name: str,
        stakeholder_view: str,
    ) -> str:
        """Generate a PIA report serialized as a JSON string.

        Args:
            tenant_id: Tenant being audited.
            budget: Current budget record.
            operations: Privacy operations for the period.
            dataset_name: Dataset name.
            stakeholder_view: Report view type.

        Returns:
            JSON-serialized report string.
        """
        ...


@runtime_checkable
class OpenDPAdapterProtocol(Protocol):
    """Contract for OpenDP framework integration.

    Provides a unified interface for applying DP mechanisms via the OpenDP library,
    with transparent fallback to vetted manual implementations when OpenDP is unavailable.
    """

    async def apply_mechanism(
        self,
        data: list[float],
        mechanism: str,
        sensitivity: float,
        epsilon: float,
        delta: float,
    ) -> Any:
        """Apply a DP mechanism asynchronously via OpenDP.

        Args:
            data: Input numerical values to privatize.
            mechanism: DP mechanism name: laplace | gaussian | subsampled.
            sensitivity: Query sensitivity (L1 for Laplace, L2 for Gaussian).
            epsilon: Privacy budget to consume.
            delta: Failure probability (0.0 for Laplace, > 0 for Gaussian).

        Returns:
            OpenDPMeasurementResult with privatized values and accounting metadata.
        """
        ...

    def compose_measurements(
        self,
        measurements: list[dict[str, Any]],
        composition_type: str,
    ) -> dict[str, Any]:
        """Compose multiple measurement specs and compute total privacy cost.

        Args:
            measurements: List of measurement dicts with keys: mechanism, epsilon, delta.
            composition_type: Composition method: sequential | parallel.

        Returns:
            Dictionary with total_epsilon, total_delta, and composition_type.
        """
        ...

    def get_library_status(self) -> dict[str, Any]:
        """Return the status and version of the OpenDP library.

        Returns:
            Dictionary with availability, version, and fallback status.
        """
        ...
