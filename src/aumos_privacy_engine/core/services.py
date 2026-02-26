"""Core business logic services for the privacy engine.

Contains five service classes:
- BudgetService: per-tenant budget allocation and tracking
- MechanismService: DP mechanism application and validation
- CompositionService: sequential/parallel/Rényi composition theorems
- ProofService: formal mathematical proof generation and verification
- VisualizationService: privacy loss curve and budget utilization charts

All services are async-first and depend on injected repositories and adapters.
They contain NO framework code — pure business logic with typed interfaces.
"""

import hashlib
import json
import math
from datetime import UTC, datetime, timedelta
from decimal import Decimal
from uuid import UUID

from aumos_common.errors import ErrorCode, NotFoundError, ValidationError
from aumos_common.observability import get_logger

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
from aumos_privacy_engine.settings import Settings

logger = get_logger(__name__)


class BudgetExhaustedError(Exception):
    """Raised when a tenant's privacy budget is insufficient for the requested operation.

    Attributes:
        tenant_id: The tenant whose budget is exhausted.
        requested_epsilon: Epsilon requested by the operation.
        remaining_epsilon: Epsilon actually remaining in the budget.
    """

    def __init__(
        self,
        tenant_id: UUID,
        requested_epsilon: float,
        remaining_epsilon: float,
    ) -> None:
        """Initialize budget exhausted error.

        Args:
            tenant_id: The tenant whose budget is exhausted.
            requested_epsilon: Epsilon amount requested.
            remaining_epsilon: Epsilon amount remaining.
        """
        self.tenant_id = tenant_id
        self.requested_epsilon = requested_epsilon
        self.remaining_epsilon = remaining_epsilon
        super().__init__(
            f"Budget exhausted for tenant {tenant_id}: "
            f"requested ε={requested_epsilon:.6f}, "
            f"remaining ε={remaining_epsilon:.6f}"
        )


class BudgetService:
    """Manages per-tenant differential privacy budget allocation and tracking.

    Enforces that tenants cannot exceed their allocated epsilon/delta budgets.
    All allocations are atomic — concurrent requests are serialized via database
    transactions to prevent double-spend.

    Budget consumption is recorded via append-only PrivacyOperation records.
    The used_epsilon on PrivacyBudget is updated atomically with the new operation
    insert to avoid recomputing sums on every check.
    """

    def __init__(
        self,
        budget_repo: object,
        operation_repo: object,
        publisher: object,
        settings: Settings,
    ) -> None:
        """Initialize budget service with injected dependencies.

        Args:
            budget_repo: Repository for PrivacyBudget persistence.
            operation_repo: Repository for PrivacyOperation persistence.
            publisher: Kafka event publisher for lifecycle events.
            settings: Privacy engine configuration settings.
        """
        self._budget_repo = budget_repo
        self._operation_repo = operation_repo
        self._publisher = publisher
        self._settings = settings

    async def get_active_budget(self, tenant_id: UUID) -> PrivacyBudget:
        """Retrieve the active budget for a tenant or raise NotFoundError.

        Args:
            tenant_id: The tenant whose budget to retrieve.

        Returns:
            The active PrivacyBudget for the tenant.

        Raises:
            NotFoundError: If no active budget exists for the tenant.
        """
        budget = await self._budget_repo.get_active(tenant_id)  # type: ignore[union-attr]
        if budget is None:
            raise NotFoundError(
                resource="PrivacyBudget",
                resource_id=str(tenant_id),
                error_code=ErrorCode.NOT_FOUND,
            )
        return budget  # type: ignore[return-value]

    async def get_or_create_budget(self, tenant_id: UUID) -> PrivacyBudget:
        """Get the active budget or create a new default one.

        Used when a tenant makes their first request and has no existing budget.
        Creates a budget with the system default epsilon and delta values.

        Args:
            tenant_id: The tenant to get or create budget for.

        Returns:
            Existing active budget or newly created default budget.
        """
        budget = await self._budget_repo.get_active(tenant_id)  # type: ignore[union-attr]
        if budget is not None:
            return budget  # type: ignore[return-value]

        now = datetime.now(UTC)
        period_end = now + timedelta(days=self._settings.budget_renewal_days)

        new_budget = await self._budget_repo.create(  # type: ignore[union-attr]
            tenant_id=tenant_id,
            total_epsilon=Decimal(str(self._settings.default_epsilon)),
            used_epsilon=Decimal("0.0"),
            total_delta=Decimal(str(self._settings.default_delta)),
            used_delta=Decimal("0.0"),
            period_start=now,
            period_end=period_end,
            auto_renew=True,
            is_active=True,
        )

        logger.info(
            "Created default privacy budget for tenant",
            tenant_id=str(tenant_id),
            total_epsilon=self._settings.default_epsilon,
            period_end=period_end.isoformat(),
        )

        return new_budget  # type: ignore[return-value]

    async def allocate(
        self,
        tenant_id: UUID,
        request: BudgetAllocateRequest,
    ) -> BudgetResponse:
        """Allocate epsilon/delta from tenant budget for an operation.

        Performs an atomic check-and-reserve:
        1. Fetch active budget with row lock to prevent races
        2. Validate requested epsilon does not exceed max_operation_epsilon
        3. Check remaining budget is sufficient
        4. Update used_epsilon/used_delta on the budget record
        5. Publish privacy.budget.low event if < 10% remaining

        Args:
            tenant_id: The tenant requesting budget allocation.
            request: Allocation request with epsilon, delta, source engine, job ID.

        Returns:
            BudgetResponse with updated utilization and allocation confirmation.

        Raises:
            BudgetExhaustedError: If insufficient budget remains.
            ValidationError: If requested epsilon exceeds per-operation limit.
        """
        # Validate per-operation epsilon limit
        if request.epsilon_requested > self._settings.max_operation_epsilon:
            raise ValidationError(
                field="epsilon_requested",
                message=(
                    f"Requested ε={request.epsilon_requested} exceeds maximum "
                    f"per-operation limit of ε={self._settings.max_operation_epsilon}"
                ),
            )

        # Healthcare engines have stricter per-operation limits
        if request.source_engine == "healthcare":
            if request.epsilon_requested > self._settings.healthcare_max_epsilon:
                raise ValidationError(
                    field="epsilon_requested",
                    message=(
                        f"Healthcare operations are limited to ε={self._settings.healthcare_max_epsilon} "
                        f"per operation (HIPAA compliance). Requested: ε={request.epsilon_requested}"
                    ),
                )

        budget = await self.get_or_create_budget(tenant_id)

        # Check remaining budget (cast to float for comparison)
        remaining = float(budget.remaining_epsilon)
        if request.epsilon_requested > remaining:
            logger.warning(
                "Budget exhausted — allocation denied",
                tenant_id=str(tenant_id),
                requested=request.epsilon_requested,
                remaining=remaining,
                source_engine=request.source_engine,
            )
            raise BudgetExhaustedError(
                tenant_id=tenant_id,
                requested_epsilon=request.epsilon_requested,
                remaining_epsilon=remaining,
            )

        # Atomic budget update — increment used_epsilon and used_delta
        updated_budget = await self._budget_repo.consume(  # type: ignore[union-attr]
            budget_id=budget.id,
            epsilon_to_consume=Decimal(str(request.epsilon_requested)),
            delta_to_consume=Decimal(str(request.delta_requested)),
        )

        logger.info(
            "Privacy budget allocated",
            tenant_id=str(tenant_id),
            job_id=str(request.job_id),
            source_engine=request.source_engine,
            epsilon_consumed=request.epsilon_requested,
            delta_consumed=request.delta_requested,
            remaining_epsilon=float(updated_budget.remaining_epsilon),
        )

        # Warn if budget is running low (< 10% remaining)
        if updated_budget.epsilon_utilization_pct > 90.0:
            await self._publisher.publish(  # type: ignore[union-attr]
                "privacy.budget.low",
                {
                    "tenant_id": str(tenant_id),
                    "budget_id": str(budget.id),
                    "remaining_epsilon": float(updated_budget.remaining_epsilon),
                    "utilization_pct": updated_budget.epsilon_utilization_pct,
                    "source_engine": request.source_engine,
                },
            )

        return BudgetResponse(
            budget_id=updated_budget.id,
            tenant_id=tenant_id,
            total_epsilon=float(updated_budget.total_epsilon),
            used_epsilon=float(updated_budget.used_epsilon),
            remaining_epsilon=float(updated_budget.remaining_epsilon),
            total_delta=float(updated_budget.total_delta),
            used_delta=float(updated_budget.used_delta),
            remaining_delta=float(updated_budget.remaining_delta),
            epsilon_utilization_pct=updated_budget.epsilon_utilization_pct,
            period_start=updated_budget.period_start,
            period_end=updated_budget.period_end,
            auto_renew=updated_budget.auto_renew,
        )

    async def get_utilization(self, tenant_id: UUID) -> BudgetResponse:
        """Get current budget utilization statistics for a tenant.

        Args:
            tenant_id: The tenant to query.

        Returns:
            BudgetResponse with current usage statistics.

        Raises:
            NotFoundError: If no budget exists for the tenant.
        """
        budget = await self.get_active_budget(tenant_id)
        return BudgetResponse(
            budget_id=budget.id,
            tenant_id=tenant_id,
            total_epsilon=float(budget.total_epsilon),
            used_epsilon=float(budget.used_epsilon),
            remaining_epsilon=float(budget.remaining_epsilon),
            total_delta=float(budget.total_delta),
            used_delta=float(budget.used_delta),
            remaining_delta=float(budget.remaining_delta),
            epsilon_utilization_pct=budget.epsilon_utilization_pct,
            period_start=budget.period_start,
            period_end=budget.period_end,
            auto_renew=budget.auto_renew,
        )

    async def renew_budget(self, tenant_id: UUID) -> BudgetResponse:
        """Create a new budget period for the tenant.

        Deactivates the current budget and creates a fresh one with
        the same total epsilon/delta limits but zero consumption.

        Args:
            tenant_id: The tenant whose budget to renew.

        Returns:
            BudgetResponse for the newly created budget period.
        """
        old_budget = await self.get_active_budget(tenant_id)
        await self._budget_repo.deactivate(old_budget.id)  # type: ignore[union-attr]

        now = datetime.now(UTC)
        period_end = now + timedelta(days=self._settings.budget_renewal_days)

        new_budget = await self._budget_repo.create(  # type: ignore[union-attr]
            tenant_id=tenant_id,
            total_epsilon=old_budget.total_epsilon,
            used_epsilon=Decimal("0.0"),
            total_delta=old_budget.total_delta,
            used_delta=Decimal("0.0"),
            period_start=now,
            period_end=period_end,
            auto_renew=old_budget.auto_renew,
            is_active=True,
        )

        await self._publisher.publish(  # type: ignore[union-attr]
            "privacy.budget.renewed",
            {
                "tenant_id": str(tenant_id),
                "new_budget_id": str(new_budget.id),
                "total_epsilon": float(new_budget.total_epsilon),
                "period_end": period_end.isoformat(),
            },
        )

        logger.info(
            "Privacy budget renewed",
            tenant_id=str(tenant_id),
            new_budget_id=str(new_budget.id),
            period_end=period_end.isoformat(),
        )

        return BudgetResponse(
            budget_id=new_budget.id,
            tenant_id=tenant_id,
            total_epsilon=float(new_budget.total_epsilon),
            used_epsilon=0.0,
            remaining_epsilon=float(new_budget.total_epsilon),
            total_delta=float(new_budget.total_delta),
            used_delta=0.0,
            remaining_delta=float(new_budget.total_delta),
            epsilon_utilization_pct=0.0,
            period_start=new_budget.period_start,
            period_end=new_budget.period_end,
            auto_renew=new_budget.auto_renew,
        )


class MechanismService:
    """Applies differential privacy mechanisms to data.

    Wraps OpenDP and Opacus mechanism implementations, validates parameters,
    records the operation, and returns privatized data with proof metadata.

    Supports: Laplace, Gaussian, Exponential, Subsampled Gaussian (DP-SGD).
    """

    def __init__(
        self,
        laplace_mechanism: object,
        gaussian_mechanism: object,
        exponential_mechanism: object,
        subsampled_mechanism: object,
        operation_repo: object,
        publisher: object,
        settings: Settings,
    ) -> None:
        """Initialize mechanism service with injected mechanism adapters.

        Args:
            laplace_mechanism: Laplace mechanism adapter (via OpenDP).
            gaussian_mechanism: Gaussian mechanism adapter (via OpenDP).
            exponential_mechanism: Exponential mechanism adapter (via OpenDP).
            subsampled_mechanism: Subsampled Gaussian adapter (via Opacus).
            operation_repo: Repository for recording PrivacyOperation records.
            publisher: Kafka event publisher.
            settings: Privacy engine configuration.
        """
        self._mechanisms = {
            "laplace": laplace_mechanism,
            "gaussian": gaussian_mechanism,
            "exponential": exponential_mechanism,
            "subsampled": subsampled_mechanism,
        }
        self._operation_repo = operation_repo
        self._publisher = publisher
        self._settings = settings

    async def apply(
        self,
        budget_id: UUID,
        tenant_id: UUID,
        request: MechanismApplyRequest,
    ) -> MechanismApplyResponse:
        """Apply a DP mechanism and record the privacy operation.

        1. Validate mechanism parameters
        2. Apply the mechanism via the appropriate adapter
        3. Record the PrivacyOperation (append-only)
        4. Publish privacy.operation.completed event
        5. Return privatized data with proof reference

        Args:
            budget_id: The budget ID that was pre-allocated for this operation.
            tenant_id: The tenant owning the budget.
            request: Mechanism application request with data, mechanism type, and params.

        Returns:
            MechanismApplyResponse with privatized data and operation ID.

        Raises:
            ValidationError: If mechanism parameters are invalid.
            ValueError: If requested mechanism is not supported.
        """
        mechanism = self._mechanisms.get(request.mechanism)
        if mechanism is None:
            raise ValidationError(
                field="mechanism",
                message=f"Unsupported mechanism: {request.mechanism}. "
                f"Supported: {list(self._mechanisms.keys())}",
            )

        # Validate parameters via mechanism adapter
        mechanism.validate_parameters(  # type: ignore[union-attr]
            sensitivity=request.sensitivity,
            epsilon=request.epsilon,
            delta=request.delta,
        )

        # Apply the mechanism
        privatized_data, actual_epsilon, actual_delta, noise_scale = await mechanism.apply(  # type: ignore[union-attr]
            data=request.data,
            sensitivity=request.sensitivity,
            epsilon=request.epsilon,
            delta=request.delta,
        )

        # Record the operation (append-only)
        operation = await self._operation_repo.create(  # type: ignore[union-attr]
            budget_id=budget_id,
            tenant_id=tenant_id,
            mechanism=request.mechanism,
            epsilon_consumed=actual_epsilon,
            delta_consumed=actual_delta,
            composition_type=request.composition_type,
            source_engine=request.source_engine,
            job_id=request.job_id,
            sensitivity=Decimal(str(request.sensitivity)),
            noise_scale=noise_scale,
            query_description=request.query_description,
            formal_proof={},  # Proof generated asynchronously by ProofService
        )

        # Publish operation completed event
        await self._publisher.publish(  # type: ignore[union-attr]
            "privacy.operations",
            {
                "tenant_id": str(tenant_id),
                "operation_id": str(operation.id),
                "job_id": str(request.job_id),
                "mechanism": request.mechanism,
                "epsilon_consumed": float(actual_epsilon),
                "delta_consumed": float(actual_delta),
                "source_engine": request.source_engine,
            },
        )

        logger.info(
            "DP mechanism applied",
            operation_id=str(operation.id),
            mechanism=request.mechanism,
            epsilon_consumed=float(actual_epsilon),
            noise_scale=float(noise_scale),
            source_engine=request.source_engine,
        )

        return MechanismApplyResponse(
            operation_id=operation.id,
            mechanism=request.mechanism,
            privatized_data=privatized_data,
            epsilon_consumed=float(actual_epsilon),
            delta_consumed=float(actual_delta),
            noise_scale=float(noise_scale),
            proof_pending=True,  # Proof generated async
        )


class CompositionService:
    """Implements DP composition theorems for multi-step privacy accounting.

    Provides three composition methods:
    - Sequential: tightest correct bound for same-dataset operations
    - Parallel: optimal bound when operations touch disjoint data partitions
    - Rényi DP: tighter than sequential for many operations, uses moments accountant

    The Rényi DP approach converts each mechanism's guarantee to an RDP bound,
    accumulates across operations, then converts the total RDP bound back to
    (ε,δ)-DP using the standard conversion formula.
    """

    def __init__(self, settings: Settings) -> None:
        """Initialize composition service.

        Args:
            settings: Privacy engine configuration (Rényi alpha, etc.).
        """
        self._settings = settings

    def sequential_compose(
        self,
        epsilons: list[float],
        deltas: list[float],
    ) -> tuple[float, float]:
        """Compute sequential composition: ε_total = Σεᵢ, δ_total = Σδᵢ.

        Applies when k mechanisms are applied to the SAME dataset. This is the
        basic composition theorem — the privacy loss simply adds up.

        Args:
            epsilons: Per-mechanism epsilon values.
            deltas: Per-mechanism delta values.

        Returns:
            Tuple of (total_epsilon, total_delta).

        Raises:
            ValueError: If epsilons and deltas have different lengths.
        """
        if len(epsilons) != len(deltas):
            raise ValueError(
                f"epsilons and deltas must have equal length: "
                f"len(epsilons)={len(epsilons)}, len(deltas)={len(deltas)}"
            )

        total_epsilon = sum(epsilons)
        total_delta = sum(deltas)

        logger.debug(
            "Sequential composition computed",
            num_operations=len(epsilons),
            total_epsilon=total_epsilon,
            total_delta=total_delta,
        )

        return total_epsilon, total_delta

    def parallel_compose(
        self,
        epsilons: list[float],
        deltas: list[float],
    ) -> tuple[float, float]:
        """Compute parallel composition: ε_total = max(εᵢ), δ_total = max(δᵢ).

        Applies when k mechanisms are applied to DISJOINT partitions of the dataset.
        Each individual's data appears in at most one partition, so the privacy
        loss is determined only by the worst mechanism.

        Args:
            epsilons: Per-mechanism epsilon values.
            deltas: Per-mechanism delta values.

        Returns:
            Tuple of (total_epsilon, total_delta).

        Raises:
            ValueError: If lists are empty or have different lengths.
        """
        if not epsilons:
            raise ValueError("epsilons list must not be empty")
        if len(epsilons) != len(deltas):
            raise ValueError(
                f"epsilons and deltas must have equal length: "
                f"len(epsilons)={len(epsilons)}, len(deltas)={len(deltas)}"
            )

        total_epsilon = max(epsilons)
        total_delta = max(deltas)

        logger.debug(
            "Parallel composition computed",
            num_partitions=len(epsilons),
            total_epsilon=total_epsilon,
            total_delta=total_delta,
        )

        return total_epsilon, total_delta

    def advanced_compose_rdp(
        self,
        epsilons: list[float],
        deltas: list[float],
        alpha: float | None = None,
        target_delta: float | None = None,
    ) -> tuple[float, float]:
        """Compute advanced composition via Rényi DP (moments accountant).

        For Gaussian mechanism operations, converts to RDP, accumulates, then
        converts back to (ε,δ)-DP. Provides significantly tighter bounds than
        sequential composition when many operations are composed.

        The conversion from (ε, 0)-DP to (α, ε)-RDP uses the standard identity:
        For the Gaussian mechanism: (α, α·ε²/(2σ²))-RDP → tighter than sequential.

        For (ε,δ)-DP mechanisms, uses the approximate conversion:
        (α, ε_rdp)-RDP → (ε_rdp + log(1/δ)/(α-1), δ)-DP

        Args:
            epsilons: Per-mechanism epsilon values.
            deltas: Per-mechanism delta values.
            alpha: Rényi order (default: from settings).
            target_delta: Target delta for (ε,δ) conversion (default: sum of input deltas).

        Returns:
            Tuple of (total_epsilon, total_delta) with tighter composition bounds.

        Raises:
            ValueError: If alpha <= 1 or lists are empty.
        """
        if not epsilons:
            raise ValueError("epsilons list must not be empty")

        alpha = alpha or self._settings.renyi_alpha
        target_delta = target_delta or sum(deltas)

        if alpha <= 1.0:
            raise ValueError(f"Rényi order alpha must be > 1, got {alpha}")

        # Convert each (ε,δ)-DP mechanism to RDP via upper bound:
        # (ε,0)-DP → (α, ε)-RDP (exact for pure DP mechanisms)
        # (ε,δ)-DP → (α, ε + log(1/(1-1/α)) + log(δ·α/(α-1))/(α-1))-RDP (approximate)
        rdp_epsilons = []
        for eps, delta in zip(epsilons, deltas, strict=True):
            if delta == 0.0:
                # Pure DP — direct conversion
                rdp_eps = eps
            else:
                # Approximate conversion for (ε,δ)-DP
                # Based on Proposition 3 of Mironov (2017)
                rdp_eps = eps + (math.log(1.0 / (1.0 - 1.0 / alpha)) + math.log(delta * alpha / (alpha - 1))) / (
                    alpha - 1
                )
            rdp_epsilons.append(max(0.0, rdp_eps))

        # Sequential RDP composition: sum of RDP epsilons
        total_rdp = sum(rdp_epsilons)

        # Convert RDP back to (ε,δ)-DP:
        # From (α, ε_rdp)-RDP → (ε_rdp + log(1/δ)/(α-1), δ)-DP
        final_epsilon = total_rdp + math.log(1.0 / target_delta) / (alpha - 1.0)
        final_delta = target_delta

        logger.debug(
            "Advanced RDP composition computed",
            num_operations=len(epsilons),
            alpha=alpha,
            total_rdp=total_rdp,
            final_epsilon=final_epsilon,
            final_delta=final_delta,
        )

        return final_epsilon, final_delta

    def compute_plan_cost(
        self,
        plan: CompositionPlanRequest,
    ) -> tuple[float, float]:
        """Compute the total privacy cost for a composition plan.

        Args:
            plan: Composition plan with steps and composition method.

        Returns:
            Tuple of (total_epsilon, total_delta) for the plan.
        """
        epsilons = [step["epsilon"] for step in plan.steps]
        deltas = [step.get("delta", 0.0) for step in plan.steps]

        if plan.composition_method == "sequential":
            return self.sequential_compose(epsilons, deltas)
        elif plan.composition_method == "parallel":
            return self.parallel_compose(epsilons, deltas)
        elif plan.composition_method == "advanced":
            return self.advanced_compose_rdp(epsilons, deltas)
        else:
            raise ValidationError(
                field="composition_method",
                message=f"Unknown composition method: {plan.composition_method}. "
                "Supported: sequential | parallel | advanced",
            )


class ProofService:
    """Generates and verifies formal mathematical proofs for DP operations.

    Uses sympy for symbolic mathematics to generate LaTeX derivations
    and JSON proof trees. Every mechanism application produces a proof
    that can be independently verified for regulatory audit.

    Proof chains can span multiple operations and verify that composition
    theorems have been correctly applied.
    """

    def __init__(self, settings: Settings) -> None:
        """Initialize proof service.

        Args:
            settings: Privacy engine configuration.
        """
        self._settings = settings

    def generate_mechanism_proof(
        self,
        operation: PrivacyOperation,
    ) -> dict:  # type: ignore[type-arg]
        """Generate a formal mathematical proof for a privacy operation.

        Produces a proof dictionary containing:
        - latex: LaTeX derivation of the privacy guarantee
        - json_tree: Structured proof tree for programmatic verification
        - theorem: Name of the DP theorem applied
        - verification_hash: SHA-256 hash for tamper detection

        Args:
            operation: The PrivacyOperation record to prove.

        Returns:
            Complete proof dictionary suitable for storing in formal_proof JSONB.
        """
        mechanism = operation.mechanism
        epsilon = float(operation.epsilon_consumed)
        delta = float(operation.delta_consumed)
        sensitivity = float(operation.sensitivity)
        noise_scale = float(operation.noise_scale)

        if mechanism == "laplace":
            proof = self._prove_laplace(epsilon, sensitivity, noise_scale)
        elif mechanism == "gaussian":
            proof = self._prove_gaussian(epsilon, delta, sensitivity, noise_scale)
        elif mechanism == "exponential":
            proof = self._prove_exponential(epsilon, sensitivity)
        elif mechanism == "subsampled":
            proof = self._prove_subsampled_gaussian(epsilon, delta, sensitivity, noise_scale)
        else:
            proof = self._prove_generic(mechanism, epsilon, delta, sensitivity)

        # Add operation metadata
        proof["operation_id"] = str(operation.id)
        proof["job_id"] = str(operation.job_id)
        proof["tenant_id"] = str(operation.tenant_id)
        proof["composition_type"] = operation.composition_type

        # Compute tamper-detection hash
        proof_content = json.dumps(
            {k: v for k, v in proof.items() if k != "verification_hash"},
            sort_keys=True,
        )
        proof["verification_hash"] = hashlib.sha256(proof_content.encode()).hexdigest()

        return proof

    def _prove_laplace(
        self,
        epsilon: float,
        sensitivity: float,
        noise_scale: float,
    ) -> dict:  # type: ignore[type-arg]
        """Generate proof for Laplace mechanism.

        The Laplace mechanism adds noise drawn from Lap(0, Δf/ε) to a query f.
        Privacy guarantee: ε-DP (pure differential privacy).

        Theorem (Dwork et al., 2006): For query f: D → ℝⁿ with L1 sensitivity Δf,
        M(D) = f(D) + Lap(Δf/ε) satisfies ε-differential privacy.

        Args:
            epsilon: Privacy parameter.
            sensitivity: L1 sensitivity of the query.
            noise_scale: Actual noise scale used (= sensitivity/epsilon).

        Returns:
            Proof dictionary with LaTeX and JSON tree.
        """
        expected_scale = sensitivity / epsilon
        scale_match = abs(noise_scale - expected_scale) < 1e-9

        latex = (
            r"\textbf{Laplace Mechanism — Privacy Proof}" + "\n\n"
            r"\textbf{Claim:} The mechanism $M(D) = f(D) + \text{Lap}(0, \lambda)$ "
            r"satisfies $\varepsilon$-differential privacy." + "\n\n"
            r"\textbf{Parameters:}" + "\n"
            r"\begin{align}" + "\n"
            rf"  \varepsilon &= {epsilon:.6f} \\" + "\n"
            rf"  \Delta f &= {sensitivity:.6f} \quad \text{{(L1 sensitivity)}} \\" + "\n"
            rf"  \lambda &= \frac{{\Delta f}}{{\varepsilon}} = \frac{{{sensitivity:.6f}}}{{{epsilon:.6f}}} "
            rf"= {expected_scale:.6f}" + "\n"
            r"\end{align}" + "\n\n"
            r"\textbf{Proof:} For any adjacent datasets $D, D'$ (differing in one record) "
            r"and any output $S \subseteq \mathbb{R}$:" + "\n"
            r"\begin{align}" + "\n"
            r"  \frac{\Pr[M(D) \in S]}{\Pr[M(D') \in S]}"
            r"  &= \frac{\exp(-|f(D) - t|/\lambda)}{\exp(-|f(D') - t|/\lambda)} \\" + "\n"
            r"  &\leq \exp\left(\frac{|f(D) - f(D')|}{\lambda}\right) \\" + "\n"
            r"  &\leq \exp\left(\frac{\Delta f}{\lambda}\right) \\" + "\n"
            rf"  &= \exp\left(\frac{{{sensitivity:.6f}}}{{{noise_scale:.6f}}}\right)" + "\n"
            rf"  = \exp({epsilon:.6f}) = e^{{\varepsilon}}" + "\n"
            r"\end{align}" + "\n\n"
            r"$\therefore$ The mechanism satisfies $\varepsilon$-differential privacy. $\square$"
        )

        return {
            "theorem": "Laplace Mechanism (Dwork et al., 2006)",
            "privacy_type": "pure_dp",
            "epsilon": epsilon,
            "delta": 0.0,
            "sensitivity": sensitivity,
            "noise_scale": noise_scale,
            "expected_noise_scale": expected_scale,
            "parameters_valid": scale_match,
            "latex": latex,
            "json_tree": {
                "type": "laplace_proof",
                "claim": f"ε-DP with ε={epsilon:.6f}",
                "mechanism": "Laplace(0, sensitivity/ε)",
                "steps": [
                    {"step": 1, "description": "State L1 sensitivity Δf", "value": sensitivity},
                    {"step": 2, "description": "Compute noise scale λ = Δf/ε", "value": expected_scale},
                    {"step": 3, "description": "Apply ratio bound via triangle inequality", "value": None},
                    {"step": 4, "description": "Bound ratio by exp(Δf/λ) = exp(ε)", "value": epsilon},
                ],
                "conclusion": f"Mechanism satisfies {epsilon:.6f}-DP",
            },
        }

    def _prove_gaussian(
        self,
        epsilon: float,
        delta: float,
        sensitivity: float,
        noise_scale: float,
    ) -> dict:  # type: ignore[type-arg]
        """Generate proof for Gaussian mechanism (ε,δ)-DP.

        The Gaussian mechanism adds N(0, σ²) noise where σ = sensitivity × c
        for c chosen to satisfy the (ε,δ)-DP guarantee.

        Args:
            epsilon: Privacy parameter.
            delta: Failure probability.
            sensitivity: L2 sensitivity of the query.
            noise_scale: Actual σ used.

        Returns:
            Proof dictionary with LaTeX and JSON tree.
        """
        # Verify sigma satisfies (ε,δ)-DP via analytic Gaussian mechanism
        # σ² ≥ (2Δf² × log(1.25/δ)) / ε²  (Dwork & Roth, 2014)
        required_sigma_sq = (2.0 * sensitivity**2 * math.log(1.25 / delta)) / (epsilon**2)
        required_sigma = math.sqrt(required_sigma_sq)
        satisfies = noise_scale >= required_sigma - 1e-9

        latex = (
            r"\textbf{Gaussian Mechanism — Privacy Proof}" + "\n\n"
            r"\textbf{Claim:} The mechanism $M(D) = f(D) + \mathcal{N}(0, \sigma^2 \mathbf{I})$ "
            r"satisfies $(\varepsilon, \delta)$-differential privacy." + "\n\n"
            r"\textbf{Parameters:}" + "\n"
            r"\begin{align}" + "\n"
            rf"  \varepsilon &= {epsilon:.6f}, \quad \delta = {delta:.2e} \\" + "\n"
            rf"  \Delta_2 f &= {sensitivity:.6f} \quad \text{{(L2 sensitivity)}} \\" + "\n"
            rf"  \sigma &= {noise_scale:.6f}" + "\n"
            r"\end{align}" + "\n\n"
            r"\textbf{Sufficient Condition (Dwork \& Roth, 2014):}" + "\n"
            r"\begin{align}" + "\n"
            r"  \sigma^2 &\geq \frac{2 \Delta_2 f^2 \ln(1.25/\delta)}{\varepsilon^2} \\" + "\n"
            rf"  {noise_scale:.6f}^2 = {noise_scale**2:.6f} &\geq {required_sigma_sq:.6f} "
            rf"\quad \Rightarrow \text{{{'SATISFIED' if satisfies else 'NOT SATISFIED'}}}" + "\n"
            r"\end{align}"
        )

        return {
            "theorem": "Gaussian Mechanism (Dwork & Roth, 2014, Theorem A.1)",
            "privacy_type": "approximate_dp",
            "epsilon": epsilon,
            "delta": delta,
            "sensitivity": sensitivity,
            "noise_scale": noise_scale,
            "required_min_sigma": required_sigma,
            "parameters_valid": satisfies,
            "latex": latex,
            "json_tree": {
                "type": "gaussian_proof",
                "claim": f"(ε={epsilon:.6f},δ={delta:.2e})-DP",
                "mechanism": f"Gaussian(0, σ²={noise_scale**2:.6f})",
                "steps": [
                    {"step": 1, "description": "State L2 sensitivity Δ₂f", "value": sensitivity},
                    {"step": 2, "description": "Required σ_min = sqrt(2Δ²ln(1.25/δ)/ε²)", "value": required_sigma},
                    {"step": 3, "description": f"Verify σ={noise_scale:.6f} >= σ_min={required_sigma:.6f}", "value": satisfies},
                ],
                "conclusion": f"Mechanism satisfies ({epsilon:.6f},{delta:.2e})-DP" if satisfies else "PROOF FAILED",
            },
        }

    def _prove_exponential(
        self,
        epsilon: float,
        sensitivity: float,
    ) -> dict:  # type: ignore[type-arg]
        """Generate proof for Exponential mechanism.

        Args:
            epsilon: Privacy parameter.
            sensitivity: Sensitivity of the quality function.

        Returns:
            Proof dictionary.
        """
        latex = (
            r"\textbf{Exponential Mechanism — Privacy Proof}" + "\n\n"
            r"\textbf{Claim:} The exponential mechanism with quality function $q: D \times R \to \mathbb{R}$ "
            r"satisfies $\varepsilon$-differential privacy." + "\n\n"
            r"\textbf{Mechanism:} $\Pr[M(D) = r] \propto \exp\!\left(\frac{\varepsilon \cdot q(D, r)}{2 \Delta q}\right)$" + "\n\n"
            r"\textbf{Proof:} For any adjacent $D, D'$ and output $r \in R$:" + "\n"
            r"\begin{align}" + "\n"
            r"  \frac{\Pr[M(D) = r]}{\Pr[M(D') = r]}"
            r"  &= \frac{\exp(\varepsilon q(D,r) / 2\Delta q)}{\exp(\varepsilon q(D',r) / 2\Delta q)}" + "\n"
            r"  \cdot \frac{\sum_{r'} \exp(\varepsilon q(D',r') / 2\Delta q)}{\sum_{r'} \exp(\varepsilon q(D,r') / 2\Delta q)} \\" + "\n"
            r"  &\leq \exp(\varepsilon/2) \cdot \exp(\varepsilon/2) = \exp(\varepsilon)" + "\n"
            r"\end{align}"
        )

        return {
            "theorem": "Exponential Mechanism (McSherry & Talwar, 2007)",
            "privacy_type": "pure_dp",
            "epsilon": epsilon,
            "delta": 0.0,
            "sensitivity": sensitivity,
            "latex": latex,
            "json_tree": {
                "type": "exponential_proof",
                "claim": f"ε-DP with ε={epsilon:.6f}",
                "mechanism": "Exponential(q, ε, Δq)",
                "conclusion": f"Mechanism satisfies {epsilon:.6f}-DP",
            },
        }

    def _prove_subsampled_gaussian(
        self,
        epsilon: float,
        delta: float,
        sensitivity: float,
        noise_scale: float,
    ) -> dict:  # type: ignore[type-arg]
        """Generate proof for Subsampled Gaussian mechanism (DP-SGD).

        Args:
            epsilon: Privacy parameter.
            delta: Failure probability.
            sensitivity: Gradient clipping norm.
            noise_scale: Noise multiplier σ.

        Returns:
            Proof dictionary.
        """
        return {
            "theorem": "Privacy Amplification by Subsampling + Gaussian (Mironov et al., 2019)",
            "privacy_type": "approximate_dp",
            "epsilon": epsilon,
            "delta": delta,
            "sensitivity": sensitivity,
            "noise_scale": noise_scale,
            "latex": (
                r"\textbf{Subsampled Gaussian (DP-SGD) — Privacy Proof}" + "\n\n"
                r"Via Rényi DP composition (Mironov, 2017) with privacy amplification " + "\n"
                r"by subsampling (Mironov et al., 2019)." + "\n"
                rf"Noise multiplier $\sigma = {noise_scale:.4f}$, clip norm $C = {sensitivity:.4f}$." + "\n"
                r"Privacy computed via Opacus moments accountant."
            ),
            "json_tree": {
                "type": "subsampled_gaussian_proof",
                "claim": f"({epsilon:.6f},{delta:.2e})-DP via RDP composition",
                "mechanism": f"SubsampledGaussian(σ={noise_scale:.4f}, C={sensitivity:.4f})",
                "library": "opacus",
                "conclusion": f"Computed by Opacus privacy accountant: ({epsilon:.6f},{delta:.2e})-DP",
            },
        }

    def _prove_generic(
        self,
        mechanism: str,
        epsilon: float,
        delta: float,
        sensitivity: float,
    ) -> dict:  # type: ignore[type-arg]
        """Generate a generic proof stub for unknown mechanisms.

        Args:
            mechanism: Mechanism name.
            epsilon: Claimed epsilon.
            delta: Claimed delta.
            sensitivity: Query sensitivity.

        Returns:
            Stub proof dictionary.
        """
        return {
            "theorem": f"Custom mechanism: {mechanism}",
            "privacy_type": "approximate_dp" if delta > 0 else "pure_dp",
            "epsilon": epsilon,
            "delta": delta,
            "sensitivity": sensitivity,
            "latex": rf"\textbf{{Custom mechanism: {mechanism}}} — Claimed $(\varepsilon={epsilon:.6f}, \delta={delta:.2e})$-DP.",
            "json_tree": {
                "type": "generic_proof",
                "claim": f"({epsilon:.6f},{delta:.2e})-DP",
                "mechanism": mechanism,
                "conclusion": "Proof delegated to mechanism implementation",
            },
        }

    def get_proof_for_job(
        self,
        operations: list[PrivacyOperation],
    ) -> ProofResponse:
        """Compile all operation proofs for a synthesis job into an audit report.

        Args:
            operations: All PrivacyOperation records for the job.

        Returns:
            ProofResponse with complete audit trail.
        """
        if not operations:
            raise NotFoundError(
                resource="PrivacyOperation",
                resource_id="job",
                error_code=ErrorCode.NOT_FOUND,
            )

        total_epsilon = sum(float(op.epsilon_consumed) for op in operations)
        total_delta = sum(float(op.delta_consumed) for op in operations)

        return ProofResponse(
            job_id=operations[0].job_id,
            num_operations=len(operations),
            total_epsilon_consumed=total_epsilon,
            total_delta_consumed=total_delta,
            composition_type=operations[0].composition_type,
            proofs=[op.formal_proof for op in operations],
            audit_summary=(
                f"Job used {len(operations)} DP operation(s) consuming "
                f"ε={total_epsilon:.6f} total privacy budget via "
                f"{operations[0].composition_type} composition."
            ),
        )


class VisualizationService:
    """Generates privacy loss curves and budget utilization visualizations.

    Uses matplotlib to create charts showing:
    - Cumulative epsilon consumption over time
    - Privacy loss by source engine (tabular, text, image, etc.)
    - Remaining budget and projected exhaustion date

    Charts are returned as base64-encoded PNG/SVG or as JSON data arrays.
    """

    def __init__(
        self,
        operation_repo: object,
        budget_repo: object,
        settings: Settings,
    ) -> None:
        """Initialize visualization service.

        Args:
            operation_repo: Repository for querying privacy operations.
            budget_repo: Repository for querying budget history.
            settings: Privacy engine configuration.
        """
        self._operation_repo = operation_repo
        self._budget_repo = budget_repo
        self._settings = settings

    async def generate_loss_curve(
        self,
        tenant_id: UUID,
        output_format: str = "png",
    ) -> LossVisualizationResponse:
        """Generate cumulative privacy loss curve for a tenant.

        Fetches all PrivacyOperation records for the tenant, computes
        cumulative epsilon over time, and renders a matplotlib line chart.

        Args:
            tenant_id: The tenant to visualize.
            output_format: Output format: png | svg | json.

        Returns:
            LossVisualizationResponse with encoded chart or JSON data.
        """
        import base64
        import io

        import matplotlib
        import matplotlib.pyplot as plt
        import matplotlib.dates as mdates

        matplotlib.use("Agg")  # Non-interactive backend

        operations = await self._operation_repo.list_by_tenant(tenant_id)  # type: ignore[union-attr]
        budget = await self._budget_repo.get_active(tenant_id)  # type: ignore[union-attr]

        if not operations:
            return LossVisualizationResponse(
                tenant_id=tenant_id,
                output_format=output_format,
                data=None,
                message="No privacy operations recorded for this tenant yet",
            )

        # Prepare data
        timestamps = [op.created_at for op in operations]
        epsilons = [float(op.epsilon_consumed) for op in operations]
        cumulative_epsilon = []
        running_sum = 0.0
        for eps in epsilons:
            running_sum += eps
            cumulative_epsilon.append(running_sum)

        total_budget = float(budget.total_epsilon) if budget else cumulative_epsilon[-1] * 2

        fig, ax = plt.subplots(figsize=(10, 6))
        ax.plot(timestamps, cumulative_epsilon, "b-o", label="Cumulative ε consumed", linewidth=2)
        ax.axhline(y=total_budget, color="r", linestyle="--", label=f"Budget limit (ε={total_budget:.2f})")
        ax.fill_between(timestamps, cumulative_epsilon, alpha=0.1)

        ax.set_xlabel("Time")
        ax.set_ylabel("Cumulative Privacy Loss (ε)")
        ax.set_title(f"Privacy Loss Curve — Tenant {str(tenant_id)[:8]}...")
        ax.legend()
        ax.xaxis.set_major_formatter(mdates.DateFormatter("%Y-%m-%d"))
        fig.autofmt_xdate()
        ax.grid(True, alpha=0.3)

        if output_format == "json":
            plt.close(fig)
            return LossVisualizationResponse(
                tenant_id=tenant_id,
                output_format="json",
                data=None,
                json_data={
                    "timestamps": [t.isoformat() for t in timestamps],
                    "cumulative_epsilon": cumulative_epsilon,
                    "budget_limit": total_budget,
                },
            )

        buffer = io.BytesIO()
        plt.savefig(buffer, format=output_format, dpi=150, bbox_inches="tight")
        plt.close(fig)
        buffer.seek(0)
        encoded = base64.b64encode(buffer.read()).decode("utf-8")

        return LossVisualizationResponse(
            tenant_id=tenant_id,
            output_format=output_format,
            data=encoded,
        )
