"""FastAPI router for the privacy engine API.

Routes are thin — they delegate all business logic to services.
Authentication and tenant context are injected via aumos-common middleware.

All endpoints follow the convention:
- GET for read operations
- POST for state-changing or computation-heavy operations
- Standard error responses via aumos-common ErrorResponse
"""

import uuid
from typing import Annotated

from fastapi import APIRouter, Depends, Query
from sqlalchemy.ext.asyncio import AsyncSession

from aumos_common.auth import TenantContext, get_current_tenant
from aumos_common.database import get_db_session
from aumos_common.observability import get_logger

from aumos_privacy_engine.adapters.repositories import (
    BudgetRepository,
    CompositionPlanRepository,
    OperationRepository,
)
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
from aumos_privacy_engine.core.services import (
    BudgetService,
    CompositionService,
    MechanismService,
    ProofService,
    VisualizationService,
)
from aumos_privacy_engine.settings import Settings

logger = get_logger(__name__)
router = APIRouter(prefix="/privacy", tags=["Privacy Engine"])

# Settings singleton for dependency injection
_settings = Settings()


def get_settings() -> Settings:
    """Provide Settings instance for dependency injection.

    Returns:
        Privacy engine settings.
    """
    return _settings


async def get_budget_service(
    session: Annotated[AsyncSession, Depends(get_db_session)],
    settings: Annotated[Settings, Depends(get_settings)],
) -> BudgetService:
    """Build and return a BudgetService with injected dependencies.

    Args:
        session: Async database session from aumos-common.
        settings: Privacy engine settings.

    Returns:
        Configured BudgetService instance.
    """
    budget_repo = BudgetRepository(session)
    operation_repo = OperationRepository(session)
    # Publisher injected from app state in production; stubbed here for simplicity
    from aumos_privacy_engine.adapters.kafka import get_publisher

    publisher = await get_publisher()
    return BudgetService(
        budget_repo=budget_repo,
        operation_repo=operation_repo,
        publisher=publisher,
        settings=settings,
    )


async def get_mechanism_service(
    session: Annotated[AsyncSession, Depends(get_db_session)],
    settings: Annotated[Settings, Depends(get_settings)],
) -> MechanismService:
    """Build MechanismService with all mechanism adapters injected.

    Args:
        session: Async database session.
        settings: Privacy engine settings.

    Returns:
        Configured MechanismService instance.
    """
    from aumos_privacy_engine.adapters.dp_mechanisms.laplace import LaplaceMechanism
    from aumos_privacy_engine.adapters.dp_mechanisms.gaussian import GaussianMechanism
    from aumos_privacy_engine.adapters.dp_mechanisms.exponential import ExponentialMechanism
    from aumos_privacy_engine.adapters.dp_mechanisms.subsampled import SubsampledGaussianMechanism
    from aumos_privacy_engine.adapters.kafka import get_publisher

    publisher = await get_publisher()
    return MechanismService(
        laplace_mechanism=LaplaceMechanism(),
        gaussian_mechanism=GaussianMechanism(),
        exponential_mechanism=ExponentialMechanism(),
        subsampled_mechanism=SubsampledGaussianMechanism(),
        operation_repo=OperationRepository(session),
        publisher=publisher,
        settings=settings,
    )


async def get_composition_service(
    settings: Annotated[Settings, Depends(get_settings)],
) -> CompositionService:
    """Build CompositionService.

    Args:
        settings: Privacy engine settings.

    Returns:
        Configured CompositionService instance.
    """
    return CompositionService(settings=settings)


async def get_proof_service(
    settings: Annotated[Settings, Depends(get_settings)],
) -> ProofService:
    """Build ProofService.

    Args:
        settings: Privacy engine settings.

    Returns:
        Configured ProofService instance.
    """
    return ProofService(settings=settings)


async def get_visualization_service(
    session: Annotated[AsyncSession, Depends(get_db_session)],
    settings: Annotated[Settings, Depends(get_settings)],
) -> VisualizationService:
    """Build VisualizationService.

    Args:
        session: Async database session.
        settings: Privacy engine settings.

    Returns:
        Configured VisualizationService instance.
    """
    return VisualizationService(
        operation_repo=OperationRepository(session),
        budget_repo=BudgetRepository(session),
        settings=settings,
    )


@router.post(
    "/budget/allocate",
    response_model=BudgetResponse,
    summary="Allocate differential privacy budget for a synthesis operation",
    description=(
        "Atomically checks and reserves epsilon/delta from the tenant's privacy budget. "
        "Must be called by synthesis engines before applying any DP mechanism. "
        "Raises 422 if budget is insufficient or parameters are invalid."
    ),
)
async def allocate_budget(
    request: BudgetAllocateRequest,
    tenant: Annotated[TenantContext, Depends(get_current_tenant)],
    budget_service: Annotated[BudgetService, Depends(get_budget_service)],
) -> BudgetResponse:
    """Allocate DP budget for a synthesis operation.

    Args:
        request: Budget allocation request with epsilon, delta, source engine.
        tenant: Current tenant context (injected by auth middleware).
        budget_service: Budget management service.

    Returns:
        Updated budget utilization after allocation.
    """
    logger.info(
        "Budget allocation request",
        tenant_id=str(tenant.tenant_id),
        job_id=str(request.job_id),
        epsilon_requested=request.epsilon_requested,
        source_engine=request.source_engine,
    )
    return await budget_service.allocate(tenant_id=tenant.tenant_id, request=request)


@router.get(
    "/budget/{tenant_id}",
    response_model=BudgetResponse,
    summary="Get current privacy budget utilization for a tenant",
)
async def get_budget_utilization(
    tenant_id: uuid.UUID,
    tenant: Annotated[TenantContext, Depends(get_current_tenant)],
    budget_service: Annotated[BudgetService, Depends(get_budget_service)],
) -> BudgetResponse:
    """Get current budget utilization for a tenant.

    Args:
        tenant_id: The tenant to query (must match authenticated tenant).
        tenant: Current tenant context.
        budget_service: Budget management service.

    Returns:
        Current budget utilization statistics.
    """
    return await budget_service.get_utilization(tenant_id=tenant_id)


@router.post(
    "/mechanism/apply",
    response_model=MechanismApplyResponse,
    summary="Apply a differential privacy mechanism to numerical data",
    description=(
        "Applies one of the supported DP mechanisms (Laplace, Gaussian, Exponential, "
        "or Subsampled Gaussian) to the provided data. Records the operation and "
        "initiates formal proof generation. Budget must be pre-allocated via /budget/allocate."
    ),
)
async def apply_mechanism(
    request: MechanismApplyRequest,
    tenant: Annotated[TenantContext, Depends(get_current_tenant)],
    budget_service: Annotated[BudgetService, Depends(get_budget_service)],
    mechanism_service: Annotated[MechanismService, Depends(get_mechanism_service)],
) -> MechanismApplyResponse:
    """Apply a DP mechanism to data.

    Args:
        request: Mechanism application request with data, mechanism type, and params.
        tenant: Current tenant context.
        budget_service: Budget management service (for budget lookup).
        mechanism_service: Mechanism application service.

    Returns:
        Privatized data with operation ID and proof status.
    """
    logger.info(
        "Mechanism apply request",
        tenant_id=str(tenant.tenant_id),
        mechanism=request.mechanism,
        epsilon=request.epsilon,
        job_id=str(request.job_id),
        data_length=len(request.data),
    )

    # Get active budget for the tenant
    budget = await budget_service.get_active_budget(tenant.tenant_id)

    return await mechanism_service.apply(
        budget_id=budget.id,
        tenant_id=tenant.tenant_id,
        request=request,
    )


@router.get(
    "/proof/{job_id}",
    response_model=ProofResponse,
    summary="Get formal mathematical proof for a synthesis job",
    description=(
        "Returns the complete formal proof chain for all DP operations associated "
        "with a synthesis job. The proof includes LaTeX derivations and JSON proof "
        "trees for each mechanism application. Used for regulatory audit."
    ),
)
async def get_proof(
    job_id: uuid.UUID,
    tenant: Annotated[TenantContext, Depends(get_current_tenant)],
    session: Annotated[AsyncSession, Depends(get_db_session)],
    proof_service: Annotated[ProofService, Depends(get_proof_service)],
    settings: Annotated[Settings, Depends(get_settings)],
) -> ProofResponse:
    """Get the formal proof for a synthesis job's privacy operations.

    Args:
        job_id: The synthesis job to retrieve proofs for.
        tenant: Current tenant context.
        session: Database session.
        proof_service: Proof generation service.
        settings: Privacy engine settings.

    Returns:
        Complete ProofResponse with all operation proofs.
    """
    operation_repo = OperationRepository(session)
    operations = await operation_repo.list_by_job(job_id=job_id, tenant_id=tenant.tenant_id)
    return proof_service.get_proof_for_job(operations=operations)


@router.get(
    "/loss/visualize",
    response_model=LossVisualizationResponse,
    summary="Get privacy loss curve visualization for a tenant",
)
async def visualize_loss(
    tenant: Annotated[TenantContext, Depends(get_current_tenant)],
    viz_service: Annotated[VisualizationService, Depends(get_visualization_service)],
    output_format: Annotated[
        str,
        Query(description="Output format: png | svg | json", pattern="^(png|svg|json)$"),
    ] = "png",
) -> LossVisualizationResponse:
    """Generate privacy loss curve visualization for a tenant.

    Args:
        tenant: Current tenant context.
        viz_service: Visualization service.
        output_format: Desired output format (png, svg, or json).

    Returns:
        LossVisualizationResponse with encoded chart or JSON data.
    """
    return await viz_service.generate_loss_curve(
        tenant_id=tenant.tenant_id,
        output_format=output_format,
    )


@router.post(
    "/composition/plan",
    response_model=CompositionPlanResponse,
    summary="Plan a multi-step DP composition and estimate total privacy cost",
    description=(
        "Allows synthesis engines to forecast the total privacy cost of a multi-step "
        "workflow before committing budget. Validates feasibility against the tenant's "
        "current budget and stores the plan for later execution tracking."
    ),
)
async def create_composition_plan(
    request: CompositionPlanRequest,
    tenant: Annotated[TenantContext, Depends(get_current_tenant)],
    session: Annotated[AsyncSession, Depends(get_db_session)],
    composition_service: Annotated[CompositionService, Depends(get_composition_service)],
    budget_service: Annotated[BudgetService, Depends(get_budget_service)],
    settings: Annotated[Settings, Depends(get_settings)],
) -> CompositionPlanResponse:
    """Create a DP composition plan with cost estimates.

    Args:
        request: Composition plan with steps and composition method.
        tenant: Current tenant context.
        session: Database session.
        composition_service: Composition theorem computation service.
        budget_service: Budget service for feasibility check.
        settings: Privacy engine settings.

    Returns:
        CompositionPlanResponse with estimated costs and feasibility.
    """
    # Compute total cost estimate
    total_epsilon, total_delta = composition_service.compute_plan_cost(request)

    # Check feasibility against current budget
    is_feasible = True
    remaining_after: float | None = None
    try:
        budget_response = await budget_service.get_utilization(tenant.tenant_id)
        remaining = budget_response.remaining_epsilon
        is_feasible = total_epsilon <= remaining
        remaining_after = remaining - total_epsilon if is_feasible else None
    except Exception:
        is_feasible = False

    # Persist the plan
    plan_repo = CompositionPlanRepository(session)
    from decimal import Decimal

    plan = await plan_repo.create(
        tenant_id=tenant.tenant_id,
        name=request.name,
        steps=[step.model_dump() for step in request.steps],
        total_epsilon_estimate=Decimal(str(total_epsilon)),
        total_delta_estimate=Decimal(str(total_delta)),
        status="pending",
        composition_method=request.composition_method,
    )

    logger.info(
        "Composition plan created",
        plan_id=str(plan.id),
        tenant_id=str(tenant.tenant_id),
        total_epsilon_estimate=total_epsilon,
        is_feasible=is_feasible,
        num_steps=len(request.steps),
    )

    return CompositionPlanResponse(
        plan_id=plan.id,
        name=plan.name,
        num_steps=len(request.steps),
        total_epsilon_estimate=total_epsilon,
        total_delta_estimate=total_delta,
        status=plan.status,
        composition_method=request.composition_method,
        is_feasible=is_feasible,
        remaining_budget_after=remaining_after,
    )


@router.get(
    "/composition/{plan_id}",
    response_model=CompositionPlanResponse,
    summary="Get composition plan status",
)
async def get_composition_plan(
    plan_id: uuid.UUID,
    tenant: Annotated[TenantContext, Depends(get_current_tenant)],
    session: Annotated[AsyncSession, Depends(get_db_session)],
) -> CompositionPlanResponse:
    """Get the current status of a composition plan.

    Args:
        plan_id: UUID of the composition plan.
        tenant: Current tenant context.
        session: Database session.

    Returns:
        CompositionPlanResponse with current status.
    """
    from aumos_common.errors import NotFoundError, ErrorCode

    plan_repo = CompositionPlanRepository(session)
    plan = await plan_repo.get_by_id(plan_id=plan_id, tenant_id=tenant.tenant_id)

    if plan is None:
        raise NotFoundError(
            resource="CompositionPlan",
            resource_id=str(plan_id),
            error_code=ErrorCode.NOT_FOUND,
        )

    return CompositionPlanResponse(
        plan_id=plan.id,
        name=plan.name,
        num_steps=len(plan.steps),
        total_epsilon_estimate=float(plan.total_epsilon_estimate),
        total_delta_estimate=float(plan.total_delta_estimate),
        status=plan.status,
        composition_method=plan.composition_method,
        is_feasible=plan.status not in ("failed", "cancelled"),
    )
