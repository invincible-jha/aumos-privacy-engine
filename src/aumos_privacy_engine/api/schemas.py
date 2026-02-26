"""Pydantic request/response schemas for the privacy engine API.

All schemas use strict validation with Field constraints to enforce
valid ε/δ ranges at the API boundary. Never accept out-of-bounds
privacy parameters — validate early and fail loudly.
"""

import uuid
from datetime import datetime
from typing import Any, Literal

from pydantic import BaseModel, Field, model_validator


class BudgetAllocateRequest(BaseModel):
    """Request to allocate differential privacy budget for a synthesis operation.

    Attributes:
        job_id: Unique ID of the synthesis job requesting budget.
        source_engine: Which synthesis engine is making the request.
        epsilon_requested: Epsilon to consume from the tenant's budget.
        delta_requested: Delta to consume (0 for pure DP mechanisms).
        mechanism: Intended DP mechanism to be applied.
        composition_type: How this operation composes with others in the job.
        query_description: Human-readable description of what is being privatized.
    """

    job_id: uuid.UUID = Field(description="Unique ID of the synthesis job requesting budget")
    source_engine: Literal["tabular", "text", "image", "audio", "video", "healthcare"] = Field(
        description="Synthesis engine making the request"
    )
    epsilon_requested: float = Field(
        gt=0.0,
        le=10.0,
        description="Epsilon to consume from tenant budget (must be > 0)",
    )
    delta_requested: float = Field(
        default=0.0,
        ge=0.0,
        lt=1.0,
        description="Delta to consume (0.0 for pure DP mechanisms like Laplace)",
    )
    mechanism: Literal["laplace", "gaussian", "exponential", "subsampled"] = Field(
        description="DP mechanism to be applied with this budget allocation"
    )
    composition_type: Literal["sequential", "parallel", "advanced"] = Field(
        default="sequential",
        description="Composition theorem to apply for this operation",
    )
    query_description: str = Field(
        default="",
        max_length=500,
        description="Human-readable description of the query being privatized",
    )


class BudgetResponse(BaseModel):
    """Current state of a tenant's differential privacy budget.

    Attributes:
        budget_id: Unique ID of the PrivacyBudget record.
        tenant_id: The owning tenant.
        total_epsilon: Total ε budget allocated for this period.
        used_epsilon: Total ε consumed so far.
        remaining_epsilon: ε still available (total - used).
        total_delta: Total δ budget.
        used_delta: Total δ consumed.
        remaining_delta: δ still available.
        epsilon_utilization_pct: Percentage of ε budget consumed (0-100).
        period_start: Start of the current budget period.
        period_end: End of the current budget period.
        auto_renew: Whether budget auto-renews after period_end.
    """

    budget_id: uuid.UUID
    tenant_id: uuid.UUID
    total_epsilon: float
    used_epsilon: float
    remaining_epsilon: float
    total_delta: float
    used_delta: float
    remaining_delta: float
    epsilon_utilization_pct: float
    period_start: datetime
    period_end: datetime
    auto_renew: bool


class MechanismApplyRequest(BaseModel):
    """Request to apply a differential privacy mechanism to numerical data.

    Attributes:
        mechanism: DP mechanism to apply.
        epsilon: Privacy budget to spend for this operation.
        delta: Failure probability (0 for pure DP mechanisms).
        sensitivity: L1 sensitivity (Laplace/Exponential) or L2 (Gaussian/Subsampled).
        data: Numerical values to privatize.
        data_type: Type of data being privatized (for proof documentation).
        composition_type: How this composes with prior operations in the job.
        source_engine: Which synthesis engine initiated this request.
        job_id: The synthesis job this operation belongs to.
        query_description: Description of the query for audit trail.
    """

    mechanism: Literal["laplace", "gaussian", "exponential", "subsampled"] = Field(
        description="DP mechanism to apply"
    )
    epsilon: float = Field(gt=0.0, le=10.0, description="Privacy budget for this operation (ε > 0)")
    delta: float = Field(
        default=0.0,
        ge=0.0,
        lt=1.0,
        description="Failure probability δ ∈ [0, 1) — use 0 for pure DP",
    )
    sensitivity: float = Field(
        gt=0.0,
        description="Query sensitivity (L1 for Laplace/Exponential, L2 for Gaussian/Subsampled)",
    )
    data: list[float] = Field(
        min_length=1,
        max_length=100000,
        description="Numerical values to privatize",
    )
    data_type: str = Field(
        default="numerical",
        description="Type of data: numerical | categorical | gradient",
    )
    composition_type: Literal["sequential", "parallel", "advanced"] = Field(
        default="sequential",
        description="Composition theorem applied",
    )
    source_engine: Literal["tabular", "text", "image", "audio", "video", "healthcare"] = Field(
        description="Synthesis engine making the request"
    )
    job_id: uuid.UUID = Field(description="Synthesis job this operation belongs to")
    query_description: str = Field(
        default="",
        max_length=500,
        description="Description of the query for audit trail",
    )

    @model_validator(mode="after")
    def validate_delta_for_mechanism(self) -> "MechanismApplyRequest":
        """Validate that delta is appropriate for the requested mechanism.

        Laplace and Exponential are pure DP and require delta=0.
        Gaussian and Subsampled are (ε,δ)-DP and require delta>0.

        Returns:
            Self after validation.

        Raises:
            ValueError: If delta is inconsistent with the mechanism.
        """
        if self.mechanism in ("laplace", "exponential") and self.delta > 0:
            raise ValueError(
                f"Mechanism '{self.mechanism}' is pure DP and requires delta=0.0, "
                f"got delta={self.delta}"
            )
        if self.mechanism in ("gaussian", "subsampled") and self.delta == 0:
            raise ValueError(
                f"Mechanism '{self.mechanism}' is (ε,δ)-DP and requires delta>0, "
                f"got delta=0. Typical values: 1e-5 to 1e-7."
            )
        return self


class MechanismApplyResponse(BaseModel):
    """Result of applying a DP mechanism to data.

    Attributes:
        operation_id: Unique ID of the recorded PrivacyOperation.
        mechanism: Mechanism that was applied.
        privatized_data: Noisy output values.
        epsilon_consumed: Actual epsilon consumed (may differ slightly from requested).
        delta_consumed: Actual delta consumed.
        noise_scale: Noise parameter used (λ for Laplace, σ for Gaussian).
        proof_pending: Whether the formal proof is being generated asynchronously.
    """

    operation_id: uuid.UUID
    mechanism: str
    privatized_data: list[float]
    epsilon_consumed: float
    delta_consumed: float
    noise_scale: float
    proof_pending: bool = True


class ProofResponse(BaseModel):
    """Formal mathematical proof for all DP operations in a synthesis job.

    Suitable for regulatory audit and compliance reporting.

    Attributes:
        job_id: The synthesis job these proofs cover.
        num_operations: Number of DP operations in the job.
        total_epsilon_consumed: Total epsilon consumed across all operations.
        total_delta_consumed: Total delta consumed across all operations.
        composition_type: Composition theorem applied.
        proofs: List of formal proof dictionaries (LaTeX + JSON tree).
        audit_summary: Human-readable audit summary.
    """

    job_id: uuid.UUID
    num_operations: int
    total_epsilon_consumed: float
    total_delta_consumed: float
    composition_type: str
    proofs: list[dict[str, Any]]
    audit_summary: str


class CompositionStepSchema(BaseModel):
    """A single step in a multi-step composition plan.

    Attributes:
        step_name: Human-readable name for this step.
        mechanism: DP mechanism to apply.
        epsilon: Epsilon for this step.
        delta: Delta for this step.
        sensitivity: Query sensitivity.
        data_description: What data this step privatizes.
    """

    step_name: str = Field(max_length=100)
    mechanism: Literal["laplace", "gaussian", "exponential", "subsampled"]
    epsilon: float = Field(gt=0.0, le=10.0)
    delta: float = Field(default=0.0, ge=0.0, lt=1.0)
    sensitivity: float = Field(gt=0.0)
    data_description: str = Field(default="", max_length=200)


class CompositionPlanRequest(BaseModel):
    """Request to plan a multi-step DP composition.

    Used by synthesis engines to forecast their total privacy cost
    before committing budget, and to validate feasibility.

    Attributes:
        name: Human-readable name for this plan.
        steps: Ordered sequence of planned mechanism applications.
        composition_method: How to compute the total cost.
    """

    name: str = Field(max_length=255, description="Name for this composition plan")
    steps: list[CompositionStepSchema] = Field(
        min_length=1,
        max_length=100,
        description="Ordered list of planned mechanism applications",
    )
    composition_method: Literal["sequential", "parallel", "advanced"] = Field(
        default="sequential",
        description="Composition theorem to apply: sequential | parallel | advanced",
    )


class CompositionPlanResponse(BaseModel):
    """Response with composition plan details and cost estimates.

    Attributes:
        plan_id: Unique ID of the created CompositionPlan record.
        name: Plan name.
        num_steps: Number of steps in the plan.
        total_epsilon_estimate: Estimated total epsilon cost.
        total_delta_estimate: Estimated total delta cost.
        status: Current plan status.
        composition_method: Method used to compute estimates.
        is_feasible: Whether the plan fits within the tenant's remaining budget.
        remaining_budget_after: Estimated remaining budget if plan executes.
    """

    plan_id: uuid.UUID
    name: str
    num_steps: int
    total_epsilon_estimate: float
    total_delta_estimate: float
    status: str
    composition_method: str
    is_feasible: bool
    remaining_budget_after: float | None = None


class LossVisualizationResponse(BaseModel):
    """Response containing a privacy loss curve visualization.

    Attributes:
        tenant_id: The tenant whose data is visualized.
        output_format: png | svg | json.
        data: Base64-encoded image data (None for json format).
        json_data: Chart data as JSON (None for image formats).
        message: Optional informational message (e.g., no data available).
    """

    tenant_id: uuid.UUID
    output_format: str
    data: str | None = None
    json_data: dict[str, Any] | None = None
    message: str | None = None
