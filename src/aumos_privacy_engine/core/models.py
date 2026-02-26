"""SQLAlchemy ORM models for the privacy engine.

All models use the `prv_` table prefix and extend AumOSModel for
automatic tenant_id, id (UUID), created_at, and updated_at fields.
PrivacyOperation is append-only — no update or delete operations.
"""

import uuid
from datetime import datetime
from decimal import Decimal

from sqlalchemy import Boolean, DateTime, ForeignKey, Numeric, String, Text
from sqlalchemy.dialects.postgresql import JSONB, UUID
from sqlalchemy.orm import Mapped, mapped_column, relationship

from aumos_common.database import AumOSModel


class PrivacyBudget(AumOSModel):
    """Per-tenant differential privacy budget with ε/δ tracking.

    Budget tracks cumulative privacy spend across all synthesis operations
    for a tenant within a defined time period. Budget auto-renewal creates
    a new PrivacyBudget record (never resets used values in place — append-only
    audit trail via PrivacyOperation).

    Attributes:
        tenant_id: Owning tenant UUID (inherited from AumOSModel).
        total_epsilon: Total ε budget allocated for this period.
        used_epsilon: Sum of epsilon consumed by all operations (computed on insert).
        total_delta: Total δ budget allocated for this period.
        used_delta: Sum of delta consumed by all operations.
        period_start: Start of the budget period (UTC).
        period_end: End of the budget period (UTC).
        auto_renew: Whether the budget renews automatically after period_end.
        is_active: Whether this is the current active budget for the tenant.
    """

    __tablename__ = "prv_privacy_budgets"

    # Budget limits
    total_epsilon: Mapped[Decimal] = mapped_column(
        Numeric(precision=10, scale=6),
        nullable=False,
        default=Decimal("10.0"),
        comment="Total epsilon (ε) budget for this period",
    )
    used_epsilon: Mapped[Decimal] = mapped_column(
        Numeric(precision=10, scale=6),
        nullable=False,
        default=Decimal("0.0"),
        comment="Epsilon consumed so far — updated on each PrivacyOperation insert",
    )
    total_delta: Mapped[Decimal] = mapped_column(
        Numeric(precision=15, scale=10),
        nullable=False,
        default=Decimal("0.00001"),
        comment="Total delta (δ) budget for this period",
    )
    used_delta: Mapped[Decimal] = mapped_column(
        Numeric(precision=15, scale=10),
        nullable=False,
        default=Decimal("0.0"),
        comment="Delta consumed so far — updated on each PrivacyOperation insert",
    )

    # Budget period
    period_start: Mapped[datetime] = mapped_column(
        DateTime(timezone=True),
        nullable=False,
        comment="Start of the budget period (UTC)",
    )
    period_end: Mapped[datetime] = mapped_column(
        DateTime(timezone=True),
        nullable=False,
        comment="End of the budget period (UTC)",
    )
    auto_renew: Mapped[bool] = mapped_column(
        Boolean,
        nullable=False,
        default=True,
        comment="Whether to automatically create a new budget after period_end",
    )
    is_active: Mapped[bool] = mapped_column(
        Boolean,
        nullable=False,
        default=True,
        comment="Whether this is the current active budget for the tenant",
    )

    # Relationships
    operations: Mapped[list["PrivacyOperation"]] = relationship(
        "PrivacyOperation",
        back_populates="budget",
        cascade="all, delete-orphan",
    )

    @property
    def remaining_epsilon(self) -> Decimal:
        """Remaining epsilon budget.

        Returns:
            Decimal representing how much epsilon is still available.
        """
        return self.total_epsilon - self.used_epsilon

    @property
    def remaining_delta(self) -> Decimal:
        """Remaining delta budget.

        Returns:
            Decimal representing how much delta is still available.
        """
        return self.total_delta - self.used_delta

    @property
    def epsilon_utilization_pct(self) -> float:
        """Percentage of epsilon budget consumed.

        Returns:
            Float from 0.0 to 100.0 representing utilization percentage.
        """
        if self.total_epsilon == 0:
            return 100.0
        return float(self.used_epsilon / self.total_epsilon * 100)


class PrivacyOperation(AumOSModel):
    """Immutable record of a single differential privacy mechanism application.

    This is an APPEND-ONLY model. Records are never updated or deleted.
    The audit trail of all privacy operations is permanent and required
    for regulatory compliance.

    Attributes:
        tenant_id: Owning tenant UUID (inherited from AumOSModel).
        budget_id: FK to the PrivacyBudget record that was consumed.
        mechanism: DP mechanism used (laplace/gaussian/exponential/subsampled).
        epsilon_consumed: Epsilon consumed by this operation.
        delta_consumed: Delta consumed by this operation (0 for pure DP).
        composition_type: How this operation composes with others.
        formal_proof: JSON + LaTeX proof of the privacy guarantee.
        source_engine: Which synthesis engine initiated this operation.
        job_id: ID of the synthesis job that triggered this operation.
        sensitivity: L1/L2 sensitivity of the query (stored for proof).
        noise_scale: Actual noise scale applied (stored for proof).
    """

    __tablename__ = "prv_privacy_operations"

    # Foreign key to budget
    budget_id: Mapped[uuid.UUID] = mapped_column(
        UUID(as_uuid=True),
        ForeignKey("prv_privacy_budgets.id", ondelete="RESTRICT"),
        nullable=False,
        index=True,
        comment="Budget record that was consumed by this operation",
    )

    # Mechanism details
    mechanism: Mapped[str] = mapped_column(
        String(50),
        nullable=False,
        comment="DP mechanism: laplace | gaussian | exponential | subsampled",
    )
    epsilon_consumed: Mapped[Decimal] = mapped_column(
        Numeric(precision=10, scale=6),
        nullable=False,
        comment="Epsilon consumed by this mechanism application",
    )
    delta_consumed: Mapped[Decimal] = mapped_column(
        Numeric(precision=15, scale=10),
        nullable=False,
        default=Decimal("0.0"),
        comment="Delta consumed (0 for pure DP mechanisms like Laplace)",
    )

    # Composition metadata
    composition_type: Mapped[str] = mapped_column(
        String(30),
        nullable=False,
        default="sequential",
        comment="Composition theorem applied: sequential | parallel | advanced",
    )

    # Formal proof (LaTeX + JSON proof tree)
    formal_proof: Mapped[dict] = mapped_column(  # type: ignore[type-arg]
        JSONB,
        nullable=False,
        default=dict,
        comment="Formal mathematical proof of the privacy guarantee (LaTeX + JSON)",
    )

    # Source tracking
    source_engine: Mapped[str] = mapped_column(
        String(50),
        nullable=False,
        comment="Synthesis engine that triggered this operation: tabular|text|image|audio|video|healthcare",
    )
    job_id: Mapped[uuid.UUID] = mapped_column(
        UUID(as_uuid=True),
        nullable=False,
        index=True,
        comment="ID of the synthesis job that triggered this operation",
    )

    # Mechanism parameters (stored for proof reconstruction)
    sensitivity: Mapped[Decimal] = mapped_column(
        Numeric(precision=10, scale=6),
        nullable=False,
        comment="Query sensitivity (L1 for Laplace, L2 for Gaussian)",
    )
    noise_scale: Mapped[Decimal] = mapped_column(
        Numeric(precision=10, scale=6),
        nullable=False,
        comment="Actual noise scale (lambda for Laplace, sigma for Gaussian)",
    )
    query_description: Mapped[str] = mapped_column(
        Text,
        nullable=False,
        default="",
        comment="Human-readable description of the query being privatized",
    )

    # Relationships
    budget: Mapped["PrivacyBudget"] = relationship("PrivacyBudget", back_populates="operations")


class CompositionPlan(AumOSModel):
    """Multi-step DP composition plan for budget forecasting.

    Allows synthesis engines to plan a sequence of DP operations and
    verify total privacy cost before committing. Used for complex
    workflows that apply multiple mechanisms to the same data.

    Attributes:
        tenant_id: Owning tenant UUID (inherited from AumOSModel).
        name: Human-readable name for this composition plan.
        steps: Ordered list of planned mechanism applications (JSONB).
        total_epsilon_estimate: Pre-computed total epsilon cost.
        total_delta_estimate: Pre-computed total delta cost.
        status: Plan lifecycle status.
        executed_at: When the plan was executed (null if pending).
    """

    __tablename__ = "prv_composition_plans"

    name: Mapped[str] = mapped_column(
        String(255),
        nullable=False,
        comment="Human-readable name for this composition plan",
    )

    # Planned steps — list of mechanism application specs
    steps: Mapped[list] = mapped_column(  # type: ignore[type-arg]
        JSONB,
        nullable=False,
        default=list,
        comment="Ordered list of planned mechanism applications with params",
    )

    # Pre-computed cost estimates
    total_epsilon_estimate: Mapped[Decimal] = mapped_column(
        Numeric(precision=10, scale=6),
        nullable=False,
        comment="Total epsilon cost estimate for all steps combined",
    )
    total_delta_estimate: Mapped[Decimal] = mapped_column(
        Numeric(precision=15, scale=10),
        nullable=False,
        default=Decimal("0.0"),
        comment="Total delta cost estimate for all steps combined",
    )

    # Lifecycle
    status: Mapped[str] = mapped_column(
        String(30),
        nullable=False,
        default="pending",
        comment="Plan status: pending | approved | executing | completed | failed | cancelled",
    )
    executed_at: Mapped[datetime | None] = mapped_column(
        DateTime(timezone=True),
        nullable=True,
        comment="Timestamp when plan execution began (null if pending/approved)",
    )
    composition_method: Mapped[str] = mapped_column(
        String(30),
        nullable=False,
        default="sequential",
        comment="Composition method for computing total cost: sequential | parallel | advanced",
    )
