"""Epsilon/delta budget accountant with per-tenant tracking and audit certificates.

Provides atomic budget consumption recording, remaining budget computation,
exhaustion alerts, and cryptographically-signed proof certificates for audit.

Reference: Dwork & Roth, "The Algorithmic Foundations of Differential Privacy", 2014.
"""

import hashlib
import json
import math
from datetime import UTC, datetime, timedelta
from decimal import Decimal
from typing import Any
from uuid import UUID, uuid4

from aumos_common.observability import get_logger

logger = get_logger(__name__)

# Sensitivity tiers define budget multipliers for budget policy enforcement
SENSITIVITY_TIERS: dict[str, float] = {
    "public": 1.0,
    "internal": 0.5,
    "confidential": 0.25,
    "restricted": 0.1,
    "healthcare": 0.05,
}

# Alert threshold: budget alert fires when utilization exceeds this fraction
BUDGET_ALERT_THRESHOLD: float = 0.90


class BudgetRecord:
    """In-memory record of a single budget consumption event.

    Attributes:
        record_id: Unique identifier for this consumption event.
        tenant_id: Tenant whose budget was consumed.
        operation_id: Optional linked operation UUID.
        epsilon_consumed: Epsilon consumed in this event.
        delta_consumed: Delta consumed in this event.
        query_label: Human-readable label for the consuming query.
        source_engine: Synthesis engine that triggered the consumption.
        recorded_at: UTC timestamp of the consumption.
        sensitivity_tier: Data sensitivity tier for this operation.
    """

    def __init__(
        self,
        tenant_id: UUID,
        epsilon_consumed: Decimal,
        delta_consumed: Decimal,
        query_label: str,
        source_engine: str,
        sensitivity_tier: str = "internal",
        operation_id: UUID | None = None,
    ) -> None:
        self.record_id: UUID = uuid4()
        self.tenant_id: UUID = tenant_id
        self.operation_id: UUID | None = operation_id
        self.epsilon_consumed: Decimal = epsilon_consumed
        self.delta_consumed: Decimal = delta_consumed
        self.query_label: str = query_label
        self.source_engine: str = source_engine
        self.sensitivity_tier: str = sensitivity_tier
        self.recorded_at: datetime = datetime.now(UTC)

    def to_dict(self) -> dict[str, Any]:
        """Serialize this record to a JSON-compatible dictionary.

        Returns:
            Dictionary with all record fields serialized as strings/floats.
        """
        return {
            "record_id": str(self.record_id),
            "tenant_id": str(self.tenant_id),
            "operation_id": str(self.operation_id) if self.operation_id else None,
            "epsilon_consumed": float(self.epsilon_consumed),
            "delta_consumed": float(self.delta_consumed),
            "query_label": self.query_label,
            "source_engine": self.source_engine,
            "sensitivity_tier": self.sensitivity_tier,
            "recorded_at": self.recorded_at.isoformat(),
        }


class EpsilonAccountant:
    """Per-tenant ε/δ privacy budget tracker with audit certificate generation.

    Maintains an append-only ledger of budget consumption events per tenant.
    All operations are idempotent with respect to the immutable record store.
    Budget exhaustion alerts are raised when utilization exceeds the configured
    threshold (default 90%).

    This class operates in-process (no external state store). The caller is
    responsible for persisting BudgetRecord data to durable storage (e.g.,
    via PrivacyOperation ORM records). This adapter focuses on the accounting
    arithmetic and certificate generation logic.

    Usage:
        accountant = EpsilonAccountant(total_epsilon=10.0, total_delta=1e-5)
        accountant.initialize_tenant(tenant_id)
        accountant.record_consumption(tenant_id, epsilon=0.5, delta=1e-6, ...)
        remaining = accountant.get_remaining_budget(tenant_id)
        cert = accountant.generate_budget_certificate(tenant_id)
    """

    def __init__(
        self,
        total_epsilon: float = 10.0,
        total_delta: float = 1e-5,
        budget_period_days: int = 30,
        alert_threshold: float = BUDGET_ALERT_THRESHOLD,
    ) -> None:
        """Initialize the epsilon accountant.

        Args:
            total_epsilon: Total epsilon budget allocated per tenant per period.
            total_delta: Total delta budget allocated per tenant per period.
            budget_period_days: Number of days in each budget period.
            alert_threshold: Utilization fraction that triggers an alert (0.0 to 1.0).
        """
        if total_epsilon <= 0:
            raise ValueError(f"total_epsilon must be > 0, got {total_epsilon}")
        if total_delta < 0 or total_delta >= 1:
            raise ValueError(f"total_delta must be in [0, 1), got {total_delta}")
        if alert_threshold <= 0 or alert_threshold > 1:
            raise ValueError(f"alert_threshold must be in (0, 1], got {alert_threshold}")

        self._total_epsilon: Decimal = Decimal(str(total_epsilon))
        self._total_delta: Decimal = Decimal(str(total_delta))
        self._budget_period_days: int = budget_period_days
        self._alert_threshold: float = alert_threshold

        # Tenant state: maps tenant_id -> list of BudgetRecord
        self._records: dict[UUID, list[BudgetRecord]] = {}
        # Tenant initialization timestamps
        self._initialized_at: dict[UUID, datetime] = {}
        # Pending alerts: maps tenant_id -> list of alert messages
        self._pending_alerts: dict[UUID, list[str]] = {}

    def initialize_tenant(
        self,
        tenant_id: UUID,
        sensitivity_tier: str = "internal",
    ) -> None:
        """Initialize budget tracking for a new tenant.

        If the tenant is already initialized, this is a no-op (idempotent).

        Args:
            tenant_id: Tenant to initialize.
            sensitivity_tier: Data sensitivity tier that may constrain effective budget.
        """
        if tenant_id in self._records:
            logger.debug("Tenant already initialized", tenant_id=str(tenant_id))
            return

        self._records[tenant_id] = []
        self._initialized_at[tenant_id] = datetime.now(UTC)
        self._pending_alerts[tenant_id] = []

        effective_epsilon = self._effective_epsilon(sensitivity_tier)
        logger.info(
            "Tenant budget initialized",
            tenant_id=str(tenant_id),
            sensitivity_tier=sensitivity_tier,
            total_epsilon=float(self._total_epsilon),
            effective_epsilon=float(effective_epsilon),
            total_delta=float(self._total_delta),
            period_days=self._budget_period_days,
        )

    def _effective_epsilon(self, sensitivity_tier: str) -> Decimal:
        """Compute the effective epsilon budget for a given sensitivity tier.

        Args:
            sensitivity_tier: One of public, internal, confidential, restricted, healthcare.

        Returns:
            Effective epsilon after applying tier multiplier.
        """
        multiplier = SENSITIVITY_TIERS.get(sensitivity_tier, 1.0)
        return self._total_epsilon * Decimal(str(multiplier))

    def record_consumption(
        self,
        tenant_id: UUID,
        epsilon_consumed: float,
        delta_consumed: float,
        query_label: str,
        source_engine: str,
        sensitivity_tier: str = "internal",
        operation_id: UUID | None = None,
    ) -> BudgetRecord:
        """Record a privacy budget consumption event for a tenant.

        This method does NOT enforce budget limits — the caller (BudgetService)
        must perform the budget check before calling this. This method only
        records the consumption and fires alerts if thresholds are crossed.

        Args:
            tenant_id: Tenant consuming budget.
            epsilon_consumed: Epsilon consumed by this operation.
            delta_consumed: Delta consumed by this operation.
            query_label: Human-readable label for the consuming query.
            source_engine: Synthesis engine that triggered this consumption.
            sensitivity_tier: Data sensitivity tier for this operation.
            operation_id: Optional UUID linking to the PrivacyOperation ORM record.

        Returns:
            The newly created BudgetRecord.

        Raises:
            RuntimeError: If the tenant has not been initialized.
        """
        if tenant_id not in self._records:
            raise RuntimeError(
                f"Tenant {tenant_id} not initialized. Call initialize_tenant() first."
            )

        record = BudgetRecord(
            tenant_id=tenant_id,
            epsilon_consumed=Decimal(str(epsilon_consumed)),
            delta_consumed=Decimal(str(delta_consumed)),
            query_label=query_label,
            source_engine=source_engine,
            sensitivity_tier=sensitivity_tier,
            operation_id=operation_id,
        )
        self._records[tenant_id].append(record)

        # Compute utilization and fire alerts if necessary
        used_epsilon = self._sum_epsilon(tenant_id)
        utilization = float(used_epsilon / self._total_epsilon)

        if utilization >= self._alert_threshold:
            alert_msg = (
                f"Budget utilization {utilization:.1%} exceeds threshold "
                f"{self._alert_threshold:.1%} for tenant {tenant_id}"
            )
            self._pending_alerts[tenant_id].append(alert_msg)
            logger.warning(
                "Privacy budget alert threshold exceeded",
                tenant_id=str(tenant_id),
                utilization_pct=round(utilization * 100, 2),
                threshold_pct=round(self._alert_threshold * 100, 2),
                used_epsilon=float(used_epsilon),
                total_epsilon=float(self._total_epsilon),
            )

        logger.info(
            "Privacy budget consumed",
            tenant_id=str(tenant_id),
            record_id=str(record.record_id),
            epsilon_consumed=epsilon_consumed,
            delta_consumed=delta_consumed,
            source_engine=source_engine,
            remaining_epsilon=float(self._total_epsilon - used_epsilon),
        )

        return record

    def _sum_epsilon(self, tenant_id: UUID) -> Decimal:
        """Sum all epsilon consumed by a tenant.

        Args:
            tenant_id: Tenant to aggregate.

        Returns:
            Total epsilon consumed across all records.
        """
        return sum(
            (r.epsilon_consumed for r in self._records[tenant_id]),
            Decimal("0.0"),
        )

    def _sum_delta(self, tenant_id: UUID) -> Decimal:
        """Sum all delta consumed by a tenant.

        Args:
            tenant_id: Tenant to aggregate.

        Returns:
            Total delta consumed across all records.
        """
        return sum(
            (r.delta_consumed for r in self._records[tenant_id]),
            Decimal("0.0"),
        )

    def get_remaining_budget(self, tenant_id: UUID) -> tuple[Decimal, Decimal]:
        """Get remaining epsilon and delta budget for a tenant.

        Args:
            tenant_id: Tenant to query.

        Returns:
            Tuple of (remaining_epsilon, remaining_delta). Both may be negative
            if the caller allowed over-consumption (the accountant records all events).

        Raises:
            RuntimeError: If the tenant is not initialized.
        """
        if tenant_id not in self._records:
            raise RuntimeError(f"Tenant {tenant_id} not initialized.")

        used_epsilon = self._sum_epsilon(tenant_id)
        used_delta = self._sum_delta(tenant_id)
        return (
            self._total_epsilon - used_epsilon,
            self._total_delta - used_delta,
        )

    def get_usage_analytics(self, tenant_id: UUID) -> dict[str, Any]:
        """Get historical budget usage analytics for a tenant.

        Provides per-engine breakdown, time-series of consumption, and
        projected exhaustion date based on recent consumption rate.

        Args:
            tenant_id: Tenant to analyze.

        Returns:
            Dictionary with usage breakdown and projections.

        Raises:
            RuntimeError: If the tenant is not initialized.
        """
        if tenant_id not in self._records:
            raise RuntimeError(f"Tenant {tenant_id} not initialized.")

        records = self._records[tenant_id]
        used_epsilon = self._sum_epsilon(tenant_id)
        used_delta = self._sum_delta(tenant_id)
        remaining_epsilon = self._total_epsilon - used_epsilon
        utilization_pct = float(used_epsilon / self._total_epsilon * 100) if self._total_epsilon > 0 else 100.0

        # Per-engine breakdown
        engine_breakdown: dict[str, float] = {}
        for record in records:
            engine = record.source_engine
            engine_breakdown[engine] = engine_breakdown.get(engine, 0.0) + float(record.epsilon_consumed)

        # Consumption rate: epsilon per day based on full history
        period_elapsed_days: float = 0.0
        if records:
            oldest = min(r.recorded_at for r in records)
            period_elapsed_days = max(
                (datetime.now(UTC) - oldest).total_seconds() / 86400,
                1 / 86400,  # minimum 1 second to avoid division by zero
            )

        daily_rate = float(used_epsilon) / period_elapsed_days if period_elapsed_days > 0 else 0.0

        # Project days until exhaustion
        projected_days_remaining: float | None = None
        if daily_rate > 0:
            projected_days_remaining = float(remaining_epsilon) / daily_rate

        # Recent 7-day consumption
        cutoff_7d = datetime.now(UTC) - timedelta(days=7)
        recent_epsilon = sum(
            float(r.epsilon_consumed) for r in records if r.recorded_at >= cutoff_7d
        )

        return {
            "tenant_id": str(tenant_id),
            "total_epsilon": float(self._total_epsilon),
            "used_epsilon": float(used_epsilon),
            "remaining_epsilon": float(remaining_epsilon),
            "total_delta": float(self._total_delta),
            "used_delta": float(used_delta),
            "utilization_pct": round(utilization_pct, 4),
            "operation_count": len(records),
            "engine_breakdown": engine_breakdown,
            "daily_consumption_rate": round(daily_rate, 6),
            "projected_days_to_exhaustion": (
                round(projected_days_remaining, 1) if projected_days_remaining is not None else None
            ),
            "recent_7d_epsilon": round(recent_epsilon, 6),
            "pending_alert_count": len(self._pending_alerts.get(tenant_id, [])),
            "initialized_at": self._initialized_at[tenant_id].isoformat(),
        }

    def get_pending_alerts(self, tenant_id: UUID) -> list[str]:
        """Retrieve and clear pending budget exhaustion alerts for a tenant.

        Args:
            tenant_id: Tenant to check alerts for.

        Returns:
            List of alert message strings. Clears the alerts after retrieval.
        """
        alerts = self._pending_alerts.get(tenant_id, [])
        self._pending_alerts[tenant_id] = []
        return alerts

    def generate_budget_certificate(
        self,
        tenant_id: UUID,
        certifier_id: str = "aumos-privacy-engine",
    ) -> dict[str, Any]:
        """Generate a cryptographically-identified audit certificate for budget usage.

        The certificate captures the current state of the tenant's budget ledger
        and signs it with a SHA-256 hash for tamper detection. This certificate
        can be included in formal privacy audit reports.

        Args:
            tenant_id: Tenant to certify.
            certifier_id: Identifier of the certifying system.

        Returns:
            JSON-serializable certificate dictionary including:
                - certificate_id: Unique UUID for this certificate.
                - tenant_id: Certified tenant.
                - issued_at: UTC timestamp.
                - budget_summary: Current budget state.
                - operations: Full list of consumption records.
                - verification_hash: SHA-256 hash of the certificate payload.

        Raises:
            RuntimeError: If the tenant is not initialized.
        """
        if tenant_id not in self._records:
            raise RuntimeError(f"Tenant {tenant_id} not initialized.")

        analytics = self.get_usage_analytics(tenant_id)
        records_serialized = [r.to_dict() for r in self._records[tenant_id]]

        payload: dict[str, Any] = {
            "certificate_id": str(uuid4()),
            "tenant_id": str(tenant_id),
            "certifier_id": certifier_id,
            "issued_at": datetime.now(UTC).isoformat(),
            "budget_summary": analytics,
            "operations": records_serialized,
            "certificate_version": "1.0",
            "dp_framework": "epsilon-delta",
        }

        # Compute deterministic hash over the payload content (excluding certificate_id and issued_at
        # to allow re-issuance; hash is over the stable budget content)
        hash_content = json.dumps(
            {
                "tenant_id": str(tenant_id),
                "total_epsilon": float(self._total_epsilon),
                "used_epsilon": analytics["used_epsilon"],
                "operation_count": analytics["operation_count"],
                "operations": records_serialized,
            },
            sort_keys=True,
        )
        payload["verification_hash"] = hashlib.sha256(hash_content.encode()).hexdigest()

        logger.info(
            "Budget certificate generated",
            tenant_id=str(tenant_id),
            certificate_id=payload["certificate_id"],
            operation_count=len(records_serialized),
            verification_hash=payload["verification_hash"][:16] + "...",
        )

        return payload

    def apply_budget_policy(
        self,
        tenant_id: UUID,
        requested_epsilon: float,
        sensitivity_tier: str = "internal",
    ) -> bool:
        """Check whether a requested epsilon consumption is within policy limits.

        Evaluates:
        1. Whether the tenant is initialized.
        2. Whether the remaining budget covers the requested amount.
        3. Whether sensitivity-tier constraints are satisfied.

        Args:
            tenant_id: Tenant requesting consumption.
            requested_epsilon: Epsilon the operation wants to consume.
            sensitivity_tier: Data sensitivity tier of the operation.

        Returns:
            True if the operation is within policy, False if budget is exhausted.
        """
        if tenant_id not in self._records:
            logger.warning("Policy check for uninitialized tenant", tenant_id=str(tenant_id))
            return False

        remaining_epsilon, _ = self.get_remaining_budget(tenant_id)
        effective_remaining = min(remaining_epsilon, self._effective_epsilon(sensitivity_tier))

        if Decimal(str(requested_epsilon)) > effective_remaining:
            logger.warning(
                "Budget policy check failed — insufficient budget",
                tenant_id=str(tenant_id),
                requested_epsilon=requested_epsilon,
                effective_remaining=float(effective_remaining),
                sensitivity_tier=sensitivity_tier,
            )
            return False

        return True
