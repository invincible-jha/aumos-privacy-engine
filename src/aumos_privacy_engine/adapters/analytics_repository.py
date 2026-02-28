"""Privacy analytics repository for budget burn-rate and utilization queries (GAP-95).

Provides read-only analytics queries over prv_privacy_operations and prv_privacy_budgets.
All queries are tenant-scoped via RLS.
"""

from __future__ import annotations

import uuid
from datetime import datetime, timedelta

from sqlalchemy import func, select, text
from sqlalchemy.ext.asyncio import AsyncSession

from aumos_common.observability import get_logger

from aumos_privacy_engine.core.models import PrivacyBudget, PrivacyOperation

logger = get_logger(__name__)


class PrivacyAnalyticsRepository:
    """Provides analytics queries over privacy operations for burn-rate tracking.

    Args:
        session: Async SQLAlchemy session with RLS context set.
    """

    def __init__(self, session: AsyncSession) -> None:
        """Initialize the analytics repository.

        Args:
            session: Async database session.
        """
        self._session = session

    async def get_daily_consumption(
        self,
        tenant_id: uuid.UUID,
        window_days: int,
    ) -> list[tuple[str, float, int, list[str]]]:
        """Query daily epsilon sums and operation counts for a tenant.

        Args:
            tenant_id: The tenant to query.
            window_days: Number of days to look back.

        Returns:
            List of (date_str, total_epsilon, op_count, engines_used) tuples.
        """
        cutoff = datetime.utcnow() - timedelta(days=window_days)
        result = await self._session.execute(
            select(
                func.date_trunc("day", PrivacyOperation.created_at).label("day"),
                func.sum(PrivacyOperation.epsilon_consumed).label("total_epsilon"),
                func.count(PrivacyOperation.id).label("op_count"),
                func.array_agg(
                    func.distinct(PrivacyOperation.source_engine)
                ).label("engines"),
            )
            .where(
                PrivacyOperation.tenant_id == tenant_id,
                PrivacyOperation.created_at >= cutoff,
            )
            .group_by("day")
            .order_by("day")
        )
        rows = result.all()
        return [
            (
                str(row.day.date()) if hasattr(row.day, "date") else str(row.day),
                float(row.total_epsilon or 0.0),
                int(row.op_count or 0),
                list(row.engines or []),
            )
            for row in rows
        ]

    async def get_consumption_by_engine(
        self,
        tenant_id: uuid.UUID,
    ) -> dict[str, float]:
        """Get total epsilon consumed per synthesis engine for the current period.

        Args:
            tenant_id: The tenant to query.

        Returns:
            Dict mapping engine_name to consumed_epsilon.
        """
        result = await self._session.execute(
            select(
                PrivacyOperation.source_engine,
                func.sum(PrivacyOperation.epsilon_consumed).label("total_epsilon"),
            )
            .where(PrivacyOperation.tenant_id == tenant_id)
            .group_by(PrivacyOperation.source_engine)
        )
        return {
            row.source_engine: float(row.total_epsilon or 0.0)
            for row in result.all()
        }

    async def get_active_budget(
        self,
        tenant_id: uuid.UUID,
    ) -> PrivacyBudget | None:
        """Get the current active budget for a tenant.

        Args:
            tenant_id: The tenant to query.

        Returns:
            Active PrivacyBudget or None if not found.
        """
        result = await self._session.execute(
            select(PrivacyBudget)
            .where(
                PrivacyBudget.tenant_id == tenant_id,
                PrivacyBudget.is_active.is_(True),
            )
            .limit(1)
        )
        return result.scalar_one_or_none()
