"""Accountant interoperability bridge for Google DP and Tumult Analytics (GAP-98).

Enables import/export of budget state between AumOS and external DP accountants,
allowing AumOS to work alongside existing privacy infrastructure.
"""

from __future__ import annotations

import uuid
from dataclasses import dataclass
from datetime import datetime
from typing import Any

from aumos_common.observability import get_logger

logger = get_logger(__name__)


@dataclass
class ExportedOperation:
    """A single exported privacy operation in a portable format.

    Attributes:
        operation_id: Unique identifier.
        mechanism: DP mechanism name.
        epsilon: Epsilon consumed.
        delta: Delta consumed.
        composition_type: Composition theorem applied.
        timestamp: When the operation occurred.
    """

    operation_id: str
    mechanism: str
    epsilon: float
    delta: float
    composition_type: str
    timestamp: str


class AccountantBridge:
    """Converts AumOS privacy operation records to/from external accountant formats.

    Supports Google's DP Accounting library format and Tumult Analytics format.
    Enables AumOS to interoperate with existing enterprise DP infrastructure.
    """

    @staticmethod
    def export_to_google_dp(
        operations: list[dict[str, Any]],
        tenant_id: uuid.UUID,
    ) -> dict[str, Any]:
        """Export operations to Google DP Accounting library format.

        The Google DP Accounting library uses a PLD (Privacy Loss Distribution)
        representation. We export as a sequence of mechanisms that can be
        re-imported into Google's accountant.

        Args:
            operations: List of operation dicts with epsilon, delta, mechanism fields.
            tenant_id: The tenant whose operations are being exported.

        Returns:
            Dict in Google DP Accounting compatible format.

        References:
            https://github.com/google/differential-privacy/tree/main/python/dp_accounting
        """
        google_dp_mechanisms = []
        for op in operations:
            mechanism = op.get("mechanism", "laplace")
            epsilon = float(op.get("epsilon_consumed", op.get("epsilon", 0.0)))
            delta = float(op.get("delta_consumed", op.get("delta", 0.0)))
            if mechanism == "laplace":
                google_dp_mechanisms.append({
                    "type": "laplace",
                    "noise_multiplier": epsilon,  # simplified
                    "num_queries": 1,
                })
            elif mechanism == "gaussian":
                # Gaussian: σ ≈ sqrt(2 * ln(1.25/delta)) / epsilon
                import math
                if delta > 0 and epsilon > 0:
                    noise_multiplier = math.sqrt(2 * math.log(1.25 / delta)) / epsilon
                else:
                    noise_multiplier = 1.0
                google_dp_mechanisms.append({
                    "type": "gaussian",
                    "noise_multiplier": noise_multiplier,
                    "num_queries": 1,
                    "delta": delta,
                })
            else:
                # For other mechanisms, represent as generic
                google_dp_mechanisms.append({
                    "type": "generic",
                    "epsilon": epsilon,
                    "delta": delta,
                    "mechanism": mechanism,
                })
        result: dict[str, Any] = {
            "schema_version": "google_dp_v1",
            "tenant_id": str(tenant_id),
            "exported_at": datetime.utcnow().isoformat(),
            "operation_count": len(operations),
            "mechanisms": google_dp_mechanisms,
        }
        logger.info(
            "accountant_exported_google_dp",
            tenant_id=str(tenant_id),
            operation_count=len(operations),
        )
        return result

    @staticmethod
    def export_to_tumult(
        operations: list[dict[str, Any]],
        tenant_id: uuid.UUID,
    ) -> dict[str, Any]:
        """Export operations to Tumult Analytics accountant format.

        Tumult Analytics uses a privacy budget specification with per-mechanism
        allocations. We export as a sequence of (epsilon, delta) pairs.

        Args:
            operations: List of operation dicts.
            tenant_id: The tenant whose operations are being exported.

        Returns:
            Dict in Tumult Analytics compatible format.
        """
        tumult_operations = [
            {
                "operation_id": str(op.get("id", uuid.uuid4())),
                "epsilon": float(op.get("epsilon_consumed", op.get("epsilon", 0.0))),
                "delta": float(op.get("delta_consumed", op.get("delta", 0.0))),
                "mechanism": op.get("mechanism", "unknown"),
                "source": op.get("source_engine", "unknown"),
                "timestamp": str(op.get("created_at", datetime.utcnow().isoformat())),
            }
            for op in operations
        ]
        total_epsilon = sum(o["epsilon"] for o in tumult_operations)
        total_delta = sum(o["delta"] for o in tumult_operations)
        result: dict[str, Any] = {
            "schema_version": "tumult_v1",
            "tenant_id": str(tenant_id),
            "exported_at": datetime.utcnow().isoformat(),
            "total_epsilon": total_epsilon,
            "total_delta": total_delta,
            "operations": tumult_operations,
        }
        logger.info(
            "accountant_exported_tumult",
            tenant_id=str(tenant_id),
            operation_count=len(operations),
        )
        return result

    @staticmethod
    def import_from_google_dp(
        google_dp_state: dict[str, Any],
    ) -> list[ExportedOperation]:
        """Import operations from Google DP Accounting library format.

        Args:
            google_dp_state: Previously exported Google DP state dict.

        Returns:
            List of ExportedOperation records.

        Raises:
            ValueError: If the schema version is not supported.
        """
        schema = google_dp_state.get("schema_version", "")
        if not schema.startswith("google_dp"):
            raise ValueError(
                f"Unsupported schema version: {schema}. Expected google_dp_v1."
            )
        operations = []
        for mech in google_dp_state.get("mechanisms", []):
            mechanism_type = mech.get("type", "generic")
            if mechanism_type == "laplace":
                epsilon = float(mech.get("noise_multiplier", 1.0))
                delta = 0.0
            elif mechanism_type == "gaussian":
                delta = float(mech.get("delta", 1e-6))
                import math
                noise_multiplier = float(mech.get("noise_multiplier", 1.0))
                if noise_multiplier > 0 and delta > 0:
                    epsilon = math.sqrt(2 * math.log(1.25 / delta)) / noise_multiplier
                else:
                    epsilon = 1.0
            else:
                epsilon = float(mech.get("epsilon", 1.0))
                delta = float(mech.get("delta", 0.0))
                mechanism_type = mech.get("mechanism", "generic")
            operations.append(
                ExportedOperation(
                    operation_id=str(uuid.uuid4()),
                    mechanism=mechanism_type,
                    epsilon=epsilon,
                    delta=delta,
                    composition_type="sequential",
                    timestamp=google_dp_state.get("exported_at", ""),
                )
            )
        logger.info(
            "accountant_imported_google_dp",
            operation_count=len(operations),
        )
        return operations

    @staticmethod
    def import_from_tumult(
        tumult_state: dict[str, Any],
    ) -> list[ExportedOperation]:
        """Import operations from Tumult Analytics accountant format.

        Args:
            tumult_state: Previously exported Tumult state dict.

        Returns:
            List of ExportedOperation records.
        """
        schema = tumult_state.get("schema_version", "")
        if not schema.startswith("tumult"):
            raise ValueError(
                f"Unsupported schema version: {schema}. Expected tumult_v1."
            )
        operations = [
            ExportedOperation(
                operation_id=op.get("operation_id", str(uuid.uuid4())),
                mechanism=op.get("mechanism", "laplace"),
                epsilon=float(op.get("epsilon", 0.0)),
                delta=float(op.get("delta", 0.0)),
                composition_type="sequential",
                timestamp=op.get("timestamp", ""),
            )
            for op in tumult_state.get("operations", [])
        ]
        logger.info(
            "accountant_imported_tumult",
            operation_count=len(operations),
        )
        return operations
