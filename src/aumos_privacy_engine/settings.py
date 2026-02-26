"""Service-specific settings extending AumOS base config."""

from pydantic import Field
from pydantic_settings import SettingsConfigDict

from aumos_common.config import AumOSSettings


class Settings(AumOSSettings):
    """Privacy engine configuration extending base AumOS settings.

    All settings can be overridden via environment variables with the
    AUMOS_PRIVACY_ prefix (e.g., AUMOS_PRIVACY_DEFAULT_EPSILON=5.0).
    """

    service_name: str = "aumos-privacy-engine"

    # Default per-tenant privacy budget (ε)
    default_epsilon: float = Field(default=10.0, gt=0.0, description="Default total epsilon budget per tenant")

    # Default delta (δ) — probability of privacy failure
    default_delta: float = Field(
        default=1e-5,
        ge=0.0,
        lt=1.0,
        description="Default delta for (ε,δ)-DP mechanisms",
    )

    # Budget renewal period in days
    budget_renewal_days: int = Field(default=30, gt=0, description="Days between automatic budget renewals")

    # Maximum epsilon allowed for a single operation
    max_operation_epsilon: float = Field(
        default=2.0, gt=0.0, description="Maximum epsilon that a single operation may consume"
    )

    # Rényi DP composition order (alpha)
    renyi_alpha: float = Field(
        default=10.0, gt=1.0, description="Rényi divergence order for advanced composition"
    )

    # Proof storage backend
    proof_backend: str = Field(
        default="postgres",
        description="Backend for storing formal proofs: memory | redis | postgres",
    )

    # Visualization output format
    viz_format: str = Field(
        default="png",
        description="Output format for privacy loss visualizations: png | svg | json",
    )

    # Healthcare mode — stricter epsilon limits
    healthcare_max_epsilon: float = Field(
        default=0.5,
        gt=0.0,
        description="Maximum epsilon per operation for healthcare synthesis (HIPAA compliance)",
    )

    model_config = SettingsConfigDict(env_prefix="AUMOS_PRIVACY_")
