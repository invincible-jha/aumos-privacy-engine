"""External integrations: DP mechanism libraries, repositories, Kafka, visualization.

Adapters implement the Protocols defined in core/interfaces.py.
They depend on third-party libraries (OpenDP, Opacus, numpy, scipy, matplotlib)
but contain NO business logic — they translate between the domain model and
external library APIs.

Adapter modules:
  dp_mechanisms/         — Core DP mechanism implementations (Laplace, Gaussian, etc.)
  epsilon_accountant     — Per-tenant ε/δ budget ledger with audit certificates
  moment_accountant      — Rényi DP moments accumulation (tight composition)
  sensitivity_analyzer   — Automatic sensitivity estimation and clip bounds
  privacy_amplifier      — Subsampling and shuffling amplification
  formal_prover          — DP verification certificates and proof chains
  audit_reporter         — Privacy Impact Assessment (PIA) report generation
  opendp_adapter         — OpenDP framework integration
"""

from aumos_privacy_engine.adapters.audit_reporter import PrivacyAuditReporter
from aumos_privacy_engine.adapters.epsilon_accountant import EpsilonAccountant
from aumos_privacy_engine.adapters.formal_prover import FormalProver
from aumos_privacy_engine.adapters.moment_accountant import MomentAccountant
from aumos_privacy_engine.adapters.opendp_adapter import OpenDPAdapter
from aumos_privacy_engine.adapters.privacy_amplifier import PrivacyAmplifier
from aumos_privacy_engine.adapters.sensitivity_analyzer import SensitivityAnalyzer

__all__ = [
    "EpsilonAccountant",
    "MomentAccountant",
    "SensitivityAnalyzer",
    "PrivacyAmplifier",
    "FormalProver",
    "PrivacyAuditReporter",
    "OpenDPAdapter",
]
