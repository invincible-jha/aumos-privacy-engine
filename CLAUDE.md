# CLAUDE.md — AumOS Privacy Engine

## Project Overview

AumOS Enterprise is a composable enterprise AI platform with 9 products + 2 services
across 62 repositories. This repo (`aumos-privacy-engine`) is part of **Tier B: Open Core**:
Phase 1A Data Factory — the CRITICAL privacy infrastructure.

**Release Tier:** B (Open Core)
**Product Mapping:** Product 1 — Data Factory (cross-cutting privacy layer)
**Phase:** 1A (Months 3-8)

## Repo Purpose

The privacy engine is the CRITICAL shared dependency that all synthesis engines (tabular,
text, image, audio, video, healthcare) call before generating any synthetic data. It enforces
per-tenant differential privacy budgets, applies formal DP mechanisms, computes composition
bounds, and generates mathematical proofs for regulatory audit. Without this service, no
synthesis engine can produce privacy-safe outputs.

## Architecture Position

```
aumos-platform-core → aumos-auth-gateway → aumos-privacy-engine
                                                ↓ (budget consumed by)
                                         aumos-tabular-engine
                                         aumos-text-engine
                                         aumos-image-engine
                                         aumos-audio-engine
                                         aumos-video-engine
                                         aumos-healthcare-synth
                                         ↓ (events published to)
                                         aumos-event-bus
                                         aumos-data-layer (stores budget/proof records)
```

**Upstream dependencies (this repo IMPORTS from):**
- `aumos-common` — auth, database, events, errors, config, health, pagination
- `aumos-proto` — Protobuf message definitions for Kafka events

**Downstream dependents (other repos IMPORT from this):**
- `aumos-tabular-engine` — calls /privacy/budget/allocate before CTGAN generation
- `aumos-text-engine` — calls /privacy/budget/allocate before LLM synthesis
- `aumos-image-engine` — calls /privacy/budget/allocate before image generation
- `aumos-audio-engine` — calls /privacy/budget/allocate before audio synthesis
- `aumos-video-engine` — calls /privacy/budget/allocate before video generation
- `aumos-healthcare-synth` — calls /privacy/budget/allocate with stricter ε limits

## Tech Stack (DO NOT DEVIATE)

| Component | Version | Purpose |
|-----------|---------|---------|
| Python | 3.11+ | Runtime |
| FastAPI | 0.110+ | REST API framework |
| SQLAlchemy | 2.0+ (async) | Database ORM |
| asyncpg | 0.29+ | PostgreSQL async driver |
| Pydantic | 2.6+ | Data validation, settings, API schemas |
| confluent-kafka | 2.3+ | Kafka producer/consumer |
| structlog | 24.1+ | Structured JSON logging |
| OpenTelemetry | 1.23+ | Distributed tracing |
| pytest | 8.0+ | Testing framework |
| ruff | 0.3+ | Linting and formatting |
| mypy | 1.8+ | Type checking |
| opendp | 0.9+ | OpenDP for Laplace/Gaussian mechanism primitives |
| opacus | 1.4+ | PyTorch DP-SGD / Subsampled Gaussian mechanism |
| sympy | 1.12+ | Symbolic math for formal proof generation |
| matplotlib | 3.8+ | Privacy loss curve visualization |
| numpy | 1.26+ | Numerical computations for DP mechanisms |
| scipy | 1.12+ | Statistical distributions for DP noise |

## Coding Standards

### ABSOLUTE RULES (violations will break integration with other repos)

1. **Import aumos-common, never reimplement.** If aumos-common provides it, use it.
   ```python
   # ✅ CORRECT
   from aumos_common.auth import get_current_tenant, get_current_user
   from aumos_common.database import get_db_session, Base, AumOSModel, BaseRepository
   from aumos_common.events import EventPublisher, Topics
   from aumos_common.errors import NotFoundError, ErrorCode
   from aumos_common.config import AumOSSettings
   from aumos_common.health import create_health_router
   from aumos_common.pagination import PageRequest, PageResponse, paginate
   from aumos_common.app import create_app
   ```

2. **Type hints on EVERY function.** No exceptions.

3. **Pydantic models for ALL API inputs/outputs.** Never return raw dicts.

4. **RLS tenant isolation via aumos-common.** Never write raw SQL that bypasses RLS.

5. **Structured logging via structlog.** Never use print() or logging.getLogger().

6. **Publish domain events to Kafka after state changes.**

7. **Async by default.** All I/O operations must be async.

8. **Google-style docstrings** on all public classes and functions.

### Privacy Engine Specific Rules

9. **VALIDATE all ε and δ parameters** before applying any mechanism.
   - ε must be > 0 and ≤ AUMOS_PRIVACY_MAX_OPERATION_EPSILON
   - δ must be ≥ 0 and < 1 (typically 1e-5 to 1e-7)
   - sensitivity must be > 0

10. **ALWAYS check remaining budget** before consuming. Raise BudgetExhaustedError if insufficient.

11. **GENERATE formal proofs** for every mechanism application. This is required for audit.

12. **NEVER implement DP mechanisms from scratch.** Use OpenDP or Opacus primitives.
    The mathematics are subtle — always use vetted library implementations.

13. **APPEND-ONLY budget consumption.** Never update used_epsilon; always insert new
    PrivacyOperation records. Budget is reconstructed by summing all operations.

14. **Test privacy accounting arithmetic.** Composition must be exact.

### Style Rules

- Max line length: **120 characters**
- Import order: stdlib → third-party → aumos-common → local
- Linter: `ruff` (select E, W, F, I, N, UP, ANN, B, A, COM, C4, PT, RUF)
- Type checker: `mypy` strict mode
- Formatter: `ruff format`

## Database Conventions

- **Table prefix:** `prv_` (e.g., `prv_privacy_budgets`, `prv_privacy_operations`)
- ALL tenant-scoped tables extend `AumOSModel`
- RLS policy on every tenant table
- Migration naming: `{timestamp}_prv_{description}.py`
- PrivacyOperation is append-only — no updates, no deletes

## Kafka Conventions

- Topic: `privacy.operations` — published after every mechanism application
- Topic: `privacy.budget.exhausted` — published when tenant budget runs low (<10% remaining)
- Topic: `privacy.budget.renewed` — published when budget auto-renews
- Always include `tenant_id`, `job_id`, `source_engine`, `epsilon_consumed` in events

## Testing

- Minimum coverage: **80%** for core modules (STRICTLY ENFORCED — this is CRITICAL infra)
- Budget accounting logic: **100% coverage** (every composition theorem must be tested)
- Mock OpenDP/Opacus in unit tests for speed
- Use `testcontainers` for integration tests
- Property-based tests for composition bounds (privacy should only get worse, never better)

## What Claude Code Should NOT Do

1. **Do NOT reimplement anything in aumos-common.**
2. **Do NOT use print().** Use `get_logger(__name__)`.
3. **Do NOT return raw dicts from API endpoints.** Use Pydantic models.
4. **Do NOT write raw SQL.** Use SQLAlchemy ORM with BaseRepository.
5. **Do NOT hardcode configuration.** Use Pydantic Settings with env vars.
6. **Do NOT skip type hints.** Every function signature must be typed.
7. **Do NOT implement DP mechanisms from scratch.** Use OpenDP or Opacus.
8. **Do NOT allow budget to go negative.** Enforce pre-consumption checks.
9. **Do NOT skip proof generation.** Every mechanism application needs a proof.
10. **Do NOT allow DELETE on PrivacyOperation records.** Audit trail is permanent.

## Repo-Specific Context

### Differential Privacy Fundamentals
- ε-DP (pure DP): Pr[M(D) ∈ S] ≤ exp(ε) × Pr[M(D') ∈ S] for any adjacent D, D'
- (ε,δ)-DP: Same bound holds with probability 1-δ (allows small failure probability)
- Rényi DP: Defined via Rényi divergence; tighter composition, convert to (ε,δ) at end
- Adjacent datasets differ by one record (add/remove semantics)

### Mechanism Selection Guide
- **Laplace**: Numerical queries, L1 sensitivity, pure ε-DP. Best for simple stats.
- **Gaussian**: Numerical queries, L2 sensitivity, (ε,δ)-DP. Better for high-dim data.
- **Exponential**: Categorical outputs (selection). Preserves utility ordering.
- **Subsampled Gaussian**: ML training (DP-SGD). Best for neural network training.

### Composition Accounting
- Sequential: Operations on SAME data → ε adds up (worst case)
- Parallel: Operations on DISJOINT partitions → max(ε) applies
- Rényi: Converts operations to RDP moments, converts to (ε,δ) at end — much tighter
- Always use the tightest applicable theorem

### Healthcare Special Case
- `aumos-healthcare-synth` uses stricter ε limits (max 0.5 per operation)
- HIPAA Safe Harbor requires formal DP proof chain for de-identification claims
- Healthcare PrivacyOperation records have 7-year retention requirement

### Performance Requirements
- Budget check + allocation: < 50ms (synthesis engines call this in hot path)
- Proof generation: < 500ms (called asynchronously after mechanism application)
- Visualization: < 2s (called on demand, not in hot path)
