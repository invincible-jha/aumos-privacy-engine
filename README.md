# aumos-privacy-engine

Composable differential privacy engine for AumOS Enterprise. Provides formal ε/δ
accounting, per-tenant budget management, and mathematical proof generation for all
synthesis engines in the Data Factory.

**Release Tier:** B (Open Core)
**Phase:** 1A (Months 3-8)
**Table Prefix:** `prv_`
**Critical Dependency:** All synthesis engines depend on this for DP guarantees.

## Overview

The privacy engine enforces differential privacy across all AumOS data synthesis
operations. It acts as a central authority for:

- Allocating and tracking per-tenant privacy budgets (ε, δ)
- Applying DP mechanisms (Laplace, Gaussian, Exponential, Subsampled Gaussian)
- Computing privacy loss via sequential, parallel, and Rényi DP composition
- Generating formal mathematical proofs for regulatory audit
- Visualizing privacy loss curves for budget planning

## Architecture

```
aumos-platform-core
  └── aumos-auth-gateway
        └── aumos-privacy-engine  ← THIS REPO
              ├── aumos-tabular-engine   (consumes budget)
              ├── aumos-text-engine      (consumes budget)
              ├── aumos-image-engine     (consumes budget)
              ├── aumos-audio-engine     (consumes budget)
              ├── aumos-video-engine     (consumes budget)
              └── aumos-healthcare-synth (consumes budget)
```

## Quick Start

```bash
# Install dependencies
make install

# Configure environment
cp .env.example .env

# Start local infrastructure
make docker-run

# Run tests
make test
```

## API Endpoints

| Method | Path | Description |
|--------|------|-------------|
| POST | `/api/v1/privacy/budget/allocate` | Allocate DP budget for an operation |
| GET | `/api/v1/privacy/budget/{tenant_id}` | Current budget utilization |
| POST | `/api/v1/privacy/mechanism/apply` | Apply DP mechanism to data |
| GET | `/api/v1/privacy/proof/{job_id}` | Formal proof for audit |
| GET | `/api/v1/privacy/loss/visualize` | Privacy loss curve visualization |
| POST | `/api/v1/privacy/composition/plan` | Plan multi-step DP composition |
| GET | `/api/v1/privacy/composition/{id}` | Get composition plan status |

## DP Mechanisms

### Laplace Mechanism
For numerical queries with L1 sensitivity:
- `noise ~ Laplace(0, sensitivity/ε)`
- ε-DP guarantee

### Gaussian Mechanism
For numerical queries with L2 sensitivity (provides (ε,δ)-DP):
- `noise ~ N(0, (sensitivity × σ)²)` where σ satisfies (ε,δ)-DP
- Tighter composition bounds via RDP

### Exponential Mechanism
For categorical selection preserving utility:
- Scores outcomes by quality function
- Selects proportional to `exp(ε × quality / 2Δu)`

### Subsampled Gaussian
For DP-SGD (stochastic gradient descent):
- Combines Opacus gradient clipping + noise addition
- Privacy amplification via subsampling

## Composition Theorems

### Sequential Composition
For k mechanisms applied to the same dataset:
- ε_total = Σεᵢ, δ_total = Σδᵢ

### Parallel Composition
For k mechanisms applied to disjoint dataset partitions:
- ε_total = max(εᵢ), δ_total = max(δᵢ)

### Rényi DP Composition
Tighter bounds using Rényi divergence:
- Convert RDP to (ε,δ)-DP at the end
- Often significantly tighter than sequential composition

## Formal Proofs

Every mechanism application generates a formal proof containing:
- LaTeX mathematical derivation
- JSON proof tree for programmatic verification
- Composition chain for multi-step operations
- Signature for tamper detection

## Development

```bash
make lint       # Ruff linting + formatting check
make typecheck  # mypy strict mode
make test       # pytest with 80% coverage gate
make format     # Auto-fix formatting
```

## License

Apache 2.0 — See [LICENSE](LICENSE)
