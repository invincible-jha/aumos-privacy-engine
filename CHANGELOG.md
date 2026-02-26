# Changelog

All notable changes to aumos-privacy-engine will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

## [0.1.0] - 2026-02-26

### Added
- Initial scaffolding for composable differential privacy engine
- Per-tenant privacy budget management with ε/δ accounting
- Laplace mechanism implementation via OpenDP
- Gaussian mechanism implementation with privacy amplification
- Exponential mechanism for categorical data selection
- Subsampled Gaussian mechanism for DP-SGD via Opacus
- Sequential composition (ε₁ + ε₂, δ₁ + δ₂)
- Parallel composition (max(ε₁, ε₂), max(δ₁, δ₂))
- Rényi DP advanced composition for tighter privacy bounds
- Formal mathematical proof generation using sympy (LaTeX + JSON)
- Privacy loss curve visualization using matplotlib
- REST API: budget allocation, mechanism application, proof retrieval, composition planning
- Hexagonal architecture: api/ + core/ + adapters/ layers
- FastAPI application with lifespan management
- SQLAlchemy async models: PrivacyBudget, PrivacyOperation, CompositionPlan
- Kafka event publishing for privacy operation lifecycle
- Full CI/CD pipeline with lint, typecheck, test, license check, Docker build
