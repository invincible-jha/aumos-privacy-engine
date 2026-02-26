# Security Policy

## Supported Versions

| Version | Supported          |
| ------- | ------------------ |
| 0.1.x   | :white_check_mark: |

## Reporting a Vulnerability

The AumOS Enterprise team takes security vulnerabilities seriously, especially given
that this service handles differential privacy budgets and formal privacy proofs.

**Do NOT open a GitHub issue for security vulnerabilities.**

### How to Report

Send a detailed report to: security@aumos.ai

Include:
- Description of the vulnerability
- Steps to reproduce
- Potential impact assessment
- Suggested remediation (if known)

### Response Timeline

- **Initial acknowledgment**: Within 48 hours
- **Severity assessment**: Within 5 business days
- **Fix timeline**: Critical (7 days), High (30 days), Medium (90 days)
- **Disclosure**: Coordinated with reporter after fix deployment

## Privacy-Specific Security Concerns

This service is the CRITICAL dependency for all differential privacy guarantees
in AumOS. The following security considerations are especially important:

### Privacy Budget Manipulation
- All budget allocations are immutable once committed
- Budget consumption is recorded in append-only audit logs
- Formal proofs are cryptographically signed

### Tenant Isolation
- Row-Level Security (RLS) enforces strict tenant boundaries
- Cross-tenant budget queries are impossible by design
- All operations are scoped to authenticated tenant context

### Proof Integrity
- Mathematical proofs use deterministic sympy computation
- Proof chains are verifiable end-to-end
- Audit logs are tamper-evident

### Mechanism Parameter Validation
- Sensitivity bounds are validated before mechanism application
- Epsilon and delta parameters are bounded to safe ranges
- Invalid parameters are rejected before any data processing

## Security Contacts

- Security Team: security@aumos.ai
- Platform Lead: platform@aumos.ai
