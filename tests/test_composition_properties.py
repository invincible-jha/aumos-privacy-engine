"""Property-based tests for DP composition correctness (GAP-94).

These tests use Hypothesis to verify mathematical invariants hold
across thousands of randomly generated inputs, as required by the
formal verification gap.

Composition theorems verified:
1. Sequential: ε(A∘B) = ε(A) + ε(B) for pure DP
2. Parallel: ε(A∥B) = max(ε(A), ε(B)) on disjoint datasets
3. Rényi composition produces tighter bounds than sequential
4. Budget monotonically increases (never decreases) with additional operations
5. Budget exhaustion detected at the exact threshold
"""

from __future__ import annotations

import pytest
from hypothesis import given, settings, strategies as st
from hypothesis.strategies import composite

from aumos_privacy_engine.core.composition_accountant import (
    BudgetExhaustedError,
    BudgetService,
    CompositionAccountant,
)


@composite
def epsilon_sequence(draw: st.DrawFn) -> list[float]:
    """Generate a sequence of valid epsilon values.

    Args:
        draw: Hypothesis draw function.

    Returns:
        List of positive epsilon values.
    """
    n = draw(st.integers(min_value=1, max_value=20))
    return [draw(st.floats(min_value=1e-6, max_value=2.0)) for _ in range(n)]


class TestSequentialComposition:
    """Pure sequential composition: ε_total = Σε_i."""

    @given(epsilons=epsilon_sequence())
    @settings(max_examples=1000)
    def test_sequential_composition_additive(self, epsilons: list[float]) -> None:
        """Sequential composition epsilon must equal sum of individual epsilons.

        Property: ε(M1 ∘ M2 ∘ ... ∘ Mk) = Σ ε(Mi) for sequential composition.
        """
        accountant = CompositionAccountant(composition_type="sequential")
        for epsilon in epsilons:
            accountant.add_operation(epsilon=epsilon, delta=0.0, mechanism="laplace")
        expected_total = sum(epsilons)
        actual_total = accountant.total_epsilon()
        assert abs(actual_total - expected_total) < 1e-10, (
            f"Sequential composition failed: expected {expected_total}, "
            f"got {actual_total}, epsilons={epsilons}"
        )

    @given(epsilons=epsilon_sequence())
    @settings(max_examples=1000)
    def test_epsilon_monotonically_increases(self, epsilons: list[float]) -> None:
        """Adding any operation must never decrease total epsilon.

        Property: ε_n ≥ ε_{n-1} for all n (budget can only worsen).
        """
        accountant = CompositionAccountant(composition_type="sequential")
        previous_epsilon = 0.0
        for epsilon in epsilons:
            accountant.add_operation(epsilon=epsilon, delta=0.0, mechanism="laplace")
            current_epsilon = accountant.total_epsilon()
            assert current_epsilon >= previous_epsilon - 1e-12, (
                f"Epsilon decreased: {previous_epsilon} -> {current_epsilon} "
                f"after adding epsilon={epsilon}"
            )
            previous_epsilon = current_epsilon

    @given(epsilons=epsilon_sequence())
    @settings(max_examples=500)
    def test_sequential_operation_count(self, epsilons: list[float]) -> None:
        """Operation count must match the number of added operations.

        Property: accountant.operation_count() == len(epsilons).
        """
        accountant = CompositionAccountant(composition_type="sequential")
        for i, epsilon in enumerate(epsilons):
            accountant.add_operation(epsilon=epsilon, delta=0.0, mechanism="laplace")
            assert accountant.operation_count() == i + 1

    @given(
        epsilon1=st.floats(min_value=1e-6, max_value=2.0),
        epsilon2=st.floats(min_value=1e-6, max_value=2.0),
    )
    @settings(max_examples=500)
    def test_two_operation_composition(self, epsilon1: float, epsilon2: float) -> None:
        """Two operations compose exactly as their sum.

        Property: ε(M1 ∘ M2) = ε1 + ε2.
        """
        accountant = CompositionAccountant(composition_type="sequential")
        accountant.add_operation(epsilon=epsilon1, delta=0.0, mechanism="laplace")
        accountant.add_operation(epsilon=epsilon2, delta=0.0, mechanism="laplace")
        expected = epsilon1 + epsilon2
        actual = accountant.total_epsilon()
        assert abs(actual - expected) < 1e-10, (
            f"Two-operation composition: expected {expected}, got {actual}"
        )


class TestParallelComposition:
    """Parallel composition: ε_total = max(ε_i) on disjoint data."""

    @given(epsilons=epsilon_sequence())
    @settings(max_examples=500)
    def test_parallel_composition_max(self, epsilons: list[float]) -> None:
        """Parallel composition epsilon must equal max of individual epsilons.

        Property: ε(M1 ∥ M2 ∥ ... ∥ Mk) = max(ε_i) when applied to disjoint partitions.
        """
        accountant = CompositionAccountant(composition_type="parallel")
        for epsilon in epsilons:
            accountant.add_operation(epsilon=epsilon, delta=0.0, mechanism="laplace")
        expected_max = max(epsilons)
        actual = accountant.total_epsilon()
        assert abs(actual - expected_max) < 1e-10, (
            f"Parallel composition failed: expected max={expected_max}, got {actual}, "
            f"epsilons={epsilons}"
        )

    @given(epsilons=epsilon_sequence())
    @settings(max_examples=500)
    def test_parallel_le_sequential(self, epsilons: list[float]) -> None:
        """Parallel composition must always produce epsilon <= sequential.

        Property: ε_parallel ≤ ε_sequential (max ≤ sum for positive values).
        """
        seq_accountant = CompositionAccountant(composition_type="sequential")
        par_accountant = CompositionAccountant(composition_type="parallel")
        for epsilon in epsilons:
            seq_accountant.add_operation(epsilon=epsilon, delta=0.0, mechanism="laplace")
            par_accountant.add_operation(epsilon=epsilon, delta=0.0, mechanism="laplace")
        assert par_accountant.total_epsilon() <= seq_accountant.total_epsilon() + 1e-12, (
            f"Parallel epsilon exceeded sequential: "
            f"parallel={par_accountant.total_epsilon()}, "
            f"sequential={seq_accountant.total_epsilon()}"
        )

    @given(
        epsilons=st.lists(
            st.floats(min_value=1e-6, max_value=2.0), min_size=2, max_size=15
        )
    )
    @settings(max_examples=500)
    def test_parallel_monotonic_with_new_element(self, epsilons: list[float]) -> None:
        """Adding a smaller epsilon in parallel should not increase total.

        Property: if new_eps <= current_max, then max stays the same.
        """
        if len(epsilons) < 2:
            return
        accountant = CompositionAccountant(composition_type="parallel")
        for epsilon in epsilons[:-1]:
            accountant.add_operation(epsilon=epsilon, delta=0.0, mechanism="laplace")
        before_max = accountant.total_epsilon()
        accountant.add_operation(epsilon=epsilons[-1], delta=0.0, mechanism="laplace")
        after_max = accountant.total_epsilon()
        expected_max = max(epsilons)
        assert abs(after_max - expected_max) < 1e-10


class TestRenyiComposition:
    """Rényi DP composition should produce tighter bounds than sequential."""

    @given(
        epsilons=st.lists(
            st.floats(min_value=0.01, max_value=1.0), min_size=2, max_size=10
        ),
        delta=st.floats(min_value=1e-8, max_value=1e-4),
    )
    @settings(max_examples=500)
    def test_renyi_tighter_than_sequential(
        self, epsilons: list[float], delta: float
    ) -> None:
        """Rényi accountant should produce epsilon <= sequential composition epsilon.

        Property: ε_renyi(δ) ≤ ε_sequential for k > 1 operations.
        This is the core value proposition of the advanced composition theorem.
        """
        sequential = CompositionAccountant(composition_type="sequential")
        renyi = CompositionAccountant(composition_type="renyi")
        for epsilon in epsilons:
            sequential.add_operation(epsilon=epsilon, delta=delta, mechanism="gaussian")
            renyi.add_operation(epsilon=epsilon, delta=delta, mechanism="gaussian")
        seq_total = sequential.total_epsilon()
        renyi_result = renyi.convert_to_epsilon_delta(delta=delta)
        renyi_total = renyi_result.epsilon
        # Rényi should be tighter (or equal) for k >= 2 operations
        # Allow small numerical tolerance
        assert renyi_total <= seq_total + 1e-6, (
            f"Rényi ({renyi_total}) not tighter than sequential ({seq_total}) "
            f"for {len(epsilons)} operations"
        )

    @given(
        epsilons=st.lists(
            st.floats(min_value=0.1, max_value=0.5), min_size=5, max_size=15
        )
    )
    @settings(max_examples=300)
    def test_renyi_empty_is_zero(self, epsilons: list[float]) -> None:
        """An empty Rényi accountant must return 0 epsilon.

        Property: ε_renyi with no operations = 0.
        """
        accountant = CompositionAccountant(composition_type="renyi")
        assert accountant.total_epsilon() == 0.0

    @given(
        epsilon=st.floats(min_value=0.01, max_value=2.0),
        delta=st.floats(min_value=1e-8, max_value=1e-4),
    )
    @settings(max_examples=300)
    def test_renyi_single_operation_equals_sequential(
        self, epsilon: float, delta: float
    ) -> None:
        """A single operation should give same result regardless of composition type.

        Property: For k=1, all composition theorems give ε = ε_1.
        """
        seq = CompositionAccountant(composition_type="sequential")
        renyi = CompositionAccountant(composition_type="renyi")
        seq.add_operation(epsilon=epsilon, delta=delta, mechanism="gaussian")
        renyi.add_operation(epsilon=epsilon, delta=delta, mechanism="gaussian")
        assert abs(seq.total_epsilon() - renyi.total_epsilon()) < 1e-10, (
            f"Single operation: sequential={seq.total_epsilon()}, "
            f"renyi={renyi.total_epsilon()}"
        )


class TestBudgetExhaustion:
    """Budget exhaustion detection must be exact — no overspend allowed."""

    @given(
        budget=st.floats(min_value=0.1, max_value=10.0),
        operations=st.lists(
            st.floats(min_value=0.01, max_value=1.0), min_size=1, max_size=50
        ),
    )
    @settings(max_examples=500)
    def test_budget_exhaustion_detected_correctly(
        self, budget: float, operations: list[float]
    ) -> None:
        """BudgetExhaustedError must be raised at the exact moment budget would be exceeded.

        Property: No operation should succeed if it would push total over budget.
        No operation should be rejected if it fits within remaining budget.
        """
        service = BudgetService(max_epsilon=budget)
        cumulative = 0.0
        for epsilon in operations:
            if cumulative + epsilon > budget + 1e-12:
                # This should raise
                with pytest.raises(BudgetExhaustedError):
                    service.check_and_consume(epsilon)
                break
            else:
                # This should succeed
                service.check_and_consume(epsilon)
                cumulative += epsilon

    @given(budget=st.floats(min_value=0.5, max_value=5.0))
    @settings(max_examples=300)
    def test_exact_budget_consumption_succeeds(self, budget: float) -> None:
        """Consuming exactly the full budget must succeed.

        Property: check_and_consume(budget) does not raise for empty service.
        """
        service = BudgetService(max_epsilon=budget)
        # Consuming the exact budget should succeed
        service.check_and_consume(budget)
        assert abs(service.consumed_epsilon - budget) < 1e-12
        assert service.remaining_epsilon >= -1e-12

    @given(
        budget=st.floats(min_value=0.5, max_value=5.0),
        overage=st.floats(min_value=1e-6, max_value=0.1),
    )
    @settings(max_examples=300)
    def test_over_budget_raises(self, budget: float, overage: float) -> None:
        """Consuming even slightly over budget must raise BudgetExhaustedError.

        Property: check_and_consume(budget + overage) raises for empty service.
        """
        service = BudgetService(max_epsilon=budget)
        with pytest.raises(BudgetExhaustedError):
            service.check_and_consume(budget + overage)

    @given(
        budget=st.floats(min_value=1.0, max_value=5.0),
        first_ops=st.lists(
            st.floats(min_value=0.01, max_value=0.1), min_size=1, max_size=5
        ),
    )
    @settings(max_examples=200)
    def test_remaining_budget_updated_correctly(
        self, budget: float, first_ops: list[float]
    ) -> None:
        """remaining_epsilon must exactly reflect consumed budget.

        Property: remaining = budget - sum(consumed_ops).
        """
        service = BudgetService(max_epsilon=budget)
        total_consumed = 0.0
        for epsilon in first_ops:
            if total_consumed + epsilon <= budget:
                service.check_and_consume(epsilon)
                total_consumed += epsilon
        expected_remaining = budget - total_consumed
        assert abs(service.remaining_epsilon - expected_remaining) < 1e-10


class TestCompositionEdgeCases:
    """Edge cases for composition accountant."""

    def test_empty_accountant_returns_zero(self) -> None:
        """Empty accountant of any type returns 0.0."""
        for composition_type in ("sequential", "parallel", "renyi"):
            accountant = CompositionAccountant(composition_type=composition_type)  # type: ignore[arg-type]
            assert accountant.total_epsilon() == 0.0

    def test_invalid_epsilon_raises(self) -> None:
        """Non-positive epsilon must raise ValueError."""
        accountant = CompositionAccountant()
        with pytest.raises(ValueError, match="Epsilon must be positive"):
            accountant.add_operation(epsilon=0.0, delta=0.0, mechanism="laplace")
        with pytest.raises(ValueError, match="Epsilon must be positive"):
            accountant.add_operation(epsilon=-1.0, delta=0.0, mechanism="laplace")

    def test_invalid_delta_raises(self) -> None:
        """Delta out of [0, 1) must raise ValueError."""
        accountant = CompositionAccountant()
        with pytest.raises(ValueError, match="Delta must be in"):
            accountant.add_operation(epsilon=1.0, delta=1.0, mechanism="gaussian")
        with pytest.raises(ValueError, match="Delta must be in"):
            accountant.add_operation(epsilon=1.0, delta=-0.1, mechanism="gaussian")

    def test_reset_clears_all_operations(self) -> None:
        """After reset, accountant returns 0 and has no operations."""
        accountant = CompositionAccountant()
        for _ in range(5):
            accountant.add_operation(epsilon=1.0, delta=0.0, mechanism="laplace")
        assert accountant.operation_count() == 5
        accountant.reset()
        assert accountant.operation_count() == 0
        assert accountant.total_epsilon() == 0.0

    def test_invalid_max_epsilon_raises(self) -> None:
        """BudgetService with non-positive max_epsilon must raise."""
        with pytest.raises(ValueError):
            BudgetService(max_epsilon=0.0)
        with pytest.raises(ValueError):
            BudgetService(max_epsilon=-1.0)
