#!/usr/bin/env python3
"""
Sanity tests for the dynamic programming algorithm in
`rental_threshold_calculator_dynamic.py`.

These tests validate small, analytically-checkable cases.
Run: python tests/dp_sanity_tests.py
"""

import os, sys

# Ensure project root is on sys.path when running as a script
CURRENT_DIR = os.path.dirname(__file__)
PROJECT_ROOT = os.path.abspath(os.path.join(CURRENT_DIR, os.pardir))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from rental_threshold_calculator_dynamic import (
    RentalConfig, PriceDistribution, RentalThresholdCalculator
)


def approx_equal(a, b, tol=1e-6):
    return abs(a - b) <= tol


def test_T1_X1_no_salvage_no_penalty():
    # One period, one unit, no penalty, salvage 0
    # Prices: [40, 60] with equal prob; cost = 50
    # Expected value: E[max(p - c, 0)] = 0.5*max(-10,0) + 0.5*max(10,0) = 5
    config = RentalConfig(
        X=1, T=1, c=50.0, s=0.0,
        arrival_rate=20.0,  # p_arrival ≈ 1
        target_leftover=3, failure_threshold=5,
        cost_floor=0.0,
    )
    price_dist = PriceDistribution([40.0, 60.0])
    calc = RentalThresholdCalculator(config, price_dist)
    dyn = calc.compute_dynamic_program()
    v = dyn.value_function[(0, 1)]
    assert approx_equal(v, 5.0, tol=1e-3), f"T1X1 expected ~5.0, got {v}"


def test_T1_X1_with_salvage():
    # One period, one unit, salvage s=5, no penalty
    # Expected value: E[max(p - c, s)] with p in {40,60}, c=50, s=5
    # Outcomes: at 40 => 5; at 60 => 10; expectation = 7.5
    config = RentalConfig(
        X=1, T=1, c=50.0, s=5.0,
        arrival_rate=20.0,  # p_arrival ≈ 1
        target_leftover=3, failure_threshold=5,
        cost_floor=0.0,
    )
    price_dist = PriceDistribution([40.0, 60.0])
    calc = RentalThresholdCalculator(config, price_dist)
    dyn = calc.compute_dynamic_program()
    v = dyn.value_function[(0, 1)]
    assert approx_equal(v, 7.5, tol=1e-3), f"T1X1 salvage expected ~7.5, got {v}"


def test_T2_X1_no_salvage_no_penalty():
    # Two periods, one unit, no penalty/salvage
    # Optimal is accept if p >= c each period. Expected value equals
    # E[max(P1, P2) - c]_+ for P in {40,60}. P(max=60) = 0.75, else 0.
    # => 0.75 * (60-50) = 7.5
    config = RentalConfig(
        X=1, T=2, c=50.0, s=0.0,
        arrival_rate=20.0,  # p_arrival ≈ 1
        target_leftover=3, failure_threshold=5,
        cost_floor=0.0,
    )
    price_dist = PriceDistribution([40.0, 60.0])
    calc = RentalThresholdCalculator(config, price_dist)
    dyn = calc.compute_dynamic_program()
    v = dyn.value_function[(0, 1)]
    assert approx_equal(v, 7.5, tol=1e-3), f"T2X1 expected ~7.5, got {v}"


def test_T2_X2_no_penalty_no_salvage():
    # Two periods, two units, no penalty/salvage
    # You can accept up to 2 sales; threshold c each period.
    # Each period E[(P-c)_+] = 0.5*10 = 5 => total expected = 10.
    config = RentalConfig(
        X=2, T=2, c=50.0, s=0.0,
        arrival_rate=20.0,  # p_arrival ≈ 1
        target_leftover=3, failure_threshold=5,
        cost_floor=0.0,
    )
    price_dist = PriceDistribution([40.0, 60.0])
    calc = RentalThresholdCalculator(config, price_dist)
    dyn = calc.compute_dynamic_program()
    v = dyn.value_function[(0, 2)]
    assert approx_equal(v, 10.0, tol=1e-3), f"T2X2 expected ~10.0, got {v}"


def test_penalty_induces_subcost_acceptance():
    # One period, two units, severe penalty for leftover (>0), no salvage.
    # target=0, failure=0, beta large => Φ(L) = 100*L^2 for L>0
    # Terminal: V_T(2) = -400, V_T(1) = -100.
    # Bid price at t=0, x=2: V_T(2) - V_T(1) = -300 => threshold = c-300.
    # With cost_floor=0, policy threshold <= 0 → always accept.
    config = RentalConfig(
        X=2, T=1, c=50.0, s=0.0,
        arrival_rate=1.0,
        target_leftover=0, failure_threshold=0,
        penalty_alpha=0.0, penalty_beta=100.0,
        cost_floor=0.0,
    )
    price_dist = PriceDistribution([1.0, 2.0, 3.0])
    calc = RentalThresholdCalculator(config, price_dist)
    dyn = calc.compute_dynamic_program()
    thr = dyn.policy[(0, 2)]
    assert thr <= 0.0, f"Penalty case threshold should be <=0, got {thr}"


def main():
    tests = [
        test_T1_X1_no_salvage_no_penalty,
        test_T1_X1_with_salvage,
        test_T2_X1_no_salvage_no_penalty,
        test_T2_X2_no_penalty_no_salvage,
        test_penalty_induces_subcost_acceptance,
    ]
    passed = 0
    for t in tests:
        t()
        passed += 1
        print(f"[PASS] {t.__name__}")
    print(f"All {passed} DP sanity tests passed.")


if __name__ == "__main__":
    main()
