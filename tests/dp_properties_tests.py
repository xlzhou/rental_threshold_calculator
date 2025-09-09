#!/usr/bin/env python3
"""
Property tests for the DP model:
 - Monotonicity: V(t, x) is nondecreasing in x
 - Thresholds relax over time: for fixed x, policy(t, x) >= policy(t+1, x)
 - Thresholds decrease with more inventory: policy(t, x) >= policy(t, x+1)

Run: python tests/dp_properties_tests.py
"""

import math
import os
import sys

CURRENT_DIR = os.path.dirname(__file__)
PROJECT_ROOT = os.path.abspath(os.path.join(CURRENT_DIR, os.pardir))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from rental_threshold_calculator_dynamic import (
    RentalConfig, PriceDistribution, RentalThresholdCalculator
)


def leq(a, b, tol=1e-9):
    return a <= b + tol


def geq(a, b, tol=1e-9):
    return a + tol >= b


def is_inf(x):
    return math.isinf(x)


def test_monotone_value_in_inventory():
    # Use zero penalties so V(t,x) should be nondecreasing in x
    config = RentalConfig(
        X=5, T=6, c=50.0, s=0.0,
        arrival_rate=0.6,  # maps to p_arrival = 1 - e^-0.6
        target_leftover=0, failure_threshold=0,
        penalty_alpha=0.0, penalty_beta=0.0,
        cost_floor=0.0,
    )
    price_dist = PriceDistribution([40.0, 60.0])
    calc = RentalThresholdCalculator(config, price_dist)
    dyn = calc.compute_dynamic_program()
    V = dyn.value_function
    # For all t, V(t, x+1) >= V(t, x)
    for t in range(config.T + 1):
        for x in range(config.X):
            assert geq(V[(t, x + 1)], V[(t, x)]), f"V not monotone at t={t}, x={x}: {V[(t, x+1)]} < {V[(t, x)]}"


def test_thresholds_relax_over_time():
    config = RentalConfig(
        X=4, T=6, c=50.0, s=0.0,
        arrival_rate=0.8,
        target_leftover=3, failure_threshold=5,
        cost_floor=0.0,
    )
    price_dist = PriceDistribution([40.0, 60.0])
    calc = RentalThresholdCalculator(config, price_dist)
    dyn = calc.compute_dynamic_program()
    policy = dyn.policy
    # For each x>=1, policy(t, x) >= policy(t+1, x) unless inf present
    for x in range(1, config.X + 1):
        for t in range(config.T - 1):
            a = policy[(t, x)]
            b = policy[(t + 1, x)]
            if is_inf(a) or is_inf(b):
                continue
            assert geq(a, b), f"Thresholds not relaxing over time at x={x}, t={t}: {a} < {b}"


def test_thresholds_drop_with_inventory():
    config = RentalConfig(
        X=5, T=5, c=50.0, s=0.0,
        arrival_rate=0.5,
        target_leftover=3, failure_threshold=5,
        cost_floor=0.0,
    )
    price_dist = PriceDistribution([40.0, 60.0])
    calc = RentalThresholdCalculator(config, price_dist)
    dyn = calc.compute_dynamic_program()
    policy = dyn.policy
    # For fixed t, policy(t, x) >= policy(t, x+1) unless inf present
    for t in range(config.T):
        for x in range(1, config.X):
            a = policy[(t, x)]
            b = policy[(t, x + 1)]
            if is_inf(a) or is_inf(b):
                continue
            assert geq(a, b), f"Thresholds not decreasing in inventory at t={t}, x={x}: {a} < {b}"


def main():
    tests = [
        test_monotone_value_in_inventory,
        test_thresholds_relax_over_time,
        test_thresholds_drop_with_inventory,
    ]
    for t in tests:
        t()
        print(f"[PASS] {t.__name__}")
    print("All DP property tests passed.")


if __name__ == "__main__":
    main()
