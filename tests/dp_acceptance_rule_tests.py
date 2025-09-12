#!/usr/bin/env python3
"""
Targeted tests to validate that the DP acceptance rule includes cost (c).

We test the internal helper for a single arrival (k=1) where the correct
acceptance is: accept iff price >= c + bid_price. At T=1 with zero penalty and
zero salvage, bid_price = V_{t+1}(x) - V_{t+1}(x-1) = 0, so the rule reduces to
accept iff price >= c. The expected value should be E[max(price - c, 0)].

The current implementation compares price >= bid_price (missing +c), so for
T=1, X=1, it accepts all prices and returns E[price - c], which differs from
E[max(price - c, 0)].
"""

import os, sys
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))
from rental_threshold_calculator_dynamic import (
    RentalConfig, PriceDistribution, RentalThresholdCalculator
)


def approx(a, b, tol=1e-6):
    return abs(a - b) <= tol


def build_terminal_V(calc: RentalThresholdCalculator):
    """Build terminal value dictionary V[(T, x)] = s*x - penalty(x)."""
    V = {}
    T = calc.config.T
    for x in range(calc.config.X + 1):
        V[(T, x)] = calc.config.s * x - calc.penalty_fn(x)
    return V


def test_k1_acceptance_rule_includes_cost():
    # T=1, X=1, s=0, penalty target >= 1 so V_T(1)=0 and V_T(0)=0
    # prices in {40, 60}, cost c=50
    # Correct EV with one arrival: 0.5*max(40-50,0) + 0.5*max(60-50,0) = 5
    config = RentalConfig(X=1, T=1, c=50.0, s=0.0, arrival_rate=1.0,
                          target_leftover=3, failure_threshold=5, cost_floor=50.0)
    price_dist = PriceDistribution([40.0, 60.0])
    calc = RentalThresholdCalculator(config, price_dist)

    V = build_terminal_V(calc)
    ev_k1 = calc._compute_expected_value_given_k_arrivals(1, current_x=1, V=V, t=0)

    expected = 5.0
    if approx(ev_k1, expected):
        print('[PASS] k=1 acceptance includes cost')
    else:
        print('[FAIL] k=1 acceptance rule: expected', expected, 'got', ev_k1)
        raise SystemExit(1)


def main():
    test_k1_acceptance_rule_includes_cost()
    print('All acceptance rule tests passed.')


if __name__ == '__main__':
    main()
