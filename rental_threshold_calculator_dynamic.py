#!/usr/bin/env python3
"""
Rental Threshold Calculator - CLI Program with Dynamic Programming (V2)

A revenue management system that uses threshold-based policies to optimize
inventory utilization for rental offers with dynamic bid pricing.
"""

import argparse
import sys
import csv
import json
import numpy as np
from dataclasses import dataclass, field
from typing import List, Tuple, Optional, Dict, Any
from enum import Enum
import math


class DecisionMode(Enum):
    BATCH_PLANNING = "batch"
    LIVE_DECISION = "live"


@dataclass
class RentalConfig:
    """Configuration parameters for the rental threshold calculator."""
    X: int  # Starting inventory
    T: int  # Number of periods (time horizon)
    c: float  # Unit cost/base rent
    s: float = 0.0  # Salvage value per unit at T
    arrival_rate: Optional[float] = None  # Arrivals per period
    total_arrivals: Optional[int] = None  # Total expected arrivals N
    target_leftover: int = 3  # L* - soft target for leftover inventory
    failure_threshold: int = 5  # L_fail - failure threshold
    holding_cost: float = 0.0  # Holding cost per period h
    penalty_alpha: float = 10.0  # Penalty parameter α for 3 < L ≤ 5
    penalty_beta: float = 100.0  # Penalty parameter β for L > 5
    cost_floor: float = None  # Minimum acceptable price (default: c)
    
    def __post_init__(self):
        if self.cost_floor is None:
            self.cost_floor = self.c
        if self.arrival_rate is None and self.total_arrivals is None:
            raise ValueError("Must specify either arrival_rate or total_arrivals")
    
    @property
    def N(self) -> float:
        """Total expected arrivals over horizon."""
        if self.total_arrivals is not None:
            return float(self.total_arrivals)
        return self.arrival_rate * self.T


@dataclass
class PriceDistribution:
    """Empirical price distribution F(p)."""
    prices: List[float]
    weights: Optional[List[float]] = None
    
    def __post_init__(self):
        if not self.prices:
            raise ValueError("Price list cannot be empty")
        self.prices = sorted(self.prices)
        if self.weights is None:
            self.weights = [1.0] * len(self.prices)
        elif len(self.weights) != len(self.prices):
            raise ValueError("Weights must match price list length")
        
        # Normalize weights to probabilities
        total_weight = sum(self.weights)
        self.probabilities = [w / total_weight for w in self.weights]
        
        # Build CDF
        self._build_cdf()
    
    def _build_cdf(self):
        """Build cumulative distribution function."""
        self.cdf_values = []
        cumulative = 0.0
        for prob in self.probabilities:
            cumulative += prob
            self.cdf_values.append(cumulative)
    
    def F(self, price: float) -> float:
        """CDF: P(P ≤ price)."""
        for i, p in enumerate(self.prices):
            if price < p:
                return self.cdf_values[i-1] if i > 0 else 0.0
        return 1.0
    
    def tail_prob(self, price: float) -> float:
        """Tail probability: P(P ≥ price) = 1 - F(price)."""
        return 1.0 - self.F(price)
    
    def conditional_mean(self, threshold: float) -> float:
        """E[P | P ≥ threshold]."""
        total_weight = 0.0
        weighted_sum = 0.0
        
        for price, prob in zip(self.prices, self.probabilities):
            if price >= threshold:
                weighted_sum += price * prob
                total_weight += prob
        
        return weighted_sum / total_weight if total_weight > 0 else 0.0
    
    def unique_prices(self) -> List[float]:
        """Get unique price levels for threshold candidates."""
        return sorted(set(self.prices))


@dataclass
class ThresholdResult:
    """Results from threshold analysis."""
    threshold: float
    operational_cutoff: float
    expected_accepts: float
    expected_leftover: float
    conditional_mean_price: float
    expected_margin: float
    expected_penalty: float
    expected_profit: float
    pacing_status: str = "on_track"


@dataclass
class DynamicResult:
    """Results from dynamic programming."""
    value_function: Dict[Tuple[int, int], float]  # V_t(x)
    bid_prices: Dict[Tuple[int, int], float]  # b_t(x)
    policy: Dict[Tuple[int, int], float]  # threshold at (t,x)


class PenaltyFunction:
    """Piecewise penalty function Φ(L)."""
    
    def __init__(self, target: int = 3, failure: int = 5, alpha: float = 10.0, beta: float = 100.0):
        self.target = target
        self.failure = failure
        self.alpha = alpha
        self.beta = beta
    
    def __call__(self, leftover: float) -> float:
        """Compute penalty Φ(L)."""
        L = leftover
        if L <= self.target:
            return 0.0
        elif L <= self.failure:
            return self.alpha * (L - self.target) ** 2
        else:
            base_penalty = self.alpha * (self.failure - self.target) ** 2
            return base_penalty + self.beta * (L - self.failure) ** 2


class RentalThresholdCalculator:
    """Main calculator implementing both static and dynamic approaches."""
    
    def __init__(self, config: RentalConfig, price_dist: PriceDistribution):
        self.config = config
        self.price_dist = price_dist
        self.penalty_fn = PenaltyFunction(
            config.target_leftover, 
            config.failure_threshold,
            config.penalty_alpha,
            config.penalty_beta
        )
    
    def compute_static_threshold(self) -> ThresholdResult:
        """Algorithm A1: Static cutoff selection."""
        candidates = self.price_dist.unique_prices()
        best_score = float('-inf')
        best_result = None
        
        for q in candidates:
            # Compute expected accepts and leftover
            tail_prob = self.price_dist.tail_prob(q)
            expected_accepts = min(self.config.N * tail_prob, self.config.X)
            expected_leftover = max(self.config.X - expected_accepts, 0)
            
            # Compute expected margin
            conditional_mean = self.price_dist.conditional_mean(q)
            unit_margin = conditional_mean - self.config.c
            
            # Compute penalty
            expected_penalty = self.penalty_fn(expected_leftover)
            
            # Score: expected revenue - penalty
            score = expected_accepts * unit_margin - expected_penalty
            
            if score > best_score:
                best_score = score
                operational_cutoff = max(q, self.config.cost_floor)
                
                best_result = ThresholdResult(
                    threshold=q,
                    operational_cutoff=operational_cutoff,
                    expected_accepts=expected_accepts,
                    expected_leftover=expected_leftover,
                    conditional_mean_price=conditional_mean,
                    expected_margin=unit_margin,
                    expected_penalty=expected_penalty,
                    expected_profit=score
                )
        
        return best_result
    
    def compute_dynamic_program(self) -> DynamicResult:
        """V2 Feature: Dynamic programming with bid prices."""
        T, X = self.config.T, self.config.X
        s = self.config.s
        
        # Initialize value function and bid prices
        V = {}  # V_t(x)
        bid_prices = {}  # b_t(x) = V_t(x) - V_t(x-1)
        
        # Terminal condition: V_T(x) = s*x - Φ(x)
        for x in range(X + 1):
            V[(T, x)] = s * x - self.penalty_fn(x)
        
        # Backward induction
        for t in range(T - 1, -1, -1):
            for x in range(X + 1):
                if x == 0:
                    # No inventory left
                    V[(t, x)] = V[(t + 1, x)]
                    bid_prices[(t, x)] = float('inf')  # Never accept
                else:
                    # Compute expected value over price distribution
                    expected_value = 0.0
                    
                    for price, prob in zip(self.price_dist.prices, self.price_dist.probabilities):
                        # Bid price: marginal value of selling one unit
                        bid_price = V[(t + 1, x)] - V[(t + 1, x - 1)]
                        
                        if price >= bid_price:
                            # Accept: get price - cost + future value with x-1
                            accept_value = price - self.config.c + V[(t + 1, x - 1)]
                        else:
                            # Reject: keep current inventory
                            accept_value = V[(t + 1, x)]
                        
                        # Take maximum of accept/reject
                        reject_value = V[(t + 1, x)]
                        expected_value += prob * max(accept_value, reject_value)
                    
                    V[(t, x)] = expected_value
                    bid_prices[(t, x)] = V[(t, x)] - V[(t, x - 1)] if x > 0 else float('inf')
        
        # Convert bid prices to thresholds (bid price + cost)
        policy = {}
        for (t, x), bid_price in bid_prices.items():
            threshold = bid_price + self.config.c if bid_price < float('inf') else float('inf')
            policy[(t, x)] = max(threshold, self.config.cost_floor)
        
        return DynamicResult(
            value_function=V,
            bid_prices=bid_prices,
            policy=policy
        )
    
    def make_decision(self, price: float, current_time: int = 0, current_inventory: int = None, 
                     use_dynamic: bool = True) -> Tuple[bool, str, float]:
        """Make accept/reject decision for a single offer."""
        if current_inventory is None:
            current_inventory = self.config.X
        
        if use_dynamic:
            # Use dynamic programming policy
            dynamic_result = self.compute_dynamic_program()
            threshold = dynamic_result.policy.get((current_time, current_inventory), float('inf'))
            
            if price >= threshold:
                margin = price - self.config.c
                rationale = f"Accept: price {price:.2f} ≥ threshold {threshold:.2f} (margin: {margin:.2f})"
                return True, rationale, margin
            else:
                rationale = f"Reject: price {price:.2f} < threshold {threshold:.2f}"
                return False, rationale, 0.0
        else:
            # Use static threshold
            static_result = self.compute_static_threshold()
            threshold = static_result.operational_cutoff
            
            if price >= threshold:
                margin = price - self.config.c
                rationale = f"Accept: price {price:.2f} ≥ cutoff {threshold:.2f} (margin: {margin:.2f})"
                return True, rationale, margin
            else:
                rationale = f"Reject: price {price:.2f} < cutoff {threshold:.2f}"
                return False, rationale, 0.0
    
    def check_pacing(self, accepted_so_far: int, current_period: int) -> Tuple[str, bool]:
        """Check pacing status and recommend adjustment."""
        if current_period < self.config.T // 2:
            return "too_early", False
        
        # Mid-horizon checkpoint
        target_accepts = (self.config.X - self.config.target_leftover) * 0.5
        
        if accepted_so_far >= target_accepts:
            return "on_track", False
        else:
            return "behind", True  # Recommend relaxing cutoff
    
    def relax_threshold(self, current_threshold: float) -> float:
        """Relax threshold to next lower price level."""
        candidates = sorted([p for p in self.price_dist.unique_prices() if p < current_threshold])
        return candidates[-1] if candidates else current_threshold * 0.9


def parse_price_list(price_input: str) -> List[float]:
    """Parse price list from string (comma-separated or file path)."""
    try:
        # Try parsing as comma-separated values
        if ',' in price_input:
            return [float(p.strip()) for p in price_input.split(',')]
        
        # Try parsing as file path
        try:
            with open(price_input, 'r') as f:
                if price_input.endswith('.csv'):
                    reader = csv.reader(f)
                    prices = []
                    for row in reader:
                        prices.extend([float(val) for val in row if val.strip()])
                    return prices
                else:
                    # Assume text file with one price per line
                    return [float(line.strip()) for line in f if line.strip()]
        except FileNotFoundError:
            # Try parsing as single float
            return [float(price_input)]
    except ValueError as e:
        raise ValueError(f"Could not parse prices: {e}")


def create_cli_parser() -> argparse.ArgumentParser:
    """Create command line argument parser."""
    parser = argparse.ArgumentParser(
        description="Rental Threshold Calculator with Dynamic Programming",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Batch planning mode
  %(prog)s --inventory 10 --periods 30 --cost 50 --arrival-rate 0.8 \\
           --prices "45,50,55,60,65,70,75,80"
  
  # Live decision mode
  %(prog)s --inventory 10 --periods 30 --cost 50 --arrival-rate 0.8 \\
           --prices prices.csv --live --offer-price 65 --current-period 5 --current-inventory 8
        """
    )
    
    # Required parameters
    parser.add_argument('-X', '--inventory', type=int, required=True,
                       help='Starting inventory (X)')
    parser.add_argument('-T', '--periods', type=int, required=True,
                       help='Number of periods/time horizon (T)')
    parser.add_argument('-c', '--cost', type=float, required=True,
                       help='Unit cost/base rent (c)')
    
    # Arrival specification (one required)
    arrival_group = parser.add_mutually_exclusive_group(required=True)
    arrival_group.add_argument('--arrival-rate', type=float,
                             help='Arrivals per period')
    arrival_group.add_argument('--total-arrivals', type=int,
                             help='Total expected arrivals (N)')
    
    # Price distribution (required)
    parser.add_argument('--prices', type=str, required=True,
                       help='Empirical prices (comma-separated or file path)')
    parser.add_argument('--weights', type=str,
                       help='Price weights (comma-separated, same length as prices)')
    
    # Optional parameters
    parser.add_argument('-s', '--salvage', type=float, default=0.0,
                       help='Salvage value per unit (default: 0)')
    parser.add_argument('--target-leftover', type=int, default=3,
                       help='Target leftover inventory L* (default: 3)')
    parser.add_argument('--failure-threshold', type=int, default=5,
                       help='Failure threshold for leftover (default: 5)')
    parser.add_argument('--penalty-alpha', type=float, default=10.0,
                       help='Penalty parameter α (default: 10)')
    parser.add_argument('--penalty-beta', type=float, default=100.0,
                       help='Penalty parameter β (default: 100)')
    parser.add_argument('--cost-floor', type=float,
                       help='Minimum acceptable price (default: cost)')
    
    # Mode selection
    parser.add_argument('--live', action='store_true',
                       help='Live decision mode (default: batch planning)')
    parser.add_argument('--offer-price', type=float,
                       help='Offer price for live decision mode')
    parser.add_argument('--current-period', type=int, default=0,
                       help='Current period for live mode (default: 0)')
    parser.add_argument('--current-inventory', type=int,
                       help='Current inventory for live mode (default: starting inventory)')
    
    # Algorithm selection
    parser.add_argument('--static-only', action='store_true',
                       help='Use only static threshold (disable dynamic programming)')
    
    # Output options
    parser.add_argument('--output', type=str,
                       help='Output file path for results')
    parser.add_argument('--format', choices=['text', 'json', 'csv'], default='text',
                       help='Output format (default: text)')
    parser.add_argument('--verbose', '-v', action='store_true',
                       help='Verbose output')
    
    return parser


def format_results(static_result: ThresholdResult, dynamic_result: Optional[DynamicResult],
                  config: RentalConfig, format_type: str = 'text') -> str:
    """Format results for output."""
    if format_type == 'json':
        data = {
            'config': {
                'inventory': config.X,
                'periods': config.T,
                'cost': config.c,
                'salvage': config.s,
                'total_arrivals': config.N,
                'target_leftover': config.target_leftover,
                'failure_threshold': config.failure_threshold
            },
            'static_analysis': {
                'threshold': static_result.threshold,
                'operational_cutoff': static_result.operational_cutoff,
                'expected_accepts': static_result.expected_accepts,
                'expected_leftover': static_result.expected_leftover,
                'conditional_mean_price': static_result.conditional_mean_price,
                'expected_margin': static_result.expected_margin,
                'expected_penalty': static_result.expected_penalty,
                'expected_profit': static_result.expected_profit
            }
        }
        
        if dynamic_result:
            data['dynamic_analysis'] = {
                'initial_value': dynamic_result.value_function.get((0, config.X), 0),
                'initial_bid_price': dynamic_result.bid_prices.get((0, config.X), 0),
                'initial_threshold': dynamic_result.policy.get((0, config.X), 0)
            }
        
        return json.dumps(data, indent=2)
    
    elif format_type == 'text':
        output = []
        output.append("=" * 60)
        output.append("RENTAL THRESHOLD CALCULATOR RESULTS")
        output.append("=" * 60)
        output.append("")
        
        # Configuration
        output.append("CONFIGURATION:")
        output.append(f"  Inventory (X): {config.X}")
        output.append(f"  Periods (T): {config.T}")
        output.append(f"  Unit Cost (c): ${config.c:.2f}")
        output.append(f"  Salvage Value (s): ${config.s:.2f}")
        output.append(f"  Expected Arrivals (N): {config.N:.1f}")
        output.append(f"  Target Leftover: {config.target_leftover}")
        output.append(f"  Failure Threshold: {config.failure_threshold}")
        output.append("")
        
        # Static Analysis
        output.append("STATIC THRESHOLD ANALYSIS:")
        output.append(f"  Optimal Threshold: ${static_result.threshold:.2f}")
        output.append(f"  Operational Cutoff: ${static_result.operational_cutoff:.2f}")
        output.append(f"  Expected Accepts: {static_result.expected_accepts:.1f}")
        output.append(f"  Expected Leftover: {static_result.expected_leftover:.1f}")
        output.append(f"  Conditional Mean Price: ${static_result.conditional_mean_price:.2f}")
        output.append(f"  Expected Unit Margin: ${static_result.expected_margin:.2f}")
        output.append(f"  Expected Penalty: ${static_result.expected_penalty:.2f}")
        output.append(f"  Expected Profit: ${static_result.expected_profit:.2f}")
        output.append("")
        
        # Dynamic Analysis
        if dynamic_result:
            initial_value = dynamic_result.value_function.get((0, config.X), 0)
            initial_bid = dynamic_result.bid_prices.get((0, config.X), 0)
            initial_threshold = dynamic_result.policy.get((0, config.X), 0)
            
            output.append("DYNAMIC PROGRAMMING ANALYSIS:")
            output.append(f"  Initial Value Function V_0({config.X}): ${initial_value:.2f}")
            output.append(f"  Initial Bid Price b_0({config.X}): ${initial_bid:.2f}")
            output.append(f"  Initial Threshold: ${initial_threshold:.2f}")
            output.append("")
        
        # Decision guidance
        output.append("DECISION GUIDANCE:")
        output.append(f"  • Accept offers ≥ ${static_result.operational_cutoff:.2f} (static)")
        if dynamic_result:
            initial_threshold = dynamic_result.policy.get((0, config.X), 0)
            output.append(f"  • Accept offers ≥ ${initial_threshold:.2f} (dynamic, t=0)")
        output.append(f"  • Target sell-through: {config.X - config.target_leftover} units")
        output.append(f"  • Monitor pacing at period {config.T // 2}")
        
        return "\n".join(output)
    
    return "Unsupported format"


def main():
    """Main CLI entry point."""
    parser = create_cli_parser()
    args = parser.parse_args()
    
    try:
        # Parse prices
        prices = parse_price_list(args.prices)
        weights = None
        if args.weights:
            weights = [float(w.strip()) for w in args.weights.split(',')]
        
        # Create configuration
        config = RentalConfig(
            X=args.inventory,
            T=args.periods,
            c=args.cost,
            s=args.salvage,
            arrival_rate=args.arrival_rate,
            total_arrivals=args.total_arrivals,
            target_leftover=args.target_leftover,
            failure_threshold=args.failure_threshold,
            penalty_alpha=args.penalty_alpha,
            penalty_beta=args.penalty_beta,
            cost_floor=args.cost_floor
        )
        
        # Create price distribution
        price_dist = PriceDistribution(prices, weights)
        
        # Create calculator
        calculator = RentalThresholdCalculator(config, price_dist)
        
        if args.live:
            # Live decision mode
            if args.offer_price is None:
                print("Error: --offer-price required for live mode", file=sys.stderr)
                sys.exit(1)
            
            current_inventory = args.current_inventory or config.X
            use_dynamic = not args.static_only
            
            accept, rationale, margin = calculator.make_decision(
                args.offer_price, 
                args.current_period, 
                current_inventory,
                use_dynamic
            )
            
            print(f"DECISION: {'ACCEPT' if accept else 'REJECT'}")
            print(f"RATIONALE: {rationale}")
            if margin > 0:
                print(f"MARGIN: ${margin:.2f}")
        
        else:
            # Batch planning mode
            static_result = calculator.compute_static_threshold()
            dynamic_result = None if args.static_only else calculator.compute_dynamic_program()
            
            # Format and output results
            output = format_results(static_result, dynamic_result, config, args.format)
            
            if args.output:
                with open(args.output, 'w') as f:
                    f.write(output)
                print(f"Results written to {args.output}")
            else:
                print(output)
    
    except Exception as e:
        print(f"Error: {e}", file=sys.stderr)
        if args.verbose:
            import traceback
            traceback.print_exc()
        sys.exit(1)


if __name__ == '__main__':
    main()