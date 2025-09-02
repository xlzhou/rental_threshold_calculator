# Rental Threshold Calculator Dynamic - Line by Line Explanation

This document provides a comprehensive line-by-line explanation of the `rental_threshold_calculator_dynamic.py` file, which implements a revenue management system for rental inventory optimization.

## Overview

The program implements both static and dynamic programming approaches to optimize rental pricing decisions using threshold-based policies. It helps determine when to accept or reject rental offers to maximize revenue while managing inventory levels.

---

## Line-by-Line Explanation

### Lines 1-18: Header and Imports

```python
#!/usr/bin/env python3
```
**Line 1**: Shebang line that makes the script executable on Unix-like systems using Python 3.

```python
"""
Rental Threshold Calculator - CLI Program with Dynamic Programming (V2)

A revenue management system that uses threshold-based policies to optimize
inventory utilization for rental offers with dynamic bid pricing.
"""
```
**Lines 2-7**: Module docstring explaining the purpose - a revenue management system using threshold policies for rental optimization.

```python
import argparse
import sys
import csv
import json
import numpy as np
from dataclasses import dataclass, field
from typing import List, Tuple, Optional, Dict, Any
from enum import Enum
import math
```
**Lines 9-18**: Import statements for:
- `argparse`: Command-line argument parsing
- `sys`: System-specific parameters and functions
- `csv`: CSV file reading/writing
- `json`: JSON data handling
- `numpy`: Numerical computations
- `dataclasses`: Data structure definitions
- `typing`: Type hints for better code documentation
- `enum`: Enumeration support
- `math`: Mathematical functions

### Lines 20-23: Decision Mode Enumeration

```python
class DecisionMode(Enum):
    BATCH_PLANNING = "batch"
    LIVE_DECISION = "live"
```
**Lines 20-23**: Defines two operating modes:
- `BATCH_PLANNING`: Offline analysis mode for strategic planning
- `LIVE_DECISION`: Real-time decision mode for individual offers

### Lines 25-53: RentalConfig Data Class

```python
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
```
**Lines 25-39**: Configuration dataclass storing all system parameters:
- `X`: Initial inventory units
- `T`: Time horizon (number of periods)
- `c`: Base cost per unit
- `s`: Salvage value at end of horizon
- Arrival parameters (rate or total)
- Penalty function parameters for excess inventory

```python
def __post_init__(self):
    if self.cost_floor is None:
        self.cost_floor = self.c
    if self.arrival_rate is None and self.total_arrivals is None:
        raise ValueError("Must specify either arrival_rate or total_arrivals")
```
**Lines 41-45**: Post-initialization validation:
- Sets cost floor to unit cost if not specified
- Ensures at least one arrival parameter is provided

```python
@property
def N(self) -> float:
    """Total expected arrivals over horizon."""
    if self.total_arrivals is not None:
        return float(self.total_arrivals)
    return self.arrival_rate * self.T
```
**Lines 47-52**: Property method calculating total expected arrivals `N`:
- Uses direct value if provided
- Otherwise calculates as arrival_rate × time_horizon

### Lines 55-111: PriceDistribution Class

```python
@dataclass
class PriceDistribution:
    """Empirical price distribution F(p)."""
    prices: List[float]
    weights: Optional[List[float]] = None
```
**Lines 55-59**: Dataclass for empirical price distribution with optional weights.

```python
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
```
**Lines 61-75**: Post-initialization setup:
- Validates non-empty price list
- Sorts prices in ascending order
- Creates uniform weights if none provided
- Normalizes weights to probabilities
- Builds cumulative distribution function

```python
def _build_cdf(self):
    """Build cumulative distribution function."""
    self.cdf_values = []
    cumulative = 0.0
    for prob in self.probabilities:
        cumulative += prob
        self.cdf_values.append(cumulative)
```
**Lines 77-83**: Builds CDF by accumulating probabilities for each price point.

```python
def F(self, price: float) -> float:
    """CDF: P(P ≤ price)."""
    for i, p in enumerate(self.prices):
        if price < p:
            return self.cdf_values[i-1] if i > 0 else 0.0
    return 1.0
```
**Lines 85-90**: Cumulative Distribution Function - returns probability that a random price is ≤ given price.

```python
def tail_prob(self, price: float) -> float:
    """Tail probability: P(P ≥ price) = 1 - F(price)."""
    return 1.0 - self.F(price)
```
**Lines 92-94**: Tail probability - probability that random price is ≥ given price.

```python
def conditional_mean(self, threshold: float) -> float:
    """E[P | P ≥ threshold]."""
    total_weight = 0.0
    weighted_sum = 0.0
    
    for price, prob in zip(self.prices, self.probabilities):
        if price >= threshold:
            weighted_sum += price * prob
            total_weight += prob
    
    return weighted_sum / total_weight if total_weight > 0 else 0.0
```
**Lines 96-106**: Conditional expectation - expected price given price ≥ threshold.

```python
def unique_prices(self) -> List[float]:
    """Get unique price levels for threshold candidates."""
    return sorted(set(self.prices))
```
**Lines 108-110**: Returns sorted unique prices for threshold optimization.

### Lines 113-133: Result Data Classes

```python
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
```
**Lines 113-124**: Stores static threshold analysis results including profitability metrics.

```python
@dataclass
class DynamicResult:
    """Results from dynamic programming."""
    value_function: Dict[Tuple[int, int], float]  # V_t(x)
    bid_prices: Dict[Tuple[int, int], float]  # b_t(x)
    policy: Dict[Tuple[int, int], float]  # threshold at (t,x)
```
**Lines 127-132**: Stores dynamic programming results with state-dependent policies.

**Class purpose**:
    DynamicResult is a container for the outputs of a dynamic programming (DP) algorithm.
    In revenue management / inventory control, the DP state is usually defined by:
        t = time (remaining periods)
        x = state variable (e.g., inventory level)
**Attributes**:
    value_function
        A mapping (t, x) → V_t(x)
        V_t(x) is the optimal expected value (e.g., maximum expected revenue) when you are at time t with inventory x.
    bid_prices
        A mapping (t, x) → b_t(x)
        Represents the opportunity cost (or shadow price) of having one more unit of inventory at state (t, x).
        Common in revenue management to decide whether to accept/reject a booking at a given price.
    policy
        A mapping (t, x) → threshold
        Represents the decision rule (like the minimum acceptable price) at each state (t, x).
        For example: “At time t with x inventory left, only accept prices ≥ policy[(t, x)].”

This class stores the results of solving the DP model:
    **value_function** → what’s the maximum expected value at each state.
    **bid_prices** → the implicit value of keeping inventory for the future.
    **policy** → the rule (threshold) to follow in each state.

### Lines 135-154: PenaltyFunction Class

```python
class PenaltyFunction:
    """Piecewise penalty function Φ(L)."""
    """Parameters:
        target: the “safe” leftover level (≤ target is acceptable, no penalty).
        failure: a higher threshold, above which leftovers are considered a serious failure.
        alpha: coefficient for the moderate penalty zone.
        beta: coefficient for the severe penalty zone."""
    def __init__(self, target: int = 3, failure: int = 5, alpha: float = 10.0, beta: float = 100.0):
        self.target = target
        self.failure = failure
        self.alpha = alpha
        self.beta = beta
```
**Lines 135-142**: Penalty function for excess leftover inventory with escalating penalties.

```python
def __call__(self, leftover: float) -> float:
    """Compute penalty Φ(L)."""
    L = leftover
    if L <= self.target:                            #Case 1: Safe zone (L ≤ target)
        return 0.0                                      #No penalty for small or acceptable leftover.
    elif L <= self.failure:                         #Case 2: Moderate zone (target < L ≤ failure)
        return self.alpha * (L - self.target) ** 2      #Quadratic penalty (convex growth) using coefficient alpha.
                                                        #Grows as leftover increases beyond target, but still below “failure” level.
    else:                                           #Case 3: Severe zone (L > failure)
        base_penalty = self.alpha * (self.failure - self.target) ** 2
                                                        #Adds the maximum “moderate penalty” (ensures continuity at L=failure).
                                                        #Then adds an even steeper quadratic term with coefficient beta.
                                                        #So beyond failure, penalties grow much faster.
        return base_penalty + self.beta * (L - self.failure) ** 2
```
**Lines 144-153**: Piecewise penalty calculation:
- No penalty if leftover ≤ target
- Quadratic penalty (α) for moderate excess
- Additional severe penalty (β) for high excess
Φ(L)=0, if 0≤L≤3
Φ(L)=α(L−3)^2, if 3<L≤5
Φ(L)=α(5-3)^2+β(L−5)^2, if L>5 (β≫α)


### Lines 156-310: RentalThresholdCalculator Class

```python
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
```
**Lines 156-167**: Main calculator class initialization with configuration and penalty function setup.

#### Static Threshold Algorithm (Lines 169-206)

```python
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
```
**Lines 169-206**: Static threshold optimization algorithm:
- Tests each unique price as potential threshold
- Calculates expected accepts based on demand probability
- Computes unit margins and penalties
- Selects threshold maximizing profit (revenue - penalty)

#### Dynamic Programming Algorithm (Lines 208-260)

```python
def compute_dynamic_program(self) -> DynamicResult:
    """V2 Feature: Dynamic programming with bid prices."""
    T, X = self.config.T, self.config.X             #Horizon T, Inventory capacity X
    s = self.config.s                               #Salvage value s
    
    # Initialize value function and bid prices
    V = {}  # V_t(x)                                #V[(t, x)]will hold the optimal value at time t with inventory x.
    bid_prices = {}  # b_t(x) = V_t(x) - V_t(x-1)   #bid_prices[(t, x)]will hold the marginal value of one extra unit (the “shadow price”):
    
    #If you end with x units, you collect salvage s⋅x. But you also pay a penalty Φ(x) (implemented by self.penalty_fn(x)), which grows if leftover is too high.
    # Terminal condition: V_T(x) = s*x - Φ(x)
    #This anchors the backward recursion: future periods t=T−1,T−2,…,0 will be computed using this terminal payoff.
    for x in range(X + 1):
        V[(T, x)] = s * x - self.penalty_fn(x)
```
**Lines 208-219**: Dynamic programming initialization:
- Creates value function dictionary V_t(x)
- Sets terminal boundary condition with salvage value and penalties


```python
# Backward induction
for t in range(T - 1, -1, -1):                  #1) Backward over time, forward over inventory
    for x in range(X + 1):                      #Classic finite-horizon DP: compute V_t(x) from the already-known V_t+1(⋅).
        if x == 0:                              #2) Zero inventory case
            # No inventory left
            V[(t, x)] = V[(t + 1, x)]           #With no units left, you can’t sell; value just carries over: V_t(0)=V_t+1(0)
            bid_prices[(t, x)] = float('inf')   # Never accept，Bid price (shadow price of inventory) is set to +∞ to ensure “never accept”.
        else:                                   #3) Positive inventory: expected value over price distribution
            # Compute expected value over price distribution
            expected_value = 0.0
            #Iterate over each possible price with probability prob (a discrete demand/offer distribution).
            #The bid price b_t+1(x) = V_t+1(x)-V_t+1(x-1) is the marginal value of keeping one more unit for the future.
            for price, prob in zip(self.price_dist.prices, self.price_dist.probabilities):
                # Bid price: marginal value of selling one unit
                bid_price = V[(t + 1, x)] - V[(t + 1, x - 1)]
                                                #4)Accept vs reject for each price
                if price >= bid_price:  
                    # Accept: get price - cost + future value with x-1
                    accept_value = price - self.config.c + V[(t + 1, x - 1)]
                else:
                    # Reject: keep current inventory
                    accept_value = V[(t + 1, x)]
                #Accept: gain price−c now and move to x−1 next period.
                #Reject: gain nothing now and stay at x next period.
                #Take the better of the two and weight by probability; sum over all prices.
                #Bellman form (for each price draw P):
                #       V_t(x) = E_p[max{P - c + V_t+1(x-1),V_t+1(x)}]
                #The acceptance threshold implied by the max is:
                #Accept if P - c + V_t+1(x - 1) >= V_t+1(x) <==> P >= c + (V_t+1(x) - V_t+1(x-1)) = c + b_t+1(x) 

                # Take maximum of accept/reject
                reject_value = V[(t + 1, x)]
                expected_value += prob * max(accept_value, reject_value)
                #Note: your if price >= bid_price: condition uses b_t+1(x) (without +c). Because
                #you still do max(accept_value, reject_value) afterwards, the logic remains correct—
                #prices below c + b will still be rejected—but you can make the intent clearer by
                #thresholding on c + b 
                                                #5) Save state value and current-period bid price
            V[(t, x)] = expected_value
            bid_prices[(t, x)] = V[(t, x)] - V[(t, x - 1)] if x > 0 else float('inf')
```
**Lines 221-248**: Backward induction through time:
- For each state (time, inventory), computes optimal value
- Calculates bid prices as marginal value of inventory
- Makes optimal accept/reject decisions for each possible price

```python
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
```
**Lines 250-260**: Converts bid prices to operational thresholds and returns complete policy.

#### Decision Making Methods (Lines 262-309)

```python
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
```
**Lines 262-291**: Real-time decision method supporting both static and dynamic policies.

```python
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
```
**Lines 293-304**: Pacing control to monitor sell-through progress.

```python
def relax_threshold(self, current_threshold: float) -> float:
    """Relax threshold to next lower price level."""
    candidates = sorted([p for p in self.price_dist.unique_prices() if p < current_threshold])
    return candidates[-1] if candidates else current_threshold * 0.9
```
**Lines 306-309**: Threshold relaxation for pacing adjustments.

### Lines 312-336: Price Parsing Utility

```python
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
```
**Lines 312-335**: Flexible price input parser supporting comma-separated strings, CSV files, and text files.

### Lines 338-412: CLI Argument Parser

```python
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
```
**Lines 338-352**: Creates argument parser with usage examples.

**Lines 355-411**: Define all CLI arguments:
- Required parameters (inventory, periods, cost, arrivals, prices)
- Optional configuration parameters
- Mode selection (batch vs live)
- Output formatting options

### Lines 415-503: Result Formatting

```python
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
```
**Lines 415-448**: JSON output formatting with structured data.

**Lines 450-501**: Text output formatting with human-readable results and decision guidance.

### Lines 506-584: Main Function

```python
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
```
**Lines 506-537**: Main function setup - parse arguments, create configuration objects.

```python
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
```
**Lines 539-573**: Execution logic for both live decision and batch planning modes.

```python
except Exception as e:
    print(f"Error: {e}", file=sys.stderr)
    if args.verbose:
        import traceback
        traceback.print_exc()
    sys.exit(1)


if __name__ == '__main__':
    main()
```
**Lines 575-584**: Error handling and script entry point.

---

## Mathematical Models

### Static Threshold Model
The static approach solves:
```
max_q { E[accepts] × E[P|P≥q] - c × E[accepts] - Φ(E[leftover]) }
```

Where:
- `E[accepts] = min(N × P(P≥q), X)`
- `E[leftover] = max(X - E[accepts], 0)`
- `Φ(L)` is the piecewise penalty function

### Dynamic Programming Model
The dynamic model uses Bellman equation:
```
V_t(x) = E_P[ max{ P - c + V_{t+1}(x-1), V_{t+1}(x) } ]
```

With terminal condition:
```
V_T(x) = s×x - Φ(x)
```

The bid price policy: `b_t(x) = V_t(x) - V_t(x-1)`

---

## Usage Examples

### Batch Planning Mode
```bash
python rental_threshold_calculator_dynamic.py \
  --inventory 10 --periods 30 --cost 50 \
  --arrival-rate 0.8 --prices "45,50,55,60,65,70,75,80"
```

### Live Decision Mode
```bash
python rental_threshold_calculator_dynamic.py \
  --inventory 10 --periods 30 --cost 50 \
  --arrival-rate 0.8 --prices "45,50,55,60,65,70,75,80" \
  --live --offer-price 65 --current-period 5 --current-inventory 8
```

This comprehensive revenue management system provides both strategic planning capabilities and real-time decision support for rental inventory optimization.