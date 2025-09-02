# Rental Threshold Calculator

A Python CLI program implementing dynamic programming-based revenue management for rental inventory optimization.

## Features

- **Static Threshold Analysis**: Empirical price distribution-based cutoff selection
- **Dynamic Programming (V2)**: Optimal bid-price policy with backward induction
- **Live Decision Mode**: Real-time accept/reject decisions for individual offers
- **Pacing Control**: Mid-horizon adjustments to stay on track
- **Penalty Management**: Piecewise penalty function for excess inventory
- **Multiple Output Formats**: Text, JSON, and CSV export options

## Installation

```bash
pip install -r requirements.txt
chmod +x rental_threshold_calculator_dynamic.py
```

## Usage

### Batch Planning Mode

Calculate optimal thresholds for the entire horizon:

```bash
python3 rental_threshold_calculator_dynamic.py \
  --inventory 10 \
  --periods 30 \
  --cost 50 \
  --arrival-rate 0.8 \
  --prices "45,50,55,60,65,70,75,80"
```

### Live Decision Mode

Make real-time accept/reject decisions:

```bash
python3 rental_threshold_calculator_dynamic.py \
  --inventory 10 \
  --periods 30 \
  --cost 50 \
  --arrival-rate 0.8 \
  --prices sample_prices.csv \
  --live \
  --offer-price 65 \
  --current-period 5 \
  --current-inventory 8
```

### Using Price Files

Create a CSV file with prices (one per line):
```
45
50
55
60
65
70
75
80
85
```

Then reference it:
```bash
python3 rental_threshold_calculator_dynamic.py \
  --inventory 10 \
  --periods 30 \
  --cost 50 \
  --arrival-rate 0.8 \
  --prices sample_prices.csv
```

### Output Formats

Export results as JSON:
```bash
python3 rental_threshold_calculator_dynamic.py \
  --inventory 10 \
  --periods 30 \
  --cost 50 \
  --arrival-rate 0.8 \
  --prices sample_prices.csv \
  --format json \
  --output results.json
```

## Key Parameters

- `--inventory, -X`: Starting inventory units
- `--periods, -T`: Time horizon (number of periods)
- `--cost, -c`: Unit cost/base rent
- `--arrival-rate`: Expected arrivals per period (or use `--total-arrivals`)
- `--prices`: Empirical price distribution (comma-separated or file)
- `--target-leftover`: Soft target for leftover inventory (default: 3)
- `--failure-threshold`: Hard failure threshold (default: 5)
- `--penalty-alpha`: Penalty coefficient for 3 < L ≤ 5 (default: 10)
- `--penalty-beta`: Penalty coefficient for L > 5 (default: 100)

## Algorithm Details

### Static Threshold (Algorithm A1)
1. Computes tail probabilities for each price level
2. Calculates expected accepts, leftover, and penalties
3. Selects threshold maximizing expected profit

### Dynamic Programming (V2)
1. Backward induction from terminal period T
2. Computes value function V_t(x) for each (time, inventory) state
3. Derives bid prices b_t(x) = V_t(x) - V_t(x-1)
4. Policy: accept if price ≥ cost + bid_price

### Penalty Function Φ(L)
- Φ(L) = 0 for L ≤ 3
- Φ(L) = α(L-3)² for 3 < L ≤ 5  
- Φ(L) = α(2)² + β(L-5)² for L > 5

## Example Output

```
============================================================
RENTAL THRESHOLD CALCULATOR RESULTS
============================================================

CONFIGURATION:
  Inventory (X): 10
  Periods (T): 30
  Unit Cost (c): $50.00
  Expected Arrivals (N): 24.0
  Target Leftover: 3

STATIC THRESHOLD ANALYSIS:
  Optimal Threshold: $65.00
  Expected Accepts: 10.0
  Expected Leftover: 0.0
  Expected Profit: $250.00

DYNAMIC PROGRAMMING ANALYSIS:
  Initial Value Function V_0(10): $284.53
  Initial Bid Price b_0(10): $22.36
  Initial Threshold: $72.36

DECISION GUIDANCE:
  • Accept offers ≥ $65.00 (static)
  • Accept offers ≥ $72.36 (dynamic, t=0)
  • Target sell-through: 7 units
  • Monitor pacing at period 15
```