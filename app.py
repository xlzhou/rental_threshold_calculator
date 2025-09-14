#!/usr/bin/env python3
"""
Flask Web Application for Rental Threshold Calculator
Integrates the existing Python calculator logic with a web interface.
"""

from flask import Flask, render_template, request, jsonify, send_file
import csv
import io
import json
import os
import sys
import tempfile
import logging
from logging.handlers import RotatingFileHandler
from werkzeug.utils import secure_filename
import psutil
import gc
import time

# Import the existing calculator logic
from rental_threshold_calculator_dynamic import (
    RentalConfig, PriceDistribution, RentalThresholdCalculator,
    PenaltyFunction, ThresholdResult, DynamicResult
)

app = Flask(__name__)
app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024  # 16MB max file size
app.config['UPLOAD_FOLDER'] = tempfile.gettempdir()

# Configure logging
if not app.debug:
    # File handler with rotation
    file_handler = RotatingFileHandler('rental_calculator.log', maxBytes=10240000, backupCount=5)
    file_handler.setFormatter(logging.Formatter(
        '%(asctime)s %(levelname)s: %(message)s [in %(pathname)s:%(lineno)d]'
    ))
    file_handler.setLevel(logging.INFO)
    app.logger.addHandler(file_handler)
    app.logger.setLevel(logging.INFO)
    app.logger.info('Rental Calculator startup')

# Cache for calculators to avoid recomputing DP
calculator_cache = {}
cache_max_size = 10  # Limit cache size to prevent memory bloat

def get_cached_calculator(config_data, prices):
    """Get or create a cached calculator instance."""
    import hashlib
    import json
    
    # Create a cache key from config and prices
    cache_key_data = {
        'config': config_data,
        'prices': sorted(prices)  # Sort for consistent hashing
    }
    cache_key = hashlib.md5(json.dumps(cache_key_data, sort_keys=True).encode()).hexdigest()
    
    if cache_key in calculator_cache:
        print(f"[CACHE DEBUG] Cache HIT for calculator {cache_key[:8]}", file=sys.stderr)
        return calculator_cache[cache_key]
    
    print(f"[CACHE DEBUG] Cache MISS - creating new calculator {cache_key[:8]}", file=sys.stderr)
    
    # Create new calculator
    config = RentalConfig(
        X=config_data['inventory'],
        T=config_data['periods'],
        c=config_data['cost'],
        s=config_data.get('salvage', 0.0),
        total_arrivals=config_data['total_arrivals'],
        target_leftover=config_data.get('target_leftover', 3),
        failure_threshold=config_data.get('failure_threshold', 5),
        cost_floor=config_data.get('cost_floor', 0.0)
    )
    
    price_dist = PriceDistribution(prices)
    calculator = RentalThresholdCalculator(config, price_dist)
    
    # Manage cache size - remove oldest entries if needed
    if len(calculator_cache) >= cache_max_size:
        oldest_key = next(iter(calculator_cache))
        print(f"[CACHE DEBUG] Evicting oldest calculator {oldest_key[:8]}", file=sys.stderr)
        del calculator_cache[oldest_key]
    
    calculator_cache[cache_key] = calculator
    print(f"[CACHE DEBUG] Cached new calculator. Cache size: {len(calculator_cache)}", file=sys.stderr)
    
    return calculator

@app.route('/')
def index():
    """Serve the main web interface."""
    return render_template('index.html')

@app.route('/help')
def help():
    """Serve the help page."""
    return render_template('help.html')

@app.route('/api/calculate', methods=['POST'])
def calculate_threshold():
    """Calculate optimal threshold based on configuration."""
    import time
    import sys
    
    start_time = time.time()
    try:
        print(f"[API DEBUG] /api/calculate started at {time.strftime('%H:%M:%S')}", file=sys.stderr)
    except (OSError, IOError):
        # Fallback to app logger if stderr is unavailable (daemon mode)
        app.logger.info(f"[API DEBUG] /api/calculate started at {time.strftime('%H:%M:%S')}")
    
    try:
        data = request.get_json()
        
        # Create configuration with cost floor (default to cost to avoid 0-threshold surprises)
        cost_floor = data.get('cost_floor')
        if cost_floor is None:
            cost_floor = data['cost']
        
        # Unit handling
        unit = data.get('unit', 'per_day')
        days_per_month = int(data.get('days_per_month', 30) or 30)

        # Convert prices/costs to per-day if inputs are monthly
        # Store original cost_floor for precision-preserving display
        original_cost_floor = cost_floor
        c = data['cost']
        cost_floor_input = cost_floor
        prices_in = list(data['prices'])
        if unit == 'per_month':
            c = c / days_per_month
            cost_floor_input = cost_floor_input / days_per_month
            prices_in = [p / days_per_month for p in prices_in]

        config = RentalConfig(
            X=data['inventory'],
            T=data['periods'],
            c=c,
            s=data.get('salvage', 0.0),
            arrival_rate=data['arrival_rate'],
            target_leftover=data.get('target_leftover', 3),
            failure_threshold=data.get('failure_threshold', 5),
            cost_floor=cost_floor_input
        )
        
        # Create price distribution
        price_dist = PriceDistribution(prices_in)
        
        # Create calculator and compute results
        calculator = RentalThresholdCalculator(config, price_dist)
        static_result = calculator.compute_static_threshold()
        dynamic_result = calculator.compute_dynamic_program()
        
        # Format response
        response = {
            'config': {
                'inventory': config.X,
                'periods': config.T,
                'cost': config.c,
                'salvage': config.s,
                'total_arrivals': config.N,
                'target_leftover': config.target_leftover,
                'failure_threshold': config.failure_threshold,
                'cost_floor': config.cost_floor,
                'unit': unit,
                'days_per_month': days_per_month
            },
            'prices': data['prices'],
            'static_analysis': {
                'threshold': static_result.threshold,
                'operational_cutoff': static_result.operational_cutoff,
                'expected_accepts': static_result.expected_accepts,
                'expected_leftover': static_result.expected_leftover,
                'conditional_mean_price': static_result.conditional_mean_price,
                'expected_margin': static_result.expected_margin,
                'expected_penalty': static_result.expected_penalty,
                'expected_profit': static_result.expected_profit
            },
            'dynamic_analysis': {
                'initial_value': dynamic_result.value_function.get((0, config.X), 0),
                'initial_bid_price': dynamic_result.bid_prices.get((0, config.X), 0),
                'initial_threshold': dynamic_result.policy.get((0, config.X), 0)
            },
            'unit': unit,
            'days_per_month': days_per_month
        }
        
        end_time = time.time()
        total_time = end_time - start_time
        print(f"[API DEBUG] /api/calculate completed in {total_time:.3f} seconds", file=sys.stderr)
        
        return jsonify(response)
        
    except Exception as e:
        end_time = time.time()
        total_time = end_time - start_time
        error_msg = f"/api/calculate FAILED in {total_time:.3f} seconds: {str(e)}"
        print(f"[API DEBUG] {error_msg}", file=sys.stderr)
        app.logger.error(error_msg)
        return jsonify({'error': str(e)}), 400

@app.route('/api/check-offer', methods=['POST'])
def check_offer():
    """Check whether to accept or reject a specific offer."""
    import time
    import sys
    
    start_time = time.time()
    
    try:
        data = request.get_json()
        
        # Use cached calculator to avoid recomputing DP
        config_data = data['config']
        unit = config_data.get('unit', 'per_day')
        days_per_month = int(config_data.get('days_per_month', 30) or 30)
        
        # Get cached calculator - this is the KEY optimization
        calculator = get_cached_calculator(config_data, data['prices'])
        
        # Check boundary conditions first
        current_period = data.get('current_period', 0)
        current_inventory = data.get('current_inventory', config_data['inventory'])
        
        if current_period >= config_data['periods']:
            # End of horizon
            accept, rationale, margin = False, f"Reject: End of horizon (period {current_period} >= {config_data['periods']})", 0.0
        elif current_inventory <= 0:
            # No inventory
            accept, rationale, margin = False, f"Reject: No inventory remaining", 0.0
        else:
            # Recreate configuration and calculator with actual cost floor
            config_data = data['config']
            
            # Convert costs to per-day if needed (same as get-dynamic-threshold API)
            c = config_data['cost']
            cost_floor_input = config_data.get('cost_floor', 0.0)
            original_cost_floor = cost_floor_input  # Store original before conversion
            if unit == 'per_month':
                c = c / days_per_month
                cost_floor_input = cost_floor_input / days_per_month
            
            config = RentalConfig(
                X=config_data['inventory'],
                T=config_data['periods'],
                c=c,
                s=config_data.get('salvage', 0.0),
                total_arrivals=config_data['total_arrivals'],
                target_leftover=config_data.get('target_leftover', 3),
                failure_threshold=config_data.get('failure_threshold', 5),
                cost_floor=cost_floor_input
            )
            
            # Convert prices to per-day if needed (same as get-dynamic-threshold API)
            prices_in = data['prices']
            if unit == 'per_month':
                prices_in = [p / days_per_month for p in prices_in]
            price_dist = PriceDistribution(prices_in)
            calculator = RentalThresholdCalculator(config, price_dist)
            
            # Make decision based on user choice
            use_dynamic = data.get('use_dynamic', True)  # Default to dynamic
            # Convert per-period offer to per-day if needed
            offer_price = data['offer_price']
            offer_type = data.get('offer_type', 'per_period')
            original_offer_price = offer_price
            if offer_type == 'per_period' and unit == 'per_month':
                offer_price = offer_price / days_per_month
            
            # Debug logs removed - unit conversion working correctly

            accept, rationale, margin = calculator.make_decision(
                offer_price,
                current_period,
                current_inventory,
                use_dynamic=use_dynamic,
                duration=data.get('duration', 1),
                offer_type=offer_type
            )
            
            # Get structured threshold data for robust rendering
            duration = data.get('duration', 1)
            
            # Calculate the threshold used in decision
            if use_dynamic:
                sobp_per, sobp_total = calculator.compute_sobp_threshold(duration, current_period, current_inventory)
                if offer_type == 'total':
                    threshold_used_per_day = sobp_total / duration  # Convert back to per-day for consistency
                    threshold_type = "sobp_total"
                else:
                    threshold_used_per_day = sobp_per
                    threshold_type = "sobp_per_period"
            else:
                static_result = calculator.compute_static_threshold()
                threshold_used_per_day = static_result.operational_cutoff
                threshold_type = "static"
            
            # Convert values for display
            display_offer_price = round(original_offer_price, 2)
            if unit == 'per_month':
                # Use higher precision rounding to avoid display artifacts
                threshold_monthly = threshold_used_per_day * days_per_month
                # If very close to cost_floor, snap to it to avoid floating point display issues
                if abs(threshold_monthly - original_cost_floor) < 0.05:
                    display_threshold = round(original_cost_floor, 2)
                else:
                    display_threshold = round(threshold_monthly, 2)
                display_margin = round(margin * days_per_month, 2) if margin > 0 else 0.0
            else:
                # Similar logic for per-day units
                if abs(threshold_used_per_day - cost_floor_input) < 0.05:
                    display_threshold = round(cost_floor_input, 2)
                else:
                    display_threshold = round(threshold_used_per_day, 2)
                display_margin = round(margin, 2) if margin > 0 else 0.0
            
            # Generate clean rationale string
            decision_text = "Accept" if accept else "Reject"
            if offer_type == 'total':
                rationale = f"{decision_text}: total {display_offer_price:.2f} {'≥' if accept else '<'} threshold {display_threshold:.2f} (D={duration}"
                if accept and display_margin > 0:
                    rationale += f", margin: {display_margin:.2f}"
                rationale += ")"
            else:
                rationale = f"{decision_text}: per-period {display_offer_price:.2f} {'≥' if accept else '<'} threshold {display_threshold:.2f} (D={duration}"
                if accept and display_margin > 0:
                    rationale += f", margin: {display_margin:.2f}/period"
                rationale += ")"
        
        end_time = time.time()
        total_time = end_time - start_time
        print(f"[API DEBUG] /api/check-offer completed in {total_time:.3f} seconds", file=sys.stderr)
        
        return jsonify({
            'accept': accept,
            'decision': 'ACCEPT' if accept else 'REJECT',
            'rationale': rationale,
            'margin': display_margin,
            # Structured data for robust rendering
            'structured': {
                'offer_price': display_offer_price,
                'threshold': display_threshold,
                'threshold_type': threshold_type,
                'duration': duration,
                'offer_type': offer_type,
                'unit': unit,
                'margin': display_margin
            }
        })
        
    except Exception as e:
        end_time = time.time()
        total_time = end_time - start_time
        print(f"[API DEBUG] /api/check-offer FAILED in {total_time:.3f} seconds: {str(e)}", file=sys.stderr)
        return jsonify({'error': str(e)}), 400

@app.route('/api/get-dynamic-threshold', methods=['POST'])
def get_dynamic_threshold():
    """Get the dynamic threshold for a specific (time, inventory) state."""
    try:
        data = request.get_json()
        
        # Recreate configuration and calculator
        config_data = data['config']
        unit = config_data.get('unit', 'per_day')
        days_per_month = int(config_data.get('days_per_month', 30) or 30)

        config = RentalConfig(
            X=config_data['inventory'],
            T=config_data['periods'],
            c=config_data['cost'],
            s=config_data.get('salvage', 0.0),
            total_arrivals=config_data['total_arrivals'],
            target_leftover=config_data.get('target_leftover', 3),
            failure_threshold=config_data.get('failure_threshold', 5),
            cost_floor=config_data.get('cost_floor', 0.0)
        )
        
        print(f"[THRESHOLD DEBUG] Config: inventory={config.X}, periods={config.T}, cost={config.c}, cost_floor={config.cost_floor}", file=sys.stderr)
        
        # Convert prices to per-day if needed
        prices_in = data['prices']
        if unit == 'per_month':
            prices_in = [p / days_per_month for p in prices_in]
        price_dist = PriceDistribution(prices_in)
        calculator = RentalThresholdCalculator(config, price_dist)
        
        print(f"[THRESHOLD DEBUG] Prices (converted): {prices_in[:5]}... (showing first 5)", file=sys.stderr)
        
        # Get dynamic programming results
        dynamic_result = calculator.compute_dynamic_program()
        static_result = calculator.compute_static_threshold()
        
        current_period = data.get('current_period', 0)
        current_inventory = data.get('current_inventory', config.X)
        
        print(f"[THRESHOLD DEBUG] State: t={current_period}, x={current_inventory}", file=sys.stderr)
        
        # Handle boundary conditions
        if current_period >= config.T:
            # At or beyond horizon end - no more offers accepted
            dynamic_threshold = float('inf')
            message = "End of horizon - no offers accepted"
        elif current_inventory <= 0:
            # No inventory left - no offers accepted  
            dynamic_threshold = float('inf')
            message = "No inventory remaining"
        else:
            # Normal case - get threshold from policy
            dynamic_threshold = dynamic_result.policy.get((current_period, current_inventory), float('inf'))
            message = "Normal operation"
        
        static_threshold = static_result.operational_cutoff
        print(f"[THRESHOLD DEBUG] Dynamic threshold: {dynamic_threshold}, Static threshold: {static_threshold}", file=sys.stderr)

        # SOBP thresholds for requested duration (defaults to 1)
        duration = data.get('duration', 1)
        print(f"[THRESHOLD DEBUG] Requested duration: {duration}", file=sys.stderr)
        sobp_per, sobp_total = calculator.compute_sobp_threshold(duration, current_period, current_inventory)
        
        print(f"[THRESHOLD DEBUG] SOBP: per={sobp_per}, total={sobp_total}, duration={duration}", file=sys.stderr)
        
        return jsonify({
            'dynamic_threshold': dynamic_threshold if dynamic_threshold != float('inf') else 999999,
            'static_threshold': static_threshold,
            'sobp_per_period_threshold': sobp_per,
            'sobp_total_threshold': sobp_total,  # Don't scale this - let frontend decide
            'duration': duration,
            'current_period': current_period,
            'current_inventory': current_inventory,
            'is_end_condition': dynamic_threshold == float('inf'),
            'message': message,
            # Add unit information for frontend scaling decisions
            'unit_info': {
                'calculation_unit': 'per_day',
                'display_unit': unit,
                'scale_factor': days_per_month if unit == 'per_month' else 1
            }
        })
        
    except Exception as e:
        print(f"[THRESHOLD DEBUG] ERROR: {str(e)}", file=sys.stderr)
        import traceback
        traceback.print_exc()
        return jsonify({'error': str(e)}), 400

@app.route('/api/check-pacing', methods=['POST'])
def check_pacing():
    """Check pacing status and recommend adjustments."""
    try:
        data = request.get_json()
        
        # Get simple config data directly (no need for full RentalConfig)
        config_data = data['config']
        inventory = config_data['inventory']
        periods = config_data['periods']
        target_leftover = config_data.get('target_leftover', 3)
        
        current_period = data['current_period']
        accepted_so_far = data['accepted_so_far']
        
        # Simple pacing logic without needing full config object
        if current_period < periods // 2:
            status = 'too_early'
            should_relax = False
        else:
            # Mid-horizon checkpoint
            target_accepts = (inventory - target_leftover) * 0.5
            if accepted_so_far >= target_accepts:
                status = 'on_track'
                should_relax = False
            else:
                status = 'behind'
                should_relax = True
        
        # Generate recommendation text
        recommendations = {
            'too_early': 'Too early for mid-horizon pacing check',
            'on_track': 'On track to meet targets. Maintain current threshold.',
            'behind': 'Behind target. Consider relaxing threshold to increase acceptance rate.'
        }
        
        return jsonify({
            'status': status,
            'should_relax': should_relax,
            'recommendation': recommendations.get(status, 'Unknown status')
        })
        
    except Exception as e:
        return jsonify({'error': str(e)}), 400

@app.route('/api/relax-threshold', methods=['POST'])
def relax_threshold():
    """Relax threshold to next lower price level."""
    try:
        data = request.get_json()
        
        current_threshold = data['current_threshold']  # This is per-day from backend
        prices = data['prices']  # These are original per-month values from user
        unit = data.get('unit', 'per_day')
        days_per_month = int(data.get('days_per_month', 30) or 30)
        
        # Convert prices to per-day for comparison with current_threshold
        if unit == 'per_month':
            comparison_prices = [p / days_per_month for p in prices]
        else:
            comparison_prices = prices
        
        # Find next lower price
        price_dist = PriceDistribution(comparison_prices)
        candidates = sorted([p for p in price_dist.unique_prices() if p < current_threshold])
        
        if candidates:
            new_threshold = candidates[-1]
        else:
            new_threshold = current_threshold * 0.9  # Fallback: reduce by 10%
        
        return jsonify({
            'new_threshold': new_threshold,  # Returns per-day value for backend consistency
            'old_threshold': current_threshold
        })
        
    except Exception as e:
        return jsonify({'error': str(e)}), 400

@app.route('/api/upload-csv', methods=['POST'])
def upload_csv():
    """Handle CSV file upload for price data."""
    try:
        if 'csv_file' not in request.files:
            return jsonify({'error': 'No file uploaded'}), 400
        
        file = request.files['csv_file']
        if file.filename == '':
            return jsonify({'error': 'No file selected'}), 400
        
        if not file.filename.lower().endswith('.csv'):
            return jsonify({'error': 'File must be a CSV'}), 400
        
        # Read and parse CSV
        stream = io.StringIO(file.stream.read().decode("UTF8"), newline=None)
        csv_reader = csv.reader(stream)
        
        prices = []
        for row in csv_reader:
            for cell in row:
                try:
                    price = float(cell.strip())
                    prices.append(price)
                except (ValueError, AttributeError):
                    continue  # Skip non-numeric values
        
        if not prices:
            return jsonify({'error': 'No valid prices found in CSV'}), 400
        
        return jsonify({
            'prices': prices,
            'count': len(prices)
        })
        
    except Exception as e:
        return jsonify({'error': str(e)}), 400

@app.route('/api/export/<format_type>', methods=['POST'])
def export_results(format_type):
    """Export results in specified format."""
    try:
        data = request.get_json()
        
        if format_type == 'csv':
            # Create CSV output
            output = io.StringIO()
            writer = csv.writer(output)
            
            # Header
            writer.writerow(['Metric', 'Value'])
            
            # Configuration
            writer.writerow(['Configuration', ''])
            config = data['config']
            writer.writerow(['Inventory', config['inventory']])
            writer.writerow(['Periods', config['periods']])
            writer.writerow(['Cost', config['cost']])
            writer.writerow(['Total Arrivals', config['total_arrivals']])
            
            # Results
            writer.writerow(['', ''])
            writer.writerow(['Results', ''])
            static = data['static_analysis']
            writer.writerow(['Optimal Threshold', static['threshold']])
            writer.writerow(['Operational Cutoff', static['operational_cutoff']])
            writer.writerow(['Expected Accepts', static['expected_accepts']])
            writer.writerow(['Expected Leftover', static['expected_leftover']])
            writer.writerow(['Conditional Mean Price', static['conditional_mean_price']])
            writer.writerow(['Expected Profit', static['expected_profit']])
            
            # Create temporary file
            temp_file = tempfile.NamedTemporaryFile(mode='w', suffix='.csv', delete=False)
            temp_file.write(output.getvalue())
            temp_file.close()
            
            return send_file(temp_file.name, as_attachment=True, download_name='results.csv')
            
        elif format_type == 'json':
            # Create temporary file
            temp_file = tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False)
            json.dump(data, temp_file, indent=2)
            temp_file.close()
            
            return send_file(temp_file.name, as_attachment=True, download_name='results.json')
            
        else:
            return jsonify({'error': 'Unsupported format'}), 400
            
    except Exception as e:
        return jsonify({'error': str(e)}), 400

@app.route('/api/health', methods=['GET'])
def health_check():
    """Health check with memory and cache statistics."""
    try:
        process = psutil.Process()
        memory_info = process.memory_info()
        
        # Get cache statistics
        cache_stats = {
            'cache_size': len(calculator_cache),
            'cache_max_size': cache_max_size
        }
        
        # Memory usage in MB
        memory_stats = {
            'rss_mb': round(memory_info.rss / 1024 / 1024, 2),
            'vms_mb': round(memory_info.vms / 1024 / 1024, 2),
            'percent': round(process.memory_percent(), 2)
        }
        
        # Trigger garbage collection if memory usage is high
        if memory_stats['percent'] > 70:
            gc.collect()
            app.logger.warning(f"High memory usage detected: {memory_stats['percent']}%, triggered GC")
        
        return jsonify({
            'status': 'healthy',
            'memory': memory_stats,
            'cache': cache_stats,
            'timestamp': time.time()
        })
    except Exception as e:
        app.logger.error(f"Health check failed: {str(e)}")
        return jsonify({'error': str(e)}), 500

@app.route('/api/clear-cache', methods=['POST'])
def clear_cache():
    """Emergency cache clearing endpoint."""
    try:
        global calculator_cache
        cache_size_before = len(calculator_cache)
        calculator_cache.clear()
        gc.collect()
        
        app.logger.info(f"Cache manually cleared: {cache_size_before} -> 0")
        return jsonify({
            'status': 'success',
            'cleared_entries': cache_size_before,
            'message': 'Cache cleared and garbage collection triggered'
        })
    except Exception as e:
        app.logger.error(f"Cache clearing failed: {str(e)}")
        return jsonify({'error': str(e)}), 500

@app.errorhandler(404)
def not_found(error):
    return jsonify({'error': 'Not found'}), 404

@app.errorhandler(500)
def internal_error(error):
    app.logger.error(f"Internal server error: {str(error)}")
    return jsonify({'error': 'Internal server error'}), 500

if __name__ == '__main__':
    # Create necessary directories
    os.makedirs('templates', exist_ok=True)
    os.makedirs('static/css', exist_ok=True)
    os.makedirs('static/js', exist_ok=True)
    
    # Run the application
    app.run(debug=True, host='0.0.0.0', port=5001)
