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
import tempfile
from werkzeug.utils import secure_filename

# Import the existing calculator logic
from rental_threshold_calculator_dynamic import (
    RentalConfig, PriceDistribution, RentalThresholdCalculator,
    PenaltyFunction, ThresholdResult, DynamicResult
)

app = Flask(__name__)
app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024  # 16MB max file size
app.config['UPLOAD_FOLDER'] = tempfile.gettempdir()

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
        
        return jsonify(response)
        
    except Exception as e:
        return jsonify({'error': str(e)}), 400

@app.route('/api/check-offer', methods=['POST'])
def check_offer():
    """Check whether to accept or reject a specific offer."""
    try:
        data = request.get_json()
        
        # Recreate configuration and calculator
        config_data = data['config']
        # Use unit information from original calculation
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
        
        price_dist = PriceDistribution(data['prices'])
        calculator = RentalThresholdCalculator(config, price_dist)
        
        # Check boundary conditions first
        current_period = data.get('current_period', 0)
        current_inventory = data.get('current_inventory', config.X)
        
        if current_period >= config.T:
            # End of horizon
            accept, rationale, margin = False, f"Reject: End of horizon (period {current_period} >= {config.T})", 0.0
        elif current_inventory <= 0:
            # No inventory
            accept, rationale, margin = False, f"Reject: No inventory remaining", 0.0
        else:
            # Recreate configuration and calculator with actual cost floor
            config_data = data['config']
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
            
            price_dist = PriceDistribution(data['prices'])
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
            
            # Convert rationale back to user-friendly units for display
            if unit == 'per_month' and offer_type == 'per_period':
                # Scale back the rationale values for per-month display
                display_rationale = rationale
                # Replace per-day values in rationale with per-month equivalents
                import re
                # Find price and threshold values in the rationale
                price_match = re.search(r'(\d+\.\d+) ≥ threshold (\d+\.\d+)', rationale)
                if not price_match:
                    price_match = re.search(r'(\d+\.\d+) < threshold (\d+\.\d+)', rationale)
                
                if price_match:
                    per_day_price = float(price_match.group(1))
                    per_day_threshold = float(price_match.group(2))
                    per_month_price = per_day_price * days_per_month
                    per_month_threshold = per_day_threshold * days_per_month
                    
                    # Replace the values in the rationale
                    display_rationale = rationale.replace(
                        f"{per_day_price:.2f}", f"{per_month_price:.2f}"
                    ).replace(
                        f"{per_day_threshold:.2f}", f"{per_month_threshold:.2f}"
                    )
                rationale = display_rationale
                
            # Scale margin for display if needed
            if unit == 'per_month' and offer_type == 'per_period':
                margin = margin * days_per_month
        
        return jsonify({
            'accept': accept,
            'decision': 'ACCEPT' if accept else 'REJECT',
            'rationale': rationale,
            'margin': margin
        })
        
    except Exception as e:
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
        
        # Convert prices to per-day if needed
        prices_in = data['prices']
        if unit == 'per_month':
            prices_in = [p / days_per_month for p in prices_in]
        price_dist = PriceDistribution(prices_in)
        calculator = RentalThresholdCalculator(config, price_dist)
        
        # Get dynamic programming results
        dynamic_result = calculator.compute_dynamic_program()
        static_result = calculator.compute_static_threshold()
        
        current_period = data.get('current_period', 0)
        current_inventory = data.get('current_inventory', config.X)
        
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

        # SOBP thresholds for requested duration (defaults to 1)
        duration = data.get('duration', 1)
        sobp_per, sobp_total = calculator.compute_sobp_threshold(duration, current_period, current_inventory)
        
        return jsonify({
            'dynamic_threshold': dynamic_threshold if dynamic_threshold != float('inf') else 999999,
            'static_threshold': static_threshold,
            'sobp_per_period_threshold': sobp_per,
            'sobp_total_threshold': sobp_total,
            'duration': duration,
            'current_period': current_period,
            'current_inventory': current_inventory,
            'is_end_condition': dynamic_threshold == float('inf'),
            'message': message
        })
        
    except Exception as e:
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
        
        current_threshold = data['current_threshold']
        prices = data['prices']
        
        # Find next lower price
        price_dist = PriceDistribution(prices)
        candidates = sorted([p for p in price_dist.unique_prices() if p < current_threshold])
        
        if candidates:
            new_threshold = candidates[-1]
        else:
            new_threshold = current_threshold * 0.9  # Fallback: reduce by 10%
        
        return jsonify({
            'new_threshold': new_threshold,
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

@app.errorhandler(404)
def not_found(error):
    return jsonify({'error': 'Not found'}), 404

@app.errorhandler(500)
def internal_error(error):
    return jsonify({'error': 'Internal server error'}), 500

if __name__ == '__main__':
    # Create necessary directories
    os.makedirs('templates', exist_ok=True)
    os.makedirs('static/css', exist_ok=True)
    os.makedirs('static/js', exist_ok=True)
    
    # Run the application
    app.run(debug=True, host='0.0.0.0', port=5001)
