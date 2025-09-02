# Rental Threshold Calculator - Web Application

A web-based rental threshold calculator that helps optimize inventory decisions using threshold-based policies. This application has been upgraded from CLI to a responsive web interface.

## Features

- **Configuration Panel**: Input inventory, periods, costs, and arrival rates
- **Price Distribution**: Enter empirical prices manually or upload CSV files
- **Threshold Analysis**: Calculate optimal thresholds and expected outcomes
- **Live Offer Checker**: Get accept/reject decisions for individual offers
- **Pacing Monitor**: Track progress and get threshold adjustment recommendations
- **Export Functionality**: Download results as CSV or JSON

## Quick Start

1. **Install Dependencies**:
   ```bash
   pip install -r requirements.txt
   ```

2. **Run the Web Application**:
   ```bash
   python app.py
   ```

3. **Open in Browser**:
   Navigate to `http://localhost:5001`

## Usage

### Basic Configuration
1. Enter your inventory parameters (starting inventory, time periods, unit cost)
2. Set arrival rate (offers per period)
3. Input empirical prices as comma-separated values or upload a CSV file
4. Click "Calculate Threshold" to get optimal recommendations

### Live Decision Making
1. After calculating thresholds, use the "Live Offer Decision" section
2. Enter an offer price and current state (period, inventory)
3. Get immediate accept/reject recommendation with rationale

### Pacing Monitoring
1. Track your progress against targets using the pacing widget
2. Enter accepted units so far and current period
3. Get status updates and threshold adjustment recommendations

### Export Results
- Click "Export CSV" or "Export JSON" to download your analysis results
- Results include configuration, calculations, and recommendations

## File Structure

```
rental_threshold_calculator/
├── app.py                              # Flask web server
├── templates/
│   └── index.html                      # Main web interface
├── static/
│   ├── css/
│   │   └── style.css                   # Responsive styling
│   └── js/
│       └── app.js                      # Frontend JavaScript
├── rental_threshold_calculator_dynamic.py  # Core calculator logic
├── requirements.txt                    # Python dependencies
└── README_WEB.md                      # This file
```

## API Endpoints

- `POST /api/calculate` - Calculate optimal threshold
- `POST /api/check-offer` - Check individual offer decision  
- `POST /api/check-pacing` - Monitor pacing status
- `POST /api/relax-threshold` - Adjust threshold down
- `POST /api/upload-csv` - Upload price data from CSV
- `POST /api/export/{format}` - Export results (csv/json)

## Example Usage

1. **Basic Setup**:
   - Inventory: 10 units
   - Periods: 30 days  
   - Cost: $70 per unit
   - Arrival Rate: 2 offers per day
   - Prices: 72, 72, 70, 68, 65, 62, 60, 60, 58, 58, 57, 56, 56, 55, 50, 50

2. **Expected Output**:
   - Optimal threshold around $65-68
   - Expected accepts: ~7-8 units
   - Expected leftover: ~2-3 units
   - Expected profit calculation with penalty adjustment

3. **Live Offer Decision Examples**:
   After calculating thresholds, test individual offers:
   
   - **Offer Price: 75K, Current Period: 5, Current Inventory: 8**
     - Expected: "ACCEPT - price 75.00 ≥ threshold 68.00 (margin: 5.00)"
   
   - **Offer Price: 55K, Current Period: 10, Current Inventory: 6** 
     - Expected: "REJECT - price 55.00 < threshold 68.00"
   
   - **Offer Price: 68K, Current Period: 15, Current Inventory: 4**
     - Expected: "ACCEPT - price 68.00 ≥ threshold 68.00 (margin: -2.00)"

4. **Pacing Status Examples**:
   The "Pacing Status" feature helps you monitor whether you're on track to meet your inventory targets during the rental period. Here's how to use it:
   
   **Minimal Setup Required:**
   - Starting Inventory: 10 servers
   - Number of Periods: 30 days
   - Target Leftover: 3 (optional, defaults to 3)
   
   **Target Logic:**
   - Target sell-through = 10 - 3 = 7 servers
   - Mid-horizon checkpoint = Day 15
   - Expected progress by day 15 = 3.5 servers (50% of target)
   
   **Usage Examples:**
   - **Day 10, Accepted: 2** → Status: "TOO EARLY" (wait until day 15+)
   - **Day 15, Accepted: 2** → Status: "BEHIND" (need 3.5+, recommend relax threshold)  
   - **Day 15, Accepted: 4** → Status: "ON TRACK" (meeting targets)
   - **Day 20, Accepted: 5** → Status: "ON TRACK" (good progress)

## Troubleshooting

- **Port 5001 in use**: Change the port in `app.py` line `app.run(port=5002)`
- **CSV upload fails**: Ensure CSV contains only numeric values
- **Calculation errors**: Check that all required fields are filled with valid numbers

## Technical Notes

- Built with Flask (Python web framework)
- Responsive design works on desktop and mobile
- Uses the existing dynamic programming calculator logic
- All calculations are performed server-side for security
- File uploads are temporarily stored and cleaned up automatically

