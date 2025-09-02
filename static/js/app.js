// Rental Threshold Calculator - Frontend JavaScript
class RentalCalculator {
    constructor() {
        this.initializeEventListeners();
        this.results = null;
    }

    initializeEventListeners() {
        // Main calculation button
        document.getElementById('calculate-btn').addEventListener('click', (e) => {
            e.preventDefault();
            console.log('Calculate button clicked'); // Debug log
            this.calculateThreshold();
        });

        // Reset button
        document.getElementById('reset-btn').addEventListener('click', (e) => {
            e.preventDefault();
            this.resetForm();
        });

        // Offer checker
        document.getElementById('check-offer').addEventListener('click', () => {
            this.checkOffer();
        });

        // Pacing checker
        document.getElementById('check-pacing').addEventListener('click', () => {
            this.checkPacing();
        });

        // Relax threshold button
        document.getElementById('relax-threshold').addEventListener('click', () => {
            this.relaxThreshold();
        });

        // Export buttons
        document.getElementById('export-csv').addEventListener('click', () => {
            this.exportResults('csv');
        });

        document.getElementById('export-json').addEventListener('click', () => {
            this.exportResults('json');
        });

        // CSV upload
        document.getElementById('csv-upload').addEventListener('change', (e) => {
            this.handleCSVUpload(e);
        });

        // Live offer input changes
        document.getElementById('current-period').addEventListener('input', () => {
            this.updateDynamicThreshold();
        });
        
        document.getElementById('current-inventory').addEventListener('input', () => {
            this.updateDynamicThreshold();
        });
    }

    async calculateThreshold() {
        try {
            const formData = this.getFormData();
            console.log('Form data:', formData); // Debug log
            
            if (!this.validateFormData(formData)) {
                return;
            }

            this.showLoading('calculate-btn');

            const response = await fetch('/api/calculate', {
                method: 'POST',
                headers: {
                    'Content-Type': 'application/json',
                },
                body: JSON.stringify(formData)
            });

            const data = await response.json();
            
            if (response.ok) {
                this.results = data;
                this.displayResults(data);
                this.showResultsPanel();
                this.showSuccess('Calculation completed successfully!');
            } else {
                this.showError('Calculation failed: ' + (data.error || 'Unknown error'));
            }
        } catch (error) {
            console.error('Error in calculateThreshold:', error);
            this.showError('Error: ' + error.message);
        } finally {
            this.hideLoading('calculate-btn');
        }
    }

    async checkOffer() {
        if (!this.results) {
            this.showError('Please calculate threshold first');
            return;
        }

        const offerPrice = parseFloat(document.getElementById('offer-price').value);
        const currentPeriod = parseInt(document.getElementById('current-period').value) || 0;
        const currentInventory = parseInt(document.getElementById('current-inventory').value) || this.results.config.inventory;
        const thresholdType = document.getElementById('threshold-type').value;

        if (isNaN(offerPrice)) {
            this.showError('Please enter a valid offer price');
            return;
        }

        this.showLoading('check-offer');

        try {
            const response = await fetch('/api/check-offer', {
                method: 'POST',
                headers: {
                    'Content-Type': 'application/json',
                },
                body: JSON.stringify({
                    config: this.results.config,
                    prices: this.results.prices,
                    offer_price: offerPrice,
                    current_period: currentPeriod,
                    current_inventory: currentInventory,
                    use_dynamic: thresholdType === 'dynamic'
                })
            });

            const data = await response.json();
            
            if (response.ok) {
                this.displayOfferResult(data);
            } else {
                this.showError('Offer check failed: ' + (data.error || 'Unknown error'));
            }
        } catch (error) {
            this.showError('Network error: ' + error.message);
        } finally {
            this.hideLoading('check-offer');
        }
    }

    async checkPacing() {
        // Get basic configuration directly from form inputs
        const inventory = parseInt(document.getElementById('inventory').value);
        const periods = parseInt(document.getElementById('periods').value);
        const targetLeftover = parseInt(document.getElementById('target-leftover').value) || 3;
        const failureThreshold = parseInt(document.getElementById('failure-threshold').value) || 5;
        
        const acceptedSoFar = parseInt(document.getElementById('accepted-so-far').value);
        const currentPeriod = parseInt(document.getElementById('current-period').value) || 0;

        // Validate required inputs
        if (!inventory || inventory <= 0) {
            this.showError('Please enter valid Starting Inventory');
            return;
        }
        
        if (!periods || periods <= 0) {
            this.showError('Please enter valid Number of Periods');
            return;
        }
        
        if (isNaN(acceptedSoFar)) {
            this.showError('Please enter valid number of accepted units');
            return;
        }

        this.showLoading('check-pacing');

        try {
            const response = await fetch('/api/check-pacing', {
                method: 'POST',
                headers: {
                    'Content-Type': 'application/json',
                },
                body: JSON.stringify({
                    config: {
                        inventory: inventory,
                        periods: periods,
                        target_leftover: targetLeftover,
                        failure_threshold: failureThreshold
                    },
                    accepted_so_far: acceptedSoFar,
                    current_period: currentPeriod
                })
            });

            const data = await response.json();
            
            if (response.ok) {
                this.displayPacingResult(data);
            } else {
                this.showError('Pacing check failed: ' + (data.error || 'Unknown error'));
            }
        } catch (error) {
            this.showError('Network error: ' + error.message);
        } finally {
            this.hideLoading('check-pacing');
        }
    }

    async relaxThreshold() {
        if (!this.results) {
            return;
        }

        try {
            const response = await fetch('/api/relax-threshold', {
                method: 'POST',
                headers: {
                    'Content-Type': 'application/json',
                },
                body: JSON.stringify({
                    current_threshold: this.results.static_analysis.operational_cutoff,
                    prices: this.results.prices
                })
            });

            const data = await response.json();
            
            if (response.ok) {
                // Update the displayed threshold
                document.getElementById('operational-cutoff').textContent = `$${data.new_threshold.toFixed(2)}`;
                this.results.static_analysis.operational_cutoff = data.new_threshold;
                this.showSuccess('Threshold relaxed to $' + data.new_threshold.toFixed(2));
            } else {
                this.showError('Failed to relax threshold: ' + (data.error || 'Unknown error'));
            }
        } catch (error) {
            this.showError('Network error: ' + error.message);
        }
    }

    getFormData() {
        return {
            inventory: parseInt(document.getElementById('inventory').value),
            periods: parseInt(document.getElementById('periods').value),
            cost: parseFloat(document.getElementById('cost').value),
            salvage: parseFloat(document.getElementById('salvage').value) || 0,
            arrival_rate: parseFloat(document.getElementById('arrival-rate').value),
            target_leftover: parseInt(document.getElementById('target-leftover').value) || 3,
            failure_threshold: parseInt(document.getElementById('failure-threshold').value) || 5,
            cost_floor: parseFloat(document.getElementById('cost-floor').value) || null,
            prices: this.parsePrices(document.getElementById('prices').value)
        };
    }

    parsePrices(priceString) {
        if (!priceString || priceString.trim() === '') {
            return [];
        }
        try {
            return priceString.split(',').map(p => parseFloat(p.trim())).filter(p => !isNaN(p) && p > 0);
        } catch (error) {
            console.error('Error parsing prices:', error);
            return [];
        }
    }

    validateFormData(data) {
        console.log('Validating form data:', data); // Debug log
        
        const required = ['inventory', 'periods', 'cost', 'arrival_rate'];
        for (let field of required) {
            if (!data[field] || data[field] <= 0 || isNaN(data[field])) {
                this.showError(`Please enter a valid ${field.replace('_', ' ')}`);
                return false;
            }
        }

        if (!data.prices || data.prices.length === 0) {
            this.showError('Please enter valid prices (comma-separated numbers)');
            return false;
        }

        return true;
    }

    displayResults(data) {
        const results = data.static_analysis;
        const dynamic = data.dynamic_analysis;
        const config = data.config;
        
        // Static analysis results
        document.getElementById('optimal-threshold').textContent = `$${results.threshold.toFixed(2)}`;
        document.getElementById('operational-cutoff').textContent = `$${results.operational_cutoff.toFixed(2)}`;
        document.getElementById('expected-accepts').textContent = results.expected_accepts.toFixed(1);
        document.getElementById('expected-leftover').textContent = results.expected_leftover.toFixed(1);
        document.getElementById('conditional-mean-price').textContent = `$${results.conditional_mean_price.toFixed(2)}`;
        document.getElementById('expected-profit').textContent = `$${results.expected_profit.toFixed(2)}`;
        
        // Dynamic analysis results
        document.getElementById('initial-value').textContent = `$${dynamic.initial_value.toFixed(2)}`;
        document.getElementById('initial-bid-price').textContent = `$${dynamic.initial_bid_price.toFixed(2)}`;
        document.getElementById('initial-threshold').textContent = `$${dynamic.initial_threshold.toFixed(2)}`;
        
        // Decision guidance
        this.displayDecisionGuidance(data);
        
        // Show threshold display container
        document.getElementById('offer-results-container').style.display = 'grid';
        this.updateDynamicThreshold();
    }
    
    displayDecisionGuidance(data) {
        const results = data.static_analysis;
        const dynamic = data.dynamic_analysis;
        const config = data.config;
        
        const guidanceList = document.getElementById('decision-guidance');
        const targetSellThrough = config.inventory - config.target_leftover;
        const pacingPeriod = Math.floor(config.periods / 2);
        
        const guidance = [
            `• Accept offers ≥ $${results.operational_cutoff.toFixed(2)} (static threshold)`,
            `• Accept offers ≥ $${dynamic.initial_threshold.toFixed(2)} (dynamic threshold, t=0)`,
            `• Target sell-through: ${targetSellThrough} units`,
            `• Monitor pacing at period ${pacingPeriod}`,
            `• Expected leftover: ${results.expected_leftover.toFixed(1)} units (target: ≤${config.target_leftover})`,
            results.expected_leftover > config.failure_threshold ? 
                `⚠️ Warning: Expected leftover (${results.expected_leftover.toFixed(1)}) exceeds failure threshold (${config.failure_threshold})` : 
                `✓ Expected leftover within acceptable range`
        ];
        
        guidanceList.innerHTML = guidance.map(item => `<li>${item}</li>`).join('');
    }

    async updateDynamicThreshold() {
        if (!this.results) return;

        const currentPeriod = parseInt(document.getElementById('current-period').value) || 0;
        const currentInventory = parseInt(document.getElementById('current-inventory').value) || this.results.config.inventory;
        
        try {
            const response = await fetch('/api/get-dynamic-threshold', {
                method: 'POST',
                headers: {
                    'Content-Type': 'application/json',
                },
                body: JSON.stringify({
                    config: this.results.config,
                    prices: this.results.prices,
                    current_period: currentPeriod,
                    current_inventory: currentInventory
                })
            });

            const data = await response.json();
            
            if (response.ok) {
                if (data.is_end_condition) {
                    document.getElementById('current-threshold').textContent = 'No offers accepted';
                    document.getElementById('current-threshold').style.color = '#e74c3c';
                } else {
                    document.getElementById('current-threshold').textContent = `$${data.dynamic_threshold.toFixed(2)}K`;
                    document.getElementById('current-threshold').style.color = '#3498db';
                }
                document.getElementById('static-comparison').textContent = `$${data.static_threshold.toFixed(2)}K`;
                
                // Update the info text with the message
                const infoElement = document.querySelector('.threshold-info p:last-child small');
                if (infoElement) {
                    infoElement.textContent = data.message;
                }
            }
        } catch (error) {
            console.error('Error updating dynamic threshold:', error);
        }
    }

    displayOfferResult(data) {
        const decisionText = document.getElementById('decision-text');
        const rationaleText = document.getElementById('rationale-text');
        const marginText = document.getElementById('margin-text');

        decisionText.textContent = data.decision;
        decisionText.className = data.accept ? 'accept' : 'reject';
        rationaleText.textContent = data.rationale;
        marginText.textContent = data.margin > 0 ? `Margin: $${data.margin.toFixed(2)}` : '';

        // The container is already visible, no need to show individually
    }

    displayPacingResult(data) {
        const resultDiv = document.getElementById('pacing-result');
        const statusText = document.getElementById('pacing-status');
        const recommendationText = document.getElementById('pacing-recommendation');
        const relaxBtn = document.getElementById('relax-threshold');

        statusText.textContent = data.status.replace('_', ' ').toUpperCase();
        recommendationText.textContent = data.recommendation;

        if (data.should_relax) {
            relaxBtn.style.display = 'inline-block';
        } else {
            relaxBtn.style.display = 'none';
        }

        resultDiv.style.display = 'block';
    }

    showResultsPanel() {
        document.getElementById('results-panel').style.display = 'block';
    }

    async handleCSVUpload(event) {
        const file = event.target.files[0];
        if (!file) return;

        const formData = new FormData();
        formData.append('csv_file', file);

        try {
            const response = await fetch('/api/upload-csv', {
                method: 'POST',
                body: formData
            });

            const data = await response.json();
            
            if (response.ok) {
                document.getElementById('prices').value = data.prices.join(', ');
                this.showSuccess('CSV uploaded successfully');
            } else {
                this.showError('CSV upload failed: ' + (data.error || 'Unknown error'));
            }
        } catch (error) {
            this.showError('Upload error: ' + error.message);
        }
    }

    async exportResults(format) {
        if (!this.results) {
            this.showError('No results to export');
            return;
        }

        try {
            const response = await fetch(`/api/export/${format}`, {
                method: 'POST',
                headers: {
                    'Content-Type': 'application/json',
                },
                body: JSON.stringify(this.results)
            });

            if (response.ok) {
                const blob = await response.blob();
                const url = window.URL.createObjectURL(blob);
                const a = document.createElement('a');
                a.style.display = 'none';
                a.href = url;
                a.download = `rental_threshold_results.${format}`;
                document.body.appendChild(a);
                a.click();
                window.URL.revokeObjectURL(url);
                document.body.removeChild(a);
            } else {
                const data = await response.json();
                this.showError('Export failed: ' + (data.error || 'Unknown error'));
            }
        } catch (error) {
            this.showError('Export error: ' + error.message);
        }
    }

    showLoading(buttonId) {
        const button = document.getElementById(buttonId);
        button.disabled = true;
        button.textContent = 'Calculating...';
        button.classList.add('loading');
        button.style.opacity = '0.7';
        
        // Show a calculation message
        this.showMessage('Calculating optimal threshold...', 'info');
    }

    hideLoading(buttonId) {
        const button = document.getElementById(buttonId);
        button.disabled = false;
        button.classList.remove('loading');
        button.style.opacity = '1';
        
        // Restore original text
        const originalTexts = {
            'calculate-btn': 'Calculate Threshold',
            'check-offer': 'Check Offer',
            'check-pacing': 'Check Pacing'
        };
        button.textContent = originalTexts[buttonId] || 'Submit';
    }

    showError(message) {
        this.showMessage(message, 'error');
    }

    showSuccess(message) {
        this.showMessage(message, 'success');
    }

    showMessage(message, type) {
        // Create or update message element
        let messageEl = document.getElementById('message');
        if (!messageEl) {
            messageEl = document.createElement('div');
            messageEl.id = 'message';
            messageEl.style.cssText = `
                position: fixed;
                top: 20px;
                right: 20px;
                padding: 15px 20px;
                border-radius: 4px;
                color: white;
                font-weight: bold;
                z-index: 1000;
                max-width: 300px;
                box-shadow: 0 4px 8px rgba(0,0,0,0.3);
                font-size: 14px;
            `;
            document.body.appendChild(messageEl);
        }

        messageEl.textContent = message;
        messageEl.className = type;
        
        // Set background color based on type
        if (type === 'error') {
            messageEl.style.backgroundColor = '#e74c3c';
        } else if (type === 'info') {
            messageEl.style.backgroundColor = '#3498db';
        } else {
            messageEl.style.backgroundColor = '#27ae60';
        }
        
        messageEl.style.display = 'block';

        // Auto-hide after different durations based on type
        const hideDelay = type === 'info' ? 2000 : 5000;
        setTimeout(() => {
            messageEl.style.display = 'none';
        }, hideDelay);
    }

    resetForm() {
        // Clear configuration form inputs
        document.getElementById('inventory').value = '';
        document.getElementById('periods').value = '';
        document.getElementById('cost').value = '';
        document.getElementById('salvage').value = '0';
        document.getElementById('arrival-rate').value = '';
        document.getElementById('target-leftover').value = '3';
        document.getElementById('failure-threshold').value = '5';
        document.getElementById('cost-floor').value = '';
        document.getElementById('prices').value = '';
        document.getElementById('csv-upload').value = '';

        // Clear live offer decision inputs
        document.getElementById('offer-price').value = '';
        document.getElementById('current-period').value = '';
        document.getElementById('current-inventory').value = '';
        document.getElementById('threshold-type').value = 'dynamic';

        // Clear pacing inputs
        document.getElementById('accepted-so-far').value = '';

        // Hide results panels
        document.getElementById('results-panel').style.display = 'none';
        document.getElementById('offer-results-container').style.display = 'none';
        document.getElementById('pacing-result').style.display = 'none';

        // Reset results data
        this.results = null;

        // Show success message
        this.showSuccess('Form reset successfully');
    }
}

// Initialize the calculator when DOM is loaded
document.addEventListener('DOMContentLoaded', () => {
    new RentalCalculator();
});