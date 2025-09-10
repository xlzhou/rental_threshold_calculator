// Rental Threshold Calculator - Frontend JavaScript
class RentalCalculator {
    constructor() {
        this.initializeEventListeners();
        this.results = null;
        this.initializeI18n();
    }

    getCurrencySymbol() {
        // Return appropriate currency symbol based on language
        const isZh = window.i18n && window.i18n.getCurrentLanguage() === 'zh';
        return isZh ? '¥' : '$';
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

        const durationEl = document.getElementById('duration');
        if (durationEl) {
            durationEl.addEventListener('input', () => this.updateDynamicThreshold());
        }

        // Language selector
        document.getElementById('language-selector').addEventListener('change', (e) => {
            this.changeLanguage(e.target.value);
        });
    }

    initializeI18n() {
        if (window.i18n) {
            // Set initial language from i18n system
            const currentLang = window.i18n.getCurrentLanguage();
            document.getElementById('language-selector').value = currentLang;
        }
    }

    changeLanguage(language) {
        if (window.i18n) {
            window.i18n.setLanguage(language);
            this.updateDynamicTexts();
        }
    }

    updateDynamicTexts() {
        // Update any dynamic text content that may have been set by JavaScript
        const calculateBtn = document.getElementById('calculate-btn');
        const checkOfferBtn = document.getElementById('check-offer');
        const checkPacingBtn = document.getElementById('check-pacing');
        
        if (calculateBtn && !calculateBtn.disabled) {
            calculateBtn.textContent = window.i18n.t('calculate_threshold');
        }
        if (checkOfferBtn && !checkOfferBtn.disabled) {
            checkOfferBtn.textContent = window.i18n.t('check_offer');
        }
        if (checkPacingBtn && !checkPacingBtn.disabled) {
            checkPacingBtn.textContent = window.i18n.t('check_pacing');
        }

        // Update currency symbols in results if available
        if (this.results) {
            this.displayResults(this.results);
            this.updateDynamicThreshold();
        }
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
                this.showSuccess(window.i18n ? window.i18n.t('calculation_success') : 'Calculation completed successfully!');
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
            this.showError(window.i18n ? window.i18n.t('no_results') : 'Please calculate threshold first');
            return;
        }

        const offerPrice = parseFloat(document.getElementById('offer-price').value);
        const currentPeriod = parseInt(document.getElementById('current-period').value) || 0;
        const currentInventory = parseInt(document.getElementById('current-inventory').value) || this.results.config.inventory;
        const thresholdType = document.getElementById('threshold-type').value;
        const duration = parseInt(document.getElementById('duration').value) || 1;
        const offerType = document.getElementById('offer-type').value || 'per_period';

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
                    use_dynamic: thresholdType === 'dynamic',
                    duration: duration,
                    offer_type: offerType
                })
            });

            const data = await response.json();
            
            if (response.ok) {
                this.displayOfferResult(data);
                // Update thresholds (SOBP may depend on duration)
                this.updateDynamicThreshold();
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
                const currency = this.getCurrencySymbol();
                const scale = (this.results.unit === 'per_month') ? (this.results.days_per_month || 30) : 1;
                document.getElementById('operational-cutoff').textContent = `${currency}${(data.new_threshold * scale).toFixed(2)}`;
                this.results.static_analysis.operational_cutoff = data.new_threshold;
                this.showSuccess('Threshold relaxed to ' + currency + (data.new_threshold * scale).toFixed(2));
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
            unit: (document.getElementById('price-unit') && document.getElementById('price-unit').value) || 'per_day',
            days_per_month: parseInt(document.getElementById('days-per-month')?.value) || 30,
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
        const currency = this.getCurrencySymbol();
        const scale = (data.unit === 'per_month') ? (data.days_per_month || 30) : 1;
        
        // Static analysis results
        document.getElementById('optimal-threshold').textContent = `${currency}${(results.threshold * scale).toFixed(2)}`;
        document.getElementById('operational-cutoff').textContent = `${currency}${(results.operational_cutoff * scale).toFixed(2)}`;
        document.getElementById('expected-accepts').textContent = results.expected_accepts.toFixed(1);
        document.getElementById('expected-leftover').textContent = results.expected_leftover.toFixed(1);
        document.getElementById('conditional-mean-price').textContent = `${currency}${(results.conditional_mean_price * scale).toFixed(2)}`;
        document.getElementById('expected-profit').textContent = `${currency}${results.expected_profit.toFixed(2)}`;
        
        // Dynamic analysis results
        document.getElementById('initial-value').textContent = `${currency}${dynamic.initial_value.toFixed(2)}`;
        document.getElementById('initial-bid-price').textContent = `${currency}${(dynamic.initial_bid_price * scale).toFixed(2)}`;
        document.getElementById('initial-threshold').textContent = `${currency}${(dynamic.initial_threshold * scale).toFixed(2)}`;
        
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
        
        // Get current language and currency symbol
        const isZh = window.i18n && window.i18n.getCurrentLanguage() === 'zh';
        const currency = this.getCurrencySymbol();
        const scale = (data.unit === 'per_month') ? (data.days_per_month || 30) : 1;
        
        const guidance = [
            isZh ? 
                `• 接受报价 ≥ ${currency}${(results.operational_cutoff * scale).toFixed(2)}（静态阈值）` :
                `• Accept offers ≥ ${currency}${(results.operational_cutoff * scale).toFixed(2)} (static threshold)`,
            isZh ?
                `• 接受报价 ≥ ${currency}${(dynamic.initial_threshold * scale).toFixed(2)}（动态阈值，t=0）` :
                `• Accept offers ≥ ${currency}${(dynamic.initial_threshold * scale).toFixed(2)} (dynamic threshold, t=0)`,
            isZh ?
                `• 目标销售量：${targetSellThrough} 个单位` :
                `• Target sell-through: ${targetSellThrough} units`,
            isZh ?
                `• 在第 ${pacingPeriod} 周期监控进度` :
                `• Monitor pacing at period ${pacingPeriod}`,
            isZh ?
                `• 预期剩余：${results.expected_leftover.toFixed(1)} 个单位（目标：≤${config.target_leftover}）` :
                `• Expected leftover: ${results.expected_leftover.toFixed(1)} units (target: ≤${config.target_leftover})`,
            results.expected_leftover > config.failure_threshold ? 
                (isZh ? 
                    `⚠️ 警告：预期剩余（${results.expected_leftover.toFixed(1)}）超过失败阈值（${config.failure_threshold}）` :
                    `⚠️ Warning: Expected leftover (${results.expected_leftover.toFixed(1)}) exceeds failure threshold (${config.failure_threshold})`) : 
                (isZh ?
                    `✓ 预期剩余在可接受范围内` :
                    `✓ Expected leftover within acceptable range`)
        ];
        
        guidanceList.innerHTML = guidance.map(item => `<li>${item}</li>`).join('');
    }

    async updateDynamicThreshold() {
        if (!this.results) return;

        const currentPeriod = parseInt(document.getElementById('current-period').value) || 0;
        const currentInventory = parseInt(document.getElementById('current-inventory').value) || this.results.config.inventory;
        const duration = parseInt(document.getElementById('duration').value) || 1;
        const currency = this.getCurrencySymbol();
        const scale = (this.results.unit === 'per_month') ? (this.results.days_per_month || 30) : 1;
        
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
                    current_inventory: currentInventory,
                    duration: duration
                })
            });

            const data = await response.json();
            
            if (response.ok) {
                if (data.is_end_condition) {
                    document.getElementById('current-threshold').textContent = 'No offers accepted';
                    document.getElementById('current-threshold').style.color = '#e74c3c';
                } else {
                    // Display SOBP per-period threshold as the primary current threshold for the selected D
                    document.getElementById('current-threshold').textContent = `${currency}${(data.sobp_per_period_threshold * scale).toFixed(2)}`;
                    document.getElementById('current-threshold').style.color = '#3498db';
                }
                document.getElementById('static-comparison').textContent = `${currency}${(data.static_threshold * scale).toFixed(2)}`;
                const sobpPer = document.getElementById('sobp-per-threshold');
                const sobpTot = document.getElementById('sobp-total-threshold');
                if (sobpPer) sobpPer.textContent = `${currency}${(data.sobp_per_period_threshold * scale).toFixed(2)} (D=${data.duration})`;
                if (sobpTot) sobpTot.textContent = `${currency}${data.sobp_total_threshold.toFixed(2)} (D=${data.duration})`;
                
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

        // Get current language
        const isZh = window.i18n && window.i18n.getCurrentLanguage() === 'zh';

        // Translate decision text
        let decisionDisplay = data.decision;
        if (isZh) {
            if (data.accept) {
                decisionDisplay = '接受报价';
            } else {
                decisionDisplay = '拒绝报价';
            }
        }

        decisionText.textContent = decisionDisplay;
        decisionText.className = data.accept ? 'accept' : 'reject';
        
        // Translate rationale (this comes from backend, so we handle common patterns)
        let rationaleDisplay = data.rationale;
        if (isZh && data.rationale) {
            // Common rationale patterns translation
            rationaleDisplay = data.rationale
                .replace(/Accept: offer price/g, '接受：报价')
                .replace(/Reject: offer price/g, '拒绝：报价')
                .replace(/exceeds threshold/g, '超过阈值')
                .replace(/below threshold/g, '低于阈值')
                .replace(/threshold/g, '阈值')
                .replace(/offer/g, '报价')
                .replace(/price/g, '价格');
        }
        
        rationaleText.textContent = rationaleDisplay;
        
        // Translate margin text
        const currency = this.getCurrencySymbol();
        const scale = (this.results && this.results.unit === 'per_month') ? (this.results.days_per_month || 30) : 1;
        const selectedOfferType = document.getElementById('offer-type') ? document.getElementById('offer-type').value : 'per_period';
        const marginValue = (selectedOfferType === 'per_period') ? (data.margin * scale) : data.margin;
        const marginDisplay = data.margin > 0 ? 
            (isZh ? `利润：${currency}${marginValue.toFixed(2)}` : `Margin: ${currency}${marginValue.toFixed(2)}`) : '';
        marginText.textContent = marginDisplay;

        // The container is already visible, no need to show individually
    }

    displayPacingResult(data) {
        const resultDiv = document.getElementById('pacing-result');
        const statusText = document.getElementById('pacing-status');
        const recommendationText = document.getElementById('pacing-recommendation');
        const relaxBtn = document.getElementById('relax-threshold');

        // Get current language
        const isZh = window.i18n && window.i18n.getCurrentLanguage() === 'zh';

        // Translate status
        let statusDisplay = data.status.replace('_', ' ').toUpperCase();
        if (isZh) {
            const statusTranslations = {
                'TOO EARLY': '太早',
                'ON TRACK': '按计划进行',
                'BEHIND': '进度落后'
            };
            statusDisplay = statusTranslations[statusDisplay] || statusDisplay;
        }
        statusText.textContent = statusDisplay;

        // Translate recommendation
        let recommendationDisplay = data.recommendation;
        if (isZh && data.recommendation) {
            recommendationDisplay = data.recommendation
                .replace(/Too early to check pacing/g, '检查进度为时过早')
                .replace(/On track with sales/g, '销售进度正常')
                .replace(/Behind on sales/g, '销售进度落后')
                .replace(/Consider relaxing threshold/g, '考虑放宽阈值')
                .replace(/target/g, '目标')
                .replace(/units/g, '单位')
                .replace(/accepted/g, '已接受')
                .replace(/so far/g, '到目前为止');
        }
        recommendationText.textContent = recommendationDisplay;

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
        button.textContent = window.i18n ? window.i18n.t('calculating') : 'Calculating...';
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
        
        // Restore original text using i18n
        const originalTexts = {
            'calculate-btn': window.i18n ? window.i18n.t('calculate_threshold') : 'Calculate Threshold',
            'check-offer': window.i18n ? window.i18n.t('check_offer') : 'Check Offer',
            'check-pacing': window.i18n ? window.i18n.t('check_pacing') : 'Check Pacing'
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
        this.showSuccess(window.i18n ? window.i18n.t('form_reset_success') : 'Form reset successfully');
    }
}

// Initialize the calculator when DOM is loaded
document.addEventListener('DOMContentLoaded', () => {
    new RentalCalculator();
});
