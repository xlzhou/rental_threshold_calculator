// Internationalization (i18n) system for Rental Threshold Calculator

class I18n {
    constructor() {
        this.currentLanguage = localStorage.getItem('language') || 'zh';
        this.translations = {
            en: {
                // Header
                'title': 'Rental Threshold Calculator',
                'subtitle': 'Optimize rental inventory decisions with threshold-based policies',
                'help': '📖 Help',
                
                // Configuration Section
                'configuration': 'Configuration',
                'starting_inventory': 'Starting Inventory (servers):',
                'number_of_periods': 'Horizon Length (days):',
                'unit_cost': 'Unit Cost (K):',
                'salvage_value': 'Salvage Value (s):',
                'arrival_rate': 'Arrival Rate (per period):',
                'price_unit': 'Price/Cost Unit:',
                'per_day': 'Per-day',
                'per_month': 'Per-month',
                'days_per_month': 'Days per month:',
                'target_leftover': 'Target Leftover (L*):',
                'failure_threshold': 'Failure Threshold:',
                'cost_floor': 'Cost Floor (optional):',
                'empirical_prices': 'Empirical Prices (comma-separated):',
                'upload_csv': 'Or upload CSV file:',
                'calculate_threshold': 'Calculate Threshold',
                'reset_form': 'Reset Form',
                
                // Static Threshold Analysis
                'static_threshold_analysis': 'Static Threshold Analysis',
                'optimal_threshold': 'Optimal Threshold (K)',
                'operational_cutoff': 'Operational Cutoff (K)',
                'expected_accepts': 'Expected Accepts',
                'expected_leftover': 'Expected Leftover',
                'conditional_mean_price': 'Conditional Mean Price (K)',
                'expected_profit': 'Expected Profit (K)',
                
                // Dynamic Programming Analysis
                'dynamic_programming_analysis': 'Dynamic Programming Analysis',
                'initial_value_function': 'Initial Value Function V₀(X)',
                'initial_bid_price': 'Initial Bid Price b₀(X)',
                'initial_threshold': 'Initial Threshold (K)',
                
                // Decision Guidance
                'decision_guidance': 'Decision Guidance',
                'calculate_first': 'Calculate threshold first to see guidance',
                'export_csv': 'Export CSV',
                'export_json': 'Export JSON',
                
                // Live Offer Decision
                'live_offer_decision': 'Live Offer Decision',
                'offer_price': 'Offer Price (K):',
                'current_period': 'Current Period:',
                'current_inventory': 'Current Inventory:',
                'rental_duration': 'Rental Duration (periods):',
                'offer_type': 'Offer Type:',
                'per_period': 'Per-period price',
                'total_price': 'Total price',
                'decision_method': 'Decision Method:',
                'dynamic_programming_recommended': 'Dynamic Programming (Recommended)',
                'static_threshold': 'Static Threshold',
                'check_offer': 'Check Offer',
                'current_dynamic_threshold': 'Current Dynamic Threshold',
                'threshold_current_state': 'Threshold for current state:',
                'sobp_threshold_per': 'SOBP per-period (D):',
                'sobp_threshold_total': 'SOBP total (D):',
                'static_comparison': 'Static threshold:',
                'threshold_auto_update': 'Threshold updates automatically based on current period and inventory',
                'decision': 'Decision:',
                'make_offer_decision': 'Make an offer decision to see results',
                
                // Pacing Status
                'pacing_status': 'Pacing Status(检查销售进度，可以忽略)',
                'pacing_hint_title': '💡 Hint:',
                'pacing_hint_text': 'For pacing check, you only need to fill:',
                'pacing_hint_inventory': '✓ Starting Inventory (servers) - Used to calculate total units that need to be sold',
                'pacing_hint_periods': '✓ Horizon Length (days) - Used to determine mid-horizon timing (days ÷ 2)',
                'pacing_hint_leftover': '✓ Target Leftover (L*) - Used in target calculation: (Inventory - Target Leftover) × 50%',
                'pacing_hint_period': '✓ Current Period - Used to check timing (defaults to 0 if empty = "too early")',
                'pacing_hint_accepted': '✓ Accepted Units So Far - Compared against target to determine if on track or behind',
                'pacing_logic_title': '📊 Pacing Logic:',
                'pacing_logic_early': 'Too Early: If Current Period < Number of Periods ÷ 2',
                'pacing_logic_target': 'Target by Mid-Horizon: (Starting Inventory - Target Leftover) × 50%',
                'pacing_logic_track': 'On Track: Accepted So Far ≥ Target',
                'pacing_logic_behind': 'Behind: Accepted So Far < Target → Recommend relaxing threshold',
                'pacing_logic_note': 'Other fields with red * are not required for pacing analysis.',
                'accepted_units_so_far': 'Accepted Units So Far:',
                'check_pacing': 'Check Pacing',
                'status': 'Status:',
                'relax_threshold': 'Relax Threshold',
                
                // Footer
                'version': 'Rental Threshold Calculator v1.0',
                
                // Messages
                'calculating': 'Calculating...',
                'form_reset_success': 'Form reset successfully',
                'calculation_success': 'Calculation completed successfully!',
                'no_results': 'Please calculate threshold first',
                'network_error': 'Network error:'
            },
            zh: {
                // Header
                'title': '租赁阈值计算器',
                'subtitle': '基于阈值策略优化租赁库存决策',
                'help': '📖 帮助',
                
                // Configuration Section
                'configuration': '配置',
                'starting_inventory': '起始库存（服务器）：',
                'number_of_periods': '销售周期内总天数：',
                'unit_cost': '单位成本（K）：',
                'salvage_value': '残值（s）：',
                'arrival_rate': '每周期报价到达率：',
                'target_leftover': '目标剩余（L*）：',
                'failure_threshold': '失败剩余库存数：',
                'cost_floor': '成本下限（不填则为成本价）：',
                'empirical_prices': '经验价格（英文逗号分隔）：',
                'upload_csv': '或上传 CSV 文件：',
                'calculate_threshold': '计算阈值',
                'reset_form': '重置表单',
                
                // Static Threshold Analysis
                'static_threshold_analysis': '静态阈值分析',
                'optimal_threshold': '最优阈值（K）',
                'operational_cutoff': '运营截止点（K）',
                'expected_accepts': '预期成交台数',
                'expected_leftover': '预期剩余',
                'conditional_mean_price': '条件均价（K）',
                'expected_profit': '预期利润（K）',
                
                // Dynamic Programming Analysis
                'dynamic_programming_analysis': '动态规划分析',
                'initial_value_function': '初始价值函数 V₀(X)',
                'initial_bid_price': '初始竞价 b₀(X)',
                'initial_threshold': '初始阈值（K）',
                
                // Decision Guidance
                'decision_guidance': '决策指导',
                'calculate_first': '请先计算阈值以查看指导',
                'export_csv': '导出 CSV',
                'export_json': '导出 JSON',
                
                // Live Offer Decision
                'live_offer_decision': '实时报价决策',
                'offer_price': '报价（K）：',
                'price_unit': '价格/成本口径：',
                'per_day': '按日',
                'per_month': '按月',
                'days_per_month': '每月折算天数：',
                'current_period': '当前周期（第几天）：',
                'current_inventory': '当前库存：',
                'rental_duration': '租期（周期数）：',
                'offer_type': '报价口径：',
                'per_period': '按期价格',
                'total_price': '整单总价',
                'decision_method': '决策方法：',
                'dynamic_programming_recommended': '动态规划（推荐）',
                'static_threshold': '静态阈值',
                'check_offer': '检查报价',
                'current_dynamic_threshold': '当前动态阈值',
                'threshold_current_state': '当前状态的阈值：',
                'sobp_threshold_per': 'SOBP 每期阈值（D）：',
                'sobp_threshold_total': 'SOBP 整单阈值（D）：',
                'static_comparison': '静态阈值：',
                'threshold_auto_update': '阈值基于当前周期和库存自动更新',
                'decision': '决策：',
                'make_offer_decision': '做出报价决策以查看结果',
                
                // Pacing Status
                'pacing_status': '进度状态（销售进度检查）',
                'pacing_hint_title': '💡 提示：',
                'pacing_hint_text': '进度检查只需要填写：',
                'pacing_hint_inventory': '✓ 起始库存（服务器）- 用于计算需要销售的总单位数',
                'pacing_hint_periods': '✓ 销售周期内总天数 - 用于确定中期时间点（天数 ÷ 2）',
                'pacing_hint_leftover': '✓ 目标剩余（L*）- 用于目标计算：（库存 - 目标剩余）× 50%',
                'pacing_hint_period': '✓ 当前周期 - 用于检查时间（如为空默认为0 = "太早"）',
                'pacing_hint_accepted': '✓ 已接受单位数 - 与目标比较以确定是否按计划进行或落后',
                'pacing_logic_title': '📊 进度逻辑：',
                'pacing_logic_early': '太早：如果当前周期 < 周期数 ÷ 2',
                'pacing_logic_target': '中期目标：（起始库存 - 目标剩余）× 50%',
                'pacing_logic_track': '按计划：已接受数 ≥ 目标',
                'pacing_logic_behind': '落后：已接受数 < 目标 → 建议放宽阈值',
                'pacing_logic_note': '其他带红色 * 的字段不是进度分析必需的。',
                'accepted_units_so_far': '到目前为止已接受单位数：',
                'check_pacing': '检查进度',
                'status': '状态：',
                'relax_threshold': '放宽阈值',
                
                // Footer
                'version': '租赁阈值计算器 v1.0',
                
                // Messages
                'calculating': '计算中...',
                'form_reset_success': '表单重置成功',
                'calculation_success': '计算完成！',
                'no_results': '请先计算阈值',
                'network_error': '网络错误：'
            }
        };
        
        this.initializeLanguage();
    }

    initializeLanguage() {
        this.updateAllTexts();
        this.updateLanguageSelector();
    }

    setLanguage(lang) {
        if (this.translations[lang]) {
            this.currentLanguage = lang;
            localStorage.setItem('language', lang);
            this.updateAllTexts();
            this.updateLanguageSelector();
        }
    }

    t(key) {
        return this.translations[this.currentLanguage][key] || this.translations['en'][key] || key;
    }

    updateAllTexts() {
        // Update all elements with data-i18n attributes
        const elements = document.querySelectorAll('[data-i18n]');
        elements.forEach(element => {
            const key = element.getAttribute('data-i18n');
            const translation = this.t(key);
            
            if (element.tagName === 'INPUT' && (element.type === 'submit' || element.type === 'button')) {
                element.value = translation;
            } else if (element.tagName === 'INPUT' && element.hasAttribute('placeholder')) {
                element.placeholder = translation;
            } else {
                element.textContent = translation;
            }
        });

        // Update document title
        document.title = this.t('title');
        
        // Update HTML lang attribute
        document.documentElement.lang = this.currentLanguage === 'zh' ? 'zh-CN' : 'en';
    }

    updateLanguageSelector() {
        const selector = document.getElementById('language-selector');
        if (selector) {
            selector.value = this.currentLanguage;
        }
    }

    getCurrentLanguage() {
        return this.currentLanguage;
    }
}

// Global i18n instance
window.i18n = new I18n();
