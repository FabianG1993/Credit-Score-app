/**
 * CreditVision AI — Frontend Application
 *
 * Handles form submission, field-level validation, API communication,
 * and result rendering (gauge chart + SHAP breakdown).
 */

let gaugeChart = null;
const API_URL = '/api/predict';

// ---------------------------------------------------------------------------
// Message Display
// ---------------------------------------------------------------------------
function displayMessage(text, type = 'info') {
    const messageBox = document.getElementById('message-box');
    messageBox.textContent = text;
    messageBox.className = `message-box ${type}`;
}

function clearMessage() {
    const messageBox = document.getElementById('message-box');
    messageBox.textContent = '';
    messageBox.className = 'message-box';
}

// ---------------------------------------------------------------------------
// Field-Level Validation
// ---------------------------------------------------------------------------
const VALIDATION_RULES = {
    age:  { min: 18, max: 100, label: 'Age', integer: true },
    MonthlyIncome: { min: 0, label: 'Monthly Income' },
    DebtRatio: { min: 0, max: 5, label: 'Debt Ratio' },
    RevolvingUtilizationOfUnsecuredLines: { min: 0, max: 1, label: 'Credit Utilization' },
    NumberOfDependents: { min: 0, label: 'Dependents' },
    NumberOfOpenCreditLinesAndLoans: { min: 0, label: 'Open Credit Lines', integer: true },
    NumberRealEstateLoansOrLines: { min: 0, label: 'Real Estate Loans', integer: true },
    NumberOfTimes90DaysLate: { min: 0, label: 'Times 90+ Days Late', integer: true },
    NumberOfTime60_89DaysPastDueNotWorse: { min: 0, label: 'Times 60-89 Days Late', integer: true },
    NumberOfTime30_59DaysPastDueNotWorse: { min: 0, label: 'Times 30-59 Days Late', integer: true },
};

function validateField(input) {
    const name = input.name;
    const rules = VALIDATION_RULES[name];
    if (!rules) return null;

    const val = parseFloat(input.value);

    if (isNaN(val)) {
        input.classList.add('input-error');
        return `${rules.label} is required.`;
    }
    if (rules.min !== undefined && val < rules.min) {
        input.classList.add('input-error');
        return `${rules.label} must be at least ${rules.min}.`;
    }
    if (rules.max !== undefined && val > rules.max) {
        input.classList.add('input-error');
        return `${rules.label} must be at most ${rules.max}.`;
    }

    input.classList.remove('input-error');
    return null;
}

function validateAllFields() {
    const inputs = document.querySelectorAll('#prediction-form input');
    let firstError = null;
    inputs.forEach(input => {
        const error = validateField(input);
        if (error && !firstError) firstError = error;
    });
    return firstError;
}

// ---------------------------------------------------------------------------
// Init
// ---------------------------------------------------------------------------
document.addEventListener('DOMContentLoaded', () => {
    const form = document.getElementById('prediction-form');
    const resetBtn = document.getElementById('reset-btn');

    // Wait for Chart.js to load (deferred script)
    waitForChartJS(() => {
        initGauge(0);
    });

    // Field-level validation on blur
    form.querySelectorAll('input').forEach(input => {
        input.addEventListener('blur', () => validateField(input));
        input.addEventListener('input', () => {
            if (input.classList.contains('input-error')) {
                validateField(input);
            }
        });
    });

    // Form submission
    form.addEventListener('submit', async (e) => {
        e.preventDefault();
        clearMessage();

        // Validate all fields first
        const validationError = validateAllFields();
        if (validationError) {
            displayMessage(validationError, 'error');
            return;
        }

        const submitBtn = document.getElementById('submit-btn');
        const btnText = submitBtn.querySelector('span');
        const loader = document.getElementById('btn-loader');

        // Collect data
        const formData = new FormData(form);
        const data = Object.fromEntries(formData.entries());

        const payload = {
            age: parseInt(data.age, 10),
            MonthlyIncome: parseFloat(data.MonthlyIncome),
            DebtRatio: parseFloat(data.DebtRatio),
            RevolvingUtilizationOfUnsecuredLines: parseFloat(data.RevolvingUtilizationOfUnsecuredLines),
            NumberOfDependents: parseFloat(data.NumberOfDependents),
            NumberOfOpenCreditLinesAndLoans: parseInt(data.NumberOfOpenCreditLinesAndLoans, 10),
            NumberRealEstateLoansOrLines: parseInt(data.NumberRealEstateLoansOrLines, 10),
            NumberOfTimes90DaysLate: parseInt(data.NumberOfTimes90DaysLate, 10),
            NumberOfTime60_89DaysPastDueNotWorse: parseInt(data.NumberOfTime60_89DaysPastDueNotWorse, 10),
            NumberOfTime30_59DaysPastDueNotWorse: parseInt(data.NumberOfTime30_59DaysPastDueNotWorse, 10)
        };

        // UI Loading state
        submitBtn.disabled = true;
        btnText.style.display = 'none';
        loader.style.display = 'block';

        try {
            const response = await fetch(API_URL, {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify(payload)
            });

            const result = await response.json();
            if (!response.ok) {
                throw new Error(result.detail || 'API Error');
            }

            updateUI(result);
            displayMessage('Prediction loaded successfully.', 'success');

        } catch (error) {
            console.error('Error fetching prediction:', error);
            const msg = error.message === 'Failed to fetch'
                ? 'Cannot reach the server. Please ensure the backend is running.'
                : error.message || 'Failed to get prediction. Check inputs and try again.';
            displayMessage(msg, 'error');
        } finally {
            submitBtn.disabled = false;
            btnText.style.display = 'block';
            loader.style.display = 'none';
        }
    });

    // Reset button
    resetBtn.addEventListener('click', () => {
        form.reset();
        clearMessage();
        // Clear validation errors
        form.querySelectorAll('.input-error').forEach(el => el.classList.remove('input-error'));
        // Reset results panels
        document.querySelectorAll('.empty-state').forEach(el => el.style.display = 'block');
        document.querySelector('.score-content').style.display = 'none';
        document.querySelector('.shap-content').style.display = 'none';
        if (gaugeChart) updateGauge(0, '#6366f1');
    });
});

// ---------------------------------------------------------------------------
// Chart.js Loader (handles deferred loading gracefully)
// ---------------------------------------------------------------------------
function waitForChartJS(callback, retries = 20) {
    if (typeof Chart !== 'undefined') {
        callback();
    } else if (retries > 0) {
        setTimeout(() => waitForChartJS(callback, retries - 1), 150);
    } else {
        console.warn('Chart.js failed to load. Gauge will not render.');
    }
}

// ---------------------------------------------------------------------------
// UI Update
// ---------------------------------------------------------------------------
function updateUI(result) {
    // Show result sections
    document.querySelectorAll('.empty-state').forEach(el => el.style.display = 'none');
    document.querySelector('.score-content').style.display = 'flex';
    document.querySelector('.shap-content').style.display = 'block';

    const scoreEl = document.getElementById('risk-percentage');
    const labelEl = document.getElementById('risk-category');
    const baselineEl = document.getElementById('baseline-value');

    scoreEl.innerText = `${result.risk_percentage.toFixed(1)}%`;
    labelEl.innerText = result.risk_category || 'Risk';

    let color = '#10b981';
    if (result.risk_percentage > 50) {
        color = '#ef4444';
    } else if (result.risk_percentage > 20) {
        color = '#f59e0b';
    }
    labelEl.style.color = color;

    baselineEl.innerText = `Baseline default probability: ${(result.base_value * 100).toFixed(1)}%`;

    if (gaugeChart) {
        updateGauge(result.risk_percentage, color);
    }
    renderShap(result);
}

// ---------------------------------------------------------------------------
// SHAP Rendering
// ---------------------------------------------------------------------------
function renderShap(result) {
    const list = document.getElementById('shap-list');
    list.innerHTML = '';

    const breakdown = result.shap_breakdown;

    // Show SHAP error if present
    if (result.shap_error) {
        const errorItem = document.createElement('li');
        errorItem.className = 'shap-empty';
        errorItem.textContent = result.shap_error;
        list.appendChild(errorItem);
        return;
    }

    if (!Array.isArray(breakdown) || breakdown.length === 0) {
        const emptyMessage = document.createElement('li');
        emptyMessage.className = 'shap-empty';
        emptyMessage.textContent = 'SHAP interpretability is not available for this prediction.';
        list.appendChild(emptyMessage);
        return;
    }

    // Use friendly_name from backend (single source of truth)
    breakdown.slice(0, 5).forEach(factor => {
        const li = document.createElement('li');
        const isIncrease = factor.impact > 0;
        li.className = `shap-item ${isIncrease ? 'increases-risk' : 'decreases-risk'}`;

        const friendlyName = factor.friendly_name || factor.feature;
        const impactText = isIncrease ? '↑ Increases Risk' : '↓ Decreases Risk';
        const impactClass = isIncrease ? 'impact-high' : 'impact-low';
        const valueText = Number.isFinite(factor.value) ? factor.value.toFixed(2) : factor.value;

        li.innerHTML = `
            <div class="shap-feature">
                <span class="feature-name">${friendlyName}</span>
                <span class="feature-val">Input Value: ${valueText}</span>
            </div>
            <div class="shap-impact ${impactClass}">
                ${impactText}
            </div>
        `;

        list.appendChild(li);
    });
}

// ---------------------------------------------------------------------------
// Gauge Chart
// ---------------------------------------------------------------------------
function initGauge(value) {
    const canvas = document.getElementById('scoreGauge');
    if (!canvas || typeof Chart === 'undefined') return;

    const ctx = canvas.getContext('2d');
    gaugeChart = new Chart(ctx, {
        type: 'doughnut',
        data: {
            datasets: [{
                data: [value, 100 - value],
                backgroundColor: ['#6366f1', 'rgba(255,255,255,0.1)'],
                borderWidth: 0,
                circumference: 180,
                rotation: 270
            }]
        },
        options: {
            responsive: true,
            maintainAspectRatio: false,
            cutout: '80%',
            plugins: {
                tooltip: { enabled: false },
                legend: { display: false }
            },
            animation: {
                animateRotate: true,
                animateScale: false
            }
        }
    });
}

function updateGauge(value, color) {
    if (!gaugeChart) return;
    gaugeChart.data.datasets[0].data = [value, 100 - value];
    gaugeChart.data.datasets[0].backgroundColor = [color, 'rgba(255,255,255,0.1)'];
    gaugeChart.update();
}
