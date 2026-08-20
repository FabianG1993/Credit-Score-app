/**
 * CreditVision AI — Frontend Application
 *
 * Handles form submission, field-level validation, API communication,
 * and result rendering (gauge chart + SHAP breakdown).
 */

let gaugeChart = null;
const API_URL = '/api/predict';
let activeRequest = null;

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
    age:  { min: 18, max: 100, label: 'Edad', integer: true },
    MonthlyIncome: { min: 0, label: 'Ingreso Mensual' },
    DebtRatio: { min: 0, max: 5, label: 'Ratio de Deuda' },
    RevolvingUtilizationOfUnsecuredLines: { min: 0, max: 1, label: 'Utilización de Crédito' },
    NumberOfDependents: { min: 0, label: 'Dependientes' },
    NumberOfOpenCreditLinesAndLoans: { min: 0, label: 'Líneas de Crédito Abiertas', integer: true },
    NumberRealEstateLoansOrLines: { min: 0, label: 'Préstamos Inmobiliarios', integer: true },
    NumberOfTimes90DaysLate: { min: 0, label: 'Veces 90+ Días de Atraso', integer: true },
    NumberOfTime60_89DaysPastDueNotWorse: { min: 0, label: 'Veces 60-89 Días de Atraso', integer: true },
    NumberOfTime30_59DaysPastDueNotWorse: { min: 0, label: 'Veces 30-59 Días de Atraso', integer: true },
};

function validateField(input) {
    const name = input.name;
    const rules = VALIDATION_RULES[name];
    if (!rules) return null;

    const val = parseFloat(input.value);

    if (isNaN(val)) {
        return setFieldError(input, `${rules.label} es requerido.`);
    }
    if (rules.integer && !Number.isInteger(val)) {
        return setFieldError(input, `${rules.label} debe ser un número entero.`);
    }
    if (rules.min !== undefined && val < rules.min) {
        return setFieldError(input, `${rules.label} debe ser al menos ${rules.min}.`);
    }
    if (rules.max !== undefined && val > rules.max) {
        return setFieldError(input, `${rules.label} debe ser como máximo ${rules.max}.`);
    }

    input.classList.remove('input-error');
    input.removeAttribute('aria-invalid');
    input.removeAttribute('aria-errormessage');
    document.getElementById(`${input.id}-error`)?.remove();
    return null;
}

function setFieldError(input, message) {
    input.classList.add('input-error');
    input.setAttribute('aria-invalid', 'true');
    input.setAttribute('aria-errormessage', `${input.id}-error`);
    let error = document.getElementById(`${input.id}-error`);
    if (!error) {
        error = document.createElement('small');
        error.id = `${input.id}-error`;
        error.className = 'field-error';
        input.closest('.input-group').appendChild(error);
    }
    error.textContent = message;
    return message;
}

function validateAllFields() {
    const inputs = document.querySelectorAll('#prediction-form input');
    let firstError = null;
    inputs.forEach(input => {
        const error = validateField(input);
        if (error && !firstError) firstError = error;
    });
    const consent = document.getElementById('demo-consent');
    if (!consent.checked && !firstError) firstError = 'Debes confirmar el uso de datos de demostración.';
    return firstError;
}

// ---------------------------------------------------------------------------
// Init
// ---------------------------------------------------------------------------
document.addEventListener('DOMContentLoaded', () => {
    const form = document.getElementById('prediction-form');
    const resetBtn = document.getElementById('reset-btn');


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
            const invalid = form.querySelector('.input-error') || document.getElementById('demo-consent');
            invalid.focus();
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
            if (activeRequest) activeRequest.abort();
            activeRequest = new AbortController();
            const timeout = setTimeout(() => activeRequest.abort(), 10000);
            const response = await fetch(API_URL, {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify(payload), signal: activeRequest.signal
            });
            clearTimeout(timeout);

            const result = await response.json();
            if (!response.ok) {
                throw new Error(result.detail || 'API Error');
            }

            updateUI(result);
            displayMessage('Resultado informativo cargado.', 'success');

        } catch (error) {
            const msg = error.name === 'AbortError' ? 'La solicitud tardó demasiado. Inténtalo de nuevo.'
                : error.message === 'Failed to fetch' ? 'No se puede conectar al servidor. Asegúrate de que el backend esté en ejecución.'
                : error.message || 'Error al obtener el resultado. Revisa los datos e intenta de nuevo.';
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
        form.querySelectorAll('.field-error').forEach(el => el.remove());
        form.querySelectorAll('[aria-invalid]').forEach(el => { el.removeAttribute('aria-invalid'); el.removeAttribute('aria-errormessage'); });
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
    const recommendationEl = document.getElementById('recommendation');

    const percentage = result.probability_of_default * 100;
    scoreEl.textContent = `${percentage.toFixed(1)}%`;
    labelEl.textContent = result.risk_band || 'Sin categoría';

    let color = '#10b981';
    if (result.risk_band === 'Riesgo alto') {
        color = '#ef4444';
    } else if (result.risk_band === 'Riesgo medio') {
        color = '#f59e0b';
    }
    labelEl.style.color = color;

    recommendationEl.textContent = `Recomendación: ${result.recommendation}.`;

    if (gaugeChart) {
        updateGauge(percentage, color);
    }
    renderShap(result);
}

// ---------------------------------------------------------------------------
// SHAP Rendering
// ---------------------------------------------------------------------------
function renderShap(result) {
    const list = document.getElementById('shap-list');
    list.replaceChildren();

    const breakdown = result.explanatory_factors;

    if (!Array.isArray(breakdown) || breakdown.length === 0) {
        const emptyMessage = document.createElement('li');
        emptyMessage.className = 'shap-empty';
        emptyMessage.textContent = 'La interpretabilidad SHAP no está disponible para esta predicción.';
        list.appendChild(emptyMessage);
        return;
    }

    // Use friendly_name from backend (single source of truth)
    breakdown.slice(0, 5).forEach(factor => {
        const li = document.createElement('li');
        const isIncrease = factor.direction === 'aumenta';
        li.className = `shap-item ${isIncrease ? 'increases-risk' : 'decreases-risk'}`;

        const friendlyName = factor.label || factor.feature;
        const impactText = isIncrease ? '↑ Aumenta el Riesgo' : '↓ Disminuye el Riesgo';
        const impactClass = isIncrease ? 'impact-high' : 'impact-low';
        const feature = document.createElement('div'); feature.className = 'shap-feature';
        const name = document.createElement('span'); name.className = 'feature-name'; name.textContent = friendlyName;
        const hint = document.createElement('span'); hint.className = 'feature-val'; hint.textContent = 'Factor técnico del modelo';
        const impact = document.createElement('div'); impact.className = `shap-impact ${impactClass}`; impact.textContent = impactText;
        feature.append(name, hint); li.append(feature, impact);

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
