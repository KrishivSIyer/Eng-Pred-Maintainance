// Chart defaults
Chart.defaults.color = '#8b9bb4';
Chart.defaults.font.family = "'Inter', sans-serif";

let maeChartInstance = null;
let predictionChartInstance = null;

// Navigation
document.querySelectorAll('.nav-item').forEach(button => {
    button.addEventListener('click', () => {
        // Update nav state
        document.querySelectorAll('.nav-item').forEach(b => b.classList.remove('active'));
        button.classList.add('active');
        
        // Update view
        const targetId = button.getAttribute('data-target');
        document.querySelectorAll('.view-section').forEach(s => s.classList.remove('active'));
        document.getElementById(targetId).classList.add('active');
    });
});

// Load Overview Data
async function loadOverviewData() {
    try {
        const response = await fetch('http://127.0.0.1:5000/api/results');
        const result = await response.json();
        
        if (result.success) {
            renderMetricsCards(result.data);
            renderMaeChart(result.data);
        } else {
            console.error('Failed to load metrics:', result.error);
        }
    } catch (e) {
        console.error('Error fetching overview data:', e);
    }
}

function renderMetricsCards(data) {
    const container = document.getElementById('metrics-container');
    container.innerHTML = '';
    
    data.forEach(item => {
        const isRFWinner = item.winner === 'Random Forest';
        const cardHtml = `
            <div class="metric-card glass-panel ${isRFWinner ? 'winner-rf' : 'winner-lstm'}">
                <div class="ds-title">
                    ${item.dataset}
                    <span class="winner-badge">${isRFWinner ? 'RF WINS' : 'LSTM WINS'}</span>
                </div>
                <div class="model-stats">
                    <div class="stat-row">
                        <span class="stat-label">RF MAE</span>
                        <span class="stat-value ${isRFWinner ? 'better' : ''}">${item.rf.mae.toFixed(2)}</span>
                    </div>
                    <div class="stat-row">
                        <span class="stat-label">LSTM MAE</span>
                        <span class="stat-value ${!isRFWinner ? 'better' : ''}">${item.lstm.mae.toFixed(2)}</span>
                    </div>
                    <div class="stat-row">
                        <span class="stat-label">Diff</span>
                        <span class="stat-value" style="color: var(--text-muted); font-size: 0.9rem;">
                            ${Math.abs(item.rf.mae - item.lstm.mae).toFixed(2)}
                        </span>
                    </div>
                </div>
            </div>
        `;
        container.insertAdjacentHTML('beforeend', cardHtml);
    });
}

function renderMaeChart(data) {
    const ctx = document.getElementById('maeChart').getContext('2d');
    
    if (maeChartInstance) {
        maeChartInstance.destroy();
    }
    
    const labels = data.map(d => d.dataset);
    const rfData = data.map(d => d.rf.mae);
    const lstmData = data.map(d => d.lstm.mae);
    
    maeChartInstance = new Chart(ctx, {
        type: 'bar',
        data: {
            labels: labels,
            datasets: [
                {
                    label: 'Random Forest MAE',
                    data: rfData,
                    backgroundColor: 'rgba(16, 185, 129, 0.8)',
                    borderRadius: 6
                },
                {
                    label: 'LSTM MAE',
                    data: lstmData,
                    backgroundColor: 'rgba(245, 158, 11, 0.8)',
                    borderRadius: 6
                }
            ]
        },
        options: {
            responsive: true,
            maintainAspectRatio: false,
            plugins: {
                legend: {
                    position: 'top',
                    labels: {
                        usePointStyle: true,
                        padding: 20
                    }
                },
                tooltip: {
                    backgroundColor: 'rgba(15, 23, 42, 0.9)',
                    titleFont: { size: 14, family: "'Outfit', sans-serif" },
                    bodyFont: { size: 13 },
                    padding: 12,
                    cornerRadius: 8
                }
            },
            scales: {
                y: {
                    beginAtZero: true,
                    grid: {
                        color: 'rgba(255, 255, 255, 0.05)'
                    }
                },
                x: {
                    grid: {
                        display: false
                    }
                }
            }
        }
    });
}

// Live Prediction Logic
document.getElementById('predict-btn').addEventListener('click', async (e) => {
    const btn = e.currentTarget;
    const ds = document.getElementById('dataset-select').value;
    
    // UI Loading state
    btn.classList.add('loading');
    btn.querySelector('.btn-text').innerText = 'Running Inference...';
    
    try {
        const response = await fetch('http://127.0.0.1:5000/api/predict', {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({ dataset: ds })
        });
        
        const result = await response.json();
        
        if (result.success) {
            displayPredictionResults(result);
        } else {
            alert('Prediction failed: ' + result.error);
        }
    } catch (err) {
        console.error(err);
        alert('Network error occurred.');
    } finally {
        btn.classList.remove('loading');
        btn.querySelector('.btn-text').innerText = 'Predict RUL';
    }
});

function displayPredictionResults(data) {
    document.getElementById('prediction-results').classList.remove('hidden');
    
    document.getElementById('res-engine-id').innerText = data.engine_id;
    document.getElementById('res-cycles').innerText = data.cycles_analyzed;
    
    document.getElementById('val-actual').innerText = Math.round(data.actual_rul);
    document.getElementById('val-rf').innerText = Math.round(data.rf_prediction);
    document.getElementById('val-lstm').innerText = Math.round(data.lstm_prediction);
    
    // Calculate errors
    const errRf = Math.abs(data.actual_rul - data.rf_prediction).toFixed(1);
    const errLstm = Math.abs(data.actual_rul - data.lstm_prediction).toFixed(1);
    
    const errRfEl = document.getElementById('err-rf');
    errRfEl.innerText = `Error: ${errRf}`;
    errRfEl.style.color = errRf < errLstm ? '#10b981' : '#f87171';
    
    const errLstmEl = document.getElementById('err-lstm');
    errLstmEl.innerText = `Error: ${errLstm}`;
    errLstmEl.style.color = errLstm < errRf ? '#10b981' : '#f87171';
    
    renderPredictionChart(data);
}

function renderPredictionChart(data) {
    const ctx = document.getElementById('predictionChart').getContext('2d');
    
    if (predictionChartInstance) {
        predictionChartInstance.destroy();
    }
    
    // Simple bar chart to show the exact difference clearly
    predictionChartInstance = new Chart(ctx, {
        type: 'bar',
        data: {
            labels: ['Actual RUL', 'Random Forest', 'LSTM'],
            datasets: [{
                label: 'Remaining Useful Life (Cycles)',
                data: [data.actual_rul, data.rf_prediction, data.lstm_prediction],
                backgroundColor: [
                    'rgba(59, 130, 246, 0.8)', // Truth
                    'rgba(16, 185, 129, 0.8)', // RF
                    'rgba(245, 158, 11, 0.8)'  // LSTM
                ],
                borderRadius: 8,
                barThickness: 60
            }]
        },
        options: {
            responsive: true,
            maintainAspectRatio: false,
            plugins: {
                legend: { display: false },
                tooltip: {
                    backgroundColor: 'rgba(15, 23, 42, 0.9)',
                    titleFont: { size: 14, family: "'Outfit', sans-serif" },
                    bodyFont: { size: 16, weight: 'bold' },
                    padding: 12,
                    cornerRadius: 8,
                    callbacks: {
                        label: function(context) {
                            return ` ${Math.round(context.raw)} Cycles`;
                        }
                    }
                }
            },
            scales: {
                y: {
                    beginAtZero: true,
                    title: {
                        display: true,
                        text: 'Cycles Remaining'
                    },
                    grid: {
                        color: 'rgba(255, 255, 255, 0.05)'
                    }
                },
                x: {
                    grid: { display: false },
                    ticks: {
                        font: { size: 14, weight: '600' }
                    }
                }
            }
        }
    });
}

// Initialize
document.addEventListener('DOMContentLoaded', () => {
    loadOverviewData();
});
