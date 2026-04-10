/* app.js - Sentinel KYC Dashboard Logic */

const velocityChart = document.getElementById('velocityChart');
const accelChart = document.getElementById('accelChart');
const datasetLogs = document.getElementById('datasetLogs');

// Simulated research data from datasets/
const mockDatasets = [
    { id: 'SMP-0492', category: '3D MASK', confidence: '99.2%', status: 'Detected' },
    { id: 'SMP-1120', category: 'DEEPFAKE', confidence: '97.8%', status: 'Detected' },
    { id: 'SMP-0381', category: 'REAL FACE', confidence: '0.2%', status: 'Verified' },
    { id: 'SMP-2294', category: 'PRINT ATTACK', confidence: '99.9%', status: 'Blocked' },
    { id: 'SMP-8831', category: 'INJECTION', confidence: '98.5%', status: 'Detected' },
];

function initializeCharts() {
    // Basic reactive chart bars
    for (let i = 0; i < 40; i++) {
        const barV = document.createElement('div');
        barV.className = 'bar';
        barV.style.height = `${Math.floor(Math.random() * 100)}%`;
        velocityChart.appendChild(barV);

        const barA = document.createElement('div');
        barA.className = 'bar';
        barA.style.height = `${Math.floor(Math.random() * 100)}%`;
        barA.style.background = 'var(--accent-violet)';
        accelChart.appendChild(barA);
    }
}

function updateCharts() {
    Array.from(velocityChart.children).forEach(bar => {
        bar.style.height = `${Math.floor(Math.random() * 100)}%`;
    });
    Array.from(accelChart.children).forEach(bar => {
        bar.style.height = `${Math.floor(Math.random() * 100)}%`;
    });
}

function renderLogs() {
    datasetLogs.innerHTML = mockDatasets.map(d => `
        <tr style="border-bottom: 1px solid rgba(255, 255, 255, 0.05);">
            <td style="padding: 1rem 0; font-family: monospace;">${d.id}</td>
            <td><span style="font-size: 0.75rem; background: rgba(255,255,255,0.05); padding: 0.2rem 0.5rem; border-radius: 4px;">${d.category}</span></td>
            <td style="color: ${parseFloat(d.confidence) > 90 ? '#ff4081' : 'var(--accent-green)'}">${d.confidence}</td>
            <td><button style="background: none; border: 1px solid var(--glass-border); color: var(--text-secondary); padding: 0.2rem 0.6rem; border-radius: 4px; cursor: pointer;">AUDIT</button></td>
        </tr>
    `).join('');
}

// Runtime
initializeCharts();
renderLogs();
setInterval(updateCharts, 1200);

// Interaction
document.querySelectorAll('.nav-item').forEach(item => {
    item.addEventListener('click', () => {
        document.querySelector('.nav-item.active').classList.remove('active');
        item.classList.add('active');
    });
});
