// --- Theme Toggle Logic ---
const themeToggleBtn = document.getElementById('theme-toggle');
const themeIcon = document.getElementById('theme-icon');
const htmlEl = document.documentElement;

// Initialize theme
const savedTheme = localStorage.getItem('theme') || 'light';
if (savedTheme === 'dark') {
    htmlEl.setAttribute('data-theme', 'dark');
    if(themeIcon) themeIcon.textContent = '☀️';
}

if(themeToggleBtn) {
    themeToggleBtn.addEventListener('click', () => {
        const currentTheme = htmlEl.getAttribute('data-theme');
        if (currentTheme === 'dark') {
            htmlEl.removeAttribute('data-theme');
            localStorage.setItem('theme', 'light');
            themeIcon.textContent = '🌙';
        } else {
            htmlEl.setAttribute('data-theme', 'dark');
            localStorage.setItem('theme', 'dark');
            themeIcon.textContent = '☀️';
        }
    });
}

// --- PDF Export Logic (Native Print) ---
function downloadPDF() {
    window.print();
}

// Initialize Sentiment Comparison Chart if data exists
document.addEventListener('DOMContentLoaded', function() {
    const ctx = document.getElementById('sentimentCompareChart');
    if (ctx) {
        const labels = ['Positive', 'Neutral', 'Negative'];
        
        // Extract data from data attributes
        const siteA = JSON.parse(ctx.getAttribute('data-site-a'));
        const siteB = JSON.parse(ctx.getAttribute('data-site-b'));

        new Chart(ctx, {
            type: 'bar',
            data: {
                labels: labels,
                datasets: [
                    {
                        label: 'Site A',
                        data: siteA,
                        backgroundColor: 'rgba(14, 165, 233, 0.7)',
                        borderColor: 'rgba(14, 165, 233, 1)',
                        borderWidth: 1,
                        borderRadius: 6
                    },
                    {
                        label: 'Site B',
                        data: siteB,
                        backgroundColor: 'rgba(99, 102, 241, 0.7)',
                        borderColor: 'rgba(99, 102, 241, 1)',
                        borderWidth: 1,
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
                        labels: { font: { family: "'Inter', sans-serif", size: 14 } }
                    }
                },
                scales: {
                    y: {
                        beginAtZero: true,
                        grid: { color: 'rgba(0,0,0,0.05)' }
                    },
                    x: {
                        grid: { display: false }
                    }
                }
            }
        });
    }

    // Initialize NLP Metrics Radar Chart
    const radarCtx = document.getElementById('radarCompareChart');
    if (radarCtx) {
        const siteAMetrics = JSON.parse(radarCtx.getAttribute('data-site-a-metrics'));
        const siteBMetrics = JSON.parse(radarCtx.getAttribute('data-site-b-metrics'));

        const calcNormalized = (m) => {
            const total = m.total || 1;
            return [
                m.pos / total,              // Positivity
                m.neg / total,              // Negativity
                m.subj,                     // Subjectivity
                1 - m.subj,                 // Objectivity
                m.clusters / 10             // Topic Diversity (max 10)
            ].map(v => Math.max(0, Math.min(1, v))); // Clamp 0-1
        };

        const siteAData = calcNormalized(siteAMetrics);
        const siteBData = calcNormalized(siteBMetrics);

        new Chart(radarCtx, {
            type: 'radar',
            data: {
                labels: ['Positivity', 'Negativity', 'Subjectivity', 'Objectivity', 'Topic Diversity'],
                datasets: [
                    {
                        label: 'Site A',
                        data: siteAData,
                        backgroundColor: 'rgba(14, 165, 233, 0.2)',
                        borderColor: 'rgba(14, 165, 233, 1)',
                        pointBackgroundColor: 'rgba(14, 165, 233, 1)',
                        borderWidth: 2,
                    },
                    {
                        label: 'Site B',
                        data: siteBData,
                        backgroundColor: 'rgba(99, 102, 241, 0.2)',
                        borderColor: 'rgba(99, 102, 241, 1)',
                        pointBackgroundColor: 'rgba(99, 102, 241, 1)',
                        borderWidth: 2,
                    }
                ]
            },
            options: {
                responsive: true,
                maintainAspectRatio: false,
                scales: {
                    r: {
                        angleLines: { color: 'rgba(0,0,0,0.1)' },
                        grid: { color: 'rgba(0,0,0,0.1)' },
                        pointLabels: {
                            font: { family: "'Inter', sans-serif", size: 12 },
                            color: '#64748b'
                        },
                        ticks: {
                            display: false,
                            min: 0,
                            max: 1
                        }
                    }
                },
                plugins: {
                    legend: {
                        position: 'top',
                        labels: { font: { family: "'Inter', sans-serif", size: 14 } }
                    },
                    tooltip: {
                        callbacks: {
                            label: function(context) {
                                return context.dataset.label + ': ' + Math.round(context.raw * 100) + '%';
                            }
                        }
                    }
                }
            }
        });
    }
});
