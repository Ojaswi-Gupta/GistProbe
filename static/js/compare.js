// --- Theme Toggle Logic ---
const htmlEl = document.documentElement;

// Initialize theme
const savedTheme = localStorage.getItem('theme') || 'dark';
if (savedTheme === 'dark') {
    htmlEl.setAttribute('data-theme', 'dark');
    const themeIcon = document.getElementById('theme-icon');
    if (themeIcon) themeIcon.textContent = '☀️';
}

// Event delegation for theme toggle (so it survives DOM replacement)
document.addEventListener('click', (e) => {
    const themeToggleBtn = e.target.closest('#theme-toggle');
    if (themeToggleBtn) {
        const currentTheme = htmlEl.getAttribute('data-theme');
        const themeIcon = document.getElementById('theme-icon');
        if (currentTheme === 'dark') {
            htmlEl.removeAttribute('data-theme');
            localStorage.setItem('theme', 'light');
            if (themeIcon) themeIcon.textContent = '🌙';
        } else {
            htmlEl.setAttribute('data-theme', 'dark');
            localStorage.setItem('theme', 'dark');
            if (themeIcon) themeIcon.textContent = '☀️';
        }
    }
});

// --- PDF Export Logic ---
function downloadPDF() {
    window.print();
}

// --- AJAX Compare Submission & Dashboard Init ---
function handleCompareSubmit(e) {
    e.preventDefault();

    const form = document.getElementById('compare-form');
    if (!form) return;

    const submitBtn = form.querySelector('button[type="submit"]');
    if (submitBtn) {
        submitBtn.disabled = true;
        submitBtn.innerHTML = 'Comparing Sites... ⏳';
    }

    // Hide old results & error alerts
    const compareResults = document.getElementById('compare-results');
    if (compareResults) compareResults.style.display = 'none';

    const errorAlert = document.querySelector('.alert.alert-danger');
    if (errorAlert) errorAlert.style.display = 'none';

    // Show compare skeleton loader
    const skeleton = document.getElementById('compare-skeleton');
    if (skeleton) {
        skeleton.style.display = 'block';
        skeleton.classList.add('fade-in-up');
    }

    const formData = new FormData(form);

    fetch('/compare', {
        method: 'POST',
        body: formData
    })
    .then(res => res.text())
    .then(html => {
        const parser = new DOMParser();
        const newDoc = parser.parseFromString(html, 'text/html');
        document.body.innerHTML = newDoc.body.innerHTML;
        
        // Re-initialize charts and event bindings
        initCompareDashboard();
    })
    .catch(err => {
        console.error("Compare Pipeline Error:", err);
        // Fallback to standard submit
        form.submit();
    });
}

function initCompareDashboard() {
    // 1. Attach AJAX submit listener
    const form = document.getElementById('compare-form');
    if (form) {
        form.removeEventListener('submit', handleCompareSubmit);
        form.addEventListener('submit', handleCompareSubmit);
    }

    // 2. Initialize Sentiment Comparison Chart if data exists
    const ctx = document.getElementById('sentimentCompareChart');
    if (ctx) {
        const labels = ['Positive', 'Neutral', 'Negative'];
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

    // 3. Initialize NLP Metrics Radar Chart
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
}

// Trigger initial dashboard bindings on DOM ready
document.addEventListener('DOMContentLoaded', initCompareDashboard);

// Helper function to autofill the form and submit it for demonstrations
function fillCompareAndProbe(url1, url2) {
    const u1 = document.querySelector('input[name="url1"]');
    const u2 = document.querySelector('input[name="url2"]');
    if (u1 && u2) {
        u1.value = url1;
        u2.value = url2;
        const form = document.getElementById('compare-form');
        if (form) {
            form.dispatchEvent(new Event('submit', { cancelable: true, bubbles: true }));
        }
    }
}
