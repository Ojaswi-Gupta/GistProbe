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
});
