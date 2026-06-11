// App Init Loader Logic
window.addEventListener('load', function () {
    const loader = document.getElementById('app-init-loader');
    const mainContent = document.getElementById('main-content');

    if (loader) {
        // If loader exists (first visit), wait 800ms, then end loader and show site
        setTimeout(function () {
            loader.style.display = 'none';
            if (mainContent) {
                mainContent.style.display = 'block';
                mainContent.classList.add('main-fade-in');
            }
        }, 800);
    }
});

// Quick Chip function
function fillAndProbe(url) {
    document.querySelector('input[name="url"]').value = url;
    document.getElementById('submit-btn').click();
}

// Form Submission Live Terminal Simulator
function handleFormSubmit(e) {
    e.preventDefault(); // Prevent native navigation to keep JS alive

    // Hide current page content under the form card
    const initialFeatures = document.getElementById('initial-features');
    if (initialFeatures) initialFeatures.style.display = 'none';

    const resultsDashboard = document.getElementById('results-dashboard');
    if (resultsDashboard) resultsDashboard.style.display = 'none';

    const errorAlert = document.querySelector('.alert.alert-danger');
    if (errorAlert) errorAlert.style.display = 'none';

    // Show skeleton loader
    const skeleton = document.getElementById('skeleton-loader');
    if (skeleton) {
        skeleton.style.display = 'block';
        skeleton.classList.add('fade-in-up');
    }

    document.getElementById('loading').style.display = 'block';
    const btn = document.getElementById('submit-btn');
    if (btn) {
        btn.disabled = true;
        btn.innerHTML = 'Probing...';
    }

    // Generate realistic backend Python logs with actual target URL
    const targetUrl = document.querySelector('input[name="url"]').value || "target_url";
    const now = new Date();
    const time = now.toTimeString().split(' ')[0]; // HH:MM:SS

    const logs = [
        `[INFO] [${time}] werkzeug: 127.0.0.1 - - "POST /process HTTP/1.1"`,
        `[INFO] [${time}] crawler.py: Initializing secure connection to ${targetUrl}...`,
        `[INFO] [${time}] crawler.py: robots.txt check: ALLOWED`,
        `[INFO] [${time}] crawler.py: Scraping raw DOM tree using BeautifulSoup... [200 OK]`,
        `[INFO] [${time}] crawler.py: Extracting text nodes...`,
        `[INFO] [${time}] analyser.py: --- Phase 2: Text Analysis & Deduplication ---`,
        `[INFO] [${time}] analyser.py: Removing exact raw duplicates...`,
        `[INFO] [${time}] analyser.py: Running TextBlob sentiment polarity metrics...`,
        `[INFO] [${time}] clustering.py: --- Phase 3: Semantic Clustering ---`,
        `[INFO] [${time}] clustering.py: Computing TF-IDF Document-Term Matrix (max_features=1000)...`,
        `[INFO] [${time}] clustering.py: Initializing MiniBatchKMeans algorithm (n_clusters=optimal)...`,
        `[INFO] [${time}] clustering.py: Generating AI Executive Summary via extractive density...`,
        `[INFO] [${time}] app.py: Saved results to results_session.csv`,
        `[INFO] [${time}] app.py: Pipeline Execution Complete. Rendering UI.`
    ];

    const termOutput = document.getElementById('term-output');
    const modalLogsBody = document.getElementById('modal-logs-body');

    if (termOutput) termOutput.innerHTML = '';
    if (modalLogsBody) modalLogsBody.innerHTML = '';

    let delay = 0;

    logs.forEach((log, index) => {
        // Fast, hacker-like typing speed
        delay += (index === 0 ? 200 : 200 + Math.random() * 400);
        setTimeout(() => {
            // Update main terminal
            if (termOutput) {
                const el = document.createElement('div');
                el.className = 'term-log';

                // Colorize [INFO] tags
                if (log.includes("Pipeline Execution Complete")) {
                    el.innerHTML = `<span style="color: #10b981;">> ${log}</span>`;
                } else {
                    el.innerHTML = `> ${log}`.replace('[INFO]', '<span style="color: #8b5cf6;">[INFO]</span>');
                }

                termOutput.appendChild(el);
                const loadingBox = document.getElementById('loading');
                if (loadingBox) loadingBox.scrollTop = loadingBox.scrollHeight;
            }

            // Mirror logs into the modal so they are preserved
            if (modalLogsBody) {
                const mEl = document.createElement('div');
                if (log.includes("Pipeline Execution Complete")) {
                    mEl.innerHTML = `<span style="color: #10b981;">> ${log}</span>`;
                } else {
                    mEl.innerHTML = `> ${log}`.replace('[INFO]', '<span style="color: #8b5cf6;">[INFO]</span>');
                }
                modalLogsBody.appendChild(mEl);
            }
        }, delay);
    });

    // Perform the actual backend request via AJAX, and enforce a minimum wait time 
    // so the cool terminal animation always plays out even if the server is super fast.
    const formData = new FormData(this);

    Promise.all([
        fetch('/process', { method: 'POST', body: formData }).then(res => res.text()),
        new Promise(resolve => setTimeout(resolve, 4500)) // Wait at least 4.5 seconds
    ])
        .then(([html]) => {
            // Save the generated logs before the DOM is replaced
            let savedLogs = '';
            const oldModalLogs = document.getElementById('modal-logs-body');
            if (oldModalLogs) {
                savedLogs = oldModalLogs.innerHTML;
            }

            const parser = new DOMParser();
            const newDoc = parser.parseFromString(html, 'text/html');
            document.body.innerHTML = newDoc.body.innerHTML;

            // Restore the generated logs into the new DOM
            const newModalLogs = document.getElementById('modal-logs-body');
            if (newModalLogs) {
                newModalLogs.innerHTML = savedLogs;
            }

            initDashboard();
        })
        .catch(err => {
            console.error("Pipeline Error:", err);
            e.target.submit();
        });
}

function initDashboard() {
    // 1. Attach form listener
    const form = document.getElementById('probe-form');
    if (form) {
        form.removeEventListener('submit', handleFormSubmit);
        form.addEventListener('submit', handleFormSubmit);
    }

    // 2. Chart.js Default Styling
    Chart.defaults.color = '#64748b';
    Chart.defaults.font.family = "'Inter', sans-serif";

    // 3. Initialize Topic Chart
    const dataElement = document.getElementById('nlp-data');
    if (dataElement) {
        const labels = JSON.parse(dataElement.getAttribute('data-labels'));
        const dataValues = JSON.parse(dataElement.getAttribute('data-values'));

        new Chart(document.getElementById('myChart'), {
            type: 'doughnut',
            data: {
                labels: labels,
                datasets: [{
                    data: dataValues,
                    backgroundColor: ['#0ea5e9', '#8b5cf6', '#10b981', '#f59e0b', '#ec4899', '#3b82f6', '#84cc16'],
                    borderWidth: 2,
                    borderColor: '#ffffff',
                    hoverOffset: 10
                }]
            },
            options: {
                responsive: true,
                maintainAspectRatio: false,
                plugins: {
                    legend: { position: 'bottom', labels: { padding: 20, usePointStyle: true } }
                },
                cutout: '70%',
                onClick: (event, elements, chart) => {
                    if (elements.length > 0) {
                        const dataIndex = elements[0].index;
                        const label = chart.data.labels[dataIndex];
                        filterTableByTopic(label);
                    }
                },
                onHover: (event, elements) => {
                    event.native.target.style.cursor = elements.length ? 'pointer' : 'default';
                }
            }
        });
    }

    // 4. Initialize Sentiment Chart
    const sentimentElement = document.getElementById('sentiment-data');
    if (sentimentElement) {
        const sentLabels = JSON.parse(sentimentElement.getAttribute('data-labels'));
        const sentValues = JSON.parse(sentimentElement.getAttribute('data-values'));

        const colorMap = { 'Positive': '#10b981', 'Negative': '#ef4444', 'Neutral': '#94a3b8' };
        const bgColors = sentLabels.map(label => colorMap[label] || '#94a3b8');

        new Chart(document.getElementById('sentimentChart'), {
            type: 'doughnut',
            data: {
                labels: sentLabels,
                datasets: [{
                    data: sentValues,
                    backgroundColor: bgColors,
                    borderWidth: 2,
                    borderColor: '#ffffff',
                    hoverOffset: 10
                }]
            },
            options: {
                responsive: true,
                maintainAspectRatio: false,
                plugins: {
                    legend: { position: 'bottom', labels: { padding: 20, usePointStyle: true } }
                },
                cutout: '70%',
                onClick: (event, elements, chart) => {
                    if (elements.length > 0) {
                        const dataIndex = elements[0].index;
                        const label = chart.data.labels[dataIndex];
                        filterTableBySentiment(label);
                    }
                },
                onHover: (event, elements) => {
                    event.native.target.style.cursor = elements.length ? 'pointer' : 'default';
                }
            }
        });
    }

    // 5. DataTables Initialization
    if ($('#resultsTable').length) {
        if ($.fn.DataTable.isDataTable('#resultsTable')) {
            $('#resultsTable').DataTable().destroy();
        }
        $('#resultsTable').DataTable({
            pageLength: 10,
            lengthMenu: [[10, 25, 50, -1], [10, 25, 50, "All"]],
            language: { search: "", searchPlaceholder: "Filter insights..." },
            order: [], // Disable initial sorting
            dom: "<'row mb-3'<'col-sm-12 col-md-6'l><'col-sm-12 col-md-6 text-end'f>>" +
                "<'row'<'col-sm-12'tr>>" +
                "<'row mt-3'<'col-sm-12 col-md-5'i><'col-sm-12 col-md-7'p>>"
        });
        $('.dataTables_filter input').addClass('form-control d-inline-block w-auto ms-2').css('border-radius', '12px');
    }
}

// Initialize on first load
$(document).ready(initDashboard);

// Interactive Takeaway Filter
function filterTableByTakeaway(element) {
    const text = element.innerText;
    const searchPhrase = text.split(' ').slice(0, 5).join(' ').replace(/[^\w\s]/gi, '').trim();
    const table = $('#resultsTable').DataTable();
    table.columns().search('');
    table.search(searchPhrase).draw();
    showFilterIndicator(`Takeaway: "${searchPhrase}..."`);
    document.getElementById('resultsTable').scrollIntoView({ behavior: 'smooth', block: 'start' });
}

// Navbar Scroll Shadow
window.addEventListener('scroll', function () {
    const navbar = document.getElementById('gp-navbar');
    if (navbar) {
        navbar.classList.toggle('scrolled', window.scrollY > 10);
    }
});

// Cursor Glow Tracker (smooth lerp via rAF)
const glow = document.getElementById('cursor-glow');
let mouseX = window.innerWidth / 2;
let mouseY = window.innerHeight / 2;
let glowX = mouseX;
let glowY = mouseY;
let isVisible = false;

// Smoothing factor: 0.08 = very smooth/laggy, 0.15 = snappier
const LERP = 0.10;

document.addEventListener('mousemove', (e) => {
    mouseX = e.clientX;
    mouseY = e.clientY;
    if (glow && !isVisible) {
        glow.style.opacity = '1';
        isVisible = true;
    }
});

document.addEventListener('mouseleave', () => {
    if (glow) {
        glow.style.opacity = '0';
        isVisible = false;
    }
});

function animateGlow() {
    if (glow) {
        // Lerp: smoothly move glowX/Y toward mouseX/Y each frame
        glowX += (mouseX - glowX) * LERP;
        glowY += (mouseY - glowY) * LERP;

        glow.style.left = glowX + 'px';
        glow.style.top = glowY + 'px';
    }
    requestAnimationFrame(animateGlow);
}

animateGlow();

// --- Theme Toggle Logic ---
const htmlEl = document.documentElement;

// Initialize theme
const savedTheme = localStorage.getItem('theme') || 'light';
if (savedTheme === 'dark') {
    htmlEl.setAttribute('data-theme', 'dark');
    const themeIcon = document.getElementById('theme-icon');
    if(themeIcon) themeIcon.textContent = '☀️';
}

// Use event delegation so the listener survives DOM replacement via fetch
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

// --- Cross-Filtering Helper Functions ---
function filterTableByTopic(label) {
    const clusterId = label.split(':')[0].trim();
    const table = $('#resultsTable').DataTable();
    
    // Clear previous search and search column 1 (Cluster) specifically
    table.search('');
    table.columns().search('');
    table.column(1).search('^' + clusterId + '$', true, false).draw();
    
    showFilterIndicator(`Cluster ${clusterId} (${label.split(':').slice(1).join(':').trim()})`);
    document.getElementById('resultsTable').scrollIntoView({ behavior: 'smooth', block: 'start' });
}

function filterTableBySentiment(label) {
    const table = $('#resultsTable').DataTable();
    
    // Clear previous search and search column 2 (Sentiment) specifically
    table.search('');
    table.columns().search('');
    table.column(2).search(label).draw();
    
    showFilterIndicator(`${label} Sentiment`);
    document.getElementById('resultsTable').scrollIntoView({ behavior: 'smooth', block: 'start' });
}

function filterTableByEntity(element) {
    // Clone to avoid modifying the original element
    const clone = element.cloneNode(true);
    const spans = clone.getElementsByTagName('span');
    while (spans.length > 0) {
        spans[0].parentNode.removeChild(spans[0]);
    }
    const entityName = clone.textContent.trim();
    const table = $('#resultsTable').DataTable();
    
    // Clear previous search and search table-wide
    table.columns().search('');
    table.search(entityName).draw();
    
    showFilterIndicator(`Entity: "${entityName}"`);
    document.getElementById('resultsTable').scrollIntoView({ behavior: 'smooth', block: 'start' });
}

function showFilterIndicator(value) {
    const indicator = document.getElementById('table-filter-indicator');
    const valSpan = document.getElementById('active-filter-val');
    if (indicator && valSpan) {
        valSpan.textContent = value;
        indicator.style.display = 'inline-flex';
        indicator.style.alignItems = 'center';
    }
}

function clearTableFilters() {
    const table = $('#resultsTable').DataTable();
    table.search('');
    table.columns().search('');
    table.draw();
    
    const indicator = document.getElementById('table-filter-indicator');
    if (indicator) {
        indicator.style.display = 'none';
    }
}
