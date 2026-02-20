/**
 * K-Space Analysis Platform - Frontend Logic
 * ============================================
 */

// State
let currentStudyId = null;
let pollInterval = null;

// ---- Tab Navigation ----
function switchTab(tab) {
    document.querySelectorAll('.tab').forEach(t => t.classList.remove('active'));
    document.querySelectorAll('.tab-content').forEach(c => c.classList.remove('active'));
    document.querySelector(`[data-tab="${tab}"]`).classList.add('active');
    document.getElementById(`tab-${tab}`).classList.add('active');
    if (tab === 'studies') loadStudies();
    if (tab === 'compare') loadComparisonOptions();
}

function goHome() {
    closeDashboard();
    closeVolumeModal();
    switchTab('upload'); // Or studies, depending on preference. Let's go to Upload as "Home"
}

// ---- Upload ----
const uploadZone = document.getElementById('uploadZone');

uploadZone.addEventListener('dragover', (e) => {
    e.preventDefault();
    uploadZone.classList.add('dragover');
});

uploadZone.addEventListener('dragleave', () => {
    uploadZone.classList.remove('dragover');
});

uploadZone.addEventListener('drop', (e) => {
    e.preventDefault();
    uploadZone.classList.remove('dragover');
    const files = e.dataTransfer.files;
    if (files.length > 0) uploadFiles(files);
});

function handleFileSelect(event) {
    const files = event.target.files;
    if (files.length > 0) uploadFiles(files);
}

async function uploadFiles(files) {
    const progressEl = document.getElementById('uploadProgress');
    const countEl = document.getElementById('uploadCount');
    progressEl.style.display = 'block';
    countEl.textContent = `${files.length} files`;

    const formData = new FormData();
    for (let i = 0; i < files.length; i++) {
        formData.append('files', files[i]);
    }

    try {
        const res = await fetch('/api/upload', { method: 'POST', body: formData });
        const data = await res.json();

        if (res.ok) {
            showToast(`${data.file_count} files uploaded! Starting analysis...`, 'success');
            progressEl.style.display = 'none';

            // Auto-start analysis
            await startAnalysis(data.study_id);
            switchTab('studies');
        } else {
            showToast('Upload failed: ' + (data.detail || 'Unknown error'), 'error');
            progressEl.style.display = 'none';
        }
    } catch (err) {
        showToast('Upload error: ' + err.message, 'error');
        progressEl.style.display = 'none';
    }

    // Reset file input
    document.getElementById('fileInput').value = '';
}

// ---- Studies ----
async function loadStudies() {
    const container = document.getElementById('studyList');

    try {
        const res = await fetch('/api/studies');
        const studies = await res.json();

        if (studies.length === 0) {
            container.innerHTML = `
        <div class="empty-state">
          <div class="icon">&#x1F50D;</div>
          <h3>No studies yet</h3>
          <p>Upload DICOM files to begin analysis</p>
        </div>`;
            return;
        }

        container.innerHTML = studies.map(s => `
      <div class="study-item" onclick="openStudy('${s.id}')">
        <div class="study-icon ${s.status}">
          ${getStatusIcon(s.status)}
        </div>
        <div class="study-info">
          <h3>${s.metadata?.series_description || s.metadata?.patient_name || 'Study ' + s.id}</h3>
          <p>${s.metadata?.modality || 'MR'} | ${s.file_count || '?'} files
             ${s.metadata?.patient_age ? ' | ' + s.metadata.patient_age : ''}</p>
        </div>
        <div class="study-meta">
          ${formatDate(s.created_at)}
        </div>
        <div>
          <span class="badge badge-${s.status}">${s.status}</span>
        </div>
      </div>
    `).join('');
    } catch (err) {
        container.innerHTML = `<div class="empty-state"><h3>Error loading studies</h3></div>`;
    }
}

function getStatusIcon(status) {
    switch (status) {
        case 'uploaded': return '&#x1F4E6;';
        case 'analyzing': return '&#x2699;';
        case 'completed': return '&#x2705;';
        case 'error': return '&#x274C;';
        default: return '&#x2753;';
    }
}

function formatDate(iso) {
    if (!iso) return '';
    const d = new Date(iso);
    return d.toLocaleDateString('tr-TR') + ' ' + d.toLocaleTimeString('tr-TR', { hour: '2-digit', minute: '2-digit' });
}

// ---- Analysis ----
async function startAnalysis(studyId) {
    try {
        const res = await fetch(`/api/analyze/${studyId}`, { method: 'POST' });
        const data = await res.json();
        showToast(data.message, 'info');

        // Start polling
        startPolling(studyId);
    } catch (err) {
        showToast('Analysis failed to start: ' + err.message, 'error');
    }
}

function startPolling(studyId) {
    if (pollInterval) clearInterval(pollInterval);

    pollInterval = setInterval(async () => {
        try {
            const res = await fetch(`/api/study/${studyId}`);
            const study = await res.json();

            if (study.status === 'completed') {
                clearInterval(pollInterval);
                pollInterval = null;
                showToast('Analysis completed!', 'success');
                loadStudies();

                // If we're viewing this study, refresh dashboard
                if (currentStudyId === studyId) {
                    renderDashboard(study);
                }
            } else if (study.status === 'error') {
                clearInterval(pollInterval);
                pollInterval = null;
                showToast('Analysis error: ' + (study.error || 'Unknown'), 'error');
                loadStudies();
            }
        } catch (err) {
            // Ignore polling errors
        }
    }, 2000);
}

// ---- Dashboard ----
async function openStudy(studyId) {
    currentStudyId = studyId;

    try {
        const res = await fetch(`/api/study/${studyId}`);
        const study = await res.json();

        if (study.status === 'uploaded') {
            // Not analyzed yet, start analysis
            await startAnalysis(studyId);
            showToast('Analysis started, please wait...', 'info');
            return;
        }

        if (study.status === 'analyzing') {
            showToast('Analysis in progress, please wait...', 'info');
            startPolling(studyId);
            return;
        }

        if (study.status === 'completed') {
            renderDashboard(study);
        }

        if (study.status === 'error') {
            showToast('This study has an error: ' + (study.error || ''), 'error');
        }
    } catch (err) {
        showToast('Error loading study: ' + err.message, 'error');
    }
}

function renderDashboard(study) {
    const mainView = document.getElementById('mainView');
    const dashboard = document.getElementById('dashboard');

    mainView.style.display = 'none';
    dashboard.classList.add('active');

    const meta = study.metadata || {};
    const results = study.results || {};
    const summary = results.summary || {};

    // Title
    document.getElementById('dashTitle').textContent =
        meta.series_description || meta.patient_name || 'Study ' + study.id;
    document.getElementById('dashSubtitle').textContent =
        `${meta.modality || 'MR'} | ${summary.slice_count || '?'} slices | ${summary.image_size || '?'} | ${formatDate(study.created_at)}`;

    // Report
    const reportRaw = summary.report_text || "Rapor bulunamadı.";
    // Simple markdown parser for bold and headers
    const reportHtml = reportRaw
        .replace(/### (.*)/g, '<h4>$1</h4>')
        .replace(/#### (.*)/g, '<h5>$1</h5>')
        .replace(/\*\*(.*?)\*\*/g, '<strong>$1</strong>')
        .replace(/> (.*)/g, '<blockquote style="border-left:3px solid #ccc; padding-left:10px; color:#555;">$1</blockquote>')
        .replace(/- (.*)/g, '<li>$1</li>');

    document.getElementById('reportText').innerHTML = reportHtml;

    // Metrics
    const metricsGrid = document.getElementById('metricsGrid');
    metricsGrid.innerHTML = `
    <div class="metric-card">
      <div class="metric-value cyan">${summary.slice_count || '-'}</div>
      <div class="metric-label">Slices</div>
    </div>
    <div class="metric-card">
      <div class="metric-value ${summary.anomalies_found > 0 ? 'red' : 'green'}">${summary.anomalies_found || 0}</div>
      <div class="metric-label">Anomalies</div>
    </div>
    <div class="metric-card">
      <div class="metric-value amber">${summary.dk_max?.toFixed(3) || '-'}</div>
      <div class="metric-label">Max dK Score</div>
    </div>
    <div class="metric-card">
      <div class="metric-value green">${summary.entropy_mean?.toFixed(2) || '-'}</div>
      <div class="metric-label">Mean Entropy</div>
    </div>
    <div class="metric-card">
      <div class="metric-value purple">${summary.lr_asymmetry_mean?.toFixed(4) || '-'}</div>
      <div class="metric-label">LR Asymmetry</div>
    </div>
    <div class="metric-card">
      <div class="metric-value blue">${summary.phase_coherence_mean?.toFixed(4) || '-'}</div>
      <div class="metric-label">Phase Coherence</div>
    </div>
  `;

    // Plots
    const plotsGrid = document.getElementById('plotsGrid');
    const plots = results.plots || {};
    const plotEntries = [
        { key: 'summary', title: '&#x1F4CA; Analysis Summary', file: plots.summary },
        { key: 'differential', title: '&#x1F50D; Differential Analysis (dK)', file: plots.differential },
        { key: 'radiomics', title: '&#x1F9EC; K-Space Radiomics', file: plots.radiomics },
        { key: 'phase', title: '&#x1F300; Phase Coherence', file: plots.phase },
        { key: 'gallery', title: '&#x1F5BC; Slice Gallery', file: plots.gallery },
    ];

    plotsGrid.innerHTML = plotEntries
        .filter(p => p.file)
        .map(p => `
      <div class="plot-card">
        <h3>${p.title}</h3>
        <img src="/static/results/${study.id}/${p.file}" alt="${p.title}" loading="lazy">
      </div>
    `).join('');

    // Transitions table
    const transitions = results.transitions || [];
    if (transitions.length > 0) {
        const tableEl = document.getElementById('transitionsTable');
        tableEl.style.display = 'block';
        const tbody = document.getElementById('transitionsBody');
        tbody.innerHTML = transitions.map(t => `
      <tr class="${t.is_anomaly ? 'anomaly' : ''}">
        <td>Slice ${t.from_slice} &#x2192; ${t.to_slice}</td>
        <td>${t.dk_score.toFixed(4)}</td>
        <td><span class="badge ${t.is_anomaly ? 'badge-error' : 'badge-completed'}">${t.status}</span></td>
      </tr>
    `).join('');
    }
}

function closeDashboard() {
    document.getElementById('mainView').style.display = 'block';
    document.getElementById('dashboard').classList.remove('active');
    document.getElementById('transitionsTable').style.display = 'none';
    currentStudyId = null;
    loadStudies(); // Refresh list logic
    switchTab('studies'); // Ensure we are on studies tab when returning
}

async function deleteCurrentStudy() {
    if (!currentStudyId) return;
    if (!confirm('Delete this study?')) return;

    try {
        await fetch(`/api/study/${currentStudyId}`, { method: 'DELETE' });
        showToast('Study deleted', 'success');
        closeDashboard();
    } catch (err) {
        showToast('Delete failed: ' + err.message, 'error');
    }
}

// ---- Toast Notifications ----
function showToast(message, type = 'info') {
    const container = document.getElementById('toasts');
    const toast = document.createElement('div');
    toast.className = `toast toast-${type}`;
    toast.innerHTML = message;
    container.appendChild(toast);

    setTimeout(() => {
        toast.style.opacity = '0';
        toast.style.transform = 'translateX(40px)';
        toast.style.transition = '0.3s ease';
        setTimeout(() => toast.remove(), 300);
    }, 4000);
}

// ---- Comparison ----
async function loadComparisonOptions() {
    try {
        const res = await fetch('/api/studies');
        const studies = await res.json();

        if (studies.length < 2) {
            document.getElementById('compareWarning').style.display = 'block';
            document.getElementById('compareBtn').disabled = true;
            document.getElementById('compareBtn').innerText = "Need 2+ Studies";
        } else {
            document.getElementById('compareWarning').style.display = 'none';
            document.getElementById('compareBtn').disabled = false;
            document.getElementById('compareBtn').innerText = "Run Comparison";
        }

        const opts = studies.map(s => `<option value="${s.id}">${s.metadata?.series_description || s.id} (${s.created_at})</option>`).join('');
        document.getElementById('studySelect1').innerHTML = opts;
        document.getElementById('studySelect2').innerHTML = opts;
    } catch (err) {
        showToast('Error loading studies for comparison', 'error');
    }
}

async function runComparison() {
    const s1 = document.getElementById('studySelect1').value;
    const s2 = document.getElementById('studySelect2').value;

    if (!s1 || !s2) {
        showToast('Please select two studies', 'warning');
        return;
    }

    if (s1 === s2) {
        showToast('Please select different studies', 'warning');
        return;
    }

    showToast('Running comparison...', 'info');

    try {
        const res = await fetch('/api/compare', {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({ study_id_1: s1, study_id_2: s2 })
        });

        if (!res.ok) throw new Error(await res.text());

        const data = await res.json();

        document.getElementById('comparisonResults').style.display = 'block';
        document.getElementById('compPlot').src = data.plot_url;

        // Render metrics
        const metricsHtml = data.metrics.map(m => `
            <div style="display:flex; justify-content:space-between; padding:4px; border-bottom:1px solid #ddd;">
                <span>Slice ${m.slice}</span>
                <span>Mag Diff: ${m.mean_mag_diff.toFixed(2)} | Phase Diff: ${m.mean_phase_diff.toFixed(2)}</span>
            </div>
        `).join('');
        document.getElementById('compMetrics').innerHTML = `<div style="max-height:200px; overflow-y:auto;">${metricsHtml}</div>`;

        showToast('Comparison complete', 'success');

    } catch (err) {
        showToast('Comparison failed: ' + err.message, 'error');
    }
}

// ---- 3D Visualization ----
let renderer, scene, camera, controls;

async function openVolumeView(studyId) {
    document.getElementById('volumeModal').style.display = 'block';
    showToast('Loading 3D data...', 'info');

    try {
        const res = await fetch(`/api/volume/${studyId}`);
        if (!res.ok) throw new Error('Failed to load volume data');
        const data = await res.json();

        initThreeJS(data.points);
    } catch (err) {
        showToast('Error: ' + err.message, 'error');
        closeVolumeModal();
    }
}

function closeVolumeModal() {
    document.getElementById('volumeModal').style.display = 'none';
    // Cleanup Three.js if needed
    if (renderer) {
        document.getElementById('volumeCanvasContainer').innerHTML = '';
        renderer = null;
    }
}

function initThreeJS(points) {
    const container = document.getElementById('volumeCanvasContainer');
    container.innerHTML = '';

    if (typeof THREE === 'undefined') {
        container.innerHTML = '<div style="color:red; padding:20px; text-align:center;">Three.js library failed to load.<br>Please check network connection or file path (/static/three.min.js).</div>';
        showToast('Three.js library not found', 'error');
        return;
    }

    // Scene
    scene = new THREE.Scene();
    scene.background = new THREE.Color(0x000000);

    // Camera
    camera = new THREE.PerspectiveCamera(75, container.clientWidth / container.clientHeight, 0.1, 1000);
    camera.position.z = 100;
    camera.position.y = 50;
    camera.position.x = 50;
    camera.lookAt(0, 0, 0);

    // Renderer
    renderer = new THREE.WebGLRenderer({ antialias: true });
    renderer.setSize(container.clientWidth, container.clientHeight);
    container.appendChild(renderer.domElement);

    // Store original points for filtering
    window.originalPoints = points;

    // Calculate center
    let cx = 0, cy = 0, cz = 0;
    points.forEach(p => { cx += p.x; cy += p.y; cz += p.z; });
    cx /= points.length; cy /= points.length; cz /= points.length;
    window.center = { x: cx, y: cy, z: cz };

    // Initial Render
    update3DView();

    // Animation Loop
    function animate() {
        if (!renderer) return;
        requestAnimationFrame(animate);

        const speed = document.getElementById('speedSlider').value / 1000;
        if (scene.getObjectByName('pointCloud')) {
            scene.getObjectByName('pointCloud').rotation.y += speed;
        }

        renderer.render(scene, camera);
    }

    animate();

    // Mouse interaction (basic rotation via event listeners)
    let isDragging = false;
    let previousMousePosition = { x: 0, y: 0 };

    renderer.domElement.addEventListener('mousedown', (e) => { isDragging = true; });
    renderer.domElement.addEventListener('mouseup', (e) => { isDragging = false; });
    renderer.domElement.addEventListener('mousemove', (e) => {
        if (isDragging) {
            const deltaMove = {
                x: e.offsetX - previousMousePosition.x,
                y: e.offsetY - previousMousePosition.y
            };

            const pc = scene.getObjectByName('pointCloud');
            if (pc) {
                pc.rotation.y += deltaMove.x * 0.01;
                pc.rotation.x += deltaMove.y * 0.01;
            }
        }
        previousMousePosition = { x: e.offsetX, y: e.offsetY };
    });
}

function update3DView() {
    if (!window.originalPoints || !scene) return;

    const threshPercent = document.getElementById('thresholdSlider').value;
    const pointSize = document.getElementById('sizeSlider').value;

    // Determine threshold value from percentage
    // Find max value in dataset to scale
    const maxVal = Math.max(...window.originalPoints.map(p => p.val));
    const minVal = Math.min(...window.originalPoints.map(p => p.val));
    const cutoff = minVal + ((maxVal - minVal) * (threshPercent / 100));

    // Filter points
    const filteredPoints = window.originalPoints.filter(p => p.val >= cutoff);

    // Create new Geometry
    const vertices = [];
    const colors = [];
    const cx = window.center.x, cy = window.center.y, cz = window.center.z;

    filteredPoints.forEach(p => {
        vertices.push(p.x - cx, p.y - cy, p.z - cz);

        const intensity = Math.min(1, p.val / 1000);
        const color = new THREE.Color();
        color.setHSL(0.7 - (intensity * 0.7), 1.0, 0.5);
        colors.push(color.r, color.g, color.b);
    });

    const geometry = new THREE.BufferGeometry();
    geometry.setAttribute('position', new THREE.Float32BufferAttribute(vertices, 3));
    geometry.setAttribute('color', new THREE.Float32BufferAttribute(colors, 3));

    const material = new THREE.PointsMaterial({
        size: parseFloat(pointSize),
        vertexColors: true,
        transparent: true,
        opacity: 0.8,
        blending: THREE.AdditiveBlending // Glow effect
    });

    // Remove old cloud
    const oldCloud = scene.getObjectByName('pointCloud');
    if (oldCloud) scene.remove(oldCloud);

    const pointCloud = new THREE.Points(geometry, material);
    pointCloud.name = 'pointCloud';
    scene.add(pointCloud);
}

// ---- Init ----
document.addEventListener('DOMContentLoaded', () => {
    loadStudies();
});
