/**
 * app.js — CV_04 Frontend Logic
 *
 * Image Thresholding & Unsupervised Segmentation
 * Handles: tab switching, method selection, file uploads, API calls,
 *          histogram drawing, seed point interaction, image zoom, save.
 */

(function () {
    'use strict';

    /* ═══════════════════════════════════════════════════════════════
     *  STATE
     * ═══════════════════════════════════════════════════════════════ */

    const state = {
        // Thresholding
        threshMethod: 'optimal',
        threshPath: '',
        threshLocalUrl: '',
        threshResultB64: '',
        // Segmentation
        segMethod: 'kmeans',
        segPath: '',
        segLocalUrl: '',
        segResultB64: '',
        // Region growing seeds: [{y, x}, ...]
        seeds: [],
        // Original image natural dimensions (for seed coordinate mapping)
        segNatW: 0,
        segNatH: 0,
    };

    /* ═══════════════════════════════════════════════════════════════
     *  DOM HELPERS
     * ═══════════════════════════════════════════════════════════════ */

    const $ = (sel) => document.querySelector(sel);
    const $$ = (sel) => document.querySelectorAll(sel);
    const show = (el) => { if (el) el.hidden = false; };
    const hide = (el) => { if (el) el.hidden = true; };

    const loader = $('#loader');
    function showLoader() { show(loader); }
    function hideLoader() { hide(loader); }

    /* ═══════════════════════════════════════════════════════════════
     *  TAB NAVIGATION
     * ═══════════════════════════════════════════════════════════════ */

    $$('.tab-btn').forEach(btn => {
        btn.addEventListener('click', () => {
            $$('.tab-btn').forEach(b => b.classList.remove('active'));
            btn.classList.add('active');
            const tab = btn.dataset.tab;
            $$('.tab-panel').forEach(p => p.classList.remove('active'));
            $(`#panel-${tab}`).classList.add('active');
        });
    });

    /* ═══════════════════════════════════════════════════════════════
     *  SLIDER VALUE DISPLAY
     * ═══════════════════════════════════════════════════════════════ */

    $$('.ctrl').forEach(ctrl => {
        const input = ctrl.querySelector('input[type="range"]');
        const valSpan = ctrl.querySelector('.val');
        if (!input || !valSpan) return;
        input.addEventListener('input', () => {
            valSpan.textContent = input.value;
        });
    });

    /* ═══════════════════════════════════════════════════════════════
     *  METHOD SELECTION (Thresholding)
     * ═══════════════════════════════════════════════════════════════ */

    const threshMethodBtns = $$('#thresh-methods .method-btn');
    threshMethodBtns.forEach(btn => {
        btn.addEventListener('click', () => {
            threshMethodBtns.forEach(b => b.classList.remove('active'));
            btn.classList.add('active');
            state.threshMethod = btn.dataset.method;
            updateThreshParams();
        });
    });

    function updateThreshParams() {
        // Hide all thresh param panels, show the active one
        ['spectral', 'local'].forEach(m => {
            const panel = $(`#params-${m}`);
            if (panel) panel.classList.remove('active');
        });
        const active = $(`#params-${state.threshMethod}`);
        if (active) active.classList.add('active');
    }

    /* ═══════════════════════════════════════════════════════════════
     *  METHOD SELECTION (Segmentation)
     * ═══════════════════════════════════════════════════════════════ */

    const segMethodBtns = $$('#seg-methods .method-btn');
    segMethodBtns.forEach(btn => {
        btn.addEventListener('click', () => {
            segMethodBtns.forEach(b => b.classList.remove('active'));
            btn.classList.add('active');
            state.segMethod = btn.dataset.method;
            updateSegParams();
        });
    });

    function updateSegParams() {
        ['kmeans', 'region_growing', 'agglomerative', 'mean_shift'].forEach(m => {
            const panel = $(`#params-${m}`);
            if (panel) panel.classList.remove('active');
        });
        const active = $(`#params-${state.segMethod}`);
        if (active) active.classList.add('active');
    }

    /* ═══════════════════════════════════════════════════════════════
     *  FILE UPLOAD (with drag-and-drop)
     * ═══════════════════════════════════════════════════════════════ */

    function setupUpload(zoneId, inputId, thumbId, textId, onUpload) {
        const zone = $(`#${zoneId}`);
        const input = $(`#${inputId}`);
        const thumb = $(`#${thumbId}`);
        const textEl = $(`#${textId}`);

        if (!zone || !input) return;

        // Drag events
        ['dragenter', 'dragover'].forEach(ev => {
            zone.addEventListener(ev, e => {
                e.preventDefault();
                zone.classList.add('dragover');
            });
        });
        ['dragleave', 'drop'].forEach(ev => {
            zone.addEventListener(ev, e => {
                e.preventDefault();
                zone.classList.remove('dragover');
            });
        });
        zone.addEventListener('drop', e => {
            const file = e.dataTransfer.files[0];
            if (file) handleFile(file);
        });

        input.addEventListener('change', () => {
            if (input.files[0]) handleFile(input.files[0]);
        });

        function handleFile(file) {
            if (!file.type.startsWith('image/')) return;

            // Show local preview
            const localUrl = URL.createObjectURL(file);
            if (thumb) {
                thumb.src = localUrl;
                show(thumb);
            }
            if (textEl) textEl.textContent = file.name;

            // Upload to server
            const fd = new FormData();
            fd.append('image', file);

            fetch('/api/upload/', { method: 'POST', body: fd })
                .then(r => r.json())
                .then(data => {
                    if (data.path) {
                        onUpload(data.path, localUrl);
                    }
                })
                .catch(err => console.error('Upload failed:', err));
        }
    }

    // Setup thresholding upload
    setupUpload('upload-zone-thresh', 'input-thresh', 'thumb-thresh', 'upload-text-thresh',
        (path, localUrl) => {
            state.threshPath = path;
            state.threshLocalUrl = localUrl;
            $('#btn-thresh').disabled = false;
        }
    );

    // Setup segmentation upload
    setupUpload('upload-zone-seg', 'input-seg', 'thumb-seg', 'upload-text-seg',
        (path, localUrl) => {
            state.segPath = path;
            state.segLocalUrl = localUrl;
            state.seeds = [];
            state.segNatW = 0;
            state.segNatH = 0;
            clearSeedMarkers();
            $('#btn-seg').disabled = false;
        }
    );

    /* ═══════════════════════════════════════════════════════════════
     *  THRESHOLDING — RUN
     * ═══════════════════════════════════════════════════════════════ */

    $('#btn-thresh').addEventListener('click', runThresholding);

    function runThresholding() {
        if (!state.threshPath) return;
        showLoader();

        const method = state.threshMethod;
        const endpointMap = {
            optimal: '/api/optimal-threshold/',
            otsu: '/api/otsu-threshold/',
            spectral: '/api/spectral-threshold/',
            local: '/api/local-threshold/',
        };

        const fd = new FormData();
        fd.append('image_path', state.threshPath);

        // Method-specific params
        if (method === 'spectral') {
            fd.append('num_classes', getParamVal('num_classes'));
        } else if (method === 'local') {
            fd.append('block_size', getParamVal('block_size'));
        }

        fetch(endpointMap[method], { method: 'POST', body: fd })
            .then(r => r.json())
            .then(data => {
                hideLoader();
                if (data.error) { alert('Error: ' + data.error); return; }
                displayThreshResult(data, method);
            })
            .catch(err => {
                hideLoader();
                console.error(err);
                alert('Processing failed.');
            });
    }

    function displayThreshResult(data, method) {
        hide($('#results-thresh-empty'));
        show($('#results-thresh'));

        // Original image
        $('#img-thresh-original').src = state.threshLocalUrl;

        // Result image
        const b64 = data.image_b64;
        state.threshResultB64 = b64;
        $('#img-thresh-result').src = 'data:image/png;base64,' + b64;

        // Method title
        const titles = {
            optimal: 'Optimal Threshold',
            otsu: "Otsu's Threshold",
            spectral: 'Spectral Threshold',
            local: 'Local (Adaptive) Threshold',
        };
        $('#thresh-method-title').textContent = titles[method] || 'Thresholded';

        // Info badge
        let info = '';
        if (method === 'optimal') {
            info = `T=${data.threshold}, ${data.iterations} iters`;
        } else if (method === 'otsu') {
            info = `T=${data.threshold}`;
        } else if (method === 'spectral') {
            info = `${data.num_classes} classes`;
        } else if (method === 'local') {
            info = `${data.num_blocks} blocks (${data.block_size}×${data.block_size})`;
        }
        $('#thresh-info').textContent = info;
        $('#thresh-time').textContent = data.time_ms + ' ms';

        // Draw histogram
        if (data.histogram) {
            const thresholds = [];
            if (data.threshold !== undefined) thresholds.push(data.threshold);
            if (data.thresholds) data.thresholds.forEach(t => thresholds.push(t));
            drawHistogram('canvas-histogram', data.histogram, thresholds);
            $('#thresh-line-info').textContent = thresholds.length > 0
                ? 'T=' + thresholds.join(', ')
                : '';
        }

        // Enable save button
        $('#btn-save-thresh').disabled = false;
    }

    /* ═══════════════════════════════════════════════════════════════
     *  SEGMENTATION — RUN
     * ═══════════════════════════════════════════════════════════════ */

    $('#btn-seg').addEventListener('click', runSegmentation);

    function runSegmentation() {
        if (!state.segPath) return;

        const method = state.segMethod;

        // For region growing, need seeds
        if (method === 'region_growing' && state.seeds.length === 0) {
            alert('Please click on the image to place at least one seed point.');
            return;
        }

        showLoader();

        const endpointMap = {
            kmeans: '/api/kmeans/',
            region_growing: '/api/region-growing/',
            agglomerative: '/api/agglomerative/',
            mean_shift: '/api/mean-shift/',
        };

        const fd = new FormData();
        fd.append('image_path', state.segPath);

        if (method === 'kmeans') {
            fd.append('k', getParamVal('k'));
            fd.append('max_iter', getParamVal('max_iter'));
        } else if (method === 'region_growing') {
            fd.append('threshold', getParamVal('rg_threshold'));
            fd.append('seeds', JSON.stringify(state.seeds.map(s => [s.y, s.x])));
        } else if (method === 'agglomerative') {
            fd.append('num_clusters', getParamVal('agg_clusters'));
        } else if (method === 'mean_shift') {
            fd.append('spatial_radius', getParamVal('spatial_radius'));
            fd.append('color_radius', getParamVal('color_radius'));
        }

        fetch(endpointMap[method], { method: 'POST', body: fd })
            .then(r => r.json())
            .then(data => {
                hideLoader();
                if (data.error) { alert('Error: ' + data.error); return; }
                displaySegResult(data, method);
            })
            .catch(err => {
                hideLoader();
                console.error(err);
                alert('Processing failed.');
            });
    }

    function displaySegResult(data, method) {
        hide($('#results-seg-empty'));
        show($('#results-seg'));

        // Original image
        const origImg = $('#img-seg-original');
        origImg.src = state.segLocalUrl;

        // Store natural dimensions after load
        origImg.onload = () => {
            state.segNatW = origImg.naturalWidth;
            state.segNatH = origImg.naturalHeight;
        };

        // Result image
        const b64 = data.image_b64;
        state.segResultB64 = b64;
        $('#img-seg-result').src = 'data:image/png;base64,' + b64;

        // Method title
        const titles = {
            kmeans: 'K-Means',
            region_growing: 'Region Growing',
            agglomerative: 'Agglomerative',
            mean_shift: 'Mean Shift',
        };
        $('#seg-method-title').textContent = titles[method] || 'Segmented';

        // Badges
        const clusters = data.num_clusters || data.num_regions || '—';
        $('#seg-clusters').textContent = clusters + ' clusters';
        $('#seg-time').textContent = data.time_ms + ' ms';

        const itersEl = $('#seg-iters');
        if (data.iterations !== undefined) {
            itersEl.textContent = data.iterations + ' iters';
            show(itersEl);
        } else {
            hide(itersEl);
        }

        // Enable save
        $('#btn-save-seg').disabled = false;
    }

    /* ═══════════════════════════════════════════════════════════════
     *  HISTOGRAM DRAWING
     * ═══════════════════════════════════════════════════════════════ */

    function drawHistogram(canvasId, histStr, thresholds = []) {
        const canvas = document.getElementById(canvasId);
        if (!canvas) return;

        const ctx = canvas.getContext('2d');
        const dpr = window.devicePixelRatio || 1;
        const rect = canvas.getBoundingClientRect();
        canvas.width = rect.width * dpr;
        canvas.height = rect.height * dpr;
        ctx.scale(dpr, dpr);

        const W = rect.width;
        const H = rect.height;

        // Parse histogram
        const hist = histStr.split(',').map(Number);
        const maxVal = Math.max(...hist, 1);

        // Clear
        ctx.clearRect(0, 0, W, H);

        // Draw bars
        const barW = W / 256;
        const gradient = ctx.createLinearGradient(0, H, 0, 0);
        gradient.addColorStop(0, 'rgba(139, 92, 246, 0.15)');
        gradient.addColorStop(1, 'rgba(6, 182, 212, 0.5)');

        ctx.fillStyle = gradient;
        for (let i = 0; i < 256; i++) {
            const barH = (hist[i] / maxVal) * (H - 10);
            ctx.fillRect(i * barW, H - barH, Math.max(barW - 0.5, 1), barH);
        }

        // Draw threshold lines
        thresholds.forEach((t, idx) => {
            const x = (t / 255) * W;
            ctx.strokeStyle = idx === 0
                ? 'rgba(239, 68, 68, 0.9)'
                : `hsla(${60 + idx * 40}, 90%, 60%, 0.85)`;
            ctx.lineWidth = 2;
            ctx.setLineDash([4, 3]);
            ctx.beginPath();
            ctx.moveTo(x, 0);
            ctx.lineTo(x, H);
            ctx.stroke();
            ctx.setLineDash([]);

            // Label
            ctx.font = '10px Inter, sans-serif';
            ctx.fillStyle = ctx.strokeStyle;
            ctx.fillText('T=' + t, x + 3, 12 + idx * 14);
        });
    }

    /* ═══════════════════════════════════════════════════════════════
     *  SEED POINT INTERACTION (Region Growing)
     * ═══════════════════════════════════════════════════════════════ */

    const seedContainer = $('#seed-image-container');
    if (seedContainer) {
        const origImg = $('#img-seg-original');

        // Left click → add seed
        seedContainer.addEventListener('click', (e) => {
            if (state.segMethod !== 'region_growing') return;
            if (!state.segPath) return;

            const rect = origImg.getBoundingClientRect();
            const relX = (e.clientX - rect.left) / rect.width;
            const relY = (e.clientY - rect.top) / rect.height;

            if (relX < 0 || relX > 1 || relY < 0 || relY > 1) return;

            // Map to natural image coordinates
            const natX = Math.round(relX * (state.segNatW || origImg.naturalWidth));
            const natY = Math.round(relY * (state.segNatH || origImg.naturalHeight));

            state.seeds.push({ x: natX, y: natY, relX, relY });
            addSeedMarker(relX, relY, state.seeds.length - 1);
        });

        // Right click → remove last seed
        seedContainer.addEventListener('contextmenu', (e) => {
            e.preventDefault();
            if (state.segMethod !== 'region_growing') return;
            if (state.seeds.length > 0) {
                state.seeds.pop();
                redrawSeedMarkers();
            }
        });
    }

    function addSeedMarker(relX, relY, index) {
        const container = $('#seed-image-container');
        if (!container) return;

        const colors = [
            '#ef4444', '#f59e0b', '#10b981', '#3b82f6',
            '#8b5cf6', '#ec4899', '#14b8a6', '#f97316'
        ];

        const marker = document.createElement('div');
        marker.className = 'seed-marker';
        marker.style.left = (relX * 100) + '%';
        marker.style.top = (relY * 100) + '%';
        marker.style.backgroundColor = colors[index % colors.length];
        marker.dataset.seedIndex = index;
        container.appendChild(marker);
    }

    function clearSeedMarkers() {
        const container = $('#seed-image-container');
        if (!container) return;
        container.querySelectorAll('.seed-marker').forEach(m => m.remove());
    }

    function redrawSeedMarkers() {
        clearSeedMarkers();
        state.seeds.forEach((s, i) => addSeedMarker(s.relX, s.relY, i));
    }

    /* ═══════════════════════════════════════════════════════════════
     *  SAVE RESULT
     * ═══════════════════════════════════════════════════════════════ */

    $('#btn-save-thresh').addEventListener('click', () => {
        saveResult(state.threshResultB64, state.threshMethod);
    });

    $('#btn-save-seg').addEventListener('click', () => {
        saveResult(state.segResultB64, state.segMethod);
    });

    function saveResult(b64, algorithm) {
        if (!b64) return;
        showLoader();

        const fd = new FormData();
        fd.append('image_b64', b64);
        fd.append('algorithm', algorithm);

        fetch('/api/save-result/', { method: 'POST', body: fd })
            .then(r => r.json())
            .then(data => {
                hideLoader();
                if (data.error) { alert('Save failed: ' + data.error); return; }
                alert(`Saved: ${data.filename} (${(data.size_bytes / 1024).toFixed(1)} KB)`);
            })
            .catch(err => {
                hideLoader();
                console.error(err);
                alert('Save failed.');
            });
    }

    /* ═══════════════════════════════════════════════════════════════
     *  IMAGE ZOOM ON CLICK
     * ═══════════════════════════════════════════════════════════════ */

    document.addEventListener('click', (e) => {
        const img = e.target;
        // Only zoom result images and non-seed-container originals
        if (img.tagName !== 'IMG') return;
        if (!img.closest('.result-card')) return;
        if (img.id === 'img-seg-original' && state.segMethod === 'region_growing') return;
        if (!img.src || img.src === window.location.href) return;

        const overlay = document.createElement('div');
        overlay.className = 'zoom-overlay';
        const zoomImg = document.createElement('img');
        zoomImg.src = img.src;
        overlay.appendChild(zoomImg);
        document.body.appendChild(overlay);

        overlay.addEventListener('click', () => overlay.remove());
        document.addEventListener('keydown', function esc(e) {
            if (e.key === 'Escape') {
                overlay.remove();
                document.removeEventListener('keydown', esc);
            }
        });
    });

    /* ═══════════════════════════════════════════════════════════════
     *  HELPERS
     * ═══════════════════════════════════════════════════════════════ */

    function getParamVal(key) {
        const ctrl = document.querySelector(`.ctrl[data-key="${key}"] input[type="range"]`);
        return ctrl ? ctrl.value : '';
    }

    // Load natural dimensions for seg original image on first load
    const segOrigImg = $('#img-seg-original');
    if (segOrigImg) {
        segOrigImg.addEventListener('load', () => {
            state.segNatW = segOrigImg.naturalWidth;
            state.segNatH = segOrigImg.naturalHeight;
        });
    }

})();
