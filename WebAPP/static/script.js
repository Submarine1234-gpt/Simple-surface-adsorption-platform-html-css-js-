document.addEventListener('DOMContentLoaded', function() {
    // Check if Plotly is loaded
    if (typeof Plotly === 'undefined') {
        console.error('Plotly.js library is not loaded. 3D Surface functionality will be disabled.');
        const surface3dBtn = document.getElementById('surface3d-btn');
        if (surface3dBtn) {
            surface3dBtn.style.display = 'none';
        }
        return;
    }

    // Button elements
    const chartControls = document.getElementById('chart-controls');
    const scatterBtn = document.getElementById('scatter-btn');
    const heatmapBtn = document.getElementById('heatmap-btn');
    const surface3dBtn = document.getElementById('surface3d-btn');
    
    // Other variable declarations
    const calculationForm = document.getElementById('calculation-form');
    const substrateFileInput = document.getElementById('substrate_file');
    const adsorbateFileInput = document.getElementById('adsorbate_file');
    const submitButton = document.getElementById('submit-button');
    const dashboard = document.getElementById('dashboard');
    const logOutput = document.getElementById('log-output');
    const heatmapContainer = document.getElementById('heatmap-container');
    const surface3dContainer = document.getElementById('surface3d-container');
    const downloadLinkContainer = document.getElementById('result-download-link');
    const downloadButton = document.getElementById('download-button');
    const historyTableBody = document.querySelector('#results-table tbody');

    // Form controls
    const rotationMethodCheckbox = document.getElementById('rotation_method');
    const rotationCountInput = document.getElementById('rotation_count');
    const rotationStepInput = document.getElementById('rotation_step');
    const rotationMethodOptions = document.getElementById('rotation-method-options');
    const hollowSitesCheckbox = document.getElementById('hollow_sites_enabled');
    const knnNeighborsInput = document.getElementById('knn_neighbors');
    const hollowDedupInput = document.getElementById('hollow_site_deduplication_distance');
    const hollowOptions = document.getElementById('hollow-options');
    const onTopSitesCheckbox = document.getElementById('on_top_sites_enabled');
    const onTopTargetInput = document.getElementById('on_top_target_atom');
    const onTopOptions = document.getElementById('on-top-options-container');
    
    // Initialize form controls
    function updateRotationMethodParams() {
        const enabled = rotationMethodCheckbox.checked;
        rotationCountInput.disabled = !enabled;
        rotationStepInput.disabled = !enabled;
        rotationMethodOptions.style.opacity = enabled ? '' : '0.6';
    }
    function updateHollowParams() {
        const enabled = hollowSitesCheckbox.checked;
        knnNeighborsInput.disabled = !enabled;
        hollowDedupInput.disabled = !enabled;
        hollowOptions.style.opacity = enabled ? '' : '0.6';
    } 
    function updateOnTopParams() {
        const enabled = onTopSitesCheckbox.checked;
        onTopTargetInput.disabled = !enabled;
        onTopOptions.style.opacity = enabled ? '' : '0.6';
    }
    
    updateRotationMethodParams();
    updateHollowParams();
    updateOnTopParams();
    rotationMethodCheckbox.addEventListener('change', updateRotationMethodParams);
    hollowSitesCheckbox.addEventListener('change', updateHollowParams);
    onTopSitesCheckbox.addEventListener('change', updateOnTopParams);

    // State variables
    let activeEventSource = null;
    let activeSessionId = null;
    let activeSurfaceAxis = '2';
    let heatmapChart = null;
    let loadedVizData = null;
    let currentChartType = 'scatter';

    // Event Listeners
    calculationForm.addEventListener('submit', handleFormSubmit);
    historyTableBody.addEventListener('click', handleHistoryViewClick);
    scatterBtn.addEventListener('click', () => switchChartView('scatter'));
    heatmapBtn.addEventListener('click', () => switchChartView('heatmap'));
    if (surface3dBtn) {
        surface3dBtn.addEventListener('click', () => switchChartView('surface3d'));
    }

    refreshHistory();

    // Switch chart view function
    function switchChartView(type) {
        currentChartType = type;
        scatterBtn.classList.toggle('active', type === 'scatter');
        heatmapBtn.classList.toggle('active', type === 'heatmap');
        if (surface3dBtn) {
            surface3dBtn.classList.toggle('active', type === 'surface3d');
        }
        
        // Hide all containers
        heatmapContainer.classList.add('hidden');
        surface3dContainer.classList.add('hidden');
        
        // Re-render the chart with the currently loaded data
        if (loadedVizData) {
            if (type === 'scatter') {
                heatmapContainer.classList.remove('hidden');
                renderScatterChart(loadedVizData.surface, loadedVizData.adsorption, loadedVizData.axis);
            } else if (type === 'heatmap') {
                heatmapContainer.classList.remove('hidden');
                renderDensityFieldHeatmap(loadedVizData.surface, loadedVizData.adsorption, loadedVizData.axis);
            } else if (type === 'surface3d' && typeof Plotly !== 'undefined') {
                surface3dContainer.classList.remove('hidden');
                render3DSurfacePlotly(loadedVizData.surface, loadedVizData.adsorption, loadedVizData.axis);
            }
        }
    }

    function handleFormSubmit(event) {
        event.preventDefault();
        if (substrateFileInput.files.length === 0 || adsorbateFileInput.files.length === 0) {
            alert('Please upload both a substrate and an adsorbate file.'); 
            return;
        }
        if (activeEventSource) activeEventSource.close();
        resetDashboard();
        const formData = new FormData(calculationForm);
        submitButton.disabled = true;
        submitButton.textContent = '计算中...';
        dashboard.classList.remove('hidden');

        fetch('/run-calculation', { method: 'POST', body: formData })
        .then(response => response.json())
        .then(data => {
            if (data.success) {
                activeSessionId = data.session_id;
                activeSurfaceAxis = data.surface_axis || '2';
                logOutput.innerHTML = `<div>[SYSTEM] Calculation started. Session ID: ${activeSessionId}</div>`;
                startLogStream(data.session_id);
            } else { 
                throw new Error(data.message); 
            }
        })
        .catch(error => {
            logOutput.innerHTML += `<div class="log-error">[ERROR] Submission failed: ${error.message}</div>`;
            resetUiAfterCalculation();
        });
    }

    function startLogStream(sessionId) {
        if (activeEventSource) activeEventSource.close();
        activeEventSource = new EventSource(`/stream-logs/${sessionId}`);
        activeEventSource.onmessage = function(event) {
            const line = event.data;
            logOutput.innerHTML += `<div>${line}</div>`;
            logOutput.scrollTop = logOutput.scrollHeight;
            if (line.includes('Log stream finished')) {
                activeEventSource.close();
                pollForCompletion(sessionId);
            }
        };
        activeEventSource.onerror = function() {
            logOutput.innerHTML += '<div class="log-error">[SYSTEM] Log stream disconnected. Checking status...</div>';
            activeEventSource.close();
            if (sessionId) pollForCompletion(sessionId);
            else resetUiAfterCalculation();
        };
    }

    function handleHistoryViewClick(event) {
        if (event.target.tagName !== 'BUTTON' || !event.target.dataset.session) return;
        if (submitButton.disabled) {
            alert("Please wait for the current calculation to finish before viewing past results."); 
            return;
        }
        const sessionId = event.target.dataset.session;
        const surfaceAxis = event.target.dataset.axis || '2';
        resetDashboard();
        dashboard.classList.remove('hidden');
        logOutput.innerHTML = `<div>[SYSTEM] Displaying results for past session: ${sessionId}</div>`;
        loadVisualization(sessionId, surfaceAxis);
    }
    
    function pollForCompletion(sessionId) {
        logOutput.innerHTML += '<div>[SYSTEM] Finalizing results...</div>';
        logOutput.scrollTop = logOutput.scrollHeight;
        let attempts = 0;
        const interval = setInterval(() => {
            fetch(`/check-status/${sessionId}`)
                .then(res => res.json())
                .then(data => {
                    attempts++;
                    if (data.status === 'complete' || attempts >= 10) {
                        clearInterval(interval);
                        loadVisualization(sessionId, activeSurfaceAxis);
                        resetUiAfterCalculation();
                    }
                })
                .catch(() => { 
                    clearInterval(interval); 
                    resetUiAfterCalculation(); 
                });
        }, 500);
    }

    function resetUiAfterCalculation() {
        submitButton.disabled = false;
        submitButton.textContent = '开始计算';
        refreshHistory();
    }
    
    function loadVisualization(sessionId, surfaceAxis) {
        downloadButton.href = `/download-result/${sessionId}`;
        downloadLinkContainer.classList.remove('hidden');
        const surfaceDataPromise = fetch(`/get-viz-data/${sessionId}/surface_atoms.json`).then(res => res.json());
        const adsorptionDataPromise = fetch(`/get-viz-data/${sessionId}/adsorption_sites.json`).then(res => res.json());
        
        Promise.all([surfaceDataPromise, adsorptionDataPromise])
            .then(([surfaceData, adsorptionData]) => {
                if (surfaceData.error || adsorptionData.error) throw new Error(surfaceData.error || adsorptionData.error);
                
                loadedVizData = { surface: surfaceData, adsorption: adsorptionData, axis: surfaceAxis };
                chartControls.classList.remove('hidden');
                heatmapContainer.classList.remove('hidden');
                switchChartView(currentChartType);
            })
            .catch(error => {
                heatmapContainer.classList.remove('hidden');
                heatmapContainer.innerHTML = `<p>Failed to load visualization data: ${error.message}</p>`;
            });
    }

    function renderScatterChart(surfaceData, adsorptionData, surfaceAxis) {
        if (heatmapChart) heatmapChart.dispose();
        heatmapChart = echarts.init(heatmapContainer);
        let plotAxes; 
        const axisNames = ['X', 'Y', 'Z']; 
        const normalAxis = parseInt(surfaceAxis, 10);
        if (normalAxis === 0) { plotAxes = { x: 1, y: 2 }; } 
        else if (normalAxis === 1) { plotAxes = { x: 0, y: 2 }; } 
        else { plotAxes = { x: 0, y: 1 }; }
        const energyValues = adsorptionData.sites.map(site => site.energy);
        
        heatmapChart.setOption({
            title: { text: 'Surface Atoms and Adsorption Sites (Scatter)', left: 'center' },
            legend: { top: 30, data: ['Surface Atoms', 'Adsorption Sites'] },
            tooltip: { 
                formatter: params => `<b>${params.seriesName}</b><br/>${axisNames[plotAxes.x]}: ${params.value[0].toFixed(2)}<br/>${axisNames[plotAxes.y]}: ${params.value[1].toFixed(2)}<br/>` + 
                (params.value[2] !== undefined ? `Energy: ${params.value[2].toFixed(4)} eV` : '') 
            },
            grid: { top: 70, right: 150, bottom: 60, left: 70 },
            xAxis: { type: 'value', name: axisNames[plotAxes.x], scale: true },
            yAxis: { type: 'value', name: axisNames[plotAxes.y], scale: true },
            visualMap: { 
                seriesIndex: 1, 
                min: Math.min(...energyValues), 
                max: Math.max(...energyValues), 
                calculable: true, 
                orient: 'vertical', 
                right: 10, 
                top: 'center', 
                text: ['High', 'Low'], 
                inRange: { color: ['#a50026', '#d73027', '#f46d43', '#fdae61', '#fee08b', '#d9ef8b', '#a6d96a', '#66bd63', '#1a9850', '#006837'] } 
            },
            series: [
                { 
                    name: 'Surface Atoms', 
                    type: 'scatter', 
                    symbolSize: 8, 
                    data: surfaceData.coords.map(c => [c[plotAxes.x], c[plotAxes.y]]), 
                    itemStyle: { color: '#999', opacity: 0.7 } 
                },
                { 
                    name: 'Adsorption Sites', 
                    type: 'scatter', 
                    symbolSize: 15, 
                    data: adsorptionData.sites.map(s => [s.coords[plotAxes.x], s.coords[plotAxes.y], s.energy]), 
                    itemStyle: { borderColor: '#555', borderWidth: 1 } 
                }
            ]
        });
    }

    // CORRECTED: Physical density field heatmap - surface atoms as heat sources, adsorption sites as energy field
    function renderDensityFieldHeatmap(surfaceData, adsorptionData, surfaceAxis) {
        // Clear previous chart
        heatmapContainer.innerHTML = '';

        // Determine plot axes
        const axisNames = ['X', 'Y', 'Z'];
        const normalAxis = parseInt(surfaceAxis, 10);
        let plotAxes;
        if (normalAxis === 0) { plotAxes = { x: 1, y: 2 }; }
        else if (normalAxis === 1) { plotAxes = { x: 0, y: 2 }; }
        else { plotAxes = { x: 0, y: 1 }; }

        // Get coordinate ranges with padding
        const allX = [...surfaceData.coords.map(c => c[plotAxes.x]), ...adsorptionData.sites.map(s => s.coords[plotAxes.x])];
        const allY = [...surfaceData.coords.map(c => c[plotAxes.y]), ...adsorptionData.sites.map(s => s.coords[plotAxes.y])];
        const xMin = Math.min(...allX) - 2;
        const xMax = Math.max(...allX) + 2;
        const yMin = Math.min(...allY) - 2;
        const yMax = Math.max(...allY) + 2;

        // Create high-resolution grid
        const gridSize = 100;
        const xStep = (xMax - xMin) / gridSize;
        const yStep = (yMax - yMin) / gridSize;

        // Initialize grids
        const surfaceDensity = Array(gridSize).fill().map(() => Array(gridSize).fill(0));
        const energyField = Array(gridSize).fill().map(() => Array(gridSize).fill(0));
        const combinedField = Array(gridSize).fill().map(() => Array(gridSize).fill(0));
        
        const x = Array.from({length: gridSize}, (_, i) => xMin + i * xStep);
        const y = Array.from({length: gridSize}, (_, i) => yMin + i * yStep);

        // Energy values for normalization
        const energyValues = adsorptionData.sites.map(site => site.energy);
        const minEnergy = Math.min(...energyValues);
        const maxEnergy = Math.max(...energyValues);

        // Step 1: Create surface atom density field (yellow regions from atoms)
        for (let i = 0; i < gridSize; i++) {
            for (let j = 0; j < gridSize; j++) {
                const gridX = x[i];
                const gridY = y[j];
                
                // Calculate surface density from all surface atoms
                surfaceData.coords.forEach(coord => {
                    const atomX = coord[plotAxes.x];
                    const atomY = coord[plotAxes.y];
                    const distance = Math.sqrt((gridX - atomX) ** 2 + (gridY - atomY) ** 2);
                    
                    // Gaussian-like influence with appropriate sigma for atomic radius
                    const sigma = 1.2; // Atomic influence radius
                    const influence = Math.exp(-distance * distance / (2 * sigma * sigma));
                    surfaceDensity[j][i] += influence;
                });
            }
        }

        // Step 2: Create adsorption energy field (energy-based color field)
        for (let i = 0; i < gridSize; i++) {
            for (let j = 0; j < gridSize; j++) {
                const gridX = x[i];
                const gridY = y[j];
                
                // Calculate energy field from all adsorption sites
                let totalEnergyInfluence = 0;
                let totalWeight = 0;
                
                adsorptionData.sites.forEach(site => {
                    const siteX = site.coords[plotAxes.x];
                    const siteY = site.coords[plotAxes.y];
                    const distance = Math.sqrt((gridX - siteX) ** 2 + (gridY - siteY) ** 2);
                    
                    // Energy influence with larger sigma for adsorption zone
                    const sigma = 0.8; // Adsorption site influence radius
                    const weight = Math.exp(-distance * distance / (2 * sigma * sigma));
                    
                    if (weight > 0.01) { // Only consider significant influences
                        // Normalize energy (lower energy = higher activity = higher value)
                        const normalizedEnergy = (maxEnergy - site.energy) / (maxEnergy - minEnergy);
                        totalEnergyInfluence += weight * normalizedEnergy;
                        totalWeight += weight;
                    }
                });
                
                // Average energy influence at this grid point
                energyField[j][i] = totalWeight > 0 ? totalEnergyInfluence / totalWeight : 0;
            }
        }

        // Step 3: Combine surface density and energy field into final field
        let maxCombined = 0;
        for (let i = 0; i < gridSize; i++) {
            for (let j = 0; j < gridSize; j++) {
                // Normalize surface density to 0-1 range
                const maxSurfaceDensity = Math.max(...surfaceDensity.flat());
                const normalizedSurface = surfaceDensity[j][i] / maxSurfaceDensity;
                
                // Combine: base blue (0) + surface atoms (yellow, 0.3-0.6) + energy sites (orange-red, 0.6-1.0)
                let value = 0.1; // Base blue level
                
                // Add surface contribution (creates yellow regions)
                if (normalizedSurface > 0.05) {
                    value = Math.max(value, 0.3 + normalizedSurface * 0.3); // 0.3-0.6 range (blue to yellow)
                }
                
                // Add energy contribution (creates orange-red regions)
                if (energyField[j][i] > 0.05) {
                    value = Math.max(value, 0.6 + energyField[j][i] * 0.4); // 0.6-1.0 range (orange to red)
                }
                
                combinedField[j][i] = value;
                maxCombined = Math.max(maxCombined, value);
            }
        }

        // Apply multi-pass Gaussian blur for ultra-smooth appearance
        const smoothedField = multiPassGaussianBlur(combinedField, 2.0, 3);

        // Create Plotly heatmap with continuous density field
        const heatmapTrace = {
            type: 'heatmap',
            x: x,
            y: y,
            z: smoothedField,
            colorscale: [
                [0, '#0f172a'],      // Very dark blue (基底深海)
                [0.1, '#1e3a8a'],    // Deep blue (基底)
                [0.15, '#2563eb'],   // Blue 
                [0.2, '#3b82f6'],    // Bright blue
                [0.25, '#60a5fa'],   // Light blue
                [0.3, '#93c5fd'],    // Very light blue (过渡)
                [0.35, '#c7d2fe'],   // Lavender blue
                [0.4, '#e0e7ff'],    // Almost white blue
                [0.45, '#fef3c7'],   // Very light yellow
                [0.5, '#fde047'],    // Yellow (表面原子区域)
                [0.55, '#facc15'],   // Golden yellow
                [0.6, '#f59e0b'],    // Orange (过渡到吸附区)
                [0.65, '#ea580c'],   // Dark orange
                [0.7, '#dc2626'],    // Red (吸附位点)
                [0.8, '#b91c1c'],    // Dark red
                [0.9, '#7f1d1d'],    // Very dark red
                [1, '#450a0a']       // Almost black red (最高活性)
            ],
            showscale: true,
            colorbar: {
                title: {
                    text: '活性密度场',
                    side: 'right',
                    font: { size: 14, color: '#1f2937' }
                },
                thickness: 25,
                len: 0.8,
                tickvals: [0.1, 0.3, 0.5, 0.7, 0.9],
                ticktext: ['基底', '表面', '原子区', '吸附区', '高活性'],
                tickfont: { size: 11, color: '#1f2937' },
                bordercolor: '#374151',
                borderwidth: 1
            },
            hoverongaps: false,
            hovertemplate: `${axisNames[plotAxes.x]}: %{x:.2f} Å<br>${axisNames[plotAxes.y]}: %{y:.2f} Å<br>活性密度: %{z:.3f}<extra></extra>`,
            zauto: false,
            zmin: 0,
            zmax: 1
        };

        const data = [heatmapTrace];

        const layout = {
            title: {
                text: '表面活性密度场分布图',
                font: { size: 18, color: '#1f2937', family: 'Arial, sans-serif' },
                x: 0.5
            },
            xaxis: {
                title: {
                    text: `${axisNames[plotAxes.x]} (Å)`,
                    font: { size: 14, color: '#374151' }
                },
                showgrid: false,
                tickfont: { size: 12, color: '#374151' },
                linecolor: '#6b7280',
                linewidth: 1
            },
            yaxis: {
                title: {
                    text: `${axisNames[plotAxes.y]} (Å)`,
                    font: { size: 14, color: '#374151' }
                },
                showgrid: false,
                tickfont: { size: 12, color: '#374151' },
                linecolor: '#6b7280',
                linewidth: 1
            },
            plot_bgcolor: '#f9fafb',
            paper_bgcolor: '#ffffff',
            margin: { l: 80, r: 120, t: 80, b: 70 },
            annotations: [
                {
                    text: '蓝色: 基底区域 | 黄色: 表面原子晕染 | 橙红: 吸附位点能量场',
                    x: 0.5,
                    y: -0.15,
                    xref: 'paper',
                    yref: 'paper',
                    showarrow: false,
                    font: { size: 12, color: '#6b7280' },
                    xanchor: 'center'
                }
            ]
        };

        const config = {
            responsive: true,
            displayModeBar: true,
            modeBarButtonsToRemove: ['pan2d', 'select2d', 'lasso2d', 'autoScale2d'],
            displaylogo: false
        };

        Plotly.newPlot(heatmapContainer, data, layout, config);
    }

    // Multi-pass Gaussian blur for ultra-smooth appearance
    function multiPassGaussianBlur(matrix, sigma, passes) {
        let result = matrix.map(row => [...row]); // Deep copy
        
        for (let pass = 0; pass < passes; pass++) {
            result = gaussianBlur(result, sigma / passes);
        }
        
        return result;
    }

    // Enhanced Gaussian blur function
    function gaussianBlur(matrix, sigma) {
        const rows = matrix.length;
        const cols = matrix[0].length;
        
        // Create Gaussian kernel
        const kernelSize = Math.ceil(sigma * 4) * 2 + 1; // Larger kernel for smoother blur
        const kernel = [];
        const center = Math.floor(kernelSize / 2);
        let sum = 0;
        
        for (let i = 0; i < kernelSize; i++) {
            const x = i - center;
            const weight = Math.exp(-(x * x) / (2 * sigma * sigma));
            kernel[i] = weight;
            sum += weight;
        }
        
        // Normalize kernel
        for (let i = 0; i < kernelSize; i++) {
            kernel[i] /= sum;
        }
        
        // Apply horizontal blur
        const temp = Array(rows).fill().map(() => Array(cols).fill(0));
        for (let i = 0; i < rows; i++) {
            for (let j = 0; j < cols; j++) {
                let value = 0;
                for (let k = 0; k < kernelSize; k++) {
                    const col = j + k - center;
                    if (col >= 0 && col < cols) {
                        value += matrix[i][col] * kernel[k];
                    } else {
                        // Handle edge cases by extending boundary
                        const boundaryCol = Math.max(0, Math.min(cols - 1, col));
                        value += matrix[i][boundaryCol] * kernel[k];
                    }
                }
                temp[i][j] = value;
            }
        }
        
        // Apply vertical blur
        const result = Array(rows).fill().map(() => Array(cols).fill(0));
        for (let i = 0; i < rows; i++) {
            for (let j = 0; j < cols; j++) {
                let value = 0;
                for (let k = 0; k < kernelSize; k++) {
                    const row = i + k - center;
                    if (row >= 0 && row < rows) {
                        value += temp[row][j] * kernel[k];
                    } else {
                        // Handle edge cases by extending boundary
                        const boundaryRow = Math.max(0, Math.min(rows - 1, row));
                        value += temp[boundaryRow][j] * kernel[k];
                    }
                }
                result[i][j] = value;
            }
        }
        
        return result;
    }

    // Plotly.js 3D Surface Visualization (unchanged)
    function render3DSurfacePlotly(surfaceData, adsorptionData, surfaceAxis) {
        if (typeof Plotly === 'undefined') {
            surface3dContainer.innerHTML = '<p style="text-align: center; color: red;">Plotly.js library is not loaded. Cannot render 3D Surface.</p>';
            return;
        }

        // Clear previous content
        surface3dContainer.innerHTML = '';

        // Determine axes based on surface normal
        const axisNames = ['X', 'Y', 'Z'];
        const normalAxis = parseInt(surfaceAxis, 10);
        let plotAxes;
        if (normalAxis === 0) { plotAxes = { x: 1, y: 2, z: 0 }; } 
        else if (normalAxis === 1) { plotAxes = { x: 0, y: 2, z: 1 }; } 
        else { plotAxes = { x: 0, y: 1, z: 2 }; }

        // Extract coordinates
        const surfaceCoords = surfaceData.coords;
        const adsorptionSites = adsorptionData.sites;

        // Create triangulated surface mesh using Delaunay triangulation
        const surfacePoints = surfaceCoords.map(coord => ({
            x: coord[plotAxes.x],
            y: coord[plotAxes.y], 
            z: coord[plotAxes.z]
        }));

        // Simple 2D triangulation for surface mesh
        const triangles = simpleDelaunayTriangulation(surfacePoints);

        // Create surface mesh trace
        const meshTrace = {
            type: 'mesh3d',
            x: surfacePoints.map(p => p.x),
            y: surfacePoints.map(p => p.y),
            z: surfacePoints.map(p => p.z),
            i: triangles.map(t => t[0]),
            j: triangles.map(t => t[1]),
            k: triangles.map(t => t[2]),
            opacity: 0.6,
            color: 'lightgray',
            name: 'Surface Mesh',
            showlegend: true
        };

        // Create surface atoms trace (red spheres)
        const atomTrace = {
            type: 'scatter3d',
            mode: 'markers',
            x: surfacePoints.map(p => p.x),
            y: surfacePoints.map(p => p.y),
            z: surfacePoints.map(p => p.z),
            marker: {
                size: 8,
                color: 'red',
                symbol: 'circle',
                opacity: 0.8
            },
            name: 'Surface Atoms',
            showlegend: true
        };

        // Create adsorption sites trace (colored by energy)
        const energyValues = adsorptionSites.map(site => site.energy);
        const minEnergy = Math.min(...energyValues);
        const maxEnergy = Math.max(...energyValues);

        const adsorptionTrace = {
            type: 'scatter3d',
            mode: 'markers',
            x: adsorptionSites.map(site => site.coords[plotAxes.x]),
            y: adsorptionSites.map(site => site.coords[plotAxes.y]),
            z: adsorptionSites.map(site => site.coords[plotAxes.z]),
            marker: {
                size: 12,
                color: energyValues,
                colorscale: [
                    [0, '#006837'],
                    [0.1, '#1a9850'],
                    [0.2, '#66bd63'],
                    [0.3, '#a6d96a'],
                    [0.4, '#d9ef8b'],
                    [0.5, '#fee08b'],
                    [0.6, '#fdae61'],
                    [0.7, '#f46d43'],
                    [0.8, '#d73027'],
                    [1, '#a50026']
                ],
                symbol: 'diamond',
                opacity: 0.9,
                colorbar: {
                    title: 'Energy (eV)',
                    titleside: 'right',
                    thickness: 20
                },
                cmin: minEnergy,
                cmax: maxEnergy
            },
            name: 'Adsorption Sites',
            showlegend: true,
            text: adsorptionSites.map(site => `Energy: ${site.energy.toFixed(4)} eV`),
            hovertemplate: '%{text}<br>X: %{x:.2f}<br>Y: %{y:.2f}<br>Z: %{z:.2f}<extra></extra>'
        };

        const data = [meshTrace, atomTrace, adsorptionTrace];

        const layout = {
            title: `Adsorption Sites on ${axisNames[normalAxis]}-Normal Surface`,
            scene: {
                xaxis: { title: `${axisNames[plotAxes.x]} (Å)` },
                yaxis: { title: `${axisNames[plotAxes.y]} (Å)` },
                zaxis: { title: `${axisNames[plotAxes.z]} (Å)` },
                camera: {
                    eye: { x: 1.5, y: 1.5, z: 1.5 }
                },
                aspectmode: 'cube'
            },
            margin: { l: 0, r: 0, b: 0, t: 50 },
            legend: {
                x: 0.02,
                y: 0.98,
                bgcolor: 'rgba(255,255,255,0.8)'
            }
        };

        const config = {
            responsive: true,
            displayModeBar: true,
            modeBarButtonsToRemove: ['pan2d', 'select2d', 'lasso2d'],
            displaylogo: false
        };

        Plotly.newPlot(surface3dContainer, data, layout, config);
    }

    // Simple Delaunay triangulation for surface mesh
    function simpleDelaunayTriangulation(points) {
        const triangles = [];
        
        // Simple triangulation - connect nearest neighbors
        for (let i = 0; i < points.length - 2; i++) {
            for (let j = i + 1; j < points.length - 1; j++) {
                for (let k = j + 1; k < points.length; k++) {
                    const p1 = points[i];
                    const p2 = points[j];
                    const p3 = points[k];
                    
                    // Check if triangle is valid (not too large)
                    const d12 = Math.sqrt((p1.x-p2.x)**2 + (p1.y-p2.y)**2);
                    const d23 = Math.sqrt((p2.x-p3.x)**2 + (p2.y-p3.y)**2);
                    const d31 = Math.sqrt((p3.x-p1.x)**2 + (p3.y-p1.y)**2);
                    
                    if (d12 < 3 && d23 < 3 && d31 < 3) { // Adjust threshold as needed
                        triangles.push([i, j, k]);
                    }
                }
            }
        }
        
        return triangles;
    }
    
    function resetDashboard() {
        dashboard.classList.add('hidden');
        chartControls.classList.add('hidden');
        heatmapContainer.classList.add('hidden');
        surface3dContainer.classList.add('hidden');
        downloadLinkContainer.classList.add('hidden');
        if (heatmapChart) { 
            heatmapChart.dispose(); 
            heatmapChart = null; 
        }
        if (activeEventSource) { 
            activeEventSource.close(); 
            activeEventSource = null; 
        }
        activeSessionId = null;
        loadedVizData = null;
    }

    function refreshHistory() {
        fetch('/get-results')
            .then(res => res.json())
            .then(data => {
                historyTableBody.innerHTML = '';
                data.forEach(result => {
                    const row = historyTableBody.insertRow();
                    row.innerHTML = `
                        <td>${result.timestamp}</td>
                        <td>${result.filename}</td>
                        <td>
                            <a href="/download-result/${result.session_id}" class="button-small" download>Download</a>
                            <button class="button-small" data-session="${result.session_id}" data-axis="${result.surface_axis || '2'}">View</button>
                        </td>`;
                });
            });
        }
});