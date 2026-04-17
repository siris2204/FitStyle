// API Base URL
const API_BASE = window.location.origin;

// DOM Elements
const tabBtns = document.querySelectorAll('.tab-btn');
const tabContents = document.querySelectorAll('.tab-content');
const dropZone = document.getElementById('dropZone');
const imageInput = document.getElementById('imageInput');
const previewSection = document.getElementById('previewSection');
const previewImage = document.getElementById('previewImage');
const annotatedImage = document.getElementById('annotatedImage');
const annotatedBox = document.getElementById('annotatedBox');
const analyzeBtn = document.getElementById('analyzeBtn');
const manualRecommendBtn = document.getElementById('manualRecommendBtn');
const loadingOverlay = document.getElementById('loadingOverlay');
const resultsSection = document.getElementById('resultsSection');
const errorMessage = document.getElementById('errorMessage');
const errorText = document.getElementById('errorText');
const recommendationsGrid = document.getElementById('recommendationsGrid');

let selectedFile = null;

// Tab Switching
tabBtns.forEach(btn => {
    btn.addEventListener('click', () => {
        const tabId = btn.dataset.tab;
        
        tabBtns.forEach(b => b.classList.remove('active'));
        tabContents.forEach(c => c.classList.remove('active'));
        
        btn.classList.add('active');
        document.getElementById(`${tabId}-tab`).classList.add('active');
        
        hideError();
        resultsSection.classList.add('hidden');
    });
});

// Drag and Drop
dropZone.addEventListener('dragover', (e) => {
    e.preventDefault();
    dropZone.classList.add('dragover');
});

dropZone.addEventListener('dragleave', () => {
    dropZone.classList.remove('dragover');
});

dropZone.addEventListener('drop', (e) => {
    e.preventDefault();
    dropZone.classList.remove('dragover');
    
    const files = e.dataTransfer.files;
    if (files.length > 0) {
        handleFileSelect(files[0]);
    }
});

// File Input Change
imageInput.addEventListener('change', (e) => {
    if (e.target.files.length > 0) {
        handleFileSelect(e.target.files[0]);
    }
});

// Handle File Selection
function handleFileSelect(file) {
    if (!file.type.startsWith('image/')) {
        showError('Please select an image file (JPG, PNG)');
        return;
    }
    
    selectedFile = file;
    
    // Show preview
    const reader = new FileReader();
    reader.onload = (e) => {
        previewImage.src = e.target.result;
        previewSection.classList.remove('hidden');
        annotatedBox.classList.add('hidden');
    };
    reader.readAsDataURL(file);
    
    hideError();
    resultsSection.classList.add('hidden');
}

// Analyze Button Click
analyzeBtn.addEventListener('click', async () => {
    if (!selectedFile) {
        showError('Please select an image first');
        return;
    }
    
    showLoading();
    hideError();
    
    try {
        const formData = new FormData();
        formData.append('file', selectedFile);
        
        const numRecs = document.getElementById('numRecommendations').value || 5;
        
        const response = await fetch(`${API_BASE}/api/recommend-from-image?k=${numRecs}`, {
            method: 'POST',
            body: formData
        });
        
        const data = await response.json();
        
        if (!response.ok || !data.success) {
            throw new Error(data.message || data.detail || 'Failed to analyze image');
        }
        
        // Show annotated image
        if (data.annotated_image) {
            annotatedImage.src = `data:image/jpeg;base64,${data.annotated_image}`;
            annotatedBox.classList.remove('hidden');
        }
        
        // Display results
        displayResults(data.measurements, data.recommendations);
        
    } catch (error) {
        showError(error.message);
    } finally {
        hideLoading();
    }
});

// Manual Recommend Button Click
manualRecommendBtn.addEventListener('click', async () => {
    const bust = parseFloat(document.getElementById('bustInput').value);
    const waist = parseFloat(document.getElementById('waistInput').value);
    const hip = parseFloat(document.getElementById('hipInput').value);
    const k = parseInt(document.getElementById('numRecommendations').value) || 5;
    
    if (isNaN(bust) || isNaN(waist) || isNaN(hip)) {
        showError('Please enter all measurements');
        return;
    }
    
    showLoading();
    hideError();
    
    try {
        const response = await fetch(`${API_BASE}/api/recommend?k=${k}`, {
            method: 'POST',
            headers: {
                'Content-Type': 'application/json'
            },
            body: JSON.stringify({ bust, waist, hip })
        });
        
        const data = await response.json();
        
        if (!response.ok || !data.success) {
            throw new Error(data.detail || 'Failed to get recommendations');
        }
        
        displayResults(data.user_measurements, data.recommendations);
        
    } catch (error) {
        showError(error.message);
    } finally {
        hideLoading();
    }
});

// Display Results
function displayResults(measurements, recommendations) {
    // Update measurements display
    document.getElementById('resultBust').textContent = 
        measurements.bust ? measurements.bust.toFixed(1) : '-';
    document.getElementById('resultWaist').textContent = 
        measurements.waist ? measurements.waist.toFixed(1) : '-';
    document.getElementById('resultHip').textContent = 
        measurements.hip ? measurements.hip.toFixed(1) : '-';
    
    // Clear and populate recommendations
    recommendationsGrid.innerHTML = '';
    
    recommendations.forEach(rec => {
        const card = document.createElement('div');
        card.className = 'recommendation-card';
        
        const imgSrc = rec.image_base64 
            ? `data:image/jpeg;base64,${rec.image_base64}`
            : 'https://via.placeholder.com/200x250?text=No+Image';
        
        card.innerHTML = `
            <img src="${imgSrc}" alt="${rec.filename}">
            <div class="recommendation-info">
                <span class="recommendation-rank">#${rec.rank} Match</span>
                <div class="recommendation-stats">
                    <p><strong>Distance:</strong> ${rec.distance.toFixed(2)}</p>
                    <p><strong>Bust:</strong> ${rec.bust.toFixed(1)}</p>
                    <p><strong>Waist:</strong> ${rec.waist.toFixed(1)}</p>
                    <p><strong>Hip:</strong> ${rec.hip.toFixed(1)}</p>
                </div>
            </div>
        `;
        
        recommendationsGrid.appendChild(card);
    });
    
    resultsSection.classList.remove('hidden');
    
    // Scroll to results
    resultsSection.scrollIntoView({ behavior: 'smooth' });
}

// Utility Functions
function showLoading() {
    loadingOverlay.classList.remove('hidden');
}

function hideLoading() {
    loadingOverlay.classList.add('hidden');
}

function showError(message) {
    errorText.textContent = message;
    errorMessage.classList.remove('hidden');
}

function hideError() {
    errorMessage.classList.add('hidden');
}

// Initialize - Check API health
async function init() {
    try {
        const response = await fetch(`${API_BASE}/health`);
        if (!response.ok) {
            showError('API server is not responding. Please make sure the backend is running.');
        }
    } catch (error) {
        showError('Cannot connect to API server. Please start the backend server.');
    }
}

init();