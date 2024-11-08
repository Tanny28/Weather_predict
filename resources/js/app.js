import './bootstrap';
import 'animate.css';

window.predictWeather = async function() {
    const city = document.getElementById('city').value;
    const duration = document.getElementById('duration').value;
    const resultsDiv = document.getElementById('prediction-results');

    try {
        const response = await fetch('/predict', {
            method: 'POST',
            headers: {
                'Content-Type': 'application/json',
                'X-CSRF-TOKEN': document.querySelector('meta[name="csrf-token"]').content
            },
            body: JSON.stringify({ city, duration })
        });

        const data = await response.json();
        resultsDiv.innerHTML = `
            <div class="animate__animated animate__fadeIn">
                <h3 class="font-semibold mt-4">Predictions for ${city}</h3>
                <div class="mt-2">
                    ${renderPredictionGraphs(data.data)}
                </div>
            </div>
        `;
    } catch (error) {
        resultsDiv.innerHTML = '<div class="text-red-500">Failed to get prediction</div>';
    }
};

window.predictImage = async function() {
    const imageInput = document.getElementById('weather-image');
    const resultsDiv = document.getElementById('image-results');
    const formData = new FormData();
    formData.append('image', imageInput.files[0]);

    try {
        const response = await fetch('/predict-image', {
            method: 'POST',
            headers: {
                'X-CSRF-TOKEN': document.querySelector('meta[name="csrf-token"]').content
            },
            body: formData
        });

        const data = await response.json();
        resultsDiv.innerHTML = `
            <div class="animate__animated animate__fadeIn">
                <h3 class="font-semibold mt-4">Image Analysis Results</h3>
                <div class="mt-2">
                    ${renderImageAnalysis(data.data)}
                </div>
            </div>
        `;
    } catch (error) {
        resultsDiv.innerHTML = '<div class="text-red-500">Failed to analyze image</div>';
    }
};

window.sendMessage = async function() {
    const input = document.getElementById('chat-input');
    const messagesDiv = document.getElementById('chat-messages');
    const message = input.value;
    input.value = '';

    // Add user message
    messagesDiv.innerHTML += `
        <div class="mb-2 animate__animated animate__fadeIn">
            <span class="font-semibold">You:</span> ${message}
        </div>
    `;

    try {
        const response = await fetch('/chat', {
            method: 'POST',
            headers: {
                'Content-Type': 'application/json',
                'X-CSRF-TOKEN': document.querySelector('meta[name="csrf-token"]').content
            },
            body: JSON.stringify({ message })
        });

        const data = await response.json();
        
        // Add bot response
        messagesDiv.innerHTML += `
            <div class="mb-2 animate__animated animate__fadeIn">
                <span class="font-semibold">Assistant:</span> ${data.message}
            </div>
        `;
        
        messagesDiv.scrollTop = messagesDiv.scrollHeight;
    } catch (error) {
        messagesDiv.innerHTML += '<div class="text-red-500">Failed to get response</div>';
    }
};

function renderPredictionGraphs(data) {
    // Implement graph rendering using Chart.js or similar
    return `<div class="h-64 bg-gray-100 rounded flex items-center justify-center">
        Graph visualization will be implemented here
    </div>`;
}

function renderImageAnalysis(data) {
    return `<div class="space-y-2">
        <p>Weather Classification: ${data.classification}</p>
        <p>Confidence: ${data.confidence}%</p>
        ${renderPredictionGraphs(data.predictions)}
    </div>`;
}