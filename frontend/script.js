// frontend/script.js
const video = document.getElementById('webcamVideo');
const canvas = document.getElementById('processingCanvas');
const context = canvas.getContext('2d');
const alertOverlay = document.getElementById('alertOverlay');
const statusMessage = document.getElementById('statusMessage');
const buzzerSound = document.getElementById('buzzerSound');

// Connect to the Flask-SocketIO backend
const socket = io('http://localhost:5000'); // Ensure this matches your backend's host and port

let alertActiveTimer = null; // To manage the 10-second alert visibility

// --- Socket.IO Event Handlers ---
socket.on('connect', () => {
    statusMessage.textContent = 'Connected to backend. Waiting for webcam.';
    console.log('Connected to backend server');
});

socket.on('disconnect', () => {
    statusMessage.textContent = 'Disconnected from backend.';
    console.log('Disconnected from backend server');
    if (alertActiveTimer) {
        clearTimeout(alertActiveTimer);
        alertOverlay.style.display = 'none';
    }
});

socket.on('alert', (data) => {
    console.log('Alert signal received:', data.action, data.reason || '');

    if (data.action === 'show_alert') {
        statusMessage.textContent = data.reason ? `Distracted: ${data.reason.trim()}` : "Distracted!";
        if (!alertActiveTimer) { // Only show/start timer if not already active
            alertOverlay.style.display = 'flex';
            playBuzzer();
            alertActiveTimer = setTimeout(() => {
                alertOverlay.style.display = 'none';
                stopBuzzer();
                alertActiveTimer = null;
            }, 10000); // Hide after 10 seconds
        }
    } else if (data.action === 'hide_alert') {
        statusMessage.textContent = 'Focused.';
        if (alertActiveTimer) {
            clearTimeout(alertActiveTimer);
            alertOverlay.style.display = 'none';
            stopBuzzer();
            alertActiveTimer = null;
        }
    } else if (data.action === 'maintain_alert') {
         // Backend tells us to keep alert, so we just make sure it's visible.
         // The JS side manages the 10-sec timeout locally too, but backend signal reinforces.
         alertOverlay.style.display = 'flex';
    } else if (data.action === 'focused') {
        // No alert to show, but confirm focused status if alert isn't currently displayed
        if (!alertActiveTimer && alertOverlay.style.display !== 'flex') {
            statusMessage.textContent = 'Focused.';
        }
    }
});

// --- Webcam and Frame Sending Logic ---
async function startWebcam() {
    try {
        const stream = await navigator.mediaDevices.getUserMedia({ video: true });
        video.srcObject = stream;
        video.play();
        statusMessage.textContent = 'Webcam streaming...';
    } catch (err) {
        statusMessage.textContent = `Error accessing webcam: ${err.message}`;
        console.error("Error accessing webcam: ", err);
    }
}

video.addEventListener('loadeddata', () => {
    canvas.width = video.videoWidth;
    canvas.height = video.videoHeight;
    
    // Start sending frames after video data is loaded
    setInterval(() => {
        if (socket.connected) {
            context.drawImage(video, 0, 0, canvas.width, canvas.height);
            // Capture image as JPEG. Quality can be adjusted (e.g., 0.7 for 70%)
            // Sending too high quality/framerate can overwhelm the server or network.
            const imageData = canvas.toDataURL('image/jpeg', 0.8);
            socket.emit('video_frame', imageData);
        }
    }, 100); // Send frames approximately every 100ms (10 FPS)
});

// --- Buzzer Sound Control ---
function playBuzzer() {
    if (buzzerSound) {
        buzzerSound.currentTime = 0; // Rewind to start if already playing
        buzzerSound.play().catch(e => console.error("Error playing buzzer sound:", e));
    }
}

function stopBuzzer() {
    if (buzzerSound) {
        buzzerSound.pause();
        buzzerSound.currentTime = 0;
    }
}

// Initialize webcam on page load
window.addEventListener('load', startWebcam);