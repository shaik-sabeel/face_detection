// frontend/script.js
const video = document.getElementById('webcamVideo');
const canvas = document.getElementById('processingCanvas');
const context = canvas.getContext('2d');
const alertOverlay = document.getElementById('alertOverlay');
const statusMessage = document.getElementById('statusMessage');
const buzzerSound = document.getElementById('buzzerSound');

// Connect to the Flask-SocketIO backend
const socket = io('http://localhost:5000'); // Ensure this matches your backend's host and port

let currentAlertState = "hidden"; // 'hidden', 'showing'

// --- Socket.IO Event Handlers ---
socket.on('connect', () => {
    statusMessage.textContent = 'Connected to backend. Waiting for webcam.';
    console.log('Connected to backend server');
});

socket.on('disconnect', () => {
    statusMessage.textContent = 'Disconnected from backend.';
    console.log('Disconnected from backend server');
    if (currentAlertState === 'showing') {
        hideAlert(); // Hide alert if disconnected
    }
});

socket.on('alert', (data) => {
    // console.log('Alert signal received from backend:', data.action, data.reason || '');

    if (data.action === 'show_alert') {
        statusMessage.textContent = data.reason ? `DISTRACTED: ${data.reason}` : "DISTRACTED!";
        if (currentAlertState !== 'showing') { // Prevent re-triggering if already showing
            showAlert();
        }
    } else if (data.action === 'hide_alert') {
        statusMessage.textContent = 'Focused.';
        hideAlert();
    } else if (data.action === 'maintain_alert') {
        statusMessage.textContent = data.reason ? `DISTRACTED: ${data.reason}` : "DISTRACTED!";
        // Ensure alert is visible if backend is maintaining it
        if (currentAlertState !== 'showing') {
            showAlert(); // Show if for some reason it got hidden
        }
    } else if (data.action === 'focused') {
        // Backend confirms focus and no alert is needed
        if (currentAlertState !== 'hidden') {
            hideAlert(); // Hide it explicitly if it was visible
        }
        statusMessage.textContent = 'Focused.';
    }
});

// --- Alert Control Functions ---
function showAlert() {
    alertOverlay.style.display = 'flex';
    playBuzzer();
    currentAlertState = 'showing';
}

function hideAlert() {
    alertOverlay.style.display = 'none';
    stopBuzzer();
    currentAlertState = 'hidden';
}

// --- Webcam and Frame Sending Logic ---
async function startWebcam() {
    try {
        const stream = await navigator.mediaDevices.getUserMedia({ video: true });
        video.srcObject = stream;
        video.play();
        statusMessage.textContent = 'Webcam streaming. Analyzing focus...';
    } catch (err) {
        statusMessage.textContent = `Error accessing webcam: ${err.message}. Please allow webcam access.`;
        console.error("Error accessing webcam: ", err);
    }
}

video.addEventListener('loadeddata', () => {
    canvas.width = video.videoWidth;
    canvas.height = video.videoHeight;
    
    // Send frames more frequently for better responsiveness (20 FPS)
    // Be mindful of server load; adjust this interval based on performance
    setInterval(() => {
        if (socket.connected) {
            context.drawImage(video, 0, 0, canvas.width, canvas.height);
            // Capture image as JPEG. Quality (0-1) can be adjusted to balance quality and data size.
            const imageData = canvas.toDataURL('image/jpeg', 0.8); // 80% quality
            socket.emit('video_frame', imageData);
        }
    }, 50); // Send frames approximately every 50ms (20 FPS)
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