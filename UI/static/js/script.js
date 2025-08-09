// DOM Elements
const inputText = document.getElementById('inputText');
const outputText = document.getElementById('outputText');
const translateBtn = document.getElementById('translateBtn');
const clearBtn = document.getElementById('clearBtn');
const copyBtn = document.getElementById('copyBtn');
const speakBtn = document.getElementById('speakBtn');
const charCount = document.getElementById('charCount');
const exampleBtns = document.querySelectorAll('.example-btn');
const loadingModal = document.getElementById('loadingModal');
const errorToast = document.getElementById('errorToast');
const errorMessage = document.getElementById('errorMessage');
const successToast = document.getElementById('successToast');
const aboutModal = document.getElementById('aboutModal');
const aboutLink = document.querySelector('.footer-link:nth-child(2)');
const closeAboutModal = document.getElementById('closeAboutModal');

// Speech recording elements
const recordBtn = document.getElementById('recordBtn');
const recordingStatus = document.getElementById('recordingStatus');
const stopRecordBtn = document.getElementById('stopRecordBtn');
const speechEditor = document.getElementById('speechEditor');
const speechText = document.getElementById('speechText');
const acceptSpeechBtn = document.getElementById('acceptSpeechBtn');
const cancelSpeechBtn = document.getElementById('cancelSpeechBtn');

// Audio recording variables
let mediaRecorder = null;
let audioChunks = [];
let isRecording = false;

// API Configuration
const API_URL = window.location.origin;

// Event Listeners
translateBtn.addEventListener('click', translateText);
clearBtn.addEventListener('click', clearText);
copyBtn.addEventListener('click', copyTranslation);
speakBtn.addEventListener('click', speakTranslation);
inputText.addEventListener('input', updateCharCount);
inputText.addEventListener('keydown', handleKeyPress);
aboutLink.addEventListener('click', () => {
    aboutModal.classList.add('show');
});
closeAboutModal.addEventListener('click', () => {
    aboutModal.classList.remove('show');
});

// Speech recording event listeners
recordBtn.addEventListener('click', toggleRecording);
stopRecordBtn.addEventListener('click', stopRecording);
acceptSpeechBtn.addEventListener('click', acceptSpeechTranscription);
cancelSpeechBtn.addEventListener('click', cancelSpeechTranscription);

// Add event listeners to example buttons
exampleBtns.forEach(btn => {
    btn.addEventListener('click', () => {
        inputText.value = btn.dataset.text;
        updateCharCount();
        translateText();
    });
});

// Update character count
function updateCharCount () {
    const count = inputText.value.length;
    charCount.textContent = count;
}

// Handle Enter key press
function handleKeyPress (e) {
    if (e.key === 'Enter' && e.ctrlKey) {
        translateText();
    }
}

// Clear input text
function clearText () {
    inputText.value = '';
    outputText.innerHTML = '<p class="placeholder-text">Translation will appear here...</p>';
    updateCharCount();
}

// Show loading modal
function showLoading () {
    loadingModal.classList.add('show');
    translateBtn.disabled = true;
}

// Hide loading modal
function hideLoading () {
    loadingModal.classList.remove('show');
    translateBtn.disabled = false;
}

// Show error toast
function showError (message) {
    errorMessage.textContent = message;
    errorToast.classList.add('show');
    setTimeout(() => {
        errorToast.classList.remove('show');
    }, 5000);
}

// Show success toast
function showSuccess () {
    successToast.classList.add('show');
    setTimeout(() => {
        successToast.classList.remove('show');
    }, 3000);
}

// Translate text
async function translateText () {
    const text = inputText.value.trim();

    if (!text) {
        showError('Please enter some text to translate');
        return;
    }

    showLoading();

    try {
        const response = await fetch(`${ API_URL }/translate`, {
            method: 'POST',
            headers: {
                'Content-Type': 'application/json',
            },
            body: JSON.stringify({ text }),
        });

        const data = await response.json();

        if (response.ok && data.success) {
            outputText.innerHTML = `<p>${ escapeHtml(data.translation) }</p>`;
        } else {
            showError(data.error || 'Translation failed');
        }
    } catch (error) {
        console.error('Translation error:', error);
        showError('Network error. Please check your connection and try again.');
    } finally {
        hideLoading();
    }
}

// Copy translation to clipboard
async function copyTranslation () {
    const translationElement = outputText.querySelector('p');

    if (!translationElement || translationElement.classList.contains('placeholder-text')) {
        showError('No translation to copy');
        return;
    }

    const translationText = translationElement.textContent;

    try {
        await navigator.clipboard.writeText(translationText);
        showSuccess();
    } catch (error) {
        // Fallback for older browsers
        const textarea = document.createElement('textarea');
        textarea.value = translationText;
        textarea.style.position = 'fixed';
        textarea.style.opacity = '0';
        document.body.appendChild(textarea);
        textarea.select();

        try {
            document.execCommand('copy');
            showSuccess();
        } catch (err) {
            showError('Failed to copy translation');
        }

        document.body.removeChild(textarea);
    }
}

// Escape HTML to prevent XSS
function escapeHtml (text) {
    const map = {
        '&': '&amp;',
        '<': '&lt;',
        '>': '&gt;',
        '"': '&quot;',
        "'": '&#39;'
    };
    return text.replace(/[&<>"']/g, m => map[m]);
}

// Check server health on load
async function checkServerHealth () {
    try {
        const response = await fetch(`${ API_URL }/health`);
        const data = await response.json();

        if (!data.model_loaded) {
            showError('Model is still loading. Please wait a moment and refresh the page.');
            translateBtn.disabled = true;
        }
    } catch (error) {
        console.error('Health check failed:', error);
        showError('Cannot connect to translation server. Please ensure the server is running.');
        translateBtn.disabled = true;
    }
}

// Audio Recording Functions
async function initializeAudioRecording () {
    try {
        const stream = await navigator.mediaDevices.getUserMedia({
            audio: {
                echoCancellation: true,
                noiseSuppression: true,
                sampleRate: 44100
            }
        });

        mediaRecorder = new MediaRecorder(stream, {
            mimeType: 'audio/webm;codecs=opus'
        });

        mediaRecorder.ondataavailable = (event) => {
            if (event.data.size > 0) {
                audioChunks.push(event.data);
            }
        };

        mediaRecorder.onstop = async () => {
            const audioBlob = new Blob(audioChunks, { type: 'audio/webm' });
            audioChunks = [];
            await transcribeAudio(audioBlob);

            // Stop all tracks to release microphone
            stream.getTracks().forEach(track => track.stop());
        };

        return true;
    } catch (error) {
        console.error('Error accessing microphone:', error);
        showError('Microphone access denied or not available. Please check your browser settings.');
        return false;
    }
}

async function toggleRecording () {
    if (isRecording) {
        stopRecording();
    } else {
        await startRecording();
    }
}

async function startRecording () {
    const initialized = await initializeAudioRecording();
    if (!initialized) return;

    try {
        audioChunks = [];
        mediaRecorder.start();
        isRecording = true;

        // Update UI
        recordBtn.style.display = 'none';
        recordingStatus.style.display = 'flex';
        recordBtn.classList.add('recording');

        // Hide speech editor if visible
        speechEditor.style.display = 'none';

    } catch (error) {
        console.error('Error starting recording:', error);
        showError('Failed to start recording. Please try again.');
        isRecording = false;
    }
}

function stopRecording () {
    if (mediaRecorder && isRecording) {
        mediaRecorder.stop();
        isRecording = false;

        // Update UI
        recordBtn.style.display = 'block';
        recordingStatus.style.display = 'none';
        recordBtn.classList.remove('recording');
    }
}

async function transcribeAudio (audioBlob) {
    showLoading();

    try {
        const formData = new FormData();
        formData.append('audio', audioBlob, 'recording.webm');

        const response = await fetch(`${ API_URL }/speech-to-text`, {
            method: 'POST',
            body: formData
        });

        const data = await response.json();

        if (response.ok && data.success) {
            // Show speech editor with transcribed text
            speechText.value = data.text;
            speechEditor.style.display = 'block';
            speechText.focus();
        } else {
            showError(data.error || 'Speech transcription failed');
        }
    } catch (error) {
        console.error('Transcription error:', error);
        showError('Network error during transcription. Please check your connection and try again.');
    } finally {
        hideLoading();
    }
}

function acceptSpeechTranscription () {
    const transcribedText = speechText.value.trim();
    if (transcribedText) {
        // Add to existing text or replace if empty
        const currentText = inputText.value.trim();
        if (currentText) {
            inputText.value = currentText + ' ' + transcribedText;
        } else {
            inputText.value = transcribedText;
        }
        updateCharCount();
    }

    // Hide speech editor
    speechEditor.style.display = 'none';
    speechText.value = '';

    // Focus back on main input
    inputText.focus();
}

function cancelSpeechTranscription () {
    speechEditor.style.display = 'none';
    speechText.value = '';
}

// Speak translation using Google Text-to-Speech
async function speakTranslation () {
    const translationElement = outputText.querySelector('p');

    if (!translationElement || translationElement.classList.contains('placeholder-text')) {
        showError('No translation to speak');
        return;
    }

    const translationText = translationElement.textContent;
    speakBtn.disabled = true;
    speakBtn.innerHTML = '<i class="fas fa-spinner fa-spin"></i>';

    try {
        // Call the text-to-speech API
        const response = await fetch(`${ API_URL }/text-to-speech`, {
            method: 'POST',
            headers: {
                'Content-Type': 'application/json',
            },
            body: JSON.stringify({
                text: translationText,
                language: 'tw'  // Specify Twi language
            }),
        });

        if (!response.ok) {
            const errorData = await response.json();
            throw new Error(errorData.error || 'Text-to-speech failed');
        }

        // Get the audio blob from the response
        const audioBlob = await response.blob();

        // Create an audio element and play the speech
        const audioUrl = URL.createObjectURL(audioBlob);
        const audio = new Audio(audioUrl);

        audio.onended = () => {
            // Clean up the URL object after playback
            URL.revokeObjectURL(audioUrl);
        };

        await audio.play();
    } catch (error) {
        console.error('Text-to-speech error:', error);
        showError(error.message || 'Failed to generate speech');
    } finally {
        speakBtn.disabled = false;
        speakBtn.innerHTML = '<i class="fas fa-volume-up"></i>';
    }
}

// Initialize on page load
document.addEventListener('DOMContentLoaded', () => {
    updateCharCount();
    checkServerHealth();

    // Set focus on input
    inputText.focus();
});