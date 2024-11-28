// /raid/vladimir_albrekht/web_demo/combined/ver6_asr_tts_rag/static/speech_utils.js

let isRecording = false;
let isTTSEnabled = false;
let isPlaying = false;
let mediaRecorder;
let audioChunks = [];
let handleTranscribedTextCallback;
let audioQueue = [];
let audioBuffer = [];
const MAX_BUFFER_SIZE = 3;  // Максимальное количество предзагруженных аудио

const supportedLanguages = ['kk', 'en', 'ru'];

// Update the setLanguage function if it exists
async function setLanguage(lang) {
    if (supportedLanguages.includes(lang)) {
        currentLanguage = lang;
        // Any other language-specific setup can be done here
    } else {
        console.error(`Unsupported language: ${lang}`);
    }
}

function initializeSpeechUtils(recordButtonId, toggleTTSButtonId, handleTranscribedTextFunc) {
    const recordButton = document.getElementById(recordButtonId);
    const toggleTTSButton = document.getElementById(toggleTTSButtonId);

    recordButton.addEventListener('click', toggleRecording);
    toggleTTSButton.addEventListener('click', toggleTTS);
    handleTranscribedTextCallback = handleTranscribedTextFunc;
}

function toggleRecording() {
    if (isRecording) {
        stopRecording();
    } else {
        startRecording();
    }
}

async function startRecording() {
    try {
        const stream = await navigator.mediaDevices.getUserMedia({ audio: true });
        mediaRecorder = new MediaRecorder(stream, { mimeType: 'audio/webm' });
        mediaRecorder.ondataavailable = event => {
            audioChunks.push(event.data);
        };
        mediaRecorder.onstop = convertAndSendAudio;
        audioChunks = [];
        mediaRecorder.start();
        isRecording = true;
        document.getElementById('record-button').innerHTML = '<i class="fas fa-stop"></i>';
        document.getElementById('record-button').classList.add('recording');
    } catch (err) {
        console.error("Error accessing microphone:", err);
    }
}

function stopRecording() {
    if (mediaRecorder && mediaRecorder.state !== 'inactive') {
        mediaRecorder.stop();
        isRecording = false;
        document.getElementById('record-button').innerHTML = '<i class="fas fa-microphone"></i>';
        document.getElementById('record-button').classList.remove('recording');
    }
}

function convertAndSendAudio() {
    const audioBlob = new Blob(audioChunks, { type: 'audio/webm' });
    
    convertToWav(audioBlob).then(wavBlob => {
        const formData = new FormData();
        formData.append('audio', wavBlob, 'recording.wav');

        fetch('/asr', {
            method: 'POST',
            body: formData
        })
        .then(response => response.json())
        .then(data => {
            if (data.transcription) {
                console.log("Транскрибированный текст:", data.transcription);
                if (handleTranscribedTextCallback) {
                    handleTranscribedTextCallback(data.transcription);
                }
            } else {
                console.error('ASR error:', data.error);
            }
        })
        .catch(error => {
            console.error('Error sending audio to ASR:', error);
        });
    });
}

function convertToWav(blob) {
    return new Promise((resolve, reject) => {
        const reader = new FileReader();
        reader.onload = function(event) {
            const audioContext = new (window.AudioContext || window.webkitAudioContext)();
            audioContext.decodeAudioData(event.target.result).then(function(buffer) {
                const wavBuffer = audioBufferToWav(buffer);
                resolve(new Blob([wavBuffer], {type: 'audio/wav'}));
            });
        };
        reader.onerror = reject;
        reader.readAsArrayBuffer(blob);
    });
}

function audioBufferToWav(buffer) {
    const numChannels = buffer.numberOfChannels;
    const sampleRate = buffer.sampleRate;
    const format = 1; // PCM
    const bitDepth = 16;

    let audioData = new Float32Array(buffer.length * numChannels);
    for (let channel = 0; channel < numChannels; channel++) {
        let channelData = buffer.getChannelData(channel);
        for (let i = 0; i < buffer.length; i++) {
            audioData[i * numChannels + channel] = channelData[i];
        }
    }

    // Convert to 16-bit PCM
    const dataLength = audioData.length * 2;
    const buffer16 = new Int16Array(audioData.length);
    for (let i = 0; i < audioData.length; i++) {
        buffer16[i] = audioData[i] < 0 ? audioData[i] * 0x8000 : audioData[i] * 0x7FFF;
    }

    const header = new ArrayBuffer(44);
    const view = new DataView(header);

    // RIFF chunk descriptor
    writeString(view, 0, 'RIFF');
    view.setUint32(4, 36 + dataLength, true);
    writeString(view, 8, 'WAVE');

    // FMT sub-chunk
    writeString(view, 12, 'fmt ');
    view.setUint32(16, 16, true); // subchunk1size
    view.setUint16(20, format, true);
    view.setUint16(22, numChannels, true);
    view.setUint32(24, sampleRate, true);
    view.setUint32(28, sampleRate * numChannels * bitDepth / 8, true); // byterate
    view.setUint16(32, numChannels * bitDepth / 8, true); // blockalign
    view.setUint16(34, bitDepth, true);

    // Data sub-chunk
    writeString(view, 36, 'data');
    view.setUint32(40, dataLength, true);

    // Write the WAV file
    const wavData = new Uint8Array(header.byteLength + dataLength);
    wavData.set(new Uint8Array(header), 0);
    wavData.set(new Uint8Array(buffer16.buffer), header.byteLength);

    return wavData.buffer;
}

function writeString(view, offset, string) {
    for (let i = 0; i < string.length; i++) {
        view.setUint8(offset + i, string.charCodeAt(i));
    }
}
function toggleTTS() {
    isTTSEnabled = !isTTSEnabled;
    document.getElementById('toggle-tts').classList.toggle('active');
    fetch('/toggle_tts', {
        method: 'POST',
        headers: {
            'Content-Type': 'application/json',
        },
        body: JSON.stringify({ enabled: isTTSEnabled }),
    })
    .then(response => response.json())
    .then(data => {
        console.log(data.message);
    })
    .catch(error => {
        console.error('Error toggling TTS:', error);
    });
}

async function playTTSStream(text) {
    if (isTTSEnabled) {
        audioQueue.push(text);
        if (!isPlaying) {
            await playNextInQueue();
        }
        await preloadAudio();
    }
}

async function fetchAudio(text) {
    const response = await fetch('/tts_stream', {
        method: 'POST',
        headers: {
            'Content-Type': 'application/json',
        },
        body: JSON.stringify({ text: text }),
    });

    if (!response.ok) {
        throw new Error(`HTTP error! status: ${response.status}`);
    }

    const audioBlob = await response.blob();
    return URL.createObjectURL(audioBlob);
}

async function preloadAudio() {
    while (audioBuffer.length < MAX_BUFFER_SIZE && audioQueue.length > 0) {
        const text = audioQueue.shift();
        try {
            const audioUrl = await fetchAudio(text);
            audioBuffer.push({ text, audioUrl });
        } catch (error) {
            console.error('Error preloading audio:', error);
            audioQueue.unshift(text);  // Возвращаем текст в очередь для повторной попытки
            break;
        }
    }
}

async function playNextInQueue() {
    if (audioBuffer.length === 0 && audioQueue.length === 0) {
        isPlaying = false;
        return;
    }

    isPlaying = true;

    let audioData;
    if (audioBuffer.length > 0) {
        audioData = audioBuffer.shift();
    } else {
        const text = audioQueue.shift();
        const audioUrl = await fetchAudio(text);
        audioData = { text, audioUrl };
    }

    const audio = new Audio(audioData.audioUrl);

    audio.onended = () => {
        URL.revokeObjectURL(audioData.audioUrl);
        playNextInQueue();
    };

    try {
        await audio.play();
        // Начинаем предзагрузку следующего аудио сразу после начала воспроизведения
        preloadAudio();
    } catch (error) {
        console.error('Error playing TTS stream:', error);
        playNextInQueue();
    }
}

export { initializeSpeechUtils, playTTSStream };