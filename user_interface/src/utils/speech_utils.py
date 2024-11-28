# /raid/vladimir_albrekht/web_demo/combined/ver6_asr_tts_rag/utils/speech_utils.py
import os
import logging
import json
import wave
import io
import numpy as np
from vosk import Model, KaldiRecognizer
from utils.tts_piper.TTS import TTS
import asyncio
from pathlib import Path
import yaml

# Configure logging
logging.basicConfig(level=logging.DEBUG)
logger = logging.getLogger(__name__)

asr_models = {}
tts_models = {}
current_language = 'kk'  # default language
tts_enabled = False

ROOT_DIR = Path(os.getenv("PROJECT_ROOT", "."))

# Path to the configuration file
CONFIG_PATH = ROOT_DIR / "conf" / "parameters_ui.yaml"

# Load the configuration
def load_config():
    with open(CONFIG_PATH, "r") as file:
        config = yaml.safe_load(file)
    return config



tts_model_paths = {
    'kk': 'src/utils/tts_piper/piper_models/voice-kk-issai-high/kk-issai-high.onnx',
    'en': 'src/utils/tts_piper/piper_models/en-us-amy-low/en-us-amy-low.onnx',
    'ru': 'src/utils/tts_piper/piper_models/ru-ruslan-medium/ru_ru_RU_ruslan_medium_ru_RU-ruslan-medium.onnx'
}

tts_lock = asyncio.Lock()

def load_asr_models():
    global asr_models

    asr_model_paths = {
        'kk': "src/utils/asr_vosk/vosk_models/vosk-model-kz-0.15",
        'en': "src/utils/asr_vosk/vosk_models/vosk-model-en-us-0.22-lgraph",
        'ru': "src/utils/asr_vosk/vosk_models/vosk-model-small-ru-0.22"   
    }
    for lang_code, model_path in asr_model_paths.items():
        if not os.path.exists(model_path):
            logger.error(f"ASR model path does not exist: {model_path}")
            raise FileNotFoundError(f"ASR model not found at {model_path}")
        logger.info(f"Loading ASR model for '{lang_code}' from {model_path}")
        asr_models[lang_code] = Model(model_path)
    logger.info("All ASR models loaded successfully")

def load_tts_models():
    global tts_models
    for lang_code, model_path in tts_model_paths.items():
        logger.info(f"Loading TTS model for '{lang_code}' from {model_path}")
        if lang_code == 'kk':
            tts_models[lang_code] = TTS(model_path, voice_id=0)
        else:
            tts_models[lang_code] = TTS(model_path, voice_id=None)
    logger.info("All TTS models loaded successfully")

# Load models
load_asr_models()
load_tts_models()

async def text_to_speech(text):
    global current_language
    logger.info(f"Generating TTS for text in language '{current_language}': {text}")
    async with tts_lock:
        try:
            tts_model = tts_models[current_language]
            audio_norm, sample_rate = tts_model.generate_audio(text)
            logger.debug(f"TTS audio generated, shape: {audio_norm.shape}, sample rate: {sample_rate}")

            buffer = io.BytesIO()
            with wave.open(buffer, "wb") as wav_file:
                wav_file.setnchannels(1)
                wav_file.setsampwidth(2)
                wav_file.setframerate(sample_rate)
                wav_file.writeframes((audio_norm * 32767).astype(np.int16).tobytes())

            logger.debug(f"TTS audio converted to WAV, size: {buffer.tell()} bytes")
            return buffer.getvalue()
        except Exception as e:
            logger.error(f"Error in text_to_speech: {str(e)}", exc_info=True)
            raise

async def transcribe_audio(audio_data):
    global current_language
    logger.debug(f"Received audio data of length: {len(audio_data)}")

    if current_language not in asr_models:
        logger.error(f"ASR model for language '{current_language}' is not loaded")
        raise RuntimeError(f"ASR model for language '{current_language}' is not initialized")

    asr_model = asr_models[current_language]

    audio_int16 = (audio_data * 32767).astype(np.int16)
    logger.debug(f"Converted to int16, shape: {audio_int16.shape}")

    sample_rate = 16000
    try:
        recognizer = KaldiRecognizer(asr_model, sample_rate)
    except Exception as e:
        logger.error(f"Error creating KaldiRecognizer: {str(e)}")
        raise

    audio_bytes = audio_int16.tobytes()

    recognizer.AcceptWaveform(audio_bytes)
    result = recognizer.FinalResult()
    logger.debug(f"Final result: {result}")

    result_dict = json.loads(result)
    text = result_dict.get("text", "")
    logger.info(f"Transcription: {text}")
    return text

def toggle_tts(enable: bool) -> bool:
    global tts_enabled
    tts_enabled = enable
    logger.info(f"TTS {'enabled' if tts_enabled else 'disabled'}")
    return tts_enabled

def is_tts_enabled() -> bool:
    return tts_enabled

async def process_tts_request(text):
    if not is_tts_enabled():
        logger.warning("TTS is not enabled. Skipping TTS generation.")
        return None

    try:
        logger.debug(f"Processing TTS request for text: {text}")
        audio_content = await text_to_speech(text)
        logger.info(f"TTS generated successfully for text: {text[:50]}...")
        return audio_content
    except Exception as e:
        logger.error(f"Failed to generate TTS for text: {text[:50]}... Error: {str(e)}", exc_info=True)
        return None  # Return None to indicate failure

async def tts_stream(text):
    text = text.strip()
    sentences = text.split(".")
    sentences = [s.strip() for s in sentences if s.strip()]  # Remove empty sentences
    sentences = sentences[:4]  # Limit to 5 sentences
    text = ". ".join(sentences) + "."  # Rejoin with periods and add final period
    
    if text:
        audio_content = await process_tts_request(text)
        if audio_content:
            yield audio_content

async def set_language(lang_code):
    global current_language, tts_models
    async with tts_lock:
        if lang_code in asr_models and lang_code in tts_models:
            current_language = lang_code
            logger.info(f"Language set to {current_language}")

            tts_model_path = tts_model_paths[current_language]

            try:
                logger.info(f"Re-initializing TTS model for '{current_language}' from {tts_model_path}")
                if lang_code == 'kk':
                    tts_models[current_language] = TTS(tts_model_path, voice_id=0)
                else:
                    tts_models[current_language] = TTS(tts_model_path, voice_id=None)
                logger.info(f"TTS model for language '{current_language}' re-initialized successfully")
            except Exception as e:
                logger.error(f"Error re-initializing TTS model for '{current_language}': {str(e)}", exc_info=True)
                raise
        else:
            logger.error(f"Language '{lang_code}' models are not loaded")
            raise ValueError(f"Language '{lang_code}' models are not loaded")

if __name__ == "__main__":
    asyncio.run(main())