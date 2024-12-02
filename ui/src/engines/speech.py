from pathlib import Path
import os
import logging
import json
import wave
import io
import numpy as np
from vosk import Model, KaldiRecognizer
from .tts_piper.TTS import TTS  # Added dot to make it relative import
import asyncio
from config.settings import get_settings

logger = logging.getLogger(__name__)

class SpeechService:
    def __init__(self):
        self.settings = get_settings()
        self.asr_models = {}
        self.tts_models = {}
        self.current_language = 'kk'
        self.tts_enabled = False
        self.tts_lock = asyncio.Lock()
        
        self.tts_model_paths = {
            'kk': 'models/tts_piper/piper_models/voice-kk-issai-high/kk-issai-high.onnx',
            'en': 'models/tts_piper/piper_models/en-us-amy-low/en-us-amy-low.onnx',
            'ru': 'models/tts_piper/piper_models/ru-ruslan-medium/ru_ru_RU_ruslan_medium_ru_RU-ruslan-medium.onnx'
        }
        
        self.asr_model_paths = {
            'kk': "models/asr_vosk/vosk_models/vosk-model-kz-0.15",
            'en': "models/asr_vosk/vosk_models/vosk-model-en-us-0.22-lgraph",
            'ru': "models/asr_vosk/vosk_models/vosk-model-small-ru-0.22"   
        }
        
        self._load_models()
        
    def _load_models(self):
        self._load_asr_models()
        self._load_tts_models()
        
    def _load_asr_models(self):
        for lang_code, model_path in self.asr_model_paths.items():
            if not os.path.exists(model_path):
                logger.error(f"ASR model path does not exist: {model_path}")
                raise FileNotFoundError(f"ASR model not found at {model_path}")
            logger.info(f"Loading ASR model for '{lang_code}' from {model_path}")
            self.asr_models[lang_code] = Model(model_path)
        logger.info("All ASR models loaded successfully")

    def _load_tts_models(self):
        for lang_code, model_path in self.tts_model_paths.items():
            logger.info(f"Loading TTS model for '{lang_code}' from {model_path}")
            voice_id = 0 if lang_code == 'kk' else None
            self.tts_models[lang_code] = TTS(model_path, voice_id=voice_id)
        logger.info("All TTS models loaded successfully")

    async def text_to_speech(self, text: str) -> bytes:
        logger.info(f"Generating TTS for text in language '{self.current_language}': {text}")
        async with self.tts_lock:
            try:
                tts_model = self.tts_models[self.current_language]
                audio_norm, sample_rate = tts_model.generate_audio(text)
                
                buffer = io.BytesIO()
                with wave.open(buffer, "wb") as wav_file:
                    wav_file.setnchannels(1)
                    wav_file.setsampwidth(2)
                    wav_file.setframerate(sample_rate)
                    wav_file.writeframes((audio_norm * 32767).astype(np.int16).tobytes())
                
                return buffer.getvalue()
            except Exception as e:
                logger.error(f"Error in text_to_speech: {str(e)}", exc_info=True)
                raise

    async def transcribe_audio(self, audio_data: np.ndarray) -> str:
        if self.current_language not in self.asr_models:
            raise RuntimeError(f"ASR model for language '{self.current_language}' is not initialized")

        audio_int16 = (audio_data * 32767).astype(np.int16)
        recognizer = KaldiRecognizer(self.asr_models[self.current_language], 16000)
        audio_bytes = audio_int16.tobytes()
        
        recognizer.AcceptWaveform(audio_bytes)
        result = recognizer.FinalResult()
        result_dict = json.loads(result)
        return result_dict.get("text", "")

    def toggle_tts(self, enable: bool) -> bool:
        self.tts_enabled = enable
        logger.info(f"TTS {'enabled' if self.tts_enabled else 'disabled'}")
        return self.tts_enabled

    def is_tts_enabled(self) -> bool:
        return self.tts_enabled

    async def process_tts_request(self, text: str) -> bytes | None:
        if not self.is_tts_enabled():
            logger.warning("TTS is not enabled. Skipping TTS generation.")
            return None

        try:
            audio_content = await self.text_to_speech(text)
            logger.info(f"TTS generated successfully for text: {text[:50]}...")
            return audio_content
        except Exception as e:
            logger.error(f"Failed to generate TTS: {str(e)}", exc_info=True)
            return None

    async def tts_stream(self, text: str):
        text = text.strip()
        sentences = text.split(".")
        sentences = [s.strip() for s in sentences if s.strip()]
        sentences = sentences[:4]
        text = ". ".join(sentences) + "."
        
        if text:
            audio_content = await self.process_tts_request(text)
            if audio_content:
                yield audio_content

    async def set_language(self, lang_code: str) -> None:
        if lang_code not in self.asr_models or lang_code not in self.tts_models:
            raise ValueError(f"Language '{lang_code}' models are not loaded")
            
        async with self.tts_lock:
            self.current_language = lang_code
            voice_id = 0 if lang_code == 'kk' else None
            self.tts_models[lang_code] = TTS(self.tts_model_paths[lang_code], voice_id=voice_id)
            logger.info(f"Language set to {lang_code} and TTS model reinitialized")

speech_service = SpeechService()