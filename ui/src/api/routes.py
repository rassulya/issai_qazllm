from pathlib import Path
from fastapi import APIRouter, HTTPException, UploadFile, File, Request, Form
from fastapi.responses import FileResponse, StreamingResponse, JSONResponse
import httpx
import logging
import soundfile as sf
import librosa
import numpy as np
import base64
import os

from schemas.base import GenerateRequest
from services.vllm import vllm_client
from engines.speech import speech_service
from config.settings import get_settings

settings = get_settings()
router = APIRouter()
logger = logging.getLogger(__name__)

@router.post("/generate")
async def generate(request: GenerateRequest):
    messages = [{"role": "system", "content": request.system_prompt}] + [
        {"role": msg.role, "content": msg.content} for msg in request.messages
    ]
    
    params = {
        "temperature": request.temperature,
        "max_tokens": request.max_tokens,
        "top_k": request.top_k,
        "best_of": request.best_of,
        "repetition_penalty": request.repetition_penalty,
    }
    
    data = await vllm_client.generate_stream(messages, params)

    async def stream_generator():
        async with httpx.AsyncClient(timeout=None) as client:
            async with client.stream('POST', f"{settings.vllm_url}/chat/completions", json=data) as response:
                if response.status_code != 200:
                    raise HTTPException(status_code=500, detail=f"LLM server error: {response.status_code}")
                async for chunk in response.aiter_bytes():
                    yield chunk

    return StreamingResponse(stream_generator(), media_type="text/event-stream")

@router.post("/asr")
async def asr(audio: UploadFile = File(...)):
    try:
        audio_content = await audio.read()
        with open("temp_audio.wav", "wb") as temp_file:
            temp_file.write(audio_content)
        
        audio_float32, sample_rate = sf.read("temp_audio.wav", dtype='float32')
        
        if len(audio_float32.shape) > 1:
            audio_float32 = np.mean(audio_float32, axis=1)
        
        if sample_rate != 16000:
            audio_float32 = librosa.resample(y=audio_float32, orig_sr=sample_rate, target_sr=16000)
        
        transcription = await speech_service.transcribe_audio(audio_float32)
        os.remove("temp_audio.wav")
        
        return {"transcription": transcription}
    except Exception as e:
        logger.error(f"ASR error: {str(e)}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))

@router.post("/tts")
async def tts_endpoint(request: Request):
    if not speech_service.is_tts_enabled():
        raise HTTPException(status_code=400, detail="TTS is not enabled")
    
    data = await request.json()
    text = data.get("text", "")
    audio_content = await speech_service.text_to_speech(text)
    audio_base64 = base64.b64encode(audio_content).decode('utf-8')
    return JSONResponse({"audio": audio_base64})

@router.post("/set_language")
async def set_language_endpoint(request: Request):
    data = await request.json()
    lang_code = data.get("language", "kk")
    
    if lang_code not in ["kk", "ru", "en"]:
        raise HTTPException(status_code=400, detail=f"Unsupported language code: {lang_code}")
        
    await speech_service.set_language(lang_code)
    return {"language": lang_code}

@router.post("/tts_stream")
async def tts_stream_endpoint(request: Request):
    data = await request.json()
    return StreamingResponse(
        speech_service.tts_stream(data.get("text", "")), 
        media_type="audio/wav"
    )

@router.post("/toggle_tts")
async def toggle_tts_endpoint(request: Request):
    data = await request.json()
    return {
        "tts_enabled": speech_service.toggle_tts(data.get("enabled", False))
    }

@router.get("/favicon.ico")
async def favicon():
    return FileResponse("static/favicon.ico")

@router.get("/")
async def read_root():
    static_dir = Path(__file__).parent.parent / "static"
    return FileResponse(static_dir / "index.html")