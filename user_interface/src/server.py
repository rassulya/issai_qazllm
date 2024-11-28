# server.py
import re
import asyncio
from fastapi import FastAPI, HTTPException, UploadFile, File, Request, Form
from fastapi.responses import FileResponse, StreamingResponse, JSONResponse
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field
from typing import List
import os
import httpx
import logging
import json
import base64
import numpy as np
import librosa
import soundfile as sf
from openai import OpenAI

# Import speech_utils functions
from utils.speech_utils import transcribe_audio, toggle_tts, is_tts_enabled, text_to_speech, tts_stream, set_language

from pathlib import Path
import yaml  # Assuming you need to parse YAML

# Get the root directory of the project
ROOT_DIR = Path(os.getenv("PROJECT_ROOT", "."))

# Path to the configuration file
CONFIG_PATH = ROOT_DIR / "conf" / "parameters_ui.yaml"


# Load the configuration
def load_config():
    with open(CONFIG_PATH, "r") as file:
        config = yaml.safe_load(file)
    return config

config = load_config()
model_config = config["model"]
model_path = model_config["model_path"]
model_port = model_config["port"]
api = config["api"]
api_port = api["port"]
api_host = api["host"]
# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Create FastAPI app
app = FastAPI()

# CORS configuration
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # For development; restrict in production
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)



# Serve static files
static_dir = os.path.join(os.path.dirname(__file__), "static")
app.mount("/static", StaticFiles(directory=static_dir), name="static")

# Configuration variables
VLLM_URL = os.getenv(f"VLLM_URL", "http://localhost:{model_port}/v1")
MODEL_PATH = os.getenv("MODEL_PATH", model_path)

SYSTEM_PROMPT = "You are a highly capable AI assistant. Respons in the language of the user query."

class Message(BaseModel):
    role: str
    content: str

class GenerateRequest(BaseModel):
    messages: List[Message]
    temperature: float = 0.6
    max_tokens: int = 256
    top_k: int = 5
    best_of: int = 1
    repetition_penalty: float = 1.0
    system_prompt: str = Field(default=SYSTEM_PROMPT)

# Initialize the OpenAI client with VLLM URL
client = OpenAI(
    api_key="EMPTY",
    base_url=VLLM_URL
)

@app.post("/generate")
async def generate(request: GenerateRequest):
    logger.info(f"Received request: {request}")

    # Use the system prompt from the request
    system_prompt = request.system_prompt

    # Prepend the system prompt to the messages
    messages = [{"role": "system", "content": system_prompt}] + [
        {"role": msg.role, "content": msg.content} for msg in request.messages
    ]

    data = {
        "model": MODEL_PATH,
        "messages": messages,
        "temperature": request.temperature,
        "max_tokens": request.max_tokens,
        "top_k": request.top_k,
        "best_of": request.best_of,
        "repetition_penalty": request.repetition_penalty,
        "stream": True,
    }

    async def stream_generator():
        async with httpx.AsyncClient(timeout=None) as client:
            try:
                async with client.stream('POST', f"{VLLM_URL}/chat/completions", json=data) as response:
                    if response.status_code != 200:
                        logger.error(f"LLM server error: {response.status_code}")
                        raise HTTPException(status_code=500, detail=f"LLM server error: {response.status_code}")

                    async for chunk in response.aiter_bytes():
                        chunk_str = chunk.decode('utf-8')
                        yield chunk  # Stream the chunk to the client

                        # Check if the chunk contains usage data
                        if '"usage"' in chunk_str:
                            try:
                                # Extract usage data and send it to the client
                                data_parts = chunk_str.strip().split('\n')
                                for part in data_parts:
                                    if part.startswith('data: '):
                                        json_str = part[len('data: '):]
                                        if json_str != '[DONE]':
                                            data_json = json.loads(json_str)
                                            if 'usage' in data_json:
                                                usage_data = data_json['usage']
                                                usage_message = f"data: {json.dumps({'usage': usage_data})}\n\n"
                                                yield usage_message.encode('utf-8')
                            except Exception as e:
                                logger.error(f"Error parsing usage data: {e}")
            except Exception as e:
                logger.exception("Error during LLM streaming")
                yield f'data: {json.dumps({"error": str(e)})}\n\n'.encode('utf-8')

    return StreamingResponse(stream_generator(), media_type="text/event-stream")

@app.post("/asr")
async def asr(audio: UploadFile = File(...)):
    try:
        logger.info(f"Получен аудиофайл: {audio.filename}, content_type: {audio.content_type}")
        audio_content = await audio.read()
        
        # Сохраняем временный файл
        with open("temp_audio.wav", "wb") as temp_file:
            temp_file.write(audio_content)
        
        # Читаем аудио файл
        audio_float32, sample_rate = sf.read("temp_audio.wav", dtype='float32')
        
        logger.debug(f"Original audio shape: {audio_float32.shape}, Sample rate: {sample_rate}")
        
        # Преобразуем в моно, если это стерео
        if len(audio_float32.shape) > 1:
            audio_float32 = np.mean(audio_float32, axis=1)
            logger.debug(f"Converted to mono. New shape: {audio_float32.shape}")
        
        # Изменяем частоту дискретизации, если она не 16000 Гц
        if sample_rate != 16000:
            logger.debug(f"Resampling from {sample_rate} to 16000 Hz")
            audio_float32 = librosa.resample(y=audio_float32, orig_sr=sample_rate, target_sr=16000)
            sample_rate = 16000
            logger.debug(f"Resampled audio shape: {audio_float32.shape}")
        
        transcription = await transcribe_audio(audio_float32)
        logger.info(f"Транскрибированный текст: {transcription}")
        
        # Удаляем временный файл
        os.remove("temp_audio.wav")
        
        return {"transcription": transcription}
    except Exception as e:
        logger.error(f"ASR error: {str(e)}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/tts")
async def tts(request: Request):
    if not is_tts_enabled():
        raise HTTPException(status_code=400, detail="TTS is not enabled")
    try:
        data = await request.json()
        text = data.get("text", "")
        audio_content = await text_to_speech(text)
        audio_base64 = base64.b64encode(audio_content).decode('utf-8')
        return JSONResponse({"audio": audio_base64})
    except Exception as e:
        logger.error(f"TTS error: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/set_language")
async def set_language_endpoint(request: Request):
    data = await request.json()
    lang_code = data.get("language", "kk")
    try:
        if lang_code not in ["kk", "ru", "en"]:
            raise ValueError(f"Unsupported language code: {lang_code}")
        await set_language(lang_code)
        return {"language": lang_code}
    except ValueError as e:
        logger.error(f"Language switch error: {str(e)}")
        raise HTTPException(status_code=400, detail=str(e))


@app.post("/tts_stream")
async def tts_stream_endpoint(request: Request):
    data = await request.json()
    text = data.get("text", "")
    
    async def generate():
        async for audio_chunk in tts_stream(text):
            yield audio_chunk

    return StreamingResponse(generate(), media_type="audio/wav")

@app.post("/toggle_tts")
async def toggle_tts_endpoint(request: Request):
    data = await request.json()
    enable = data.get("enabled", False)
    result = toggle_tts(enable)
    return {"tts_enabled": result}

@app.get("/health")
async def health_check():
    return {"status": "ok"}
@app.get("/")
async def read_root():
    return FileResponse(os.path.join(static_dir, "index.html"))
@app.middleware("http")
async def add_no_cache_headers(request: Request, call_next):
    response = await call_next(request)
    response.headers["Cache-Control"] = "no-cache, no-store, must-revalidate"
    response.headers["Pragma"] = "no-cache"
    response.headers["Expires"] = "0"
    return response
@app.get("/favicon.ico")
async def favicon():
    return {"message": "No favicon"}

@app.post("/v1/chat/completions")
async def chat_completions(request: Request, image: UploadFile = File(None), question: str = Form(None)):
    try:
        if image:
            # Handle image upload
            contents = await image.read()
            base64_image = base64.b64encode(contents).decode('utf-8')
            image_url = f"data:image/jpeg;base64,{base64_image}"
            
            messages = [{
                "role": "user",
                "content": [
                    {"type": "text", "text": question or "Describe the main elements in this image"},
                    {"type": "image_url", "image_url": {"url": image_url}}
                ]
            }]
        else:
            # Handle regular chat request
            data = await request.json()
            messages = data.get("messages", [])
        
        # Extract other parameters
        model = data.get("model", MODEL_PATH) if 'data' in locals() else MODEL_PATH
        max_tokens = data.get("max_tokens", 300) if 'data' in locals() else 300
        temperature = data.get("temperature", 0.7) if 'data' in locals() else 0.7
        stop = data.get("stop", ["<|eot_id|>"]) if 'data' in locals() else ["<|eot_id|>"]
        
        # Prepare the payload for the VLLM service
        payload = {
            "model": model,
            "messages": messages,
            "max_tokens": max_tokens,
            "temperature": temperature,
            "stop": stop
        }
        
        # Make the API call to VLLM
        async with httpx.AsyncClient() as client:
            response = await client.post(f"{VLLM_URL}/chat/completions", json=payload)
            response.raise_for_status()
            result = response.json()
        
        # Return the result directly
        return result
    
    except Exception as e:
        logger.error(f"Chat completion error: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

if __name__ == "__main__":
    import uvicorn
    logger.info("Starting ASR_TTS server...")
    uvicorn.run(app, host=api_host, port=api_port) 
