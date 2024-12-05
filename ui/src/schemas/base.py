from pydantic import BaseModel, Field
from typing import List

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
    system_prompt: str = Field(default="You are a highly capable AI assistant. Respons in the language of the user query.")