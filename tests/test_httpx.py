import httpx
import asyncio

async def stream_generator():
    print("stream_generator function is being called...")
    data = {
        'model': 'models/llama8b',
        'messages': [
            {'role': 'system', 'content': 'You are a highly knowledgeable assistant who should use language of the user query. Please answer in 3 sentences at most.'},
            {'role': 'user', 'content': 'hi'}
        ],
        'temperature': 0.6,
        'max_tokens': 512,
        'top_k': 5,
        'best_of': 1,
        'repetition_penalty': 1.0,
        'stream': True
    }
    vllm_url = 'http://172.28.0.2:8003/v1'

    async with httpx.AsyncClient(timeout=None) as client:
        try:
            print(f"Connecting to {vllm_url}/chat/completions")
            async with client.stream('POST', f"{vllm_url}/chat/completions", json=data) as response:
                if response.status_code != 200:
                    print(f"LLM server error: {response.status_code}")
                    return
                async for chunk in response.aiter_bytes():
                    print(f"Received chunk: {chunk}")
        except Exception as e:
            print(f"Error during streaming: {e}")

async def main():
    print("Starting stream generator...")
    await stream_generator()

if __name__ == "__main__":
    asyncio.run(main())
