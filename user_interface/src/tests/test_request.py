from openai import OpenAI
import base64

# Initialize the OpenAI client with the server URL
client = OpenAI(
    api_key="EMPTY",
    base_url="http://localhost:8036/v1"  # Make sure this matches your server's address and port
)

# Function to encode the image
def encode_image(image_path):
    with open(image_path, "rb") as image_file:
        return base64.b64encode(image_file.read()).decode('utf-8')

# Path to your image
image_path = "/raid/vladimir_albrekht/web_demo/combined/ver6_asr_tts_rag/vlm/goddes.jpg"

# Getting the base64 string
base64_image = encode_image(image_path)

# Creating the API request
response = client.chat.completions.create(
    model="/raid/vladimir_albrekht/vision_lm/models/MiniCPM_db7_8B-Inst-27000_stable",  # Use your actual model path
    messages=[
        {
            "role": "user",
            "content": [
                {"type": "text", "text": "Describe this image in detail"},
                {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{base64_image}"}}
            ]
        }
    ],
    max_tokens=300,
    temperature=0.8
)

# Print the response
print("Chat completion output:", response.choices[0].message.content)