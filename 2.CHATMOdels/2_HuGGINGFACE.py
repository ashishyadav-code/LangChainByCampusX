from huggingface_hub import InferenceClient
from dotenv import load_dotenv
load_dotenv()
import os

HF_TOKEN = os.getenv("HF_TOKEN")

client = InferenceClient(
    token=HF_TOKEN
)

res = client.chat.completions.create(
    model="meta-llama/Llama-3.1-8B-Instruct",
    messages=[{"role": "user", "content": "Hello"}],
)

print(res.choices[0].message["content"])