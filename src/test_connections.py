from pinecone import Pinecone
from groq import Groq
from openai import OpenAI
import voyageai
from src.config import (
    PINECONE_API_KEY,
    VOYAGE_API_KEY,
    GROQ_API_KEY,
    OPENROUTER_API_KEY,
)


def test_voyage():
    print("--- 1. Testing Voyage AI ---")
    vo = voyageai.Client(api_key=VOYAGE_API_KEY)
    res = vo.embed(["Hello world"], model="voyage-3-lite")
    print(f"Voyage AI connected! Embedding dim: {len(res.embeddings[0])}")


def test_pinecone():
    print("\n--- 2. Testing Pinecone ---")
    pc = Pinecone(api_key=PINECONE_API_KEY)
    print("Connected to Pinecone.")
    print("Indexes:", pc.list_indexes())


def test_groq():
    print("\n--- 3. Testing Groq ---")
    groq_client = Groq(api_key=GROQ_API_KEY)
    response = groq_client.chat.completions.create(
        model="llama-3.1-8b-instant",
        messages=[
            {
                "role": "user",
                "content": "Reply with exactly: Groq connection successful.",
            }
        ],
    )
    print("Groq Response:", response.choices[0].message.content)


def test_openrouter():
    print("\n--- 4. Testing OpenRouter ---")
    openrouter_client = OpenAI(
        base_url="https://openrouter.ai/api/v1",
        api_key=OPENROUTER_API_KEY,
    )
    response = openrouter_client.chat.completions.create(
        model="meta-llama/llama-3.1-8b-instruct",
        messages=[
            {
                "role": "user",
                "content": "Reply with exactly: OpenRouter connection successful.",
            }
        ],
    )
    print("OpenRouter Response:", response.choices[0].message.content)


if __name__ == "__main__":
    test_voyage()
    test_pinecone()
    test_groq()
    test_openrouter()
    print("\n[OK] All connection tests passed successfully!")
