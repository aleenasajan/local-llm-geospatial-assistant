import ollama
import numpy as np
from pathlib import Path

KNOWLEDGE_FILE = Path("rag/knowledge/project_knowledge.txt")
EMBED_MODEL = "nomic-embed-text"
LLM_MODEL = "gemma4:latest"

text = KNOWLEDGE_FILE.read_text()

chunks = [
    chunk.strip()
    for chunk in text.split("\n\n")
    if chunk.strip()
]


def embed(text):
    response = ollama.embeddings(
        model=EMBED_MODEL,
        prompt=text
    )
    return np.array(response["embedding"])


chunk_embeddings = [
    embed(chunk)
    for chunk in chunks
]

query = input("👤 Ask a question: ")

query_embedding = embed(query)

scores = []

for i, chunk_embedding in enumerate(chunk_embeddings):
    similarity = np.dot(
        query_embedding,
        chunk_embedding
    ) / (
        np.linalg.norm(query_embedding)
        * np.linalg.norm(chunk_embedding)
    )

    scores.append(
        (similarity, chunks[i])
    )

scores.sort(
    reverse=True,
    key=lambda x: x[0]
)

best_score, best_chunk = scores[0]

print("\n🔎 RAG retrieved:\n")
print(best_chunk)

print("\n📊 Similarity score:")
print(round(float(best_score), 4))


prompt = f"""
You are an assistant for the QGIS2OllamaMCP project.

Use the retrieved project knowledge below to answer the user's question.

Retrieved context:
{best_chunk}

User question:
{query}

Answer clearly and concisely.
"""


response = ollama.chat(
    model=LLM_MODEL,
    messages=[
        {
            "role": "user",
            "content": prompt
        }
    ]
)

print("\n🤖 Gemma response:\n")
print(response["message"]["content"])
