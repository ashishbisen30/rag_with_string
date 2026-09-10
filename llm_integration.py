import os
from dotenv import load_dotenv
from openai import OpenAI

from simple_rag import create_embeddings, load_documents, retrieve

# 1. Force environment reload
load_dotenv(override=True)

# 2. Initialize OpenAI client
client = OpenAI()


def generate_answer_with_llm(query: str, retrieved_context: list[str]) -> str:
    """Formats prompt with retrieved context and fetches response from OpenAI LLM."""
    context_text = "\n".join(retrieved_context)

    prompt = f"""You are a grounded assistant. Answer the question using ONLY the provided context below.
If the answer cannot be found in the context, strictly reply with: "I cannot answer this based on the provided documents."

Context:
{context_text}

Question: {query}
Answer:"""

    response = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[{"role": "user", "content": prompt}],
        temperature=0,  # Deterministic output
    )

    return response.choices[0].message.content.strip()


if __name__ == "__main__":
    file_name = "knowledge.txt"

    # Step 1 & 2: Load and Embed
    docs = load_documents(file_name)
    model, doc_embeddings = create_embeddings(docs)

    # Test prompts
    test_questions = [
        "Q1: Who created Python?",
        "Q2: What is FastAPI?",
        "Q3: What does RAG stand for?",
        "Q4: What does LangChain provide?",
        "Q5: Who created Java?",
    ]

    print("\n" + "=" * 50)
    print("ACTIVE LLM INTEGRATION TEST (OpenAI gpt-4o-mini)")
    print("=" * 50)

    for q in test_questions:
        context = retrieve(q, model, doc_embeddings, docs, top_k=1)
        answer = generate_answer_with_llm(q, context)

        print(f"\nQuestion: {q}")
        print(f"Retrieved Context: {context[0]}")
        print(f"Real LLM Response: {answer}")