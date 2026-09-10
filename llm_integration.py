import os
from dotenv import load_dotenv
from google import genai

from simple_rag import create_embeddings, load_documents, retrieve

# 1. Load environment variables
load_dotenv(override=True)

# 2. Initialize Gemini Client
client = genai.Client(api_key=os.getenv("GEMINI_API_KEY"))


def generate_answer_with_llm(query: str, retrieved_context: list[str]) -> str:
    """Formats prompt with retrieved context and fetches response from Gemini."""
    context_text = "\n".join(retrieved_context)

    prompt = f"""You are a grounded assistant. Answer the question using ONLY the provided context below.
If the answer cannot be found in the context, strictly reply with: "I cannot answer this based on the provided documents."

Context:
{context_text}

Question: {query}
Answer:"""

    response = client.models.generate_content(
        model="gemini-3.6-flash",
        contents=prompt,
    )

    return response.text.strip()


if __name__ == "__main__":
    file_name = "knowledge.txt"

    # Step 1 & 2: Load and Embed
    docs = load_documents(file_name)
    model, doc_embeddings = create_embeddings(docs)

    # All 5 Test Questions
    test_questions = [
        "Q1: Who created Python?",
        "Q2: What is FastAPI?",
        "Q3: What does RAG stand for?",
        "Q4: What does LangChain provide?",
        "Q5: Who created Java?",
        "Q6: What is microservice architecture?",
    ]

    print("\n" + "=" * 50)
    print("ACTIVE LLM INTEGRATION TEST (Google Gemini 3.6 Flash)")
    print("=" * 50)

    for q in test_questions:
        context = retrieve(q, model, doc_embeddings, docs, top_k=1)
        answer = generate_answer_with_llm(q, context)

        print(f"\nQuestion: {q}")
        print(f"Retrieved Context: {context[0]}")
        print(f"Real LLM Response: {answer}")