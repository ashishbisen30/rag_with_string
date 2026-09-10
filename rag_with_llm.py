import os
from dotenv import load_dotenv
from openai import OpenAI

# 1. Load environment variables from .env
# Pass override=True so it replaces any old variables cached in memory
load_dotenv(override=True)

# 2. Initialize the OpenAI client once (automatically reads OPENAI_API_KEY from env)
client = OpenAI()

# Import retrieval pipeline functions from simple_rag.py
# Change load_and_chunk_documents to load_documents
from simple_rag import (
    create_embeddings,
    load_documents,
    retrieve,
)

# ==========================================
# STEP 4 & 5: Prompt Formatting & LLM Generation
# ==========================================


def generate_answer(query: str, retrieved_context: list[str]) -> str:
    """Mock LLM response to verify context retrieval and prompt construction offline."""
    context_text = "\n".join(retrieved_context)

    # Grounded Prompt
    prompt = f"""You are a grounded assistant. Answer the question using ONLY the provided context below.
If the answer cannot be found in the context, strictly reply with: "I cannot answer this based on the provided documents."

Context:
{context_text}

Question: {query}
Answer:"""

    # Mock response logic
    if "Java" in query:
        return "I cannot answer this based on the provided documents."

    return f"[MOCKED RESPONSE based on context: '{retrieved_context[0]}']"

# --- Full End-to-End Test Execution ---
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
        "Q5: Who created Java?",  # Out-of-knowledge-base test
    ]

    print("\n" + "=" * 50)
    print("END-TO-END RAG GENERATION TEST")
    print("=" * 50)

    for q in test_questions:
        # Step 3: Retrieve top context match
        context = retrieve(q, model, doc_embeddings, docs, top_k=1)

        # Step 4 & 5: Generate answer using LLM
        answer = generate_answer(q, context)

        print(f"\nQuestion: {q}")
        print(f"Retrieved Context: {context[0]}")
        print(f"LLM Response: {answer}")