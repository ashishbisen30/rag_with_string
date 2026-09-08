import numpy as np
from sentence_transformers import SentenceTransformer


# ==========================================
# STEP 1: Document Ingestion
# ==========================================
def load_documents(file_path: str) -> list[str]:
    """Reads a text file line-by-line, stripping whitespace and skipping empty lines."""
    with open(file_path, "r", encoding="utf-8") as f:
        documents = [line.strip() for line in f if line.strip()]
    return documents


# ==========================================
# STEP 2: Create Vector Embeddings
# ==========================================
# WHAT: Convert text strings into lists of floating-point numbers (dense vectors).
# WHY: Allows us to perform semantic similarity searches using vector geometry (dot product).
# PURPOSE: Builds numerical representations for all document chunks.


def create_embeddings(documents: list[str]):
    # Load a free, lightweight embedding model locally
    # "all-MiniLM-L6-v2" creates 384-dimensional vectors
    print("\nLoading embedding model (runs locally)...")
    model = SentenceTransformer("all-MiniLM-L6-v2")

    # Generate embeddings for each document
    # normalize_embeddings=True scales vectors to length 1 (simplifies similarity calculations)
    print("Generating embeddings for documents...")
    doc_embeddings = model.encode(documents, normalize_embeddings=True)

    return model, doc_embeddings


# ==========================================
# STEP 3: Retrieve Relevant Chunks
# ==========================================
# WHAT: Compare the query vector against all document vectors using mathematical dot product.
# WHY: Finds which text lines contain the answers relative to the user query.
# PURPOSE: Extracts only relevant context out of the entire database.


def retrieve(
    query: str,
    model: SentenceTransformer,
    doc_embeddings: np.ndarray,
    documents: list[str],
    top_k: int = 1,
) -> list[str]:
    """Encodes query, calculates similarity against stored document vectors, and returns top matches."""
    # 1. Convert the user question into a vector
    query_embedding = model.encode([query], normalize_embeddings=True)

    # 2. Calculate cosine similarity using dot product
    # (Since vectors are normalized, dot product equals cosine similarity)
    similarities = np.dot(doc_embeddings, query_embedding.T).flatten()

    # 3. Sort document indices by highest similarity score
    top_indices = np.argsort(similarities)[::-1][:top_k]

    # 4. Extract and return top matching text lines
    retrieved_docs = [documents[i] for i in top_indices]
    return retrieved_docs


# --- Execution and Verification ---
if __name__ == "__main__":
    # Steps 1 & 2
    file_name = "knowledge.txt"
    docs = load_documents(file_name)
    model, doc_embeddings = create_embeddings(docs)

    # Step 3 Test Execution
    print("\n--- Step 3 Verification ---")
    test_query = "Who created Python?"
    retrieved_context = retrieve(test_query, model, doc_embeddings, docs, top_k=1)

    print(f"Query: '{test_query}'")
    print(f"Retrieved Match: '{retrieved_context[0]}'")

    #Python web framework
    # Step 3 Test Execution
    print("\n--- Step 3.1 Verification ---")
    test_query = "Python web framework?"
    retrieved_context = retrieve(test_query, model, doc_embeddings, docs, top_k=1)

    print(f"Query: '{test_query}'")
    print(f"Retrieved Match: '{retrieved_context[0]}'")