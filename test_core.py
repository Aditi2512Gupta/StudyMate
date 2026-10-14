"""
Pipeline test script to verify end-to-end functionality of StudyMate core components.
"""

from core.pdf_processor import extract_text_from_pdf, chunk_text
from core.vector_store import FAISSVectorStore
from core.gemini_client import generate_offline_fallback_embeddings
from core.rag_engine import answer_query_with_rag
from core.quiz_manager import evaluate_quiz

def run_test():
    print("Step 1: Testing PDF Text Extraction...")
    pages, meta = extract_text_from_pdf("sample_data/sample_study_notes.pdf")
    print(f" -> Extracted {len(pages)} pages. Total chars: {meta['total_characters']}")
    assert len(pages) == 4, "Expected 4 pages"

    print("\nStep 2: Testing NLP Chunking...")
    chunks = chunk_text(pages, chunk_size=800, chunk_overlap=150)
    print(f" -> Generated {len(chunks)} chunks.")
    assert len(chunks) > 0, "Expected non-zero chunks"

    print("\nStep 3: Testing Embedding & FAISS Vector Store...")
    texts = [c["text"] for c in chunks]
    embeddings = generate_offline_fallback_embeddings(texts, dimension=768)
    assert embeddings.shape == (len(chunks), 768), f"Unexpected embedding shape {embeddings.shape}"

    store = FAISSVectorStore(dimension=768)
    store.build_index(chunks, embeddings)
    stats = store.get_stats()
    print(f" -> FAISS Stats: {stats}")
    assert stats["total_vectors"] == len(chunks), "Vector count mismatch"

    print("\nStep 4: Testing RAG Query & Cosine Similarity Search...")
    query = "What is reinforcement learning and reward signal?"
    answer, sources = answer_query_with_rag(query, store, api_key=None, top_k=3)
    print(f" -> Retrieved {len(sources)} relevant chunks.")
    for idx, s in enumerate(sources, 1):
        print(f"    Source #{idx}: Page {s['page_number']} (Similarity: {s['similarity_score']}): {s['text'][:65]}...")
    assert len(sources) > 0, "Expected at least 1 retrieved source"

    print("\nStep 5: Testing Quiz Evaluation Logic...")
    mock_questions = [
        {
            "id": 1,
            "question": "What is the primary goal of Supervised Learning?",
            "options": {
                "A": "Learning from labeled datasets",
                "B": "Discovering hidden clusters",
                "C": "Trial and error reward maximization",
                "D": "Compressing images"
            },
            "correct_answer": "A",
            "explanation": "Supervised learning trains on labeled input-output pairs.",
            "reference_topic": "ML Paradigms"
        },
        {
            "id": 2,
            "question": "Which FAISS index type computes exact Inner Product?",
            "options": {
                "A": "IndexIVFFlat",
                "B": "IndexFlatIP",
                "C": "IndexFlatL2",
                "D": "IndexHNSW"
            },
            "correct_answer": "B",
            "explanation": "IndexFlatIP computes exact Inner Product (Cosine similarity when normalized).",
            "reference_topic": "Vector Databases"
        }
    ]
    # Test perfect score
    result_perfect = evaluate_quiz(mock_questions, {1: "A", 2: "B"})
    print(f" -> Mock Quiz Perfect Score: {result_perfect['score_percentage']}% (Grade: {result_perfect['grade']})")
    assert result_perfect["score_percentage"] == 100.0
    assert result_perfect["grade"] == "A+"

    # Test partial score
    result_partial = evaluate_quiz(mock_questions, {1: "A", 2: "C"})
    print(f" -> Mock Quiz Partial Score: {result_partial['score_percentage']}% (Grade: {result_partial['grade']})")
    assert result_partial["score_percentage"] == 50.0
    assert result_partial["grade"] == "D"

    print("\nALL CORE PIPELINE TESTS PASSED SUCCESSFULLY! Ready for Streamlit UI.")

if __name__ == "__main__":
    run_test()
