# 📚 StudyMate – AI-Powered Study Assistant
### *Infosys Summer Internship Project*

[![Python](https://img.shields.io/badge/Python-3.10%20%7C%203.11%20%7C%203.12%20%7C%203.14-blue.svg)](https://www.python.org/)
[![Streamlit](https://img.shields.io/badge/Streamlit-1.65.0-FF4B4B.svg)](https://streamlit.io/)
[![PyMuPDF](https://img.shields.io/badge/PyMuPDF-1.28.2-red.svg)](https://pymupdf.readthedocs.io/)
[![FAISS](https://img.shields.io/badge/FAISS-CPU%201.15.1-008080.svg)](https://github.com/facebookresearch/faiss)
[![Gemini](https://img.shields.io/badge/Google%20GenAI-Gemini%20API-4285F4.svg)](https://ai.google.dev/)

---

## 📌 Executive Summary & Project Overview

**StudyMate** is an intelligent, full-stack educational assistant developed as an **Infosys Summer Internship project**. It empowers students, researchers, and educators to transform dense academic textbooks, lecture slides, and research papers into interactive, high-retention learning experiences.

By combining modern **Natural Language Processing (NLP)**, **dense vector embeddings**, **Facebook AI Similarity Search (FAISS)**, and **Retrieval-Augmented Generation (RAG)** powered by Google's **Gemini LLM**, StudyMate eliminates the traditional problems of hallucinations and knowledge cutoffs in educational AI.

---

## 🛠️ Complete Technology Stack

| Layer | Technology | Primary Function in StudyMate |
| :--- | :--- | :--- |
| **Language** | **Python 3.10+** | Core logic, modular software engineering, clean object-oriented architecture |
| **User Interface** | **Streamlit** | Modern, reactive multi-tab web dashboard, state management, interactive quiz forms |
| **PDF Extraction** | **PyMuPDF (fitz)** | High-speed page-by-page text extraction, coordinate inspection, metadata extraction |
| **NLP & Chunking** | **Custom NLP Pipeline** | Regex text cleaning, hyphenated line break stitching, sliding-window chunking with overlap |
| **Embeddings** | **Google text-embedding-004** | 768-dimensional dense semantic vector space representations |
| **Vector Database** | **FAISS (IndexFlatIP)** | In-memory indexing, L2 normalization, microsecond Cosine Similarity search |
| **LLM Reasoning** | **Gemini 2.5 Flash / 2.0 Flash** | Academic summarization, context-grounded Q&A, structured quiz generation |
| **Prompt Engineering** | **Structured Templates** | System personas, few-shot conditioning, strict hallucination guardrails, JSON schema enforcement |
| **Environment** | **python-dotenv** | Secure API key isolation and environment configuration |

---

## ✨ Core Features & Functional Capabilities

### 1. 📄 Document Ingestion & Text Preprocessing
- Upload multi-page academic textbooks, lecture notes, or research papers in `.pdf` format.
- Instant fallback demo button: **"Load Sample Academic Guide"** (pre-bundled 4-page academic guide on AI, ML, NLP & RAG).
- Robust edge-case guards: validates encrypted files, corrupted streams, scanned PDFs lacking OCR, and empty pages.
- Calculates document statistics: page count, total words, character count, and estimated reading time.

### 2. 🧠 Multi-Style AI Summarizer
- **Key Takeaways & Core Terms**: Concise bullet-point summary highlighting bold academic terminology and mnemonic revision tips.
- **Executive / Conceptual Study Guide**: Structured theoretical frameworks, mechanism explanations, and real-world significance.
- **High-Yield Exam Revision Cheat-Sheet**: Core definitions, algorithmic principles, common pitfalls, and top 3 likely exam questions with model answers.
- One-click export to download summaries in Markdown (`.md`).

### 3. 💬 Context-Grounded AI Study Chat (RAG Pipeline)
- Conversational study assistant answering conceptual and technical questions.
- **Zero Hallucination Guardrail**: Instructs the model to rely solely on the document chunks. If information is absent, the model states so explicitly rather than fabricating facts.
- **Transparent Source Citations**: Every answer includes an expandable card displaying the exact retrieved chunks, source page numbers, and cosine similarity relevance percentages.
- Quick study prompt chips for immediate demonstration.

### 4. 📝 Interactive Smart Quiz & Assessment Engine
- Generates curriculum-aligned Multiple-Choice Questions (MCQs) customized by **number of questions** (3, 5, 8, 10), **difficulty** (Easy, Medium, Hard), and **focus topic**.
- Two-layer JSON guarantee (`response_mime_type="application/json"` + regex parser + schema validator) prevents broken formatting.
- Interactive question cards with radio option selection.
- Automated score calculation with percentage, letter grade (A+, A, B, C, D, F), pass/fail status, and celebratory feedback (balloons/confetti on $\ge 80\%$).
- Comprehensive post-quiz review displaying user answers vs. correct answers with in-depth academic explanations for every option.

---

## 🏗️ System Architecture & RAG Pipeline

```mermaid
flowchart TD
    A[📄 PDF Study Material] --> B[PyMuPDF Page Extractor]
    B --> C[NLP Sanitizer & Hyphen Reassembler]
    C --> D[Sliding-Window Text Chunker<br/>Chunk Size: 800 | Overlap: 150]
    
    subgraph Offline_or_Ingestion [Document Ingestion Pipeline]
        D --> E[Gemini text-embedding-004]
        E --> F[L2 Vector Normalization]
        F --> G[(FAISS Vector Index<br/>IndexFlatIP - Cosine Similarity)]
    end

    subgraph RAG_Query_Synthesis [Query & Synthesis Pipeline]
        H[💬 Student Question] --> I[Gemini Embedding Model]
        I --> J[L2 Vector Normalization]
        J --> K[FAISS Top-K Nearest Neighbor Search]
        G -.-> K
        K --> L[Retrieved Top-K Context Chunks + Page Citations]
        L --> M[Engineered RAG Grounding Prompt Template]
        H --> M
        M --> N[Gemini 2.5 Flash LLM]
        N --> O[✨ Grounded Answer + Source Verification Cards]
    end
```

---

## 🔬 Role of Each AI Concept in StudyMate

### 1. Natural Language Processing (NLP) & Sliding-Window Chunking
Raw text extracted from PDFs often contains formatting noise: words split by hyphens across line breaks (e.g., `ma-\nchine` $\rightarrow$ `machine`), irregular spacing, and control codes. The NLP module sanitizes this stream.
Because LLMs have context limits and embedding models produce diluted vectors for overly long documents, the text is chunked into cohesive semantic segments (default: 800 characters). A **sliding-window overlap** (default: 150 characters) ensures sentences spanning paragraph boundaries are preserved in both chunks, preventing loss of context.

### 2. Dense Vector Embeddings
Unlike sparse keyword models (TF-IDF or Bag-of-Words) that only match exact words, dense vector embeddings map textual semantics into a continuous 768-dimensional latent geometric space. Words and sentences with similar meanings (e.g., *"neural network weights"* and *"synaptic connection parameters"*) are placed near each other, enabling semantic search even when the student uses different vocabulary than the textbook author.

### 3. FAISS Vector Database & Cosine Similarity
Meta AI's **FAISS (Facebook AI Similarity Search)** provides sub-millisecond vector indexing. StudyMate uses `faiss.IndexFlatIP` (Inner Product). By applying $L_2$ vector normalization prior to indexing:
$$\|u\| = 1, \quad \|v\| = 1 \implies \text{Cosine Similarity}(u, v) = \frac{u \cdot v}{\|u\| \|v\|} = u \cdot v = \text{Inner Product}(u, v)$$
This yields exact cosine similarity scores bounded in $[-1.0, 1.0]$, ensuring fast retrieval and accurate relevance scoring.

### 4. Prompt Engineering & Guardrails
Prompt engineering governs how the LLM reasons. StudyMate employs:
- **System Instructions**: Defines persona as a rigorous academic tutor.
- **Few-Shot Context Formatting**: Clearly separates retrieved context from user questions.
- **Negative Constraints / Guardrails**: Explicitly orders the model to state *"Based on the provided excerpts, there is insufficient information to answer this question"* if the fact is not in the text, preventing academic hallucinations.
- **JSON Schema Enforcement**: Instructs the model to output valid JSON for quizzes, coupled with Gemini's native `response_mime_type="application/json"`.

### 5. Retrieval-Augmented Generation (RAG)
Standalone LLMs suffer from knowledge cutoffs and cannot verify private lecture notes or proprietary materials. RAG bridges this gap: instead of relying on parameterized memory, it fetches the authoritative text directly from the FAISS index and injects it into the prompt. This ensures 100% academic grounding and transparent verification.

---

## 📂 Project Structure

```
StudyMate/
├── app.py                      # Main Streamlit web application & UI state management
├── config.py                   # Central settings, environment loader & model defaults
├── requirements.txt            # Python package dependencies
├── .env.example                # Sample environment file template
├── .gitignore                  # Git ignore rules for virtualenvs, keys & caches
├── README.md                   # Comprehensive internship project documentation
├── test_core.py                # Automated end-to-end pipeline verification test
│
├── core/                       # Modular business logic
│   ├── __init__.py             # Package initializer
│   ├── pdf_processor.py        # PyMuPDF text extraction, NLP text cleaning & chunking
│   ├── gemini_client.py        # Gemini API client (google-genai), embeddings & error handlers
│   ├── vector_store.py         # FAISS vector database (IndexFlatIP & L2 normalization)
│   ├── prompts.py              # Prompt engineering templates (Summary, RAG Q&A, Quiz JSON)
│   ├── summary_manager.py      # Multi-format academic summary generator
│   ├── quiz_manager.py         # MCQ generation, schema validator & grade calculation
│   └── rag_engine.py           # End-to-end RAG coordinator
│
└── sample_data/                # Built-in demonstration assets
    ├── create_sample_pdf.py    # Generator script for academic sample notes
    └── sample_study_notes.pdf  # 4-page test PDF covering AI, ML, NLP, FAISS & RAG
```

---

## 🚀 Step-by-Step Installation & Setup

### Prerequisites
- Python 3.10, 3.11, 3.12, or 3.14 installed on Windows, macOS, or Linux.
- A free Google Gemini API Key from [Google AI Studio](https://aistudio.google.com/app/apikey).

### 1. Clone or Open the Project
```bash
cd c:\StudyMate
```

### 2. Create and Activate a Virtual Environment (Recommended)
```bash
# Windows (PowerShell)
python -m venv venv
.\venv\Scripts\Activate.ps1

# Linux / macOS
python3 -m venv venv
source venv/bin/activate
```

### 3. Install Dependencies
```bash
pip install -r requirements.txt
```

### 4. Configure Environment Variables
Copy `.env.example` to `.env`:
```bash
# Windows
copy .env.example .env

# Linux / macOS
cp .env.example .env
```
Open `.env` and add your Gemini API Key:
```env
GEMINI_API_KEY=AIzaSyYourActualKeyHere
```
*(Note: You can also enter the API key directly in the web app's sidebar!)*

### 5. Run Automated Verification Tests
```bash
python test_core.py
```
*Expected output:*
```
ALL CORE PIPELINE TESTS PASSED SUCCESSFULLY! Ready for Streamlit UI.
```

### 6. Launch the Streamlit Web Application
```bash
streamlit run app.py
```
The browser will automatically open to `http://localhost:8501`.

---

## 🧪 Sample Demonstration & Testing Workflow

Follow this 5-minute walkthrough to demonstrate all features during your internship review or presentation:

1. **Start the Application**: Run `streamlit run app.py`.
2. **API Key Setup**:
   - If `GEMINI_API_KEY` is in `.env`, the sidebar shows a green **"API Key Active & Configured"** badge.
   - Alternatively, type your key directly into the sidebar text box.
3. **Load Study Material**:
   - Click the blue **"📘 Load Sample Academic Guide"** button in the sidebar.
   - Watch the status indicator: PyMuPDF extracts 4 pages, generates 14 NLP chunks, and indexes them into FAISS.
   - Notice the **Document Metrics** card populate: Total Pages: 4, Total Words: ~1,085, Text Chunks: 14, FAISS Vectors: 14.
4. **Test Summarization (Tab 1)**:
   - Select **"📌 Quick Key Takeaways & Core Terms"** or **"🎓 High-Yield Exam Revision Cheat-Sheet"**.
   - Click **"✨ Generate AI Summary"**.
   - Review the generated academic summary with bold terms and review tips. Click **"💾 Download Summary as Markdown"**.
5. **Test RAG Study Chat (Tab 2)**:
   - Click the quick suggestion button: *"How does FAISS vector search compute Cosine Similarity?"*
   - Observe the real-time answer citing `[Page 2]` and `[Page 4]`.
   - Click the **"🔍 Verified Sources"** expander below the answer to inspect the exact retrieved text chunks and relevance scores.
6. **Test Smart Quiz & Assessment (Tab 3)**:
   - Select 5 questions, Medium difficulty, and click **"🚀 Generate Quiz from Document"**.
   - Answer the multiple-choice questions using the radio buttons.
   - Click **"📤 Submit Quiz & Calculate Score"**.
   - View your score percentage, letter grade (e.g., A+), performance feedback, and celebratory balloons.
   - Review the detailed explanations provided for each question.
7. **Inspect Technical Architecture (Tab 4)**:
   - Walk through the visual pipeline diagram and reference the Viva Q&A during your project evaluation!

---

## 🛡️ Error Handling & Edge Cases

| Scenario | System Behavior & Mitigation |
| :--- | :--- |
| **Missing API Key** | The application does not crash. It displays an informational badge and runs an offline semantic fallback mode for instant UI preview. |
| **Invalid or Expired API Key** | Displays a clear, actionable error banner with a direct link to Google AI Studio to generate a fresh key. |
| **Corrupted / Invalid PDF** | Caught by `pdf_processor.PDFProcessingError` with a friendly error message informing the user to upload a valid file. |
| **Password-Protected PDF** | PyMuPDF checks `doc.is_encrypted` and halts before reading, warning the user to provide an unencrypted document. |
| **Scanned / Image-Only PDF** | Detects extractable character count ($< 15$ chars) and alerts the user that OCR text layer is required. |
| **Rate Limit / Quota Exceeded (429)** | `gemini_client` automatically falls back across candidate models (`gemini-2.5-flash` $\rightarrow$ `gemini-2.0-flash` $\rightarrow$ `gemini-1.5-flash`). |
| **Broken LLM JSON Output** | Sanitized via regex stripping of code fences and trailing commas, then validated against the question schema. |

---

## 🎓 Internship Viva / Defense Cheat-Sheet

Use these concise talking points during your project viva or presentation:

- **Q: Why use FAISS over a cloud vector database like Pinecone?**  
  *Answer:* For a local, privacy-preserving study assistant, FAISS runs directly in-memory with zero network latency, zero hosting cost, and microsecond query times via optimized C++ SIMD routines.

- **Q: How does Cosine Similarity differ from Euclidean Distance in RAG?**  
  *Answer:* Euclidean distance measures the straight-line distance between vector coordinates, which can be heavily distorted if one chunk is longer than another. Cosine similarity measures the angle between vectors, capturing semantic meaning invariant to document length.

- **Q: How does StudyMate guard against LLM hallucinations?**  
  *Answer:* Through strict prompt engineering guardrails instructing the LLM to ground its answers strictly in the retrieved FAISS chunks and explicitly state when information is missing, coupled with transparent source citation expanders for the student.

- **Q: How does the quiz engine ensure reliable scoring?**  
  *Answer:* The prompt enforces a strict JSON schema with keys `question`, `options`, `correct_answer`, and `explanation`. Gemini's `response_mime_type="application/json"` guarantees valid syntax. User selections are matched deterministically against the answer key.

---

## 👨‍💻 Author & Acknowledgments
- **Project:** StudyMate – AI-Powered Study Assistant
- **Program:** Infosys Summer Internship Project
- **Domain:** Artificial Intelligence, Natural Language Processing, and Full-Stack Machine Learning
