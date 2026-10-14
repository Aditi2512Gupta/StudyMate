"""
StudyMate – AI-Powered Study Assistant
Infosys Summer Internship Project

Tech Stack: Python, Streamlit, PyMuPDF, Gemini API, Embeddings, FAISS.
Modules: Document Text Extraction, NLP Chunking, Vector Storage, Summarization, RAG Q&A, and Quiz Engine.
"""

import os
import streamlit as st
import numpy as np

import config
from core.pdf_processor import extract_text_from_pdf, chunk_text, PDFProcessingError
from core.vector_store import FAISSVectorStore
from core import gemini_client
from core import summary_manager
from core import quiz_manager
from core.rag_engine import answer_query_with_rag

# ---------------------------------------------------------------------------
# Streamlit Page Configuration
# ---------------------------------------------------------------------------
st.set_page_config(
    page_title=f"{config.APP_NAME} – {config.APP_SUBTITLE}",
    page_icon=config.APP_ICON,
    layout="wide",
    initial_sidebar_state="expanded"
)

# ---------------------------------------------------------------------------
# Custom CSS for Premium Design & Visual Polish
# ---------------------------------------------------------------------------
st.markdown("""
<style>
    /* Google Font Import */
    @import url('https://fonts.googleapis.com/css2?family=Plus+Jakarta+Sans:wght@400;500;600;700;800&display=swap');

    html, body, [class*="css"] {
        font-family: 'Plus Jakarta Sans', -apple-system, BlinkMacSystemFont, sans-serif;
    }

    /* Main Container Padding */
    .block-container {
        padding-top: 1.8rem;
        padding-bottom: 3rem;
    }

    /* App Header Banner */
    .app-header-container {
        background: linear-gradient(135deg, #1E1B4B 0%, #312E81 50%, #4338CA 100%);
        padding: 1.6rem 2rem;
        border-radius: 16px;
        color: white;
        margin-bottom: 1.8rem;
        box-shadow: 0 10px 25px -5px rgba(67, 56, 202, 0.3);
        border: 1px solid rgba(255, 255, 255, 0.1);
    }
    .app-header-title {
        font-size: 2.1rem;
        font-weight: 800;
        margin: 0;
        letter-spacing: -0.5px;
        display: flex;
        align-items: center;
        gap: 12px;
    }
    .app-header-badge {
        display: inline-block;
        background: rgba(255, 255, 255, 0.2);
        backdrop-filter: blur(8px);
        padding: 4px 12px;
        border-radius: 9999px;
        font-size: 0.8rem;
        font-weight: 600;
        letter-spacing: 0.5px;
        text-transform: uppercase;
        margin-top: 8px;
    }
    .app-header-desc {
        color: #E0E7FF;
        font-size: 0.95rem;
        margin-top: 8px;
        margin-bottom: 0;
    }

    /* Metric Cards */
    .stat-card {
        background: #F8FAFC;
        border: 1px solid #E2E8F0;
        border-radius: 12px;
        padding: 12px 16px;
        text-align: center;
        box-shadow: 0 1px 3px rgba(0,0,0,0.05);
    }
    .stat-card-val {
        font-size: 1.5rem;
        font-weight: 700;
        color: #1E293B;
    }
    .stat-card-lbl {
        font-size: 0.78rem;
        font-weight: 600;
        color: #64748B;
        text-transform: uppercase;
        letter-spacing: 0.5px;
    }

    /* Source Citation Card */
    .source-box {
        background: #F1F5F9;
        border-left: 4px solid #4F46E5;
        padding: 10px 14px;
        border-radius: 6px;
        margin-bottom: 8px;
        font-size: 0.88rem;
    }
    .source-meta {
        font-weight: 700;
        color: #3730A3;
        margin-bottom: 4px;
        display: flex;
        justify-content: space-between;
    }

    /* Quiz Question Card */
    .quiz-card {
        background: #FFFFFF;
        border: 1px solid #E2E8F0;
        border-radius: 12px;
        padding: 18px 22px;
        margin-bottom: 18px;
        box-shadow: 0 2px 6px rgba(0,0,0,0.04);
    }
    .quiz-q-num {
        font-size: 0.85rem;
        font-weight: 700;
        color: #4F46E5;
        text-transform: uppercase;
        margin-bottom: 6px;
    }
    .quiz-q-title {
        font-size: 1.05rem;
        font-weight: 600;
        color: #0F172A;
        margin-bottom: 12px;
    }

    /* Quiz Feedback Badges */
    .badge-correct {
        background: #DEF7EC;
        color: #03543F;
        border: 1px solid #84E1BC;
        padding: 4px 10px;
        border-radius: 6px;
        font-weight: 700;
        font-size: 0.85rem;
    }
    .badge-incorrect {
        background: #FDE8E8;
        color: #9B1C1C;
        border: 1px solid #F8B4B4;
        padding: 4px 10px;
        border-radius: 6px;
        font-weight: 700;
        font-size: 0.85rem;
    }

    /* Scoreboard Card */
    .scoreboard-box {
        background: linear-gradient(135deg, #EEF2FF 0%, #E0E7FF 100%);
        border: 1px solid #C7D2FE;
        border-radius: 14px;
        padding: 24px;
        text-align: center;
        margin-bottom: 24px;
    }
    .scoreboard-score {
        font-size: 3rem;
        font-weight: 800;
        color: #3730A3;
        line-height: 1;
    }
    .scoreboard-grade {
        font-size: 1.4rem;
        font-weight: 700;
        color: #4338CA;
        margin-top: 6px;
    }

    /* Pipeline Flow Chart Box */
    .pipeline-step {
        background: #F8FAFC;
        border: 1px solid #CBD5E1;
        border-radius: 10px;
        padding: 14px;
        margin-bottom: 12px;
    }
    .pipeline-step-title {
        font-weight: 700;
        color: #1E293B;
        font-size: 0.95rem;
        display: flex;
        align-items: center;
        gap: 8px;
    }
    .pipeline-step-body {
        font-size: 0.85rem;
        color: #475569;
        margin-top: 4px;
    }

    /* Tab styling */
    .stTabs [data-baseweb="tab-list"] {
        gap: 8px;
    }
    .stTabs [data-baseweb="tab"] {
        padding: 10px 18px;
        border-radius: 8px;
        font-weight: 600;
    }
</style>
""", unsafe_allow_html=True)


# ---------------------------------------------------------------------------
# Global Session State Initialization
# ---------------------------------------------------------------------------
def init_session_state():
    """Initializes persistent variables in Streamlit session_state."""
    defaults = {
        "api_key": config.get_api_key(),
        "processed_file_name": None,
        "pages_data": [],
        "doc_metadata": {},
        "chunks": [],
        "vector_store": FAISSVectorStore(dimension=768),
        "active_summary": {},
        "chat_messages": [],
        "active_quiz": [],
        "user_answers": {},
        "quiz_submitted": False,
        "quiz_results": None,
        "chunk_size": config.DEFAULT_CHUNK_SIZE,
        "chunk_overlap": config.DEFAULT_CHUNK_OVERLAP,
        "top_k": config.DEFAULT_TOP_K,
        "selected_model": config.DEFAULT_LLM_MODEL
    }
    for key, val in defaults.items():
        if key not in st.session_state:
            st.session_state[key] = val

init_session_state()


# ---------------------------------------------------------------------------
# Helper Functions for Document Ingestion
# ---------------------------------------------------------------------------
def process_pdf_document(source, filename: str):
    """Processes uploaded PDF or sample file: extracts text, chunks, embeds, and builds FAISS index."""
    with st.spinner(f"Extracting and indexing text from '{filename}' using PyMuPDF and NLP chunking..."):
        try:
            pages, meta = extract_text_from_pdf(source)
            chunks = chunk_text(
                pages, 
                chunk_size=st.session_state.chunk_size, 
                chunk_overlap=st.session_state.chunk_overlap
            )
            
            # Embeddings step
            effective_key = config.get_api_key(st.session_state.api_key)
            if effective_key:
                try:
                    embeddings = gemini_client.generate_embeddings([c["text"] for c in chunks], api_key=effective_key)
                except Exception as emb_err:
                    st.warning(f"Could not reach Gemini Embedding API ({str(emb_err)}). Using local semantic vector fallback.")
                    embeddings = gemini_client.generate_offline_fallback_embeddings([c["text"] for c in chunks])
            else:
                embeddings = gemini_client.generate_offline_fallback_embeddings([c["text"] for c in chunks])

            # Build FAISS Vector Index
            vstore = FAISSVectorStore(dimension=embeddings.shape[1])
            vstore.build_index(chunks, embeddings)

            # Update session state
            st.session_state.processed_file_name = filename
            st.session_state.pages_data = pages
            st.session_state.doc_metadata = meta
            st.session_state.chunks = chunks
            st.session_state.vector_store = vstore
            st.session_state.active_summary = {}
            st.session_state.active_quiz = []
            st.session_state.quiz_submitted = False
            st.session_state.quiz_results = None
            st.session_state.user_answers = {}

            st.success(f"Successfully processed '{filename}': {meta['total_pages']} pages, {len(chunks)} chunks indexed in FAISS!")
        except PDFProcessingError as pe:
            st.error(f"PDF Processing Error: {str(pe)}")
        except Exception as e:
            st.error(f"Unexpected error while processing document: {str(e)}")


def reset_application():
    """Resets all session state to clear current study material."""
    st.session_state.processed_file_name = None
    st.session_state.pages_data = []
    st.session_state.doc_metadata = {}
    st.session_state.chunks = []
    st.session_state.vector_store = FAISSVectorStore(dimension=768)
    st.session_state.active_summary = {}
    st.session_state.chat_messages = []
    st.session_state.active_quiz = []
    st.session_state.user_answers = {}
    st.session_state.quiz_submitted = False
    st.session_state.quiz_results = None
    st.rerun()


# ---------------------------------------------------------------------------
# Sidebar: Configuration, Document Upload, & Controls
# ---------------------------------------------------------------------------
with st.sidebar:
    st.markdown(f"### {config.APP_ICON} {config.APP_NAME}")
    st.caption(f"**{config.PROJECT_TAG}** | v{config.APP_VERSION}")
    st.markdown("---")

    # 1. API Key Section
    st.markdown("#### 🔑 Gemini API Configuration")
    user_key_input = st.text_input(
        "Gemini API Key",
        value=st.session_state.api_key,
        type="password",
        placeholder="AIzaSy...",
        help="Paste your Google Gemini API key here or define GEMINI_API_KEY in .env"
    )
    if user_key_input != st.session_state.api_key:
        st.session_state.api_key = user_key_input

    effective_api_key = config.get_api_key(st.session_state.api_key)
    if effective_api_key:
        st.success("API Key Active & Configured", icon="✅")
    else:
        st.warning("Running in Offline Preview mode. Enter an API key to enable live Gemini AI generation.", icon="⚠️")
        st.markdown("[Get a Free Gemini API Key](https://aistudio.google.com/app/apikey)")

    st.markdown("---")

    # 2. Document Upload Section
    st.markdown("#### 📄 Document Ingestion")
    uploaded_file = st.file_uploader("Upload PDF Study Material", type=["pdf"], help="Upload academic lecture notes, textbooks, or research papers")

    if uploaded_file is not None:
        if st.session_state.processed_file_name != uploaded_file.name:
            process_pdf_document(uploaded_file, uploaded_file.name)

    # Quick Demo / Sample Document Loader Button
    sample_pdf_path = config.BASE_DIR / "sample_data" / "sample_study_notes.pdf"
    if sample_pdf_path.exists():
        if st.button("📘 Load Sample Academic Guide", use_container_width=True, help="Load pre-built 4-page course guide on AI, ML, NLP & RAG"):
            with open(sample_pdf_path, "rb") as f:
                process_pdf_document(f.read(), "AI_and_ML_Study_Guide.pdf")

    # 3. Document Statistics (if loaded)
    if st.session_state.processed_file_name:
        st.markdown("---")
        st.markdown("#### 📊 Document Metrics")
        meta = st.session_state.doc_metadata
        vstats = st.session_state.vector_store.get_stats()
        
        st.markdown(f"**File:** `{st.session_state.processed_file_name}`")
        col_s1, col_s2 = st.columns(2)
        with col_s1:
            st.metric("Total Pages", meta.get("total_pages", 0))
            st.metric("Total Words", f"{meta.get('total_words', 0):,}")
        with col_s2:
            st.metric("Text Chunks", len(st.session_state.chunks))
            st.metric("FAISS Vectors", vstats.get("total_vectors", 0))

    # 4. Advanced RAG Tuning Expander
    st.markdown("---")
    with st.expander("⚙️ Advanced RAG & NLP Settings"):
        st.session_state.selected_model = st.selectbox(
            "Gemini Model",
            options=["gemini-2.5-flash", "gemini-2.0-flash", "gemini-1.5-flash"],
            index=0
        )
        st.session_state.chunk_size = st.slider("Chunk Size (characters)", min_value=300, max_value=1500, value=st.session_state.chunk_size, step=50)
        st.session_state.chunk_overlap = st.slider("Chunk Overlap (characters)", min_value=50, max_value=300, value=st.session_state.chunk_overlap, step=25)
        st.session_state.top_k = st.slider("FAISS Top-K Retrieval", min_value=1, max_value=8, value=st.session_state.top_k, step=1)

    # 5. Reset App Button
    if st.session_state.processed_file_name:
        st.markdown("---")
        if st.button("🗑️ Clear & Reset Document", use_container_width=True, type="secondary"):
            reset_application()


# ---------------------------------------------------------------------------
# Main Content Header
# ---------------------------------------------------------------------------
st.markdown(f"""
<div class="app-header-container">
    <div class="app-header-title">
        <span>{config.APP_ICON}</span> {config.APP_NAME}
    </div>
    <div class="app-header-badge">{config.PROJECT_TAG}</div>
    <p class="app-header-desc">
        Intelligent academic assistant combining <b>PyMuPDF</b> text extraction, 
        <b>NLP text chunking</b>, <b>Gemini embeddings</b>, <b>FAISS vector search</b>, and 
        <b>context-grounded RAG</b> for summarization, intelligent Q&A, and interactive quizzes.
    </p>
</div>
""", unsafe_allow_html=True)


# ---------------------------------------------------------------------------
# Check if Document is Loaded
# ---------------------------------------------------------------------------
if not st.session_state.processed_file_name:
    st.info("👋 **Welcome to StudyMate!** To get started, please upload a study material PDF in the left sidebar, or click **'Load Sample Academic Guide'** for an immediate hands-on walkthrough.", icon="📘")
    
    # Showcase Cards on empty state
    col1, col2, col3 = st.columns(3)
    with col1:
        st.markdown("""
        ### 📄 AI Summarizer
        Extract multi-page study notes and synthesize:
        - Quick bullet-point takeaways
        - Executive conceptual overviews
        - High-yield exam cheat-sheets
        """)
    with col2:
        st.markdown("""
        ### 💬 Context-Grounded Chat
        Ask any conceptual or technical question:
        - Retrieves top relevant passages via FAISS
        - Direct source page citations
        - Zero hallucinations via strict guardrails
        """)
    with col3:
        st.markdown("""
        ### 📝 Interactive Quiz
        Test your mastery dynamically:
        - Custom question counts & difficulty
        - Instant scoring and letter grade
        - Full answer breakdown with explanations
        """)
    
    st.stop()


# ---------------------------------------------------------------------------
# Main Application Tabs
# ---------------------------------------------------------------------------
tab_summary, tab_chat, tab_quiz, tab_architecture = st.tabs([
    "📄 1. Document & Summary",
    "💬 2. AI Study Chat (RAG)",
    "📝 3. Smart Quiz & Assessment",
    "🧠 4. Architecture & Viva Guide"
])


# ===========================================================================
# TAB 1: Document & Summary
# ===========================================================================
with tab_summary:
    st.markdown("### 📄 Document Overview & AI Summarizer")
    
    # Document Metric Cards
    m_col1, m_col2, m_col3, m_col4 = st.columns(4)
    with m_col1:
        st.markdown(f"""
        <div class="stat-card">
            <div class="stat-card-val">{st.session_state.doc_metadata.get('total_pages', 0)}</div>
            <div class="stat-card-lbl">Total Pages</div>
        </div>
        """, unsafe_allow_html=True)
    with m_col2:
        st.markdown(f"""
        <div class="stat-card">
            <div class="stat-card-val">{st.session_state.doc_metadata.get('total_words', 0):,}</div>
            <div class="stat-card-lbl">Word Count</div>
        </div>
        """, unsafe_allow_html=True)
    with m_col3:
        st.markdown(f"""
        <div class="stat-card">
            <div class="stat-card-val">{len(st.session_state.chunks)}</div>
            <div class="stat-card-lbl">NLP Chunks</div>
        </div>
        """, unsafe_allow_html=True)
    with m_col4:
        st.markdown(f"""
        <div class="stat-card">
            <div class="stat-card-val">{st.session_state.doc_metadata.get('estimated_reading_minutes', 1)} min</div>
            <div class="stat-card-lbl">Estimated Read Time</div>
        </div>
        """, unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)

    # Summarization Form
    st.markdown("#### 🧠 Generate Study Summary")
    sum_col1, sum_col2 = st.columns([2, 1])
    
    with sum_col1:
        summary_style_choice = st.selectbox(
            "Select Summary Format",
            options=[
                ("key_takeaways", "📌 Quick Key Takeaways & Core Terms"),
                ("executive", "📖 Executive / Conceptual Study Guide"),
                ("exam_cheatsheet", "🎓 High-Yield Exam Revision Cheat-Sheet")
            ],
            format_func=lambda x: x[1]
        )[0]
    
    with sum_col2:
        st.write("") # vertical spacing
        generate_sum_btn = st.button("✨ Generate AI Summary", type="primary", use_container_width=True)

    if generate_sum_btn:
        if not effective_api_key:
            st.error("Please enter your Gemini API Key in the sidebar to generate AI summaries.")
        else:
            with st.spinner("Synthesizing document with Gemini LLM..."):
                try:
                    summary_output = summary_manager.generate_summary(
                        pages=st.session_state.pages_data,
                        summary_style=summary_style_choice,
                        api_key=effective_api_key,
                        model=st.session_state.selected_model
                    )
                    st.session_state.active_summary[summary_style_choice] = summary_output
                except Exception as e:
                    st.error(f"Summarization Failed: {str(e)}")

    # Display Current Summary (if exists)
    current_summary = st.session_state.active_summary.get(summary_style_choice)
    if current_summary:
        st.markdown("---")
        st.markdown("#### 📋 Generated Summary")
        st.markdown(current_summary)
        
        # Download summary
        st.download_button(
            label="💾 Download Summary as Markdown",
            data=current_summary,
            file_name=f"{st.session_state.processed_file_name}_{summary_style_choice}_summary.md",
            mime="text/markdown"
        )

    # Document Text Inspector Expander
    st.markdown("---")
    with st.expander("🔍 Inspect Extracted PDF Pages & NLP Chunks"):
        tab_p, tab_c = st.tabs(["📄 Pages Preview", "🧩 NLP Chunks View"])
        with tab_p:
            for p in st.session_state.pages_data:
                st.markdown(f"**Page {p['page_number']}** ({p['char_count']} characters)")
                st.text_area(f"Page {p['page_number']} Text", value=p["text"], height=120, disabled=True, key=f"preview_p_{p['page_number']}")
        with tab_c:
            for c in st.session_state.chunks[:10]:
                st.caption(f"Chunk #{c['chunk_id']} • Page {c['page_number']} • {c['char_count']} chars")
                st.info(c["text"])
            if len(st.session_state.chunks) > 10:
                st.caption(f"... and {len(st.session_state.chunks) - 10} more chunks indexed in FAISS.")


# ===========================================================================
# TAB 2: AI Study Chat (RAG)
# ===========================================================================
with tab_chat:
    st.markdown("### 💬 AI Study Chat (RAG-Powered)")
    st.caption("Ask questions about the uploaded document. Answers are strictly grounded in retrieved document chunks with verifiable page citations.")

    # Quick Suggested Questions based on document
    st.markdown("**💡 Quick Study Prompts:**")
    quick_cols = st.columns(3)
    sample_queries = [
        "What are the three main machine learning paradigms?",
        "How does FAISS vector search compute Cosine Similarity?",
        "What are the key limitations of standalone LLMs and how does RAG solve them?"
    ]
    
    selected_quick_query = None
    for i, col in enumerate(quick_cols):
        with col:
            if st.button(sample_queries[i], use_container_width=True, key=f"quick_q_{i}"):
                selected_quick_query = sample_queries[i]

    # Chat Clear Button
    if st.session_state.chat_messages:
        if st.button("🧹 Clear Chat History", type="secondary"):
            st.session_state.chat_messages = []
            st.rerun()

    # Render Existing Chat History
    for msg in st.session_state.chat_messages:
        with st.chat_message(msg["role"]):
            st.markdown(msg["content"])
            if msg.get("sources"):
                with st.expander(f"🔍 Verified Sources ({len(msg['sources'])} Chunks Retrieved from FAISS)"):
                    for s in msg["sources"]:
                        sim_pct = int(s.get("similarity_score", 0.0) * 100)
                        st.markdown(f"""
                        <div class="source-box">
                            <div class="source-meta">
                                <span>📄 Page {s['page_number']} • Chunk #{s['chunk_id']}</span>
                                <span>Relevance Score: {sim_pct}%</span>
                            </div>
                            <div>{s['text']}</div>
                        </div>
                        """, unsafe_allow_html=True)

    # Chat Input Box
    user_input = st.chat_input("Ask a question about your study material...") or selected_quick_query

    if user_input:
        # Append User Message
        st.session_state.chat_messages.append({"role": "user", "content": user_input})
        with st.chat_message("user"):
            st.markdown(user_input)

        # Generate Grounded RAG Answer
        with st.chat_message("assistant"):
            with st.spinner("Searching FAISS vector database and formulating answer..."):
                try:
                    answer_text, sources = answer_query_with_rag(
                        query=user_input,
                        vector_store=st.session_state.vector_store,
                        api_key=effective_api_key,
                        top_k=st.session_state.top_k,
                        model=st.session_state.selected_model
                    )
                    st.markdown(answer_text)
                    
                    if sources:
                        with st.expander(f"🔍 Verified Sources ({len(sources)} Chunks Retrieved from FAISS)"):
                            for s in sources:
                                sim_pct = int(s.get("similarity_score", 0.0) * 100)
                                st.markdown(f"""
                                <div class="source-box">
                                    <div class="source-meta">
                                        <span>📄 Page {s['page_number']} • Chunk #{s['chunk_id']}</span>
                                        <span>Relevance Score: {sim_pct}%</span>
                                    </div>
                                    <div>{s['text']}</div>
                                </div>
                                """, unsafe_allow_html=True)

                    # Save to state
                    st.session_state.chat_messages.append({
                        "role": "assistant",
                        "content": answer_text,
                        "sources": sources
                    })
                except Exception as e:
                    err_msg = f"RAG Error: {str(e)}"
                    st.error(err_msg)
                    st.session_state.chat_messages.append({"role": "assistant", "content": err_msg})


# ===========================================================================
# TAB 3: Smart Quiz & Assessment
# ===========================================================================
with tab_quiz:
    st.markdown("### 📝 Smart Multiple-Choice Quiz (MCQ) & Assessment")
    st.caption("Generate dynamic, curriculum-aligned multiple-choice quizzes directly from the uploaded material to test your mastery.")

    # Quiz Generator Form
    with st.expander("⚙️ Quiz Generation Parameters", expanded=(len(st.session_state.active_quiz) == 0)):
        q_col1, q_col2, q_col3 = st.columns(3)
        with q_col1:
            quiz_num_q = st.selectbox("Number of Questions", options=[3, 5, 8, 10], index=1)
        with q_col2:
            quiz_diff = st.selectbox("Difficulty Level", options=["Easy", "Medium", "Hard"], index=1)
        with q_col3:
            quiz_topic = st.text_input("Focus Topic (Optional)", value="All Document Topics")

        gen_quiz_btn = st.button("🚀 Generate Quiz from Document", type="primary", use_container_width=True)

    if gen_quiz_btn:
        if not effective_api_key:
            st.error("Please enter your Gemini API Key in the sidebar to generate AI quizzes.")
        else:
            with st.spinner(f"Generating {quiz_num_q} {quiz_diff}-level questions from document content..."):
                try:
                    questions = quiz_manager.generate_quiz(
                        pages=st.session_state.pages_data,
                        num_questions=quiz_num_q,
                        difficulty=quiz_diff,
                        topic=quiz_topic,
                        api_key=effective_api_key,
                        model=st.session_state.selected_model
                    )
                    st.session_state.active_quiz = questions
                    st.session_state.user_answers = {}
                    st.session_state.quiz_submitted = False
                    st.session_state.quiz_results = None
                    st.success(f"Generated {len(questions)} quiz questions! Answer below.")
                except Exception as e:
                    st.error(f"Quiz Generation Error: {str(e)}")

    # Display Quiz Questions Form
    if st.session_state.active_quiz:
        questions = st.session_state.active_quiz
        
        # If already submitted, show scoreboard at top
        if st.session_state.quiz_submitted and st.session_state.quiz_results:
            res = st.session_state.quiz_results
            
            st.markdown(f"""
            <div class="scoreboard-box">
                <div class="scoreboard-score">{res['score_percentage']}%</div>
                <div class="scoreboard-grade">Grade: {res['grade']} • {'PASSED ✅' if res['passed'] else 'REVISION NEEDED ⚠️'}</div>
                <p style="margin-top: 10px; font-weight: 500; color: #4338CA;">{res['feedback']}</p>
                <div style="display: flex; justify-content: center; gap: 24px; margin-top: 14px; font-weight: 600; font-size: 0.95rem;">
                    <span style="color: #059669;">Correct: {res['correct_count']}</span>
                    <span style="color: #DC2626;">Incorrect: {res['incorrect_count']}</span>
                    <span style="color: #D97706;">Unanswered: {res['unanswered_count']}</span>
                    <span style="color: #4B5563;">Total: {res['total_questions']}</span>
                </div>
            </div>
            """, unsafe_allow_html=True)

            if res["score_percentage"] >= 80:
                st.balloons()

        st.markdown("---")
        st.markdown("#### 📝 Answer the Questions:")

        # Render questions
        for idx, q in enumerate(questions, 1):
            q_id = q["id"]
            topic_tag = q.get("reference_topic", "General")
            
            st.markdown(f"""
            <div class="quiz-card">
                <div class="quiz-q-num">Question {idx} of {len(questions)} • <span style="color: #64748B;">{topic_tag}</span></div>
                <div class="quiz-q-title">{q['question']}</div>
            </div>
            """, unsafe_allow_html=True)

            options_dict = q["options"]
            option_keys = ["A", "B", "C", "D"]
            
            # Format option label
            formatted_options = [f"{k}) {options_dict[k]}" for k in option_keys]

            # If quiz already submitted, show disabled selection
            if st.session_state.quiz_submitted:
                user_selected = st.session_state.user_answers.get(q_id, "")
                correct_answer = q["correct_answer"]
                
                # Check status
                if user_selected == correct_answer:
                    st.markdown(f"<span class='badge-correct'>✅ Your Answer: {user_selected}) {options_dict.get(user_selected, '')} — CORRECT</span>", unsafe_allow_html=True)
                elif user_selected:
                    st.markdown(f"<span class='badge-incorrect'>❌ Your Answer: {user_selected}) {options_dict.get(user_selected, '')} — INCORRECT</span>", unsafe_allow_html=True)
                    st.markdown(f"<span class='badge-correct' style='margin-top: 6px; display: inline-block;'>Correct Answer: {correct_answer}) {options_dict.get(correct_answer, '')}</span>", unsafe_allow_html=True)
                else:
                    st.markdown(f"<span class='badge-incorrect'>⚠️ Not Answered</span>", unsafe_allow_html=True)
                    st.markdown(f"<span class='badge-correct' style='margin-top: 6px; display: inline-block;'>Correct Answer: {correct_answer}) {options_dict.get(correct_answer, '')}</span>", unsafe_allow_html=True)

                st.info(f"💡 **Explanation:** {q['explanation']}")
            else:
                # Active selection mode
                current_choice = st.session_state.user_answers.get(q_id, None)
                preselect_idx = option_keys.index(current_choice) if current_choice in option_keys else None
                
                selected_radio = st.radio(
                    f"Select option for Q{idx}",
                    options=option_keys,
                    format_func=lambda k: f"{k}) {options_dict[k]}",
                    index=preselect_idx,
                    key=f"radio_q_{q_id}",
                    label_visibility="collapsed"
                )
                if selected_radio:
                    st.session_state.user_answers[q_id] = selected_radio

            st.write("") # spacing

        # Action Buttons
        st.markdown("---")
        btn_col1, btn_col2, btn_col3 = st.columns([2, 1, 1])

        with btn_col1:
            if not st.session_state.quiz_submitted:
                if st.button("📤 Submit Quiz & Calculate Score", type="primary", use_container_width=True):
                    # Evaluate answers
                    results = quiz_manager.evaluate_quiz(questions, st.session_state.user_answers)
                    st.session_state.quiz_results = results
                    st.session_state.quiz_submitted = True
                    st.rerun()
            else:
                if st.button("🔄 Retake This Quiz", type="primary", use_container_width=True):
                    st.session_state.user_answers = {}
                    st.session_state.quiz_submitted = False
                    st.session_state.quiz_results = None
                    st.rerun()

        with btn_col2:
            if st.session_state.quiz_submitted:
                if st.button("✨ Generate New Quiz", use_container_width=True):
                    st.session_state.active_quiz = []
                    st.session_state.user_answers = {}
                    st.session_state.quiz_submitted = False
                    st.session_state.quiz_results = None
                    st.rerun()


# ===========================================================================
# TAB 4: Architecture & Viva Guide
# ===========================================================================
with tab_architecture:
    st.markdown("### 🧠 StudyMate System Architecture & Viva Guide")
    st.caption("Technical deep dive into the engineering principles, AI algorithms, and implementation details for your Infosys Internship evaluation.")

    # Visual Pipeline Architecture
    st.markdown("#### 🏗️ End-to-End RAG Architecture Pipeline")
    
    st.markdown("""
    <div class="pipeline-step">
        <div class="pipeline-step-title">1. Document Ingestion & NLP Cleaning (PyMuPDF + Python)</div>
        <div class="pipeline-step-body">
            Raw PDF files are ingested and parsed page-by-page. Text is sanitized: broken hyphenated words across line boundaries are reassembled (<code>re.sub</code>), control characters are filtered, and paragraph formatting is preserved.
        </div>
    </div>
    
    <div class="pipeline-step">
        <div class="pipeline-step-title">2. Semantic Text Chunking with Sliding Window Overlap (NLP)</div>
        <div class="pipeline-step-body">
            Long documents are split into cohesive chunks (default: 800 characters) with a configurable overlap (default: 150 characters). The overlap preserves semantic continuity across chunk boundaries so concepts spanning consecutive paragraphs are never severed.
        </div>
    </div>

    <div class="pipeline-step">
        <div class="pipeline-step-title">3. Dense Vector Embeddings (Gemini text-embedding-004)</div>
        <div class="pipeline-step-body">
            Each text chunk is mapped into a 768-dimensional continuous latent space using Google's embedding model. In this vector space, semantically related ideas reside in close geometric proximity.
        </div>
    </div>

    <div class="pipeline-step">
        <div class="pipeline-step-title">4. High-Performance Vector Indexing (FAISS IndexFlatIP)</div>
        <div class="pipeline-step-body">
            Vectors are L2-normalized (unit length) and indexed using Facebook AI Similarity Search (<code>faiss.IndexFlatIP</code>). Because vectors are normalized, inner product computation is mathematically equivalent to exact <b>Cosine Similarity</b>, enabling microsecond similarity search.
        </div>
    </div>

    <div class="pipeline-step">
        <div class="pipeline-step-title">5. Grounded Prompt Engineering & Synthesis (Gemini LLM)</div>
        <div class="pipeline-step-body">
            User queries are embedded into the same vector space. FAISS retrieves the Top-K most relevant chunks. An engineered prompt template injects these excerpts as grounded context, instructing Gemini to cite specific pages and strictly forbidding hallucinations.
        </div>
    </div>
    """, unsafe_allow_html=True)

    st.markdown("---")

    # Required Skills Mapping Table
    st.markdown("#### 🎯 Explicit Mapping of Required Skills to Code Implementation")
    
    st.markdown("""
| Required Skill | Role in StudyMate | Implementation File | Key Libraries & Functions |
| :--- | :--- | :--- | :--- |
| **Python** | Core application logic, modular OOP structure, error handling | `app.py`, `config.py` | Type hinting, dataclasses, modular packages |
| **NLP** | Text cleaning, hyphen joining, sentence boundary detection, sliding window chunking | `core/pdf_processor.py` | `re`, `pymupdf` (fitz), sliding window tokenizer |
| **Embeddings** | Mapping natural language text to 768-d semantic space | `core/gemini_client.py` | `google.genai.Client.models.embed_content` (`text-embedding-004`) |
| **FAISS** | In-memory vector database, L2 normalization, Top-K Cosine Similarity search | `core/vector_store.py` | `faiss.IndexFlatIP`, `faiss.normalize_L2` |
| **Prompt Engineering** | Few-shot templates, hallucination guardrails, strict JSON schema output | `core/prompts.py` | System instructions, context grounding, JSON schema constraints |
| **LLMs** | Text summarization, question answering, automated MCQ generation | `core/gemini_client.py`, `core/quiz_manager.py` | `gemini-2.5-flash`, `gemini-2.0-flash` |
| **RAG** | Full retrieval-augmented generation workflow with transparent citations | `core/rag_engine.py` | End-to-end vector retrieval + grounded prompt synthesis |
| **Streamlit** | Modern responsive web UI, session state, interactive quiz scoring & cards | `app.py` | `st.session_state`, `st.tabs`, `st.chat_message`, `st.radio` |
    """)

    st.markdown("---")

    # Viva Questions & Answers
    st.markdown("#### 🎓 Sample Internship Viva / Defense Questions & Model Answers")
    
    with st.expander("Q1: What is RAG and why is it preferred over fine-tuning for study materials?"):
        st.markdown("""
        **Answer:**
        **Retrieval-Augmented Generation (RAG)** dynamically injects relevant document chunks into the prompt context at query time. It is preferred over fine-tuning for study materials because:
        1. **Instant Updates**: New PDFs can be queried immediately without expensive re-training or weight updates.
        2. **Hallucination Prevention**: The LLM is instructed to answer strictly based on retrieved chunks.
        3. **Verifiable Citations**: Every answer can cite exact source pages and paragraph numbers.
        4. **Cost & Compute Efficiency**: Running a vector search takes milliseconds and costs virtually nothing compared to fine-tuning.
        """)

    with st.expander("Q2: Why use `faiss.IndexFlatIP` with L2-normalized vectors instead of `IndexFlatL2`?"):
        st.markdown("""
        **Answer:**
        Cosine similarity measures the angle between two semantic vectors regardless of text length:
        $$\\text{Cosine Similarity}(u, v) = \\frac{u \\cdot v}{\\|u\\| \\|v\\|}$$
        When vectors are pre-normalized using $L_2$ normalization ($\\|u\\| = 1, \\|v\\| = 1$), the denominator becomes 1:
        $$\\text{Cosine Similarity}(u, v) = u \\cdot v = \\text{Inner Product}(u, v)$$
        `IndexFlatIP` performs direct dot products using optimized BLAS matrix routines, which is faster and provides bounded similarity scores in the $[-1, 1]$ range.
        """)

    with st.expander("Q3: How does StudyMate ensure the quiz generation does not return broken JSON?"):
        st.markdown("""
        **Answer:**
        StudyMate applies a two-layer reliability architecture:
        1. **API Level**: It passes `response_mime_type="application/json"` to Gemini's `GenerateContentConfig`, forcing the model's token decoding to adhere strictly to JSON syntax.
        2. **Application Level**: In `core/quiz_manager.py`, the response undergoes regex sanitization (removing any accidental markdown fences ````json ... ```` and fixing trailing commas), followed by strict schema validation (`question`, 4 `options`, `correct_answer`, `explanation`).
        """)

    with st.expander("Q4: Why is chunk overlap necessary in the NLP pipeline?"):
        st.markdown("""
        **Answer:**
        If documents are cut abruptly at chunk boundaries without overlap, an important sentence or definition might be split across two separate chunks (e.g., the subject in Chunk 1 and the predicate/conclusion in Chunk 2). Neither chunk on its own would contain sufficient semantic context to match a query. A sliding window overlap (e.g., 150 characters) ensures transitional context is preserved in both chunks.
        """)
