"""
Prompt Engineering Module for StudyMate.
Contains carefully designed, structured prompt templates for:
- Academic document summarization across multiple study formats
- Strict context-grounded RAG Question & Answering with source citations
- Structured Multiple-Choice Quiz (MCQ) generation with JSON output validation
"""

# Base System Instruction for StudyMate
SYSTEM_INSTRUCTION = """You are StudyMate, an expert academic AI tutor designed for students and researchers.
Your mission is to provide clear, pedagogically sound, and strictly grounded explanations based on study materials.
You value factual accuracy, intellectual rigor, clear conceptual clarity, and structured formatting."""

# ---------------------------------------------------------------------------
# Summarization Prompt Templates
# ---------------------------------------------------------------------------

SUMMARY_TEMPLATES = {
    "key_takeaways": """You are an academic study assistant.
Generate a structured, high-yield summary of the provided study material.

Guidelines:
1. Start with a 2-3 sentence Core Theme overview.
2. Provide Key Takeaways organized as bullet points grouped under logical conceptual headings.
3. Highlight critical terminology, definitions, and acronyms in **bold**.
4. Conclude with a "Quick Review Tip" that helps a student remember the main idea.

Format the output in clean, readable Markdown.

--- STUDY MATERIAL ---
{document_text}
--- END OF MATERIAL ---

Generate the Key Takeaways Summary now:""",

    "executive": """You are an academic summarization engine.
Synthesize the provided study material into an Executive / Conceptual Study Guide.

Guidelines:
1. Structure the summary with:
   - **Executive Abstract**: High-level synthesis of what this material covers.
   - **Core Concepts & Theoretical Frameworks**: Detailed paragraphs explaining the fundamental mechanisms.
   - **Real-World Applications & Significance**: Why these concepts matter in practical engineering or scientific contexts.
   - **Key Takeaways & Synthesized Conclusions**.
2. Maintain high academic standards, logical flow, and professional tone.

Format the output in clean, readable Markdown.

--- STUDY MATERIAL ---
{document_text}
--- END OF MATERIAL ---

Generate the Executive Study Guide now:""",

    "exam_cheatsheet": """You are an expert university exam tutor.
Convert the provided study material into a high-yield "Exam Revision Cheat-Sheet".

Guidelines:
1. **Essential Definitions**: Bullet list of key terms and their precise academic definitions.
2. **Core Principles / Theorems / Formulas**: Any formulas, algorithmic steps, or fundamental laws mentioned.
3. **Common Pitfalls & Distinctions**: Highlighting easily confused concepts (e.g., Overfitting vs Underfitting, Precision vs Recall).
4. **Top 3 Potential Exam Questions & Model Answers**: Short, high-probability exam questions based directly on this text.

Format the output in clean, readable Markdown with tables or bullet lists where appropriate.

--- STUDY MATERIAL ---
{document_text}
--- END OF MATERIAL ---

Generate the Exam Revision Cheat-Sheet now:"""
}


# ---------------------------------------------------------------------------
# RAG Q&A Grounded Prompt Template
# ---------------------------------------------------------------------------

RAG_QA_PROMPT_TEMPLATE = """You are StudyMate's RAG (Retrieval-Augmented Generation) Academic Assistant.
Your task is to answer the student's question accurately and objectively, grounded STRICTLY in the retrieved document chunks below.

### STRICT RULES & GUARDRAILS:
1. **Context Grounding**: Rely SOLELY on the information presented in the "RETRIEVED CONTEXT CHUNKS" section.
2. **No Hallucinations**: Do NOT assume, extrapolate, or bring in external knowledge not verified in the context.
3. **Insufficient Information**: If the provided chunks do not contain enough information to answer the question, state:
   "Based on the provided document excerpts, there is insufficient information to answer this question." Do not attempt to guess.
4. **Citations**: Whenever referencing specific facts, explicitly mention the source page (e.g., "[Page X]").
5. **Tone**: Educational, encouraging, and clear. Use bullet points or code formatting when explaining technical concepts.

--- RETRIEVED CONTEXT CHUNKS ---
{context_chunks}
--- END OF RETRIEVED CONTEXT ---

STUDENT'S QUESTION:
{question}

Provide your grounded answer below:"""


# ---------------------------------------------------------------------------
# Multiple Choice Quiz (MCQ) Generation Prompt Template
# ---------------------------------------------------------------------------

QUIZ_GENERATION_PROMPT_TEMPLATE = """You are an academic assessment designer.
Generate a high-quality {num_questions}-question Multiple-Choice Quiz (MCQ) based STRICTLY on the provided study content.

Difficulty Level: {difficulty}
Target Topic: {topic}

### JSON FORMAT REQUIREMENT:
You must respond with a VALID JSON array containing exactly {num_questions} objects.
Do NOT include any preamble, conversational text, or explanation outside the JSON array.

Each question object must adhere to this exact schema:
[
  {{
    "question": "Clear, grammatically correct question text?",
    "options": {{
      "A": "Option text A",
      "B": "Option text B",
      "C": "Option text C",
      "D": "Option text D"
    }},
    "correct_answer": "A",
    "explanation": "Clear explanation of why this option is correct and others are incorrect based on the text.",
    "reference_topic": "Specific subtopic or chapter referenced"
  }}
]

### ASSESSMENT QUALITY GUIDELINES:
1. Every question must be directly answerable from the provided text.
2. Plausible distractors: The incorrect options (distractors) should be believable but unambiguously false.
3. Randomize the distribution of correct letters (A, B, C, D) so the answer is not always the same choice.
4. Ensure the output is 100% valid JSON that can be parsed by `json.loads()`.

--- SOURCE STUDY MATERIAL ---
{document_text}
--- END OF MATERIAL ---

Generate the JSON array of {num_questions} questions now:"""
