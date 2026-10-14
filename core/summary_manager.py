"""
Summary Manager Module.
Generates structured, academic summaries from extracted PDF text using Gemini LLM.
Supports Key Takeaways, Executive Summaries, and Exam Cheat-Sheets.
"""

from typing import List, Dict, Any, Optional
from core import prompts
from core import gemini_client
import config


def prepare_document_text(pages: List[Dict[str, Any]], max_chars: int = 40000) -> str:
    """
    Assembles extracted page texts into a unified document string with page markers.
    Trims if the document exceeds safe single-call context limits while preserving distribution.
    """
    total_len = sum(p.get("char_count", len(p.get("text", ""))) for p in pages)
    
    if total_len <= max_chars:
        # Include all pages with clear page headers
        assembled = []
        for p in pages:
            page_num = p["page_number"]
            text = p["text"].strip()
            if text:
                assembled.append(f"--- [Page {page_num}] ---\n{text}")
        return "\n\n".join(assembled)

    # For very large documents, sample proportionally across beginning, middle, and end
    sampled_pages = []
    step = max(1, len(pages) // 10)
    for i in range(0, len(pages), step):
        p = pages[i]
        sampled_pages.append(f"--- [Page {p['page_number']}] ---\n{p['text'][:2500]}")
    
    return "\n\n".join(sampled_pages)


def generate_summary(
    pages: List[Dict[str, Any]],
    summary_style: str = "key_takeaways",
    api_key: Optional[str] = None,
    model: str = config.DEFAULT_LLM_MODEL
) -> str:
    """
    Generates a structured academic summary using Gemini LLM and prompt engineering templates.
    
    Args:
        pages: List of page dictionaries extracted from PDF.
        summary_style: One of 'key_takeaways', 'executive', 'exam_cheatsheet'.
        api_key: Gemini API key.
        model: Gemini model identifier.
        
    Returns:
        Markdown formatted summary string.
    """
    if not pages:
        raise ValueError("Cannot summarize empty document. Please upload a valid PDF first.")

    doc_text = prepare_document_text(pages)
    if not doc_text.strip():
        raise ValueError("Document has no readable text to summarize.")

    template = prompts.SUMMARY_TEMPLATES.get(summary_style, prompts.SUMMARY_TEMPLATES["key_takeaways"])
    formatted_prompt = template.format(document_text=doc_text)

    summary = gemini_client.generate_text(
        prompt=formatted_prompt,
        api_key=api_key,
        model=model,
        temperature=config.TEMPERATURE_SUMMARY,
        max_output_tokens=2500
    )

    return summary
