"""
PDF Processing & NLP Text Preparation Module.
Handles document ingestion via PyMuPDF (fitz), text cleaning, validation,
and semantic text chunking with sliding window overlap for RAG.
"""

import re
from typing import List, Dict, Any, Tuple
import pymupdf


class PDFProcessingError(Exception):
    """Custom exception raised when PDF processing fails."""
    pass


def clean_text(text: str) -> str:
    """
    NLP text cleaning function:
    - Normalizes Unicode and control characters.
    - Joins words broken by hyphenation at line breaks (e.g. 'algo-\nrithm' -> 'algorithm').
    - Collapses multiple whitespace/newlines into standard paragraph spacing.
    - Preserves sentence structure and punctuation.
    """
    if not text:
        return ""
    
    # Remove control characters except standard whitespace
    text = re.sub(r"[\x00-\x08\x0b\x0c\x0e-\x1f\x7f-\x9f]", "", text)
    
    # Fix hyphenated words at line endings: e.g. "com-\nputer" -> "computer"
    text = re.sub(r"(\b[A-Za-z]+)-\s*\n\s*([A-Za-z]+\b)", r"\1\2", text)
    
    # Replace single line-breaks within sentences with a single space
    # (Leaves double line breaks for paragraph separation)
    lines = text.split("\n")
    cleaned_paragraphs = []
    current_para = []
    
    for line in lines:
        stripped = line.strip()
        if not stripped:
            if current_para:
                cleaned_paragraphs.append(" ".join(current_para))
                current_para = []
        else:
            current_para.append(stripped)
            
    if current_para:
        cleaned_paragraphs.append(" ".join(current_para))
        
    normalized = "\n\n".join(cleaned_paragraphs)
    # Collapse 3+ newlines into 2
    normalized = re.sub(r"\n{3,}", "\n\n", normalized)
    return normalized.strip()


def extract_text_from_pdf(pdf_source: Any) -> Tuple[List[Dict[str, Any]], Dict[str, Any]]:
    """
    Extracts text page-by-page from a PDF file using PyMuPDF.
    
    Args:
        pdf_source: Can be a filepath (str/Path), file-like object, or bytes buffer.
        
    Returns:
        Tuple of:
          - pages: List of dicts [{"page_number": int, "text": str, "char_count": int}]
          - metadata: Dict of document-level metadata (total_pages, total_chars, etc.)
          
    Raises:
        PDFProcessingError: If the file is corrupted, encrypted, or contains no readable text.
    """
    try:
        if isinstance(pdf_source, bytes):
            doc = pymupdf.open(stream=pdf_source, filetype="pdf")
        elif hasattr(pdf_source, "read"):
            # Streamlit UploadedFile or file-like object
            data = pdf_source.read()
            # Reset pointer if possible
            if hasattr(pdf_source, "seek"):
                pdf_source.seek(0)
            doc = pymupdf.open(stream=data, filetype="pdf")
        else:
            doc = pymupdf.open(str(pdf_source))
    except Exception as e:
        raise PDFProcessingError(f"Unable to open PDF document: {str(e)}")

    if doc.is_encrypted:
        doc.close()
        raise PDFProcessingError("The uploaded PDF is password protected or encrypted. Please provide an unencrypted PDF.")

    total_pages = len(doc)
    if total_pages == 0:
        doc.close()
        raise PDFProcessingError("The uploaded PDF document contains 0 pages.")

    pages_data = []
    total_characters = 0

    for page_idx in range(total_pages):
        page_num = page_idx + 1
        page = doc[page_idx]
        raw_text = page.get_text("text") or ""
        cleaned = clean_text(raw_text)
        
        char_count = len(cleaned)
        total_characters += char_count
        
        pages_data.append({
            "page_number": page_num,
            "text": cleaned,
            "raw_text": raw_text,
            "char_count": char_count
        })

    doc.close()

    # Check for empty text (e.g., scanned image-only PDF without OCR)
    if total_characters < 15:
        raise PDFProcessingError(
            "No extractable text found in this PDF. It may be a scanned image document without an OCR text layer. "
            "Please upload a text-based PDF."
        )

    metadata = {
        "total_pages": total_pages,
        "total_characters": total_characters,
        "total_words": sum(len(p["text"].split()) for p in pages_data),
        "estimated_reading_minutes": max(1, round(sum(len(p["text"].split()) for p in pages_data) / 200))
    }

    return pages_data, metadata


def chunk_text(
    pages: List[Dict[str, Any]], 
    chunk_size: int = 800, 
    chunk_overlap: int = 150
) -> List[Dict[str, Any]]:
    """
    NLP sliding-window chunking strategy with overlap.
    Segments document pages into semantically cohesive chunks for vector embedding and FAISS retrieval.
    
    Args:
        pages: Output from extract_text_from_pdf
        chunk_size: Target character length per chunk (default 800 chars)
        chunk_overlap: Character overlap between consecutive chunks (default 150 chars)
        
    Returns:
        List of chunk dicts:
        [{
            "chunk_id": int,
            "page_number": int,
            "text": str,
            "char_count": int,
            "word_count": int
        }, ...]
    """
    if chunk_size <= 0:
        chunk_size = 800
    if chunk_overlap >= chunk_size:
        chunk_overlap = chunk_size // 4

    chunks = []
    chunk_id = 1

    for page in pages:
        page_num = page["page_number"]
        page_text = page["text"]
        
        if not page_text:
            continue

        # Split text primarily by paragraphs to keep semantic units together
        paragraphs = [p.strip() for p in page_text.split("\n\n") if p.strip()]
        
        current_chunk_text = ""
        
        for para in paragraphs:
            # If paragraph itself is larger than chunk_size, split by sentences or fixed window
            if len(para) > chunk_size:
                # Flush existing buffer
                if current_chunk_text:
                    chunks.append({
                        "chunk_id": chunk_id,
                        "page_number": page_num,
                        "text": current_chunk_text.strip(),
                        "char_count": len(current_chunk_text.strip()),
                        "word_count": len(current_chunk_text.strip().split())
                    })
                    chunk_id += 1
                    # Keep overlap from end of flushed chunk
                    overlap_seed = current_chunk_text[-chunk_overlap:] if len(current_chunk_text) > chunk_overlap else ""
                    current_chunk_text = overlap_seed
                
                # Split large paragraph into sentences
                sentences = re.split(r"(?<=[.!?])\s+", para)
                for sentence in sentences:
                    sentence = sentence.strip()
                    if not sentence:
                        continue
                    if len(current_chunk_text) + len(sentence) + 1 <= chunk_size:
                        current_chunk_text = (current_chunk_text + " " + sentence).strip()
                    else:
                        if current_chunk_text:
                            chunks.append({
                                "chunk_id": chunk_id,
                                "page_number": page_num,
                                "text": current_chunk_text.strip(),
                                "char_count": len(current_chunk_text.strip()),
                                "word_count": len(current_chunk_text.strip().split())
                            })
                            chunk_id += 1
                            overlap_seed = current_chunk_text[-chunk_overlap:] if len(current_chunk_text) > chunk_overlap else ""
                            current_chunk_text = (overlap_seed + " " + sentence).strip()
                        else:
                            # Single sentence larger than chunk_size: hard window split
                            start = 0
                            while start < len(sentence):
                                sub = sentence[start : start + chunk_size]
                                chunks.append({
                                    "chunk_id": chunk_id,
                                    "page_number": page_num,
                                    "text": sub.strip(),
                                    "char_count": len(sub.strip()),
                                    "word_count": len(sub.strip().split())
                                })
                                chunk_id += 1
                                start += (chunk_size - chunk_overlap)
                            current_chunk_text = ""
            else:
                # Standard paragraph
                candidate = (current_chunk_text + "\n\n" + para).strip() if current_chunk_text else para
                if len(candidate) <= chunk_size:
                    current_chunk_text = candidate
                else:
                    # Save current chunk
                    if current_chunk_text:
                        chunks.append({
                            "chunk_id": chunk_id,
                            "page_number": page_num,
                            "text": current_chunk_text.strip(),
                            "char_count": len(current_chunk_text.strip()),
                            "word_count": len(current_chunk_text.strip().split())
                        })
                        chunk_id += 1
                        overlap_seed = current_chunk_text[-chunk_overlap:] if len(current_chunk_text) > chunk_overlap else ""
                        current_chunk_text = (overlap_seed + "\n\n" + para).strip()
                    else:
                        current_chunk_text = para

        # Flush any remaining text on this page
        if current_chunk_text.strip():
            chunks.append({
                "chunk_id": chunk_id,
                "page_number": page_num,
                "text": current_chunk_text.strip(),
                "char_count": len(current_chunk_text.strip()),
                "word_count": len(current_chunk_text.strip().split())
            })
            chunk_id += 1

    return chunks
