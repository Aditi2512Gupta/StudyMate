"""
Gemini API Client Module.
Wraps the Google GenAI SDK (google-genai) for LLM text generation and vector embeddings.
Includes model fallbacks, error handling (rate limits, invalid keys, missing keys),
and a deterministic offline embedding fallback for initial preview mode.
"""

import os
import hashlib
from typing import List, Tuple, Optional
import numpy as np
from google import genai
from google.genai import types
from google.genai.errors import APIError

import config


class GeminiClientError(Exception):
    """Custom exception for Gemini API related failures."""
    pass


def get_client(api_key: Optional[str] = None) -> genai.Client:
    """
    Instantiates a google.genai Client with the provided or configured API key.
    
    Raises:
        GeminiClientError: If no API key is provided or found in the environment.
    """
    key = config.get_api_key(api_key)
    if not key:
        raise GeminiClientError(
            "Gemini API Key is missing. Please enter your API key in the sidebar "
            "or set GEMINI_API_KEY in your .env file. "
            "You can obtain a free key from Google AI Studio: https://aistudio.google.com/app/apikey"
        )
    return genai.Client(api_key=key)


def validate_api_key(api_key: str) -> Tuple[bool, str]:
    """
    Validates a Gemini API key by making a lightweight model call.
    Returns (is_valid: bool, message: str).
    """
    if not api_key or not api_key.strip():
        return False, "API key cannot be empty."
    
    try:
        client = genai.Client(api_key=api_key.strip())
        response = client.models.generate_content(
            model="gemini-2.5-flash",
            contents="Say 'OK'",
            config=types.GenerateContentConfig(max_output_tokens=10, temperature=0.0)
        )
        if response and response.text:
            return True, "API Key is valid and active!"
        return False, "Unexpected empty response from Gemini API."
    except APIError as e:
        return False, f"Gemini API Error: {e.message if hasattr(e, 'message') else str(e)}"
    except Exception as e:
        return False, f"Connection/Validation Error: {str(e)}"


def generate_text(
    prompt: str,
    api_key: Optional[str] = None,
    model: str = config.DEFAULT_LLM_MODEL,
    temperature: float = 0.3,
    max_output_tokens: Optional[int] = None,
    response_mime_type: Optional[str] = None
) -> str:
    """
    Generates text from Gemini using the specified prompt, with automatic fallback models.
    
    Args:
        prompt: User or template prompt string.
        api_key: Gemini API key.
        model: Model name (e.g., 'gemini-2.5-flash').
        temperature: Sampling temperature (0.0 to 1.0).
        max_output_tokens: Token generation limit.
        response_mime_type: e.g., 'application/json' for structured JSON output.
        
    Returns:
        Generated text response string.
    """
    client = get_client(api_key)
    models_to_try = [model] + [m for m in config.FALLBACK_LLM_MODELS if m != model]
    
    config_args = {
        "temperature": temperature,
    }
    if max_output_tokens is not None:
        config_args["max_output_tokens"] = max_output_tokens
    if response_mime_type is not None:
        config_args["response_mime_type"] = response_mime_type

    gen_config = types.GenerateContentConfig(**config_args)

    last_error = None
    for candidate_model in models_to_try:
        try:
            response = client.models.generate_content(
                model=candidate_model,
                contents=prompt,
                config=gen_config
            )
            if response and response.text is not None:
                return response.text.strip()
            raise GeminiClientError("Received empty response from Gemini model.")
        except APIError as e:
            err_msg = str(e)
            last_error = e
            # If model not found or quota exhausted on this model, try fallback
            if "RESOURCE_EXHAUSTED" in err_msg or "429" in err_msg or "NOT_FOUND" in err_msg:
                continue
            raise GeminiClientError(f"Gemini API Error ({candidate_model}): {err_msg}")
        except Exception as e:
            last_error = e
            continue

    raise GeminiClientError(f"Failed to generate response after trying models {models_to_try}. Error: {str(last_error)}")


def generate_embeddings(
    texts: List[str],
    api_key: Optional[str] = None,
    model: str = config.DEFAULT_EMBEDDING_MODEL
) -> np.ndarray:
    """
    Generates dense vector embeddings using Google's text-embedding-004 model.
    Batches texts to comply with API payload limits.
    
    Args:
        texts: List of text strings to embed.
        api_key: Gemini API key.
        model: Embedding model name.
        
    Returns:
        2D numpy array of shape (len(texts), embedding_dimension) with float32 type.
    """
    if not texts:
        return np.empty((0, 768), dtype="float32")

    client = get_client(api_key)
    batch_size = 16  # Safe batch size for text-embedding-004
    all_vectors = []

    for i in range(0, len(texts), batch_size):
        batch = texts[i : i + batch_size]
        try:
            response = client.models.embed_content(
                model=model,
                contents=batch
            )
            if hasattr(response, "embeddings") and response.embeddings:
                for emb in response.embeddings:
                    all_vectors.append(emb.values)
            elif hasattr(response, "embedding") and response.embedding:
                all_vectors.append(response.embedding.values)
            else:
                raise GeminiClientError("Embedding response did not contain expected embedding data.")
        except APIError as e:
            raise GeminiClientError(f"Gemini Embedding API Error: {str(e)}")
        except Exception as e:
            raise GeminiClientError(f"Error during embedding generation: {str(e)}")

    vectors_array = np.array(all_vectors, dtype="float32")
    return vectors_array


def generate_offline_fallback_embeddings(texts: List[str], dimension: int = 768) -> np.ndarray:
    """
    Deterministic offline semantic embedding fallback using TF-IDF and feature hashing.
    Used when no Gemini API key is configured yet, allowing full UI exploration and vector store testing.
    """
    if not texts:
        return np.empty((0, dimension), dtype="float32")

    vectors = np.zeros((len(texts), dimension), dtype="float32")
    for doc_idx, text in enumerate(texts):
        words = text.lower().split()
        for word in words:
            # Hash word into dimension bucket
            h = int(hashlib.md5(word.encode("utf-8")).hexdigest(), 16) % dimension
            vectors[doc_idx, h] += 1.0

        # L2 normalize
        norm = np.linalg.norm(vectors[doc_idx])
        if norm > 0:
            vectors[doc_idx] /= norm

    return vectors
