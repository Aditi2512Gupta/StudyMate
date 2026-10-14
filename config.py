"""
Configuration module for StudyMate – AI-Powered Study Assistant.
Handles environment variables, default model settings, chunking parameters, and app metadata.
"""

import os
from pathlib import Path
from dotenv import load_dotenv

# Base paths
BASE_DIR = Path(__file__).resolve().parent

# Load .env file if present
load_dotenv(BASE_DIR / ".env")

# App Metadata
APP_NAME = "StudyMate"
APP_SUBTITLE = "AI-Powered Study Assistant"
APP_VERSION = "1.0.0"
PROJECT_TAG = "Infosys Summer Internship Project"
APP_ICON = "📚"

# Gemini LLM Models
# Primary recommended fast reasoning model
DEFAULT_LLM_MODEL = os.getenv("DEFAULT_LLM_MODEL", "gemini-2.5-flash")
FALLBACK_LLM_MODELS = ["gemini-2.0-flash", "gemini-1.5-flash"]

# Gemini Embedding Model
DEFAULT_EMBEDDING_MODEL = os.getenv("DEFAULT_EMBEDDING_MODEL", "text-embedding-004")

# NLP & Chunking Defaults
DEFAULT_CHUNK_SIZE = int(os.getenv("DEFAULT_CHUNK_SIZE", 800))
DEFAULT_CHUNK_OVERLAP = int(os.getenv("DEFAULT_CHUNK_OVERLAP", 150))

# Vector Search / Retrieval Defaults
DEFAULT_TOP_K = int(os.getenv("DEFAULT_TOP_K", 4))
SIMILARITY_THRESHOLD = float(os.getenv("SIMILARITY_THRESHOLD", 0.25))

# Temperature Defaults
TEMPERATURE_SUMMARY = 0.2
TEMPERATURE_QA = 0.3
TEMPERATURE_QUIZ = 0.2

def get_api_key(session_key: str | None = None) -> str:
    """
    Retrieve Gemini API key with priority:
    1. Key explicitly provided in the UI session
    2. Environment variable GEMINI_API_KEY from .env or OS
    """
    if session_key and session_key.strip():
        return session_key.strip()
    
    env_key = os.getenv("GEMINI_API_KEY", "").strip()
    return env_key
