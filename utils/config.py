import streamlit as st
import os
from typing import Dict, Any

def setup_page_config() -> None:
    """Configure the Streamlit page settings."""
    st.set_page_config(
        page_title="Mark Musk - CreditChek API Assistant",
        page_icon="🤖",
        layout="wide",
        initial_sidebar_state="auto"
    )

def get_openai_config() -> Dict[str, Any]:
    """Get OpenAI configuration from environment variables."""
    return {
        "api_key": os.getenv("OPENAI_API_KEY"),
        "model": os.getenv("OPENAI_MODEL", "gpt-4-turbo-preview"),
        "temperature": float(os.getenv("OPENAI_TEMPERATURE", "0.7")),
        "max_tokens": int(os.getenv("OPENAI_MAX_TOKENS", "2000"))
    }

def get_pinecone_config() -> Dict[str, str]:
    """Get Pinecone configuration from environment variables."""
    return {
        "api_key": os.getenv("PINECONE_API_KEY"),
        "environment": os.getenv("PINECONE_ENVIRONMENT"),
        "index_name": os.getenv("PINECONE_INDEX_NAME", "creditchek-docs")
    }

def get_app_config() -> Dict[str, Any]:
    """Get application configuration."""
    return {
        "doc_url": "https://docs.creditchek.africa",
        "supported_languages": [
            "Python",
            "NodeJS",
            "PHP Laravel",
            "GoLang"
        ],
        "chunk_size": 1000,
        "chunk_overlap": 200
    } 