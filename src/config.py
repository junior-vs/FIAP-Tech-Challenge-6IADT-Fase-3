"""
Módulo: src/config.py
Descrição: Gerenciamento centralizado de configurações e variáveis de ambiente.
Motivo da alteração: Substituição de URLs de livros por caminhos de diretórios locais de protocolos médicos.
"""

from dotenv import load_dotenv
from pydantic_settings import BaseSettings, SettingsConfigDict

# Load environment variables from .env file
load_dotenv()

class Settings(BaseSettings):
    gemini_api_key: str
    model_name: str = "gemini-1.5-flash" # Atualizado para um modelo mais recente/rápido se disponível
    temperature: float = 0.0
    
    # Configurações de RAG
    chunk_size: int = 1000
    chunk_overlap: int = 200
    
    # Caminhos do Sistema Médico
    docs_path: str = "docs/knowledge_base"  # Pasta onde ficarão os PDFs/MDs
    vector_db_path: str = "data/chroma_db"  # Pasta onde o ChromaDB salvará os índices
    
    # Remover configurações antigas de URL se não forem mais usadas
    # book_url: str = ... (Removido)
    # storage_path: str = ... (Removido)

    model_config = SettingsConfigDict(env_file=".env", extra="ignore")

settings = Settings()