"""
Módulo: src/config.py
"""
from dotenv import load_dotenv
from pydantic_settings import BaseSettings
from pathlib import Path
import logging

load_dotenv()

logger = logging.getLogger(__name__)

class Settings(BaseSettings):
    """Configurações centralizadas da aplicação médica."""
    
    # ===== LLM Configuration =====
    gemini_api_key: str              # Obrigatório (sem default)
    model_name: str = "gemini-2.0-flash"                  # Obrigatório (sem default)
    temperature: float = 0.0               # Obrigatório (sem default)
    
    # ===== Vector Store Configuration (Chroma) =====
    vector_db_path: str = "data/chroma_db"
    chunk_size: int = 1000
    chunk_overlap: int = 200
    
    # ===== Knowledge Base =====
    docs_path: str = "docs/knowledge_base/7_SeniorHealth_QA"
    
    # ===== Cache Configuration =====
    redis_url: str = "redis://localhost:6379"
    cache_ttl: int = 3600
    
    class Config:
        env_file = ".env"
        case_sensitive = False
        extra = "forbid"
    
    @property
    def docs_full_path(self) -> Path:
        """Retorna caminho completo para base de conhecimento com validação."""
        path = Path(self.docs_path).resolve()
        if not path.exists():
            logger.warning(f"⚠️ Base de conhecimento não encontrada em: {path}")
            logger.info("📂 Criando diretório...")
            path.mkdir(parents=True, exist_ok=True)
        return path
    
    @property
    def vector_db_full_path(self) -> Path:
        """Retorna caminho completo para vector DB com criação automática."""
        path = Path(self.vector_db_path).resolve()
        path.mkdir(parents=True, exist_ok=True)
        return path


try:
    settings = Settings()  # type: ignore
except Exception as e:
    logger.error(f"❌ Erro ao carregar configurações:\n{e}")
    raise RuntimeError(
        f"❌ Erro ao carregar configurações:\n{e}\n"
        f"Verifique .env e remova campos legados (book_url, storage_path, faiss_index_path)"
    )