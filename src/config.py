"""
Módulo: src/config.py
"""
from dotenv import load_dotenv
from pydantic_settings import BaseSettings, SettingsConfigDict

load_dotenv()

class Settings(BaseSettings):
    gemini_api_key: str
    model_name: str = "gemini-1.5-flash"
    temperature: float = 0.0
    
    chunk_size: int = 1000
    chunk_overlap: int = 200
    
    # CORREÇÃO CRÍTICA AQUI: Adicionado '/data' ao caminho
    docs_path: str = "docs/data/knowledge_base"  
    
    vector_db_path: str = "data/chroma_db"

    model_config = SettingsConfigDict(env_file=".env", extra="ignore")

settings = Settings() # type: ignore