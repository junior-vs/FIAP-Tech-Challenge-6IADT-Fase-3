import logging
from langchain_google_genai import ChatGoogleGenerativeAI, GoogleGenerativeAIEmbeddings
from src.config import settings

logger = logging.getLogger(__name__)


class LLMFactory:
    """Factory para criar instâncias de LLM e embeddings."""
    
    _llm_instance = None
    _embeddings_instance = None
    
    @classmethod
    def get_llm(cls) -> ChatGoogleGenerativeAI:
        """Retorna instância singleton do LLM."""
        if cls._llm_instance is None:
            logger.info(f"🤖 Inicializando {settings.model_name}...")
            try:
                cls._llm_instance = ChatGoogleGenerativeAI(
                    model=settings.model_name,
                    google_api_key=settings.gemini_api_key,
                    temperature=settings.temperature,
                    top_p=0.95,
                    max_output_tokens=2048,
                )
                logger.info(f"✅ {settings.model_name} inicializado com sucesso")
            except Exception as e:
                logger.error(f"❌ Erro ao inicializar LLM: {e}")
                logger.info(
                    "💡 Dica: Verifique se o modelo está disponível executando:\n"
                    "  .venv/bin/python -c "
                    '"import google.genai as genai; [print(m.id) for m in genai.models.list()]"'
                )
                raise
        
        return cls._llm_instance
    
    @classmethod
    def get_embeddings(cls) -> GoogleGenerativeAIEmbeddings:
        """Retorna instância singleton de embeddings."""
        if cls._embeddings_instance is None:
            logger.info("📊 Inicializando embeddings...")
            try:
                cls._embeddings_instance = GoogleGenerativeAIEmbeddings(
                    model="models/embedding-001",
                    google_api_key=settings.gemini_api_key,
                )
                logger.info("✅ Embeddings inicializados com sucesso")
            except Exception as e:
                logger.error(f"❌ Erro ao inicializar embeddings: {e}")
                raise
        
        return cls._embeddings_instance
    
    @classmethod
    def reset(cls):
        """Reseta instâncias (útil para testes)."""
        cls._llm_instance = None
        cls._embeddings_instance = None