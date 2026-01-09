"""
Módulo: src/infrastructure/llm_factory.py

Factory para criar e gerenciar instâncias de LLM e embeddings do Google Gemini.
Suporta modelos base e modelos fine-tuned.
"""

import logging
import os
from typing import Optional
from langchain_google_genai import ChatGoogleGenerativeAI, GoogleGenerativeAIEmbeddings
from src.config import settings
from pathlib import Path

try:
    from src.infrastructure.local_llama import load_finetuned
except Exception:
    load_finetuned = None

logger = logging.getLogger(__name__)


class LLMFactory:
    """
    Factory para criar instâncias de LLM e embeddings.
    
    Mantém instâncias singleton para evitar múltiplas inicializações.
    Suporta mudança entre modelos base e fine-tuned.
    """
    
    _llm_instance = None
    _embeddings_instance = None
    _current_model = None
    
    @classmethod
    def get_llm(cls, model_name: Optional[str] = None) -> ChatGoogleGenerativeAI:
        """
        Retorna instância do LLM.
        
        Args:
            model_name: Nome do modelo a usar. Se não fornecido, usa settings.model_name.
                       Exemplos: "gemini-2.0-flash", "tunedModels/..."
        
        Returns:
            Instância de ChatGoogleGenerativeAI
        """
        model_to_use = model_name or settings.model_name
        
        # Reseta se o modelo mudou
        if cls._current_model != model_to_use:
            cls._llm_instance = None
            cls._current_model = model_to_use
        
        if cls._llm_instance is None:
            # Suporte para modelo local (transformers fine-tuned)
            if isinstance(model_to_use, str) and model_to_use.startswith("local-llama:"):
                local_path = model_to_use.split("local-llama:", 1)[1].strip()
                logger.info(f"🤖 Carregando modelo local LLaMA em: {local_path}")
                if load_finetuned is None:
                    raise RuntimeError("local_llama não está disponível. Instale 'transformers' e carregue o módulo.")

                model_path = Path(local_path)
                if not model_path.exists():
                    raise FileNotFoundError(f"Modelo local não encontrado em: {model_path}")

                try:
                    cls._llm_instance = load_finetuned(str(model_path))
                    logger.info(f"✅ Modelo local carregado: {model_path}")
                except Exception as e:
                    logger.error(f"❌ Erro ao carregar modelo local: {e}")
                    raise
            else:
                logger.info(f"🤖 Inicializando {model_to_use} (remoto)...")
                try:
                    cls._llm_instance = ChatGoogleGenerativeAI(
                        model=model_to_use,
                        google_api_key=settings.gemini_api_key,
                        temperature=settings.temperature,
                        top_p=0.95,
                        max_output_tokens=2048,
                        request_timeout=settings.request_timeout,
                        retries=settings.retries,
                    )
                    
                    # Identifica se é modelo fine-tuned
                    is_fine_tuned = model_to_use.startswith("tunedModels/")
                    model_type = "fine-tuned" if is_fine_tuned else "base"
                    
                    logger.info(f"✅ {model_to_use} ({model_type}) inicializado com sucesso")
                except Exception as e:
                    logger.error(f"❌ Erro ao inicializar LLM: {e}")
                    logger.info(
                        "💡 Dica: Verifique se o modelo está disponível executando:\n"
                        "  python -c \"import google.generativeai as genai; "
                        "[print(m.name) for m in genai.list_tuned_models()]\""
                    )
                    raise
        
        return cls._llm_instance
    
    @classmethod
    def get_embeddings(cls) -> GoogleGenerativeAIEmbeddings:
        """
        Retorna instância singleton de embeddings.
        
        Returns:
            Instância de GoogleGenerativeAIEmbeddings
        """
        if cls._embeddings_instance is None:
            logger.info("📊 Inicializando embeddings...")
            try:
                cls._embeddings_instance = GoogleGenerativeAIEmbeddings(
                    model="models/embedding-001",
                    google_api_key=settings.gemini_api_key,
                    request_options={"timeout": settings.request_timeout},
                )
                logger.info("✅ Embeddings inicializados com sucesso")
            except Exception as e:
                logger.error(f"❌ Erro ao inicializar embeddings: {e}")
                raise
        
        return cls._embeddings_instance
    
    @classmethod
    def switch_model(cls, model_name: str) -> ChatGoogleGenerativeAI:
        """
        Muda o modelo LLM em tempo de execução.
        
        Útil para testar diferentes modelos ou trocar de modelo base para fine-tuned.
        
        Args:
            model_name: Nome do novo modelo (ex: "tunedModels/medical-assistant-abc123")
        
        Returns:
            Nova instância do LLM com o modelo especificado
        
        Example:
            >>> factory = LLMFactory()
            >>> llm = factory.switch_model("tunedModels/my-fine-tuned-model")
        """
        logger.info(f"🔄 Trocando modelo para: {model_name}")
        cls._llm_instance = None
        cls._current_model = None
        return cls.get_llm(model_name)
    
    @classmethod
    def get_current_model(cls) -> str:
        """Retorna o nome do modelo atualmente em uso."""
        return cls._current_model or settings.model_name
    
    @classmethod
    def reset(cls):
        """
        Reseta instâncias singleton (útil para testes).
        
        Força a reinicialização de LLM e embeddings na próxima chamada.
        """
        logger.debug("🔄 Resetando instâncias de LLM e embeddings...")
        cls._llm_instance = None
        cls._embeddings_instance = None
        cls._current_model = None
    
    @classmethod
    def list_available_tuned_models(cls) -> list[str]:
        """
        Lista todos os modelos fine-tuned disponíveis.
        
        Returns:
            Lista com nomes dos modelos fine-tuned
        """
        try:
            import google.generativeai as genai
            genai.configure(api_key=settings.gemini_api_key)
            
            tuned_models = []
            for model in genai.list_tuned_models():
                tuned_models.append(model.name)
            
            return tuned_models
        except Exception as e:
            logger.error(f"Erro ao listar modelos fine-tuned: {e}")
            return []