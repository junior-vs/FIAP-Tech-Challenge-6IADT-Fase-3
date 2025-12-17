#!/usr/bin/env python3
"""
Script para inicializar o sistema de assistência médica.
Configura Chroma com protocolos médicos internos.
"""

import sys
from pathlib import Path

project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))

from src.config import settings
from src.infrastructure.vector_store import VectorStoreRepository
from src.infrastructure.llm_factory import LLMFactory


def main():
    print("🔧 Inicializando Sistema de Assistência Médica...\n")
    
    try:
        # Validar configurações
        print("✅ Verificando configurações...")
        if not settings.gemini_api_key:
            raise ValueError("GEMINI_API_KEY não configurada")
        print("✅ GEMINI_API_KEY encontrada\n")
        
        # Testar LLM
        print("🤖 Testando conexão com Google Gemini...")
        llm = LLMFactory.get_llm()
        test_response = llm.invoke("Teste de conexão")
        print("✅ Conexão com Google Gemini estabelecida\n")
        
        # Inicializar vectorstore
        print("📚 Inicializando Vectorstore...")
        vector_repo = VectorStoreRepository()
        retriever = vector_repo.get_retriever()
        print("✅ Vectorstore inicializado com sucesso\n")
        
        # Teste de retrieval
        print("🔍 Testando busca vetorial...")
        test_docs = retriever.invoke("sepse em idosos")
        print(f"✅ Teste OK: {len(test_docs)} documentos encontrados\n")
        
        print("=" * 60)
        print("✅ INICIALIZAÇÃO CONCLUÍDA COM SUCESSO!")
        print("=" * 60)
        print("\nPróxima etapa:")
        print("  .venv/bin/python src/main.py")
        print("\n")
        
    except Exception as e:
        print(f"\n❌ ERRO durante inicialização: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()
