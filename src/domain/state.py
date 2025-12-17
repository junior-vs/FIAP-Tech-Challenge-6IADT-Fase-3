"""
Módulo: src/domain/state.py
Descrição: Define a estrutura de dados (Estado) que trafega pelo grafo de decisão.
Motivo da alteração: Correção de tipagem (documents) e inclusão de campos faltantes (chat_history).
"""

from typing import TypedDict, List, Optional
from langchain_core.documents import Document

class AgentState(TypedDict):
    """
    Estado central da aplicação que flui através do grafo RAG.
    
    WHEN [pergunta médica é submetida]
    THE SYSTEM SHALL [manter estado consistente através de todos os nós]
    """
    
    # Input
    medical_question: str  # Pergunta do usuário (idioma original)
    language: str  # Idioma detectado: "pt" ou "en"
    
    # Processing
    medical_question_en: str  # Pergunta traduzida para inglês (para busca)
    is_safe: bool  # Passou na validação de guardrails
    documents: List[Document]  # Documentos recuperados
    
    # Output
    generation: str  # Resposta em inglês (antes de tradução final)
    generation_final: str  # Resposta final no idioma original
    hallucination_check: str  # Resultado da validação