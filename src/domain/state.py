"""
Módulo: src/domain/state.py
Descrição: Define a estrutura de dados (Estado) que trafega pelo grafo de decisão.
Motivo da alteração: Correção de tipagem e inclusão de campos para risco e validação.
"""

from typing import Any, List, Optional, TypedDict
from langchain_core.documents import Document


class AgentState(TypedDict):
    """
    Estado do agente RAG para manter contexto durante processamento.
    Todas as chaves são obrigatórias exceto quando marcadas como Optional.
    """
    medical_question: str
    
    # Contexto do paciente (se houver)
    context_data: Optional[str]
    
    # Documentos recuperados são objetos do LangChain
    documents: List[Document] 
    
    generation: str
    
    # Flags de Segurança e Controle
    is_safe: bool
    risk_level: Optional[str]  # Campo para nível de risco
    
    # Controle de histórico e loops
    chat_history: Optional[List[Any]]
    loop_count: Optional[int]
    
    # Campos para validação de alucinação
    is_valid: Optional[bool]
    hallucination_check: Optional[str]