"""
Módulo: src/domain/state.py
Descrição: Define a estrutura de dados (Estado) que trafega pelo grafo de decisão.
Motivo da alteração: Correção de tipagem (documents) e inclusão de campos faltantes (chat_history).
"""

from typing import List, Any, Optional, Annotated
import operator
from typing_extensions import TypedDict

class AgentState(TypedDict):
    """
    O estado do agente rastreia o fluxo da conversa médica e a tomada de decisão.
    """
    medical_question: str
    
    # Contexto do paciente (se houver)
    context_data: Optional[str]
    
    # Documentos recuperados são objetos do LangChain, não apenas strings
    documents: List[Any] 
    
    generation: str
    
    # Flags de Segurança e Controle
    is_safe: bool
    risk_level: str
    
    # CRÍTICO: Adicionado para manter o histórico e controle de loops
    chat_history: List[Any]
    loop_count: int