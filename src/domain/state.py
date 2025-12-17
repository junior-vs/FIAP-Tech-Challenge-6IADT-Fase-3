from typing import List, Any, Optional, Tuple
from typing_extensions import TypedDict


"""
Módulo: src/domain/state.py
Descrição: Define a estrutura de dados (Estado) que trafega pelo grafo de decisão.
Motivo da alteração: Adaptação para contexto médico e inclusão de flags de segurança.
"""

class AgentState(TypedDict):
    """
    O estado do agente rastreia o fluxo da conversa médica e a tomada de decisão.
    
    Atributos:
        medical_question (str): A pergunta clínica feita pelo usuário (médico/enfermeiro).
        context_data (Optional[str]): Informações anonimizadas do paciente, se houver.
        
        documents (List[str]): Lista de trechos de protocolos/bulas recuperados pelo RAG.
        
        generation (str): A resposta gerada pelo LLM.
        
        is_safe (bool): Flag de segurança. True se a pergunta não contém dados sensíveis (PII) 
                        ou se já foi anonimizada. False se precisa bloquear.
                        
        risk_level (str): Classificação de risco da pergunta (ex: 'informativo', 'emergencia').
                          Útil para direcionar para fluxos urgentes.
    """
    medical_question: str
    context_data: Optional[str]
    documents: List[str]
    generation: str
    is_safe: bool
    risk_level: str