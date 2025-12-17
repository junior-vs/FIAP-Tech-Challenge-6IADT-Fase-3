"""
Módulo: src/use_cases/graph.py
Descrição: Define o grafo de execução (workflow) do assistente.
Motivo da alteração: Implementação do fluxo linear com verificação de segurança (Guardrails) 
e validação de resposta, orquestrando os nós criados anteriormente.
"""

import logging
from langgraph.graph import StateGraph
from src.domain.state import AgentState
from src.use_cases.nodes import RAGNodes

logger = logging.getLogger(__name__)


class GraphBuilder:
    """Constrói o grafo de orquestração RAG com suporte multilíngue."""
    
    def __init__(self):
        self.nodes = RAGNodes()
    
    def build(self):
        """Constrói e retorna o grafo de execução."""
        workflow = StateGraph(AgentState)
        
        # Adicionar nós (com nós de tradução)
        workflow.add_node("detect_language", self._log_node("detect_language", self.nodes.detect_language))
        workflow.add_node("translate_to_en", self._log_node("translate_to_en", self.nodes.translate_question_to_english))
        workflow.add_node("guardrails", self._log_node("guardrails", self.nodes.guardrails_check))
        workflow.add_node("retrieve", self._log_node("retrieve", self.nodes.retrieve))
        workflow.add_node("grade", self._log_node("grade", self.nodes.grade_documents))
        workflow.add_node("generate", self._log_node("generate", self.nodes.generate))
        workflow.add_node("translate_response", self._log_node("translate_response", self.nodes.translate_response_to_original_language))
        workflow.add_node("validate", self._log_node("validate", self.nodes.validate_hallucination))
        
        # Definir edges (com nós de tradução)
        workflow.add_edge("detect_language", "translate_to_en")
        workflow.add_edge("translate_to_en", "guardrails")
        workflow.add_edge("guardrails", "retrieve")
        workflow.add_edge("retrieve", "grade")
        workflow.add_edge("grade", "generate")
        workflow.add_edge("generate", "translate_response")
        workflow.add_edge("translate_response", "validate")
        
        # Ponto de entrada
        workflow.set_entry_point("detect_language")
        
        # Ponto final
        workflow.set_finish_point("validate")
        
        logger.info("✅ Grafo RAG com tradução construído")
        
        return workflow.compile()
    
    def _log_node(self, node_name: str, node_fn):
        """Wrapper que loga entrada e saída."""
        def wrapper(state: AgentState) -> dict:
            logger.debug(f"▶️ Nó: {node_name}")
            result = node_fn(state)
            logger.debug(f"◀️ Saída: {list(result.keys())}")
            return result
        return wrapper