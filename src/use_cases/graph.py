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
from src.infrastructure.llm_factory import LLMFactory
from src.infrastructure.vector_store import VectorStoreRepository

logger = logging.getLogger(__name__)

class GraphBuilder:
    """Constrói o grafo de orquestração RAG com LangGraph."""
    
    def __init__(self):
        # Inicializar dependências necessárias
        self.llm = LLMFactory.get_llm()
        vector_repo = VectorStoreRepository()
        self.retriever = vector_repo.get_retriever()
        self.nodes = RAGNodes(retriever=self.retriever, llm=self.llm)
    
    def build(self):
        """Constrói e retorna o grafo de execução."""
        workflow = StateGraph(AgentState)
        
        # Adicionar nós
        workflow.add_node("guardrails", self.nodes.guardrails)
        workflow.add_node("retrieve", self.nodes.retrieve)
        workflow.add_node("grade", self.nodes.grade_documents)
        workflow.add_node("generate", self.nodes.generate)
        workflow.add_node("validate", self.nodes.validate_response)
        
        # Definir edges com roteamento condicional
        workflow.add_edge("guardrails", "retrieve")
        workflow.add_edge("retrieve", "grade")
        workflow.add_edge("grade", "generate")
        workflow.add_edge("generate", "validate")
        
        # Ponto de entrada
        workflow.set_entry_point("guardrails")
        
        # Ponto final
        workflow.set_finish_point("validate")
        
        logger.info("✅ Grafo RAG construído com sucesso")
        
        return workflow.compile()