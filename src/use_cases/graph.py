"""
Módulo: src/use_cases/graph.py
Descrição: Define o grafo de execução (workflow) do assistente.
Motivo da alteração: Implementação do fluxo linear com verificação de segurança (Guardrails) 
e validação de resposta, orquestrando os nós criados anteriormente.
"""

from langgraph.graph import END, StateGraph

# Importações dos nossos módulos
from src.domain.state import AgentState
from src.use_cases.nodes import RAGNodes
from src.infrastructure.vector_store import VectorStoreRepository
from src.utils.logging import get_logger

logger = get_logger()

class GraphBuilder:
    def __init__(self):
        # Inicializa o repositório e os nós
        # O VectorStoreRepository já carrega os XMLs e prepara o retriever
        self.vector_store = VectorStoreRepository()
        self.nodes = RAGNodes(self.vector_store.get_retriever())

    def build(self):
        """
        Constrói e compila o grafo de decisão do Assistente Médico.
        """
        # Define que o estado que trafega no grafo é o nosso AgentState
        workflow = StateGraph(AgentState)

        # --- 1. Adicionando os Nós (Nodes) ---
        workflow.add_node("guardrails", self.nodes.guardrails_check)
        workflow.add_node("retrieve", self.nodes.retrieve)
        workflow.add_node("grade", self.nodes.grade_documents)
        workflow.add_node("generate", self.nodes.generate)
        workflow.add_node("validate", self.nodes.validate_generation)

        # --- 2. Definindo o Fluxo (Edges) ---
        
        # Ponto de partida: Sempre verifica a segurança primeiro
        workflow.set_entry_point("guardrails")

        # Lógica Condicional: Onde ir depois do Guardrail?
        def router_check_safety(state: AgentState):
            # Se is_safe for True, segue para buscar documentos
            if state.get("is_safe"):
                return "go_to_retrieve"
            # Se não, encerra o fluxo imediatamente (já temos a msg de recusa)
            return "stop"

        workflow.add_conditional_edges(
            "guardrails",
            router_check_safety,
            {
                "go_to_retrieve": "retrieve",
                "stop": END
            }
        )

        # Fluxo Linear (Caminho Feliz)
        # Busca -> Avalia Relevância -> Gera Resposta -> Valida Alucinação -> Fim
        workflow.add_edge("retrieve", "grade")
        workflow.add_edge("grade", "generate")
        workflow.add_edge("generate", "validate")
        workflow.add_edge("validate", END)

        # Compila o grafo para execução
        app = workflow.compile()
        return app