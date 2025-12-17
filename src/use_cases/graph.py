from langgraph.graph import StateGraph, END
from src.domain.state import GraphState
from src.use_cases.nodes import RAGNodes
from langgraph.checkpoint.memory import MemorySaver
from src.utils.logging import logger

class RAGGraphBuilder:
    def __init__(self, retriever, max_loops: int = 3):
        self.nodes = RAGNodes(retriever)
        self.max_loops = max_loops

    def _check_loop_limit(self, state: GraphState) -> bool:
        """
        Verifica se ainda há tentativas disponíveis.
        
        Returns:
            True se ainda há tentativas, False se limite foi atingido
        """
        current_loop = state.get("loop_count", 0)
        max_allowed = state.get("max_loops", self.max_loops)
        
        if current_loop >= max_allowed:
            logger.warning(
                f"⚠️ Loop limit reached",
                extra={
                    "current_loop": current_loop,
                    "max_loops": max_allowed,
                    "question": state.get("question", "")[:50]
                }
            )
            return False
        return True

    def _check_hallucination(self, state: GraphState):
        """
        Decide se vai tentar corrigir alucinação ou finalizar.
        Garante que nunca exceda o limite de loops.
        """
        if not state.get("hallucination", False):
            # Sem alucinação, retorna resposta
            logger.info("✅ Generation validated - no hallucination detected")
            return "end"

        # Alucinação detectada: verifica se pode tentar novamente
        if self._check_loop_limit(state):
            logger.info(
                f"🔄 Hallucination detected, attempting fix",
                extra={"loop_count": state.get("loop_count", 0)}
            )
            return "transform_query"
        else:
            # Limite atingido: retorna resposta como está
            logger.warning("🛑 Hallucination detected but loop limit reached - returning degraded response")
            return "end"

    def _check_guardrail_result(self, state: GraphState):
        """Verifica se guardrail rejeitou a entrada."""
        if state.get("generation"):
            # Guardrail rejeitou, geração tem mensagem de erro
            logger.warning("🛑 Input rejected by guardrails", 
                          extra={"reason": state.get("generation", "")[:100]})
            return "end"
        return "retrieve"

    def _decide_next_step(self, state: GraphState):
        """
        Decide entre reformular query ou gerar resposta.
        Verifica limite de loops antes de reformular.
        """
        documents = state.get("documents", [])
        
        if not documents:
            if not self._check_loop_limit(state):
                logger.warning("⚠️ No documents found and loop limit reached - falling back to generation")
                return "generate"
            
            logger.info(f"🔄 No relevant documents, reformulating query (loop {state.get('loop_count', 0)})")
            return "transform_query"
        
        return "generate"

    def _store_original_question(self, state: GraphState):
        """Armazena a pergunta original e inicializa contadores."""
        updates = {}
        
        if "original_question" not in state:
            updates["original_question"] = state["question"]
        
        if "loop_count" not in state:
            updates["loop_count"] = 0
        
        if "max_loops" not in state:
            updates["max_loops"] = self.max_loops
        
        return updates if updates else {}

    def build(self):
        workflow = StateGraph(GraphState)

        # Adiciona nós
        workflow.add_node("store_question", self._store_original_question)
        workflow.add_node("guardrails", self.nodes.guardrails_check)
        workflow.add_node("retrieve", self.nodes.retrieve)
        workflow.add_node("grade_documents", self.nodes.grade_documents)
        workflow.add_node("generate", self.nodes.generate)
        workflow.add_node("validate_gen", self.nodes.validate_generation)
        workflow.add_node("transform_query", self.nodes.transform_query)

        # Define entrada
        workflow.set_entry_point("store_question")

        # Fluxo de edges
        workflow.add_edge("store_question", "guardrails")
        
        workflow.add_conditional_edges(
            "guardrails",
            self._check_guardrail_result,
            {
                "end": END,
                "retrieve": "retrieve"
            }
        )

        workflow.add_edge("retrieve", "grade_documents")
        
        workflow.add_conditional_edges(
            "grade_documents",
            self._decide_next_step,
            {
                "transform_query": "transform_query",
                "generate": "generate"
            }
        )
        
        workflow.add_edge("transform_query", "retrieve")
        workflow.add_edge("generate", "validate_gen")
        
        workflow.add_conditional_edges(
            "validate_gen",
            self._check_hallucination,
            {
                "transform_query": "transform_query",
                "end": END
            }
        )

        memory = MemorySaver()
        return workflow.compile(checkpointer=memory)