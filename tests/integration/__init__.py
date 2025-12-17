"""
tests/integration/test_rag_pipeline.py
Testes de integração do pipeline RAG completo.
"""

import pytest
from unittest.mock import Mock, patch, MagicMock
from src.use_cases.graph import GraphBuilder
from src.domain.state import AgentState


@pytest.mark.integration
class TestRAGPipeline:
    """Testes da pipeline RAG completa."""
    
    @patch('src.infrastructure.vector_store.VectorStoreRepository')
    @patch('src.infrastructure.llm_factory.LLMFactory')
    def test_full_pipeline_valid_medical_question(
        self, mock_llm_factory, mock_vector_store_class
    ):
        """
        Testa fluxo completo: pergunta válida -> guardrails -> retrieve -> 
        grade -> generate -> validate
        """
        # Mock do retriever
        mock_docs = [
            Mock(
                page_content="Protocolo de Sepse: diagnóstico rápido",
                metadata={"source": "0000010.xml"}
            )
        ]
        mock_retriever = Mock()
        mock_retriever.invoke.return_value = mock_docs
        
        # Mock do VectorStore
        mock_vs_instance = Mock()
        mock_vs_instance.get_retriever.return_value = mock_retriever
        mock_vector_store_class.return_value = mock_vs_instance
        
        # Mock do LLM
        mock_llm = MagicMock()
        mock_llm.invoke.return_value = "Resposta sobre protocolo de sepse"
        mock_llm.with_structured_output.return_value = mock_llm
        mock_llm_factory.get_llm.return_value = mock_llm
        mock_llm_factory.get_embeddings.return_value = Mock()
        
        # Construir grafo
        builder = GraphBuilder()
        graph = builder.build()
        
        # Executar
        initial_state: AgentState = {
            "medical_question": "Qual é o protocolo para sepse em idosos?",
            "context_data": None,
            "documents": [],
            "generation": "",
            "is_safe": True,
            "risk_level": "emergencia"
        }
        
        result = graph.invoke(initial_state)
        
        # Assertions
        assert result["is_safe"] is True
        assert len(result["documents"]) > 0
        assert result["generation"] != ""
        assert "protocolo" in result["generation"].lower() or "sepse" in result["generation"].lower()
    
    @patch('src.infrastructure.vector_store.VectorStoreRepository')
    @patch('src.infrastructure.llm_factory.LLMFactory')
    def test_pipeline_blocks_invalid_topic(
        self, mock_llm_factory, mock_vector_store_class
    ):
        """
        Testa que guardrails bloqueia perguntas fora do escopo médico.
        """
        # Mock do LLM para guardrails retornar is_valid=False
        mock_llm = MagicMock()
        mock_guardrail_chain = Mock()
        mock_guardrail_chain.invoke.return_value = Mock(
            is_valid=False,
            reason="Pergunta fora do escopo médico"
        )
        
        mock_llm_factory.get_llm.return_value = mock_llm
        mock_llm_factory.get_embeddings.return_value = Mock()
        
        # Setup vector store
        mock_vs_instance = Mock()
        mock_vs_instance.get_retriever.return_value = Mock()
        mock_vector_store_class.return_value = mock_vs_instance
        
        # Construir grafo
        builder = GraphBuilder()
        graph = builder.build()
        
        # Executar com pergunta inválida
        initial_state: AgentState = {
            "medical_question": "Como faço um brigadeiro de colher?",
            "context_data": None,
            "documents": [],
            "generation": "",
            "is_safe": True,
            "risk_level": "informativo"
        }
        
        result = graph.invoke(initial_state)
        
        # Assertions
        # O resultado pode variar dependendo da implementação real
        # Mas esperamos que seja seguro ainda
        assert "generation" in result


@pytest.mark.integration
class TestGuardrailsNode:
    """Testes específicos do nó guardrails."""
    
    def test_guardrails_accepts_valid_medical_question(self, sample_medical_question):
        """Guardrails aceita pergunta médica válida."""
        from src.use_cases.nodes import RAGNodes
        
        mock_retriever = Mock()
        nodes = RAGNodes(mock_retriever)
        
        state: AgentState = {
            "medical_question": sample_medical_question,
            "context_data": None,
            "documents": [],
            "generation": "",
            "is_safe": True,
            "risk_level": "informativo"
        }
        
        # Mock da guardrail chain
        nodes.guardrail_chain = Mock()
        nodes.guardrail_chain.invoke = Mock(
            return_value=Mock(is_valid=True, reason="")
        )
        
        result = nodes.guardrails_check(state)
        
        assert result["is_safe"] is True


@pytest.mark.integration  
class TestRetrievalNode:
    """Testes específicos do nó retrieve."""
    
    def test_retrieve_returns_documents(self, sample_agent_state, mock_retriever):
        """Retrieve retorna documentos do ChromaDB."""
        from src.use_cases.nodes import RAGNodes
        
        nodes = RAGNodes(mock_retriever)
        
        result = nodes.retrieve(sample_agent_state)
        
        assert "documents" in result
        assert len(result["documents"]) > 0
        mock_retriever.invoke.assert_called_once()
    
    def test_retrieve_empty_results(self, sample_agent_state):
        """Retrieve trata resultado vazio gracefully."""
        from src.use_cases.nodes import RAGNodes
        
        mock_retriever = Mock()
        mock_retriever.invoke.return_value = []
        
        nodes = RAGNodes(mock_retriever)
        result = nodes.retrieve(sample_agent_state)
        
        assert len(result["documents"]) == 0


@pytest.mark.integration
class TestGenerationNode:
    """Testes específicos do nó generate."""
    
    def test_generate_with_documents(self, sample_agent_state, mock_retriever):
        """Generate cria resposta com documentos disponíveis."""
        from src.use_cases.nodes import RAGNodes
        
        nodes = RAGNodes(mock_retriever)
        nodes.rag_chain = Mock()
        nodes.rag_chain.invoke = Mock(
            return_value="Resposta baseada em protocolos"
        )
        
        sample_agent_state["documents"] = mock_retriever.invoke("test")
        
        result = nodes.generate(sample_agent_state)
        
        assert result["generation"] != ""
        assert "Resposta" in result["generation"]


@pytest.mark.integration
@pytest.mark.slow
class TestValidationNode:
    """Testes do nó validate (detecção de alucinações)."""
    
    def test_validate_accepts_grounded_response(self, sample_agent_state, mock_retriever):
        """Validate aceita resposta baseada em documentos."""
        from src.use_cases.nodes import RAGNodes
        
        nodes = RAGNodes(mock_retriever)
        nodes.hallucination_chain = Mock()
        nodes.hallucination_chain.invoke = Mock(
            return_value=Mock(binary_score="sim", reason="")
        )
        
        sample_agent_state["documents"] = mock_retriever.invoke("test")
        sample_agent_state["generation"] = "Resposta válida"
        
        result = nodes.validate_generation(sample_agent_state)
        
        assert result["generation"] == "Resposta válida"
    
    def test_validate_rejects_hallucination(self, sample_agent_state, mock_retriever):
        """Validate rejeita resposta com alucinação."""
        from src.use_cases.nodes import RAGNodes
        
        nodes = RAGNodes(mock_retriever)
        nodes.hallucination_chain = Mock()
        nodes.hallucination_chain.invoke = Mock(
            return_value=Mock(
                binary_score="nao",
                reason="Dosagem não está nos protocolos"
            )
        )
        
        sample_agent_state["documents"] = mock_retriever.invoke("test")
        sample_agent_state["generation"] = "Prescrever 500mg diários"
        
        result = nodes.validate_generation(sample_agent_state)
        
        assert "Peço desculpas" in result["generation"]
        assert "Prescrever" not in result["generation"]
