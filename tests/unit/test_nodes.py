"""
Testes unitários para os nós do grafo RAG.
"""

import pytest
from unittest.mock import Mock, patch
from src.domain.state import AgentState
from src.use_cases.nodes import RAGNodes


@pytest.fixture
def mock_retriever():
    """Mock do retriever para testes."""
    return Mock()


@pytest.fixture
def rag_nodes(mock_retriever):
    """Instância de RAGNodes com dependências mockadas."""
    return RAGNodes(mock_retriever)


def test_guardrails_check_valid_medical_question(rag_nodes):
    """Deve aceitar pergunta médica válida."""
    state = {
        "medical_question": "Qual é o protocolo de tratamento para IAM?",
        "is_safe": True,
    }
    
    result = rag_nodes.guardrails_check(state)
    
    assert result["is_safe"] is True
    assert "generation" not in result or result["generation"] == ""


def test_guardrails_check_invalid_topic(rag_nodes):
    """Deve rejeitar pergunta fora do escopo médico."""
    state = {
        "medical_question": "Qual é a receita de brigadeiro?",
        "is_safe": True,
    }
    
    result = rag_nodes.guardrails_check(state)
    
    assert result["is_safe"] is False
    assert "Desculpe" in result["generation"]


def test_retrieve_documents(rag_nodes, mock_retriever):
    """Deve recuperar documentos da base."""
    mock_docs = [Mock(page_content="Protocolo X")]
    mock_retriever.invoke.return_value = mock_docs
    
    state = {"medical_question": "Teste"}
    result = rag_nodes.retrieve(state)
    
    assert len(result["documents"]) == 1
    mock_retriever.invoke.assert_called_once()