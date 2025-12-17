"""
conftest.py - Configuração compartilhada para testes pytest
"""

import pytest
from unittest.mock import Mock, MagicMock
from pathlib import Path
import sys

# Adicionar src ao path para importações
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.domain.state import AgentState


@pytest.fixture
def sample_medical_question():
    """Pergunta médica válida para testes."""
    return "Qual é o protocolo de tratamento para sepse em pacientes idosos?"


@pytest.fixture
def sample_invalid_question():
    """Pergunta fora do escopo médico."""
    return "Qual é a receita do pudim de leite condensado?"


@pytest.fixture
def sample_agent_state() -> AgentState:
    """AgentState padrão para testes."""
    return {
        "medical_question": "Qual é o protocolo para IAM?",
        "context_data": None,
        "documents": [],
        "generation": "",
        "is_safe": True,
        "risk_level": "informativo"
    }


@pytest.fixture
def mock_retriever():
    """Mock do retriever do ChromaDB."""
    mock = Mock()
    mock.invoke = Mock(return_value=[
        Mock(
            page_content="Protocolo X: Tratamento de Infarto Agudo do Miocárdio",
            metadata={"source": "0000002.xml", "specialty": "Cardiologia"}
        ),
        Mock(
            page_content="Protocolo Y: ECG em situação de emergência",
            metadata={"source": "0000050.xml", "specialty": "Cardiologia"}
        ),
    ])
    return mock


@pytest.fixture
def mock_llm():
    """Mock do LLM Google Gemini."""
    mock = MagicMock()
    mock.invoke = Mock(return_value="Resposta gerada pelo LLM")
    mock.with_structured_output = Mock(return_value=mock)
    return mock


@pytest.fixture
def mock_vector_store(mock_retriever):
    """Mock do VectorStoreRepository."""
    mock = Mock()
    mock.get_retriever = Mock(return_value=mock_retriever)
    return mock


# Markers customizados
def pytest_configure(config):
    """Registra custom markers."""
    config.addinivalue_line(
        "markers", "integration: marca testes de integração"
    )
    config.addinivalue_line(
        "markers", "slow: marca testes lentos"
    )
    config.addinivalue_line(
        "markers", "security: marca testes de segurança"
    )
