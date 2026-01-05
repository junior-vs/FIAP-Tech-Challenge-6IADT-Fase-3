"""Smoke test do pipeline sem vectorstore/embeddings.

Objetivo: validar rapidamente as alterações de segurança/logging/explainability
sem depender de quota de embeddings (Chroma/Gemini embeddings).

O teste executa:
- setup de logging (incluindo request_id)
- guardrails (local + LLM)
- generate (com fontes)
- validate_response (checagem de groundedness)

Uso (PowerShell):
    ./.venv/Scripts/python.exe ./smoke_test.py
"""

from __future__ import annotations

import uuid

from langchain_core.documents import Document

from src.config import settings
from src.infrastructure.llm_factory import LLMFactory
from src.use_cases.nodes import RAGNodes
from src.utils.logging import setup_logging


def main() -> int:
    setup_logging(level=getattr(settings, "log_level", "INFO"))

    llm = LLMFactory.get_llm()
    nodes = RAGNodes(retriever=None, llm=llm)

    request_id = str(uuid.uuid4())
    question = "Quais são sinais e cuidados iniciais para sepse em idosos?"

    docs = [
        Document(
            page_content=(
                "O reconhecimento precoce de sepse inclui avaliação de sinais vitais, "
                "alteração do estado mental e marcadores de disfunção orgânica. "
                "Medidas iniciais podem incluir suporte hemodinâmico e investigação de foco infeccioso, "
                "conforme protocolo institucional."
            ),
            metadata={"source": "0000001.xml"},
        ),
        Document(
            page_content=(
                "Em idosos, a sepse pode se manifestar com sintomas atípicos, como confusão mental, "
                "queda e piora funcional. A conduta deve seguir protocolos locais e avaliação médica."
            ),
            metadata={"source": "0000002.xml"},
        ),
    ]

    state = {
        "request_id": request_id,
        "medical_question": question,
        "context_data": None,
        "documents": docs,
        "generation": "",
        "is_safe": True,
        "risk_level": None,
        "requires_human_validation": None,
        "safety_reason": None,
        "chat_history": None,
        "loop_count": None,
        "is_valid": None,
        "hallucination_check": None,
    }

    state = nodes.guardrails(state)
    if not state.get("is_safe", False):
        print("\n[SMOKE] Pergunta bloqueada pelos guardrails:")
        print(state.get("generation"))
        return 2

    state = nodes.generate(state)
    state = nodes.validate_response(state)

    print("\n[SMOKE] Resultado final:")
    print(f"- request_id: {request_id}")
    print(f"- is_safe: {state.get('is_safe')}")
    print(f"- requires_human_validation: {state.get('requires_human_validation')}")
    print(f"- hallucination_check: {state.get('hallucination_check')}")

    generation = (state.get("generation") or "").strip()
    preview = generation.replace("\n", " ")
    if len(preview) > 260:
        preview = preview[:260] + "..."
    print(f"- preview: {preview}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
