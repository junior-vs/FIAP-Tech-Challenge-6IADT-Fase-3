"""
Módulo: src/domain/guardrails_check.py
Descrição: Define os modelos de dados para validação estruturada (Output Parsers).
Motivo da alteração: Adaptação das descrições para validar perguntas e respostas médicas.
"""

from pydantic import BaseModel, Field

class InputGuardrail(BaseModel):
    is_valid: bool = Field(
        description="A pergunta é válida para o contexto médico/hospitalar? True se for sobre saúde, procedimentos ou protocolos. False se for sobre outros temas."
    )
    reason: str = Field(
        default="",
        description="Explicação curta se a pergunta for inválida (ex: tema fora do escopo médico)."
    )

class RetrievalGrader(BaseModel):
    binary_score: str = Field(
        description="O documento recuperado contém informações médicas relevantes para responder à pergunta? Responda 'sim' ou 'nao'"
    )

class HallucinationGrade(BaseModel):
    binary_score: str = Field(
        description="A resposta gerada é 100% baseada nos protocolos fornecidos? Responda 'sim' ou 'nao'"
    )
    reason: str = Field(
        description="Explicação se a resposta contiver informações não presentes nos documentos (alucinação)."
    )