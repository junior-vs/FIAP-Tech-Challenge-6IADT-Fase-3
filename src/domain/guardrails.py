"""
Guardrails de validação para segurança e conformidade médica.
Responsável por validar entrada de usuários antes do processamento.
"""

import logging
import re
from typing import Optional
from pydantic import BaseModel, Field

logger = logging.getLogger(__name__)


class GuardrailsValidationResult(BaseModel):
    """Resultado da validação de guardrails."""
    
    is_valid: bool = Field(..., description="Se a entrada passou na validação")
    reason: Optional[str] = Field(default=None, description="Motivo da rejeição")
    has_pii: bool = Field(default=False, description="Se detectou PII na entrada")
    is_medical_relevant: bool = Field(default=True, description="Se é relevante ao contexto médico")


class GuardrailsValidator:
    """
    Valida perguntas médicas para segurança, conformidade e pertinência clínica.
    
    WHEN [usuário submete pergunta médica] 
    THE SYSTEM SHALL [validar entrada contra PII, relevância médica, e conformidade]
    """
    
    # Padrões PII (Informações Pessoalmente Identificáveis)
    PII_PATTERNS = {
        "cpf": r"\d{3}\.\d{3}\.\d{3}-\d{2}",
        "cnpj": r"\d{2}\.\d{3}\.\d{3}/\d{4}-\d{2}",
        "phone": r"(\+\d{1,3})?\s?\(?\d{2}\)?\s?\d{4,5}-?\d{4}",
        "email": r"[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}",
        "patient_name": r"(?i)(paciente|patient|Sr\.|Dra?\.|Mrs?\.)\s+[A-Z][a-z]+(?:\s+[A-Z][a-z]+)*",
    }
    
    # Termos médicos mínimos para relevância
    MEDICAL_KEYWORDS = {
        "diagnóstico", "diagnóstico", "sintoma", "tratamento", "protocolo",
        "medicamento", "droga", "terapia", "paciente", "saúde", "doença",
        "infecção", "inflamação", "alergia", "cirurgia", "hospital",
        "médico", "clínico", "clínica", "pressão", "diabetes", "hipertensão",
        "sepse", "pneumonia", "insuficiência", "cardíaco", "renal", "hepático",
        "antibiótico", "vacinação", "vacina", "febre", "dor", "fadiga",
        "dispneia", "tosse", "náusea", "vômito", "diarreia", "anemia",
        "angiografia", "radiografia", "ressonância", "ultrassom", "tomografia",
        "análise", "exame", "laboratorio", "teste", "cultura", "hemograma",
        "idoso", "geriátrico", "criança", "neonato", "gestante", "pós-operatório"
    }
    
    # Tópicos explicitamente não-médicos
    NON_MEDICAL_TOPICS = {
        "receita", "brigadeiro", "bolo", "livro", "romance", "filme",
        "política", "economia", "futebol", "música", "história",
        "matemática", "física", "programação", "código", "javascript"
    }
    
    def __init__(self):
        self.max_question_length = 500
        self.min_question_length = 5
    
    def validate(self, question: str) -> bool:
        """
        Valida pergunta do usuário contra múltiplos critérios de segurança.
        
        WHEN [pergunta é submetida]
        THE SYSTEM SHALL [retornar True se válida, False caso contrário]
        
        Args:
            question: Texto da pergunta do usuário
            
        Returns:
            bool: True se pergunta passa em todas as validações, False caso contrário
        """
        logger.debug(f"Validando pergunta: {question[:60]}...")
        
        result = self._run_validations(question)
        
        if not result.is_valid:
            logger.warning(f"Validação rejeitada: {result.reason}")
            return False
        
        logger.info("✅ Pergunta passou em todas as validações")
        return True
    
    def _run_validations(self, question: str) -> GuardrailsValidationResult:
        """Executa todas as validações em sequência."""
        
        # Validação 1: Comprimento
        if not self._validate_length(question):
            return GuardrailsValidationResult(
                is_valid=False,
                reason=f"Pergunta deve ter entre {self.min_question_length} e {self.max_question_length} caracteres"
            )
        
        # Validação 2: PII Detection
        pii_found = self._detect_pii(question)
        if pii_found:
            return GuardrailsValidationResult(
                is_valid=False,
                reason=f"Detectadas informações pessoais ({pii_found}) na pergunta. Remova dados sensíveis.",
                has_pii=True
            )
        
        # Validação 3: Relevância Médica
        if not self._is_medically_relevant(question):
            return GuardrailsValidationResult(
                is_valid=False,
                reason="Pergunta não é relevante ao contexto médico. Formule uma pergunta sobre saúde ou protocolos clínicos.",
                is_medical_relevant=False
            )
        
        # Todas as validações passaram
        return GuardrailsValidationResult(is_valid=True)
    
    def _validate_length(self, question: str) -> bool:
        """
        WHEN [pergunta é recebida]
        THE SYSTEM SHALL [rejeitar se menor que min ou maior que max]
        """
        length = len(question.strip())
        return self.min_question_length <= length <= self.max_question_length
    
    def _detect_pii(self, text: str) -> Optional[str]:
        """
        WHEN [texto é analisado]
        THE SYSTEM SHALL [detectar PII usando padrões regex]
        
        Returns:
            str: Tipo de PII detectado (ex: "cpf", "email") ou None
        """
        text_lower = text.lower()
        
        for pii_type, pattern in self.PII_PATTERNS.items():
            if re.search(pattern, text):
                logger.warning(f"PII detectado: {pii_type}")
                return pii_type
        
        return None
    
    def _is_medically_relevant(self, question: str) -> bool:
        """
        WHEN [pergunta é recebida]
        THE SYSTEM SHALL [verificar se contém termos médicos relevantes]
        
        Estratégia: Buscar por palavras-chave médicas ou rejeitar tópicos não-médicos óbvios.
        """
        question_lower = question.lower()
        
        # Rejeição explícita: tópicos claramente não-médicos
        for topic in self.NON_MEDICAL_TOPICS:
            if topic in question_lower:
                logger.debug(f"Tópico não-médico detectado: {topic}")
                return False
        
        # Aceição: contém palavras-chave médicas
        for keyword in self.MEDICAL_KEYWORDS:
            if keyword in question_lower:
                logger.debug(f"Palavra-chave médica encontrada: {keyword}")
                return True
        
        # Heurística: Se tem muitos números ou menciona "protocolo/tratamento/sintoma"
        if any(word in question_lower for word in ["protocolo", "tratamento", "sintoma", "medicação"]):
            return True
        
        # Se nada foi encontrado, rejeitar por segurança
        logger.debug("Nenhuma palavra-chave médica detectada")
        return False
    
    def get_validation_result_message(self, result: GuardrailsValidationResult) -> str:
        """Gera mensagem amigável baseada no resultado de validação."""
        if result.is_valid:
            return "✅ Pergunta validada com sucesso"
        elif result.has_pii:
            return f"⚠️ {result.reason}"
        elif not result.is_medical_relevant:
            return f"⚠️ {result.reason}"
        else:
            return f"❌ {result.reason}"


# ========== Pydantic Models para LangChain ==========

from typing import Literal

class GuardrailsGrade(BaseModel):
    """
    Modelo para classificação de relevância médica de perguntas.
    Usado pelo chain estruturado de guardrails como fallback.
    """
    is_safe: Literal["sim", "não"] = Field(
        description="Indica se a pergunta é segura e apropriada para o contexto médico"
    )
    risk_level: Literal["baixo", "médio", "alto"] = Field(
        description="Nível de risco da pergunta para o sistema médico",
        default="baixo"
    )
    reasoning: str = Field(
        description="Explicação da decisão de classificação",
        default=""
    )


class HallucinationGrade(BaseModel):
    """
    Modelo para detecção de alucinações na resposta do LLM.
    Verifica se a resposta está baseada nos documentos fornecidos.
    """
    is_grounded: Literal["sim", "não"] = Field(
        description="Indica se a resposta está baseada nos documentos fornecidos"
    )
    confidence: Literal["baixa", "média", "alta"] = Field(
        description="Nível de confiança na avaliação",
        default="média"
    )
    issues: str = Field(
        description="Problemas identificados na resposta, se houver",
        default=""
    )


class DocumentGrade(BaseModel):
    """
    Modelo para classificação de relevância de documentos.
    Usado para filtrar documentos recuperados.
    """
    is_relevant: Literal["sim", "não"] = Field(
        description="Indica se o documento é relevante para a pergunta"
    )
    relevance_score: Literal["baixa", "média", "alta"] = Field(
        description="Pontuação de relevância do documento",
        default="média"
    )