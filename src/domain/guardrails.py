"""
Guardrails de validação para segurança e conformidade médica.
Usa LLM para análise semântica real da pergunta (não keywords).
"""

import logging
from typing import Optional
from pydantic import BaseModel, Field
import re
from src.infrastructure.llm_factory import LLMFactory

logger = logging.getLogger(__name__)


class GuardrailsValidationResult(BaseModel):
    """Resultado da validação de guardrails."""
    
    is_valid: bool = Field(..., description="Se a entrada passou na validação")
    reason: Optional[str] = Field(default=None, description="Motivo da rejeição")
    has_pii: bool = Field(default=False, description="Se detectou PII na entrada")
    is_medical_relevant: bool = Field(default=True, description="Se é relevante ao contexto médico")


class GuardrailsValidator:
    """
    Valida perguntas médicas usando análise semântica com LLM.
    
    WHEN [usuário submete pergunta médica] 
    THE SYSTEM SHALL [validar entrada contra PII e relevância médica real]
    """
    
    # Padrões PII - mantidos simples e eficientes
    PII_PATTERNS = {
        "cpf": r"\d{3}\.\d{3}\.\d{3}-\d{2}",
        "cnpj": r"\d{2}\.\d{3}\.\d{3}/\d{4}-\d{2}",
        "phone": r"(\+\d{1,3})?\s?\(?\d{2}\)?\s?\d{4,5}-?\d{4}",
        "email": r"[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}",
        "patient_name": r"(?i)(paciente|patient|Sr\.|Dra?\.|Mrs?\.)\s+[A-Z][a-z]+(?:\s+[A-Z][a-z]+)*",
    }
    
    def __init__(self):
        self.max_question_length = 500
        self.min_question_length = 5
        self.llm = LLMFactory.get_llm()
        # Cache simples para evitar chamar LLM repetidas vezes
        self._cache = {}
    
    def validate(self, question: str) -> bool:
        """
        Valida pergunta do usuário contra múltiplos critérios de segurança.
        
        WHEN [pergunta é submetida]
        THE SYSTEM SHALL [validar usando análise LLM de relevância médica]
        
        Args:
            question: Texto da pergunta do usuário
            
        Returns:
            bool: True se pergunta passa em todas as validações, False caso contrário
        """
        logger.debug(f"Validando pergunta: {question[:60]}...")
        
        result = self._run_validations(question)
        
        if not result.is_valid:
            logger.warning(f"❌ Validação rejeitada: {result.reason}")
            return False
        
        logger.info(f"✅ Pergunta passou em todas as validações")
        return True
    
    def _run_validations(self, question: str) -> GuardrailsValidationResult:
        """Executa todas as validações em sequência."""
        
        # Validação 1: Comprimento
        if not self._validate_length(question):
            return GuardrailsValidationResult(
                is_valid=False,
                reason=f"Pergunta deve ter entre {self.min_question_length} e {self.max_question_length} caracteres"
            )
        
        # Validação 2: PII Detection (rápido, regex)
        pii_found = self._detect_pii(question)
        if pii_found:
            return GuardrailsValidationResult(
                is_valid=False,
                reason=f"Detectadas informações pessoais ({pii_found}) na pergunta. Remova dados sensíveis.",
                has_pii=True
            )
        
        # Validação 3: Relevância Médica (usando LLM)
        is_relevant = self._is_medically_relevant(question)
        if not is_relevant:
            return GuardrailsValidationResult(
                is_valid=False,
                reason="Pergunta não é sobre medicina ou saúde. Por favor, faça uma pergunta sobre saúde, doenças, tratamentos ou protocolos médicos.",
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
        result = self.min_question_length <= length <= self.max_question_length
        
        if not result:
            logger.warning(f"⚠️ Comprimento inválido: {length} caracteres (esperado: {self.min_question_length}-{self.max_question_length})")
        
        return result
    
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
                logger.warning(f"🔐 PII detectado: {pii_type}")
                return pii_type
        
        return None
    
    def _is_medically_relevant(self, question: str) -> bool:
        """
        WHEN [pergunta é recebida]
        THE SYSTEM SHALL [usar LLM para analisar se é pergunta médica]
        
        ✅ NOVO: Análise semântica com LLM, não keywords
        
        Args:
            question: Pergunta a validar
            
        Returns:
            bool: True se é pergunta médica, False caso contrário
        """
        # Verificar cache primeiro (para evitar múltiplas chamadas ao LLM)
        cache_key = hash(question)
        if cache_key in self._cache:
            logger.debug(f"✅ Usando resposta em cache")
            return self._cache[cache_key]
        
        try:
            logger.debug(f"🤖 Analisando pergunta com LLM...")
            
            # Prompt simples e claro para o LLM
            # Instruir para responder APENAS com "sim" ou "não"
            prompt = f"""Analise a seguinte pergunta e responda APENAS com "sim" ou "não".

A pergunta é sobre medicina, saúde, doenças, tratamentos, protocolos médicos, 
diagnósticos, sintomas, medicamentos, cirurgias, ou tópicos clínicos similares?

Pergunta: "{question}"

Responda APENAS com "sim" ou "não":"""
            
            response = self.llm.invoke(prompt)
            response_text = response.content if hasattr(response, 'content') else str(response)
            response_text = response_text.strip().lower()
            
            logger.debug(f"🤖 Resposta do LLM: {response_text}")
            
            # Analisar resposta
            is_medical = "sim" in response_text or "yes" in response_text
            
            # Cachear resultado
            self._cache[cache_key] = is_medical
            
            if is_medical:
                logger.debug(f"✅ Pergunta reconhecida como médica")
            else:
                logger.debug(f"❌ Pergunta NÃO reconhecida como médica")
            
            return is_medical
        
        except Exception as e:
            logger.error(f"❌ Erro ao analisar com LLM: {e}")
            # Em caso de erro, ser permissivo (assumir que é relevante)
            # Melhor deixar passar do que rejeitar com erro
            logger.warning(f"⚠️ Erro na análise LLM - assumindo pergunta válida por segurança")
            return True
    
    def clear_cache(self):
        """Limpa cache de análises."""
        self._cache.clear()
        logger.debug("🗑️ Cache de validação limpo")