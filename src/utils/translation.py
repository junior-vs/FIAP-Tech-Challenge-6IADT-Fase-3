"""
Módulo de tradução automática e detecção de idioma.
Suporta português ↔ inglês para integração com base de conhecimento em inglês.
"""

import logging
import re
from typing import Literal
from langdetect import detect, DetectorFactory
from src.infrastructure.llm_factory import LLMFactory

logger = logging.getLogger(__name__)

# Determinístico para detecção de idioma
DetectorFactory.seed = 0


class LanguageDetector:
    """Detecta o idioma de um texto."""
    
    # Palavras-chave em português médico
    PORTUGUESE_MEDICAL_KEYWORDS = {
        "protocolo", "tratamento", "medicamento", "paciente", "saúde",
        "diagnóstico", "sintoma", "doença", "infecção", "hospital",
        "médico", "idoso", "sepse", "pneumonia", "pressão",
        "diabetes", "hipertensão", "febre", "dor", "fadiga",
        "qual", "como", "quando", "onde", "por que", "o que",
        "você", "seus", "sua", "dele", "dela", "pode", "deve",
        "é", "são", "está", "estão", "foi", "foram"
    }
    
    # Palavras-chave em inglês médico
    ENGLISH_MEDICAL_KEYWORDS = {
        "protocol", "treatment", "medication", "patient", "health",
        "diagnosis", "symptom", "disease", "infection", "hospital",
        "doctor", "elderly", "sepsis", "pneumonia", "pressure",
        "diabetes", "hypertension", "fever", "pain", "fatigue",
        "what", "how", "when", "where", "why", "which",
        "you", "your", "his", "her", "can", "should",
        "is", "are", "was", "were", "be", "been"
    }
    
    @staticmethod
    def detect_language(text: str) -> Literal["pt", "en"]:
        """
        Detecta se texto é português ou inglês.
        
        WHEN [texto é fornecido]
        THE SYSTEM SHALL [detectar idioma com alta confiança]
        
        Args:
            text: Texto a analisar
            
        Returns:
            "pt" para português, "en" para inglês
        """
        if not text or len(text) < 3:
            logger.warning("⚠️ Texto muito curto para detecção de idioma")
            return "en"  # Default para inglês
        
        try:
            # Usar langdetect para detecção inicial
            detected = detect(text)
            logger.debug(f"🔍 Idioma detectado por langdetect: {detected}")
            
            # Mapear para pt/en
            if detected in ("pt", "pt-BR", "pt-PT"):
                return "pt"
            elif detected in ("en", "en-US", "en-GB"):
                return "en"
            
            # Fallback: contar palavras-chave
            text_lower = text.lower()
            
            pt_score = sum(1 for word in text_lower.split() 
                          if word in LanguageDetector.PORTUGUESE_MEDICAL_KEYWORDS)
            en_score = sum(1 for word in text_lower.split() 
                          if word in LanguageDetector.ENGLISH_MEDICAL_KEYWORDS)
            
            logger.debug(f"Scores: PT={pt_score}, EN={en_score}")
            
            if pt_score > en_score:
                return "pt"
            else:
                return "en"
        
        except Exception as e:
            logger.warning(f"⚠️ Erro na detecção de idioma: {e}")
            return "en"  # Default


class Translator:
    """Traduz textos entre português e inglês usando LLM."""
    
    def __init__(self):
        self.llm = LLMFactory.get_llm()
    
    def translate_pt_to_en(self, text: str) -> str:
        """
        Traduz português para inglês.
        
        WHEN [texto em português é fornecido]
        THE SYSTEM SHALL [traduzir para inglês mantendo precisão médica]
        
        Args:
            text: Texto em português
            
        Returns:
            Texto traduzido para inglês
        """
        if not text or len(text) < 2:
            return text
        
        try:
            logger.debug(f"🔄 Traduzindo para inglês: {text[:50]}...")
            
            prompt = f"""Translate the following medical question from Portuguese to English. 
Maintain medical terminology accuracy. Return ONLY the translation, nothing else.

Portuguese: {text}

English:"""
            
            response = self.llm.invoke(prompt)
            translation = response.content if hasattr(response, 'content') else str(response)
            translation = translation.strip() # type: ignore
            
            logger.debug(f"✅ Tradução: {translation[:50]}...")
            return translation
        
        except Exception as e:
            logger.error(f"❌ Erro ao traduzir para inglês: {e}")
            return text  # Fallback: retornar original
    
    def translate_en_to_pt(self, text: str) -> str:
        """
        Traduz inglês para português.
        
        WHEN [texto em inglês é fornecido]
        THE SYSTEM SHALL [traduzir para português mantendo precisão médica]
        
        Args:
            text: Texto em inglês
            
        Returns:
            Texto traduzido para português
        """
        if not text or len(text) < 2:
            return text
        
        try:
            logger.debug(f"🔄 Traduzindo para português: {text[:50]}...")
            
            prompt = f"""Translate the following medical response from English to Portuguese (Brazilian Portuguese - pt-BR).
Maintain medical terminology accuracy and clarity. Return ONLY the translation, nothing else.

English: {text}

Portuguese (pt-BR):"""
            
            response = self.llm.invoke(prompt)
            translation = response.content if hasattr(response, 'content') else str(response)
            translation = translation.strip() # type: ignore
            
            logger.debug(f"✅ Tradução: {translation[:50]}...")
            return translation
        
        except Exception as e:
            logger.error(f"❌ Erro ao traduzir para português: {e}")
            return text  # Fallback: retornar original
    
    def translate(self, text: str, source_lang: Literal["pt", "en"], 
                 target_lang: Literal["pt", "en"]) -> str:
        """
        Traduz texto entre idiomas.
        
        Args:
            text: Texto a traduzir
            source_lang: Idioma origem (pt/en)
            target_lang: Idioma destino (pt/en)
            
        Returns:
            Texto traduzido
        """
        if source_lang == target_lang:
            return text  # Sem tradução necessária
        
        if source_lang == "pt" and target_lang == "en":
            return self.translate_pt_to_en(text)
        elif source_lang == "en" and target_lang == "pt":
            return self.translate_en_to_pt(text)
        else:
            logger.warning(f"⚠️ Combinação de idioma não suportada: {source_lang} → {target_lang}")
            return text