"""
Módulo de anonimização de documentos médicos.
Pré-processa XMLs antes de ingestão no vectorstore.
"""

import re
from typing import List
from presidio_analyzer import AnalyzerEngine
from presidio_anonymizer import AnonymizerEngine

class MedicalDataAnonymizer:
    """Anonymiza PII em documentos médicos usando Microsoft Presidio."""
    
    def __init__(self):
        self.analyzer = AnalyzerEngine()
        self.anonymizer = AnonymizerEngine()
        self.pii_patterns = {
            "CPF": r"\d{3}\.\d{3}\.\d{3}-\d{2}",
            "CNPJ": r"\d{2}\.\d{3}\.\d{3}/\d{4}-\d{2}",
            "PHONE": r"\(\d{2}\)\s?\d{4,5}-\d{4}",
            "EMAIL": r"[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}",
        }
    
    def anonymize_document(self, text: str) -> str:
        """Anonymiza PII no texto mantendo estrutura clínica."""
        try:
            # Presidio detecção automática
            results = self.analyzer.analyze(text, language="pt")
            
            if results:
                anonymized = self.anonymizer.anonymize(
                    text=text,
                    analyzer_results=results
                )
                return anonymized.text
            return text
        except Exception as e:
            logger.warning(f"Erro ao anonymizar: {e}. Usando texto original.")
            return text
    
    def anonymize_batch(self, documents: List[str]) -> List[str]:
        """Processa lote de documentos."""
        return [self.anonymize_document(doc) for doc in documents]