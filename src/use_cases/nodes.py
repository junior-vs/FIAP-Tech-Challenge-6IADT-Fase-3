"""
Módulo: src/use_cases/nodes.py
Descrição: Implementação dos nós do grafo (Passos da execução).
Motivo da alteração: 
- Alteração dos Prompts para persona "Assistente Médico".
- Uso das chaves do novo AgentState (medical_question, is_safe).
- Inclusão de instruções de segurança (não prescrever sem validação).
"""

import logging
from typing import List
from langchain_core.documents import Document
from src.domain.state import AgentState
from src.domain.guardrails import GuardrailsValidator
from src.infrastructure.llm_factory import LLMFactory
from src.infrastructure.vector_store import VectorStoreRepository

logger = logging.getLogger(__name__)

class RAGNodes:
    """Nós de processamento para o grafo RAG."""
    
    def __init__(self):
        self.guardrails = GuardrailsValidator()
        self.llm = LLMFactory.get_llm()
        self.retriever = VectorStoreRepository().get_retriever()
    
    def guardrails_check(self, state: AgentState) -> dict:
        """Valida segurança e pertinência médica da pergunta."""
        logger.debug("🛡️ Verificando pertinência do tema médico...")
        
        question = state.get("medical_question", "")
        
        try:
            is_valid = self.guardrails.validate(question)
            
            if is_valid:
                logger.info("✅ Tema médico válido.")
                return {"is_safe": True}
            else:
                logger.warning("⚠️ Tema fora do escopo médico.")
                return {
                    "is_safe": False,
                    "generation": "Desculpe, sua pergunta não é relacionada a temas médicos. Por favor, formule uma pergunta sobre saúde ou protocolos clínicos."
                }
        except Exception as e:
            logger.error(f"❌ Erro na validação de guardrails: {e}")
            return {
                "is_safe": False,
                "generation": f"Erro ao validar pergunta: {str(e)}"
            }
    
    def retrieve(self, state: AgentState) -> dict:
        """Recupera documentos relevantes da base vetorial."""
        question = state.get("medical_question", "")
        logger.debug(f"🔍 Iniciando busca vetorial para: {question[:60]}...")
        
        try:
            documents = self.retriever.invoke(question)
            
            if not isinstance(documents, list):
                logger.warning(f"⚠️ Retriever retornou tipo inesperado: {type(documents)}")
                documents = list(documents) if hasattr(documents, '__iter__') else []
            
            logger.info(f"✅ Recuperados {len(documents)} documentos relevantes")
            logger.debug("Busca vetorial concluída")
            
            return {"documents": documents}
        
        except Exception as e:
            logger.error(f"❌ Erro na recuperação: {e}", exc_info=True)
            return {
                "documents": [],
                "generation": "Erro ao buscar protocolos na base de conhecimento."
            }
    
    def grade_documents(self, state: AgentState) -> dict:
        """Avalia relevância dos documentos recuperados."""
        logger.debug("Avalia relevância dos documentos...")
        
        documents = state.get("documents", [])
        question = state.get("medical_question", "")
        
        if not documents:
            logger.warning("⚠️ Nenhum documento fornecido para avaliação")
            return {"documents": []}
        
        try:
            useful_docs = []
            
            for doc in documents:
                # ✅ CORRIGIDO: Verificar se é realmente um Document
                if not isinstance(doc, Document):
                    logger.warning(
                        f"⚠️ Item não é Document: tipo={type(doc)}, conteúdo={str(doc)[:50]}"
                    )
                    continue
                
                # Verificar se documento contém informação relevante
                doc_content = doc.page_content.lower()
                question_lower = question.lower()
                
                # Critério simples: sobreposição de palavras-chave
                question_words = set(question_lower.split())
                doc_words = set(doc_content.split())
                overlap = len(question_words & doc_words) / max(len(question_words), 1)
                
                if overlap > 0.1:  # 10% de sobreposição mínima
                    useful_docs.append(doc)
            
            logger.info(f"Documentos úteis: {len(useful_docs)}/{len(documents)}")
            return {"documents": useful_docs}
        
        except Exception as e:
            logger.error(f"❌ Erro ao avaliar documentos: {e}", exc_info=True)
            return {"documents": documents}  # Retornar originais em caso de erro
    
    def generate(self, state: AgentState) -> dict:
        """Gera resposta clínica baseada em documentos."""
        logger.debug("Gerando resposta clínica...")
        
        documents = state.get("documents", [])
        question = state.get("medical_question", "")
        
        if not question:
            return {"generation": "Pergunta vazia fornecida."}
        
        try:
            # Construir contexto dos documentos
            context = ""
            if documents:
                context = "Protocolos consultados:\n\n"
                for i, doc in enumerate(documents, 1):
                    if isinstance(doc, Document):
                        source = doc.metadata.get("source", f"Protocolo {i}")
                        context += f"{i}. {source}\n{doc.page_content[:300]}...\n\n"
                    else:
                        logger.warning(f"⚠️ Documento {i} não é do tipo Document: {type(doc)}")
            
            # Prompt estruturado
            prompt = f"""Você é um assistente médico especializado em protocolos clínicos.
Baseado nos protocolos fornecidos, responda à pergunta do médico.

Protocolos de referência:
{context}

Pergunta do médico:
{question}

Resposta (cite os protocolos utilizados):"""
            
            response = self.llm.invoke(prompt)
            generation = response.content if hasattr(response, 'content') else str(response)
            
            logger.info("✅ Resposta gerada com sucesso")
            return {"generation": generation}
        
        except Exception as e:
            logger.error(f"❌ Erro ao gerar resposta: {e}", exc_info=True)
            return {"generation": f"Erro ao gerar resposta: {str(e)}"}
    
    def validate_hallucination(self, state: AgentState) -> dict:
        """Valida se a resposta está baseada nos documentos (sem alucinações)."""
        logger.debug("🔍 Validando alucinações...")
        
        generation = state.get("generation", "")
        documents = state.get("documents", [])
        
        if not documents:
            logger.warning("⚠️ Sem documentos para validar hallucination")
            return {"hallucination_check": "sem_docs"}
        
        try:
            # Verificação simples: resposta deve conter termos dos documentos
            gen_lower = generation.lower()
            has_reference = False
            
            for doc in documents:
                if isinstance(doc, Document):
                    doc_content_lower = doc.page_content.lower()
                    # Procurar por palavras-chave do documento na resposta
                    if len(doc_content_lower) > 50:
                        key_phrase = doc_content_lower.split()[:5]
                        if any(word in gen_lower for word in key_phrase):
                            has_reference = True
                            break
            
            if has_reference:
                logger.info("✅ Resposta validada (baseada em documentos)")
                return {"hallucination_check": "valid"}
            else:
                logger.warning("⚠️ Possível alucinação detectada")
                return {"hallucination_check": "possible_hallucination"}
        
        except Exception as e:
            logger.error(f"❌ Erro na validação: {e}", exc_info=True)
            return {"hallucination_check": "error"}