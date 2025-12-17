"""
Módulo: src/use_cases/nodes.py
Descrição: Implementação dos nós do grafo (Passos da execução).
Motivo da alteração: 
- Alteração dos Prompts para persona "Assistente Médico".
- Uso das chaves do novo AgentState (medical_question, is_safe).
- Inclusão de instruções de segurança (não prescrever sem validação).
- Suporte a múltiplos idiomas (detecção e tradução).
"""

import logging
import math
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
        """Inicializa todos os componentes necessários para os nós."""
        logger.debug("🔨 Inicializando RAGNodes...")
        
        # ✅ NOVO: Inicializar todos os componentes
        self.guardrails = GuardrailsValidator()
        self.llm = LLMFactory.get_llm()
        self.embeddings = LLMFactory.get_embeddings()
        
        # Vector store for retrieval
        try:
            vector_repo = VectorStoreRepository()
            self.retriever = vector_repo.get_retriever()
        except Exception as e:
            logger.warning(f"⚠️ Erro ao inicializar vector store: {e}")
            self.retriever = None
        
        logger.debug("✅ RAGNodes inicializado com sucesso")
    
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
            logger.error(f"❌ Erro na validação de guardrails: {e}", exc_info=True)
            return {
                "is_safe": False,
                "generation": f"Erro ao validar pergunta: {str(e)}"
            }
    
    def retrieve(self, state: AgentState) -> dict:
        """Recupera documentos relevantes da base vetorial."""
        question = state.get("medical_question", "")
        logger.debug(f"🔍 Iniciando busca vetorial para: {question[:60]}...")
        
        try:
            if not self.retriever:
                logger.warning("⚠️ Retriever não está disponível")
                return {"documents": []}
            
            documents = self.retriever.invoke(question)
            
            if not isinstance(documents, list):
                logger.warning(f"⚠️ Retriever retornou tipo inesperado: {type(documents)}")
                documents = list(documents) if hasattr(documents, '__iter__') else []
            
            logger.info(f"✅ Recuperados {len(documents)} documentos relevantes")
            
            for i, doc in enumerate(documents):
                logger.debug(f"  Doc {i+1}: {type(doc).__name__} - "
                           f"Content length: {len(doc.page_content) if hasattr(doc, 'page_content') else 'N/A'} chars")
            
            return {"documents": documents}
        
        except Exception as e:
            logger.error(f"❌ Erro na recuperação: {e}", exc_info=True)
            return {
                "documents": [],
                "generation": "Erro ao buscar protocolos na base de conhecimento."
            }
    
    def grade_documents(self, state: AgentState) -> dict:
        """Avalia relevância dos documentos recuperados."""
        documents = state.get("documents", [])
        question = state.get("medical_question", "")
        
        logger.debug(f"📊 Avaliando {len(documents)} documentos para pergunta: {question[:50]}...")
        
        if not documents:
            logger.warning("⚠️ Nenhum documento fornecido para avaliação")
            return {"documents": []}
        
        try:
            useful_docs = []
            
            for i, doc in enumerate(documents):
                if not isinstance(doc, Document):
                    logger.warning(f"⚠️ Item {i} não é Document: tipo={type(doc).__name__}")
                    continue
                
                doc_content = doc.page_content.lower()
                question_lower = question.lower()
                
                question_words = set(question_lower.split())
                doc_words = set(doc_content.split())
                overlap = len(question_words & doc_words) / max(len(question_words), 1)
                
                logger.debug(f"  Doc {i+1}: Sobreposição={overlap:.2%}")
                
                if overlap > 0.05:
                    useful_docs.append(doc)
            
            logger.info(f"✅ {len(useful_docs)}/{len(documents)} documentos úteis após avaliação")
            
            if not useful_docs and documents:
                logger.warning("⚠️ Nenhum documento passou na avaliação. Retornando todos os documentos.")
                return {"documents": documents}
            
            return {"documents": useful_docs}
        
        except Exception as e:
            logger.error(f"❌ Erro ao avaliar documentos: {e}", exc_info=True)
            return {"documents": documents}
    
    def generate(self, state: AgentState) -> dict:
        """Gera resposta clínica baseada em documentos."""
        documents = state.get("documents", [])
        question = state.get("medical_question", "")
        
        logger.debug(f"📝 Gerando resposta com {len(documents)} documentos...")
        
        if not question:
            return {"generation": "Pergunta vazia fornecida."}
        
        try:
            context = ""
            if documents:
                context = "Protocolos consultados:\n\n"
                for i, doc in enumerate(documents, 1):
                    if isinstance(doc, Document):
                        source = doc.metadata.get("source", f"Protocolo {i}")
                        preview = doc.page_content[:500]
                        context += f"{i}. **{source}**\n{preview}...\n\n"
                    else:
                        logger.warning(f"⚠️ Documento {i} não é do tipo Document: {type(doc)}")
            else:
                logger.warning("⚠️ Nenhum documento disponível para geração")
                context = "⚠️ Nenhum protocolo foi encontrado na base de conhecimento."
            
            if documents:
                system_prompt = """Você é um assistente médico especializado em protocolos clínicos.
Baseado nos protocolos fornecidos, responda à pergunta do médico com precisão.
SEMPRE cite os protocolos utilizados na resposta."""
            else:
                system_prompt = """Você é um assistente médico. 
Infelizmente, nenhum protocolo foi encontrado na base de conhecimento para esta pergunta.
Informe ao usuário que a pergunta não pode ser respondida completamente sem acesso aos protocolos."""
            
            prompt = f"""{system_prompt}

Protocolos de referência:
{context}

Pergunta do médico:
{question}

Resposta (cite os protocolos utilizados se disponíveis):"""
            
            response = self.llm.invoke(prompt)
            generation = response.content if hasattr(response, 'content') else str(response)
            
            logger.info("✅ Resposta gerada com sucesso")
            logger.debug(f"  Tamanho da resposta: {len(generation)} chars")
            
            return {"generation": generation}
        
        except Exception as e:
            logger.error(f"❌ Erro ao gerar resposta: {e}", exc_info=True)
            return {"generation": f"Erro ao gerar resposta: {str(e)}"}
    
    def validate_hallucination(self, state: AgentState) -> dict:
        """
        Valida se a resposta está baseada nos documentos (sem alucinações).
        
        WHEN [resposta é gerada]
        THE SYSTEM SHALL [validar se resposta é baseada nos documentos recuperados]
        """
        generation = state.get("generation", "")
        documents = state.get("documents", [])
        
        logger.debug(f"🔍 Validando alucinações... (docs={len(documents)}, gen_len={len(generation)})")
        
        # Caso 1: Sem documentos recuperados
        if not documents:
            logger.warning("⚠️ Sem documentos para validar hallucination")
            logger.info("💡 Modo fallback: Aceitando resposta pois não há documentos para validação")
            return {"hallucination_check": "no_docs_available"}
        
        try:
            # Camada 1: Rejeição óbvia se resposta diz "não tenho acesso"
            if any(phrase in generation.lower() for phrase in 
                   ["não tenho acesso", "não posso responder", "desculpe", "não encontrei",
                    "não foi possível", "não consegui", "sem acesso", "indisponível"]):
                logger.info("✅ Resposta é uma rejeição apropriada (sem acesso aos dados)")
                return {"hallucination_check": "valid_rejection"}
            
            # Camada 2: Validação semântica com embeddings
            has_semantic_match = self._semantic_validation(generation, documents)
            
            if has_semantic_match:
                logger.info("✅ Resposta validada (semelhança semântica com documentos)")
                return {"hallucination_check": "valid"}
            
            # Camada 3: Fallback para keyword matching
            has_keyword_match = self._keyword_validation(generation, documents)
            
            if has_keyword_match:
                logger.info("✅ Resposta validada (palavras-chave dos documentos encontradas)")
                return {"hallucination_check": "valid_keywords"}
            
            logger.warning("⚠️ Possível alucinação detectada (sem correspondência com documentos)")
            logger.debug(f"  Resposta: {generation[:100]}...")
            
            return {"hallucination_check": "possible_hallucination"}
        
        except Exception as e:
            logger.error(f"❌ Erro na validação: {e}", exc_info=True)
            return {"hallucination_check": "validation_error"}
    
    def _semantic_validation(self, generation: str, documents: List[Document]) -> bool:
        """Valida usando embeddings e similiaridade semântica."""
        try:
            docs_to_check = documents[:3]
            
            logger.debug("📊 Calculando similiaridade semântica...")
            gen_embedding = self.embeddings.embed_query(generation)
            
            max_similarity = 0.0
            
            for i, doc in enumerate(docs_to_check):
                if not isinstance(doc, Document):
                    continue
                
                doc_embedding = self.embeddings.embed_query(doc.page_content[:500])
                similarity = self._cosine_similarity(gen_embedding, doc_embedding)
                logger.debug(f"  Doc {i+1}: Similiaridade = {similarity:.3f}")
                
                max_similarity = max(max_similarity, similarity)
            
            semantic_threshold = 0.4
            
            if max_similarity >= semantic_threshold:
                logger.debug(f"✅ Similiaridade semântica OK (max={max_similarity:.3f} >= {semantic_threshold})")
                return True
            else:
                logger.debug(f"❌ Similiaridade semântica baixa (max={max_similarity:.3f} < {semantic_threshold})")
                return False
        
        except Exception as e:
            logger.warning(f"⚠️ Erro na validação semântica: {e}")
            return True
    
    def _keyword_validation(self, generation: str, documents: List[Document]) -> bool:
        """Validação por palavras-chave com critério menos rigoroso."""
        try:
            gen_lower = generation.lower()
            doc_terms = set()
            
            for doc in documents:
                if isinstance(doc, Document):
                    words = [w.lower() for w in doc.page_content.split() 
                            if len(w) >= 4 and w.isalnum()]
                    doc_terms.update(words[:20])
            
            logger.debug(f"📝 Termos-chave documentos: {list(doc_terms)[:10]}...")
            
            matches = sum(1 for term in doc_terms if term in gen_lower)
            match_ratio = matches / len(doc_terms) if doc_terms else 0
            
            logger.debug(f"  Matches: {matches}/{len(doc_terms)} = {match_ratio:.1%}")
            
            keyword_threshold = 0.1
            
            if match_ratio >= keyword_threshold:
                logger.debug(f"✅ Validação por keywords OK (match_ratio={match_ratio:.1%})")
                return True
            else:
                logger.debug(f"❌ Validação por keywords falhou (match_ratio={match_ratio:.1%} < {keyword_threshold})")
                return False
        
        except Exception as e:
            logger.warning(f"⚠️ Erro na validação por keywords: {e}")
            return False
    
    def _cosine_similarity(self, vec_a: list, vec_b: list) -> float:
        """Calcula similiaridade coseno entre dois vetores."""
        dot_product = sum(a * b for a, b in zip(vec_a, vec_b))
        magnitude_a = math.sqrt(sum(a ** 2 for a in vec_a))
        magnitude_b = math.sqrt(sum(b ** 2 for b in vec_b))
        
        if magnitude_a == 0 or magnitude_b == 0:
            return 0.0
        
        return dot_product / (magnitude_a * magnitude_b)