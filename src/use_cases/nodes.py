"""
Módulo: src/use_cases/nodes.py
Descrição: Implementação dos nós do grafo (Passos da execução).
Motivo da alteração: 
- Alteração dos Prompts para persona "Assistente Médico".
- Uso das chaves do novo AgentState (medical_question, is_safe).
- Inclusão de instruções de segurança (não prescrever sem validação).
"""

import logging
import re
from typing import List, Dict, Any
from langchain_core.documents import Document
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from src.domain.state import AgentState
from src.domain.guardrails import GuardrailsGrade, HallucinationGrade, DocumentGrade
from src.infrastructure.llm_factory import LLMFactory
from src.infrastructure.vector_store import VectorStoreRepository
from src.utils.logging import logger

logger = logging.getLogger(__name__)

class RAGNodes:
    """
    Nós do grafo RAG para processamento de perguntas médicas.
    Implementa validação, recuperação, classificação e geração de respostas.
    """
    
    def __init__(self, retriever, llm):
        self.retriever = retriever
        self.llm = llm
        
        # Chain para validação de guardrails com prompt aprimorado
        self.guardrails_prompt = ChatPromptTemplate.from_messages([
            ("system", """Você é um classificador especializado em identificar perguntas médicas e clínicas.

Sua tarefa é determinar se uma pergunta está relacionada ao contexto médico, de saúde ou clínico.

CRITÉRIOS PARA PERGUNTAS VÁLIDAS:
- Perguntas sobre condições médicas, doenças, sintomas
- Perguntas sobre tratamentos, medicamentos, protocolos clínicos
- Perguntas sobre anatomia, fisiologia, patologia
- Perguntas sobre diagnósticos, exames, procedimentos
- Perguntas sobre saúde preventiva, cuidados de saúde
- Perguntas sobre especialidades médicas
- Perguntas sobre questões de saúde específicas para diferentes populações (idosos, crianças, etc.)

CRITÉRIOS PARA REJEIÇÃO:
- Perguntas sobre assuntos completamente não-médicos (esportes, culinária, tecnologia geral)
- Solicitações para atividades ilegais ou perigosas
- Perguntas com conteúdo ofensivo ou inadequado

IMPORTANTE: 
- A pergunta pode estar em qualquer idioma (português, inglês, espanhol, etc.)
- Analise o CONTEÚDO SEMÂNTICO, não apenas palavras-chave
- Seja PERMISSIVO para temas relacionados à saúde
- Em caso de dúvida, ACEITE a pergunta

Responda apenas com:
- "válida" se a pergunta está relacionada ao contexto médico/saúde
- "inválida" se a pergunta está claramente fora do escopo médico"""),
            ("human", "Pergunta: {question}")
        ])
        
        self.guardrails_chain = (
            self.guardrails_prompt 
            | self.llm 
            | StrOutputParser()
        )
        
        # Chain para validação estruturada (backup)
        self.structured_guardrails_prompt = ChatPromptTemplate.from_messages([
            ("system", """Você é um auditor de conformidade médica. Analise se a pergunta está relacionada ao contexto médico/saúde.

Considere válidas perguntas sobre:
- Condições médicas e doenças
- Tratamentos e medicamentos  
- Sintomas e diagnósticos
- Protocolos clínicos
- Anatomia e fisiologia
- Saúde preventiva
- Especialidades médicas
- Cuidados de saúde para populações específicas

A pergunta pode estar em qualquer idioma. Analise o significado semântico.

Responda no formato JSON especificado."""),
            ("human", "Pergunta: {question}")
        ])
        
        self.structured_guardrails_chain = (
            self.structured_guardrails_prompt 
            | self.llm.with_structured_output(GuardrailsGrade)
        )
        
        # Chain para classificação de documentos
        self.grader_prompt = ChatPromptTemplate.from_messages([
            ("system", """Você é um classificador que avalia se um documento recuperado é relevante para uma pergunta médica.

Analise o conteúdo do documento e determine se ele contém informações úteis para responder à pergunta.

Responda no formato JSON especificado."""),
            ("human", "Pergunta: {question}\n\nDocumento: {document}")
        ])
        
        self.retrieval_grader = (
            self.grader_prompt 
            | self.llm.with_structured_output(DocumentGrade)
        )
        
        # Chain para geração de respostas
        self.rag_prompt = ChatPromptTemplate.from_messages([
            ("system", """Você é um assistente médico especializado que fornece informações baseadas em protocolos clínicos.

INSTRUÇÕES:
1. Use APENAS as informações dos protocolos fornecidos no contexto
2. Seja preciso e objetivo nas suas respostas
3. Sempre cite a fonte (nome do protocolo) das informações
4. Se a pergunta estiver em outro idioma, responda no mesmo idioma da pergunta
5. Se não houver informação suficiente no contexto, indique claramente

FORMATO DA RESPOSTA:
- Responda de forma clara e estruturada
- Cite as fontes: (Protocolo: nome_do_arquivo.xml)
- Use linguagem profissional mas acessível

IMPORTANTE: Esta é uma ferramenta de apoio à decisão médica. Sempre recomende consulta com profissional de saúde para decisões clínicas."""),
            ("human", "Pergunta: {question}\n\nContexto dos protocolos:\n{context}")
        ])
        
        self.rag_chain = self.rag_prompt | self.llm | StrOutputParser()
        
        # Chain para detecção de alucinações
        self.hallucination_prompt = ChatPromptTemplate.from_messages([
            ("system", """Você é um verificador que determina se uma resposta do LLM está baseada nos documentos fornecidos.

Analise se a resposta contém APENAS informações presentes nos documentos ou se há conteúdo adicional não fundamentado.

Responda no formato JSON especificado com:
- is_grounded: "sim" se a resposta está totalmente baseada nos documentos, "não" se contém informações extras
- confidence: seu nível de confiança na avaliação
- issues: problemas específicos encontrados (se houver)"""),
            ("human", "Documentos: {documents}\n\nResposta do LLM: {generation}")
        ])
        
        self.hallucination_grader = (
            self.hallucination_prompt 
            | self.llm.with_structured_output(HallucinationGrade)
        )

    def guardrails(self, state: AgentState) -> AgentState:
        """
        Valida se a pergunta é apropriada para o contexto médico.
        Implementa verificação de segurança e relevância com análise semântica.
        """
        question = state["medical_question"]
        logger.info(f"🛡️ Validando pergunta: {question}")
        
        try:
            # Primeira tentativa com chain simples
            try:
                result = self.guardrails_chain.invoke({"question": question})
                result_clean = result.strip().lower()
                
                # Análise mais flexível do resultado
                is_valid = any(term in result_clean for term in ['válida', 'valid', 'sim', 'yes', 'aceita', 'accept'])
                is_invalid = any(term in result_clean for term in ['inválida', 'invalid', 'não', 'no', 'rejeita', 'reject'])
                
                if is_valid and not is_invalid:
                    logger.info("✅ Pergunta aprovada pelos guardrails")
                    return {**state, "is_safe": True, "risk_level": "baixo"}
                elif is_invalid and not is_valid:
                    logger.warning(f"⚠️ Pergunta rejeitada: {result}")
                    return {
                        **state, 
                        "is_safe": False, 
                        "risk_level": "alto",
                        "generation": "Desculpe, mas essa pergunta está fora do escopo médico que posso ajudar. Por favor, faça uma pergunta relacionada à saúde ou medicina."
                    }
                else:
                    # Resultado ambíguo, usar chain estruturada como backup
                    logger.info("🔄 Resultado ambíguo, usando validação estruturada")
                    raise Exception("Resultado ambíguo")
                    
            except Exception as e:
                logger.info(f"🔄 Fallback para validação estruturada: {str(e)}")
                
                # Usar chain estruturada como backup
                structured_result = self.structured_guardrails_chain.invoke({"question": question})
                
                # Verificar se a resposta está fundamentada nos documentos
                if hasattr(structured_result, 'is_safe') and structured_result.is_safe == "sim":
                    logger.info("✅ Pergunta aprovada pelos guardrails estruturados")
                    return {
                        **state, 
                        "is_safe": True, 
                        "risk_level": getattr(structured_result, 'risk_level', 'baixo')
                    }
                elif hasattr(structured_result, 'is_safe'):
                    logger.warning("⚠️ Pergunta rejeitada pelos guardrails estruturados")
                    return {
                        **state, 
                        "is_safe": False, 
                        "risk_level": getattr(structured_result, 'risk_level', 'alto'),
                        "generation": "Desculpe, mas essa pergunta está fora do escopo médico que posso ajudar. Por favor, faça uma pergunta relacionada à saúde ou medicina."
                    }
                else:
                    # Fallback: tentar acessar como dict
                    if isinstance(structured_result, dict) and structured_result.get('is_safe') == "sim":
                        logger.info("✅ Pergunta aprovada pelos guardrails (dict format)")
                        return {
                            **state, 
                            "is_safe": True, 
                            "risk_level": structured_result.get('risk_level', 'baixo')
                        }
                    else:
                        logger.warning("⚠️ Formato de resposta inesperado do validador")
                        return {
                            **state, 
                            "is_safe": False, 
                            "risk_level": "alto",
                            "generation": "Erro na validação da pergunta. Por favor, tente novamente."
                        }
        
        except Exception as e:
            logger.error(f"❌ Erro na validação de guardrails: {str(e)}")
            # Em caso de erro, assumir que é seguro para não bloquear perguntas médicas válidas
            logger.warning("⚠️ Erro na validação - assumindo pergunta como válida por segurança")
            return {**state, "is_safe": True, "risk_level": "baixo"}

    def retrieve(self, state: AgentState) -> AgentState:
        """
        Recupera documentos relevantes usando busca semântica por vetor.
        """
        question = state["medical_question"]
        logger.info(f"🔍 Buscando documentos para: {question}")
        
        try:
            documents = self.retriever.invoke(question)
            logger.info(f"✅ Recuperados {len(documents)} documentos relevantes")
            
            return {**state, "documents": documents}
        
        except Exception as e:
            logger.error(f"❌ Erro na recuperação de documentos: {str(e)}")
            return {**state, "documents": []}

    def grade_documents(self, state: AgentState) -> AgentState:
        """
        Classifica documentos recuperados quanto à relevância para a pergunta.
        """
        question = state["medical_question"]
        documents = state["documents"]
        
        logger.info(f"📊 Classificando {len(documents)} documentos")
        
        try:
            filtered_docs = []
            
            for doc in documents:
                try:
                    grade = self.retrieval_grader.invoke({
                        "question": question,
                        "document": doc.page_content
                    })
                    
                    # Verificar se o documento é relevante
                    if hasattr(grade, 'is_relevant') and grade.is_relevant == "sim":
                        filtered_docs.append(doc)
                    elif isinstance(grade, dict) and grade.get('is_relevant') == "sim":
                        filtered_docs.append(doc)
                
                except Exception as e:
                    logger.warning(f"⚠️ Erro ao classificar documento: {str(e)}")
                    # Em caso de erro, manter o documento
                    filtered_docs.append(doc)
            
            logger.info(f"Documentos úteis: {len(filtered_docs)}/{len(documents)}")
            
            return {**state, "documents": filtered_docs}
        
        except Exception as e:
            logger.error(f"❌ Erro na classificação de documentos: {str(e)}")
            return state

    def generate(self, state: AgentState) -> AgentState:
        """
        Gera resposta baseada nos documentos recuperados e na pergunta.
        """
        question = state["medical_question"]
        documents = state["documents"]
        
        logger.info("🤖 Gerando resposta baseada nos protocolos")
        
        try:
            # Preparar contexto dos documentos
            context = "\n\n".join([
                f"Protocolo {i+1}. {doc.metadata.get('source', 'fonte_desconhecida')}: {doc.page_content}"
                for i, doc in enumerate(documents)
            ])
            
            # Gerar resposta
            generation = self.rag_chain.invoke({
                "question": question,
                "context": context
            })
            
            logger.info("✅ Resposta gerada com sucesso")
            
            return {**state, "generation": generation}
        
        except Exception as e:
            logger.error(f"❌ Erro na geração de resposta: {str(e)}")
            return {**state, "generation": "Desculpe, ocorreu um erro ao gerar a resposta. Tente novamente."}

    def validate_response(self, state: AgentState) -> AgentState:
        """
        Valida se a resposta gerada é baseada nos documentos fornecidos.
        Detecta possíveis alucinações do modelo.
        """
        generation = state["generation"]
        documents = state["documents"]
        
        logger.info("🔍 Validando resposta contra documentos fonte")
        
        try:
            # Preparar contexto dos documentos para verificação
            docs_content = "\n".join([doc.page_content for doc in documents])
            
            # Verificar se há alucinação usando chain estruturada
            grade = self.hallucination_grader.invoke({
                "documents": docs_content,
                "generation": generation
            })
            
            # Verificar se a resposta está fundamentada nos documentos
            if hasattr(grade, 'is_grounded') and grade.is_grounded == "sim":
                logger.info("✅ Resposta validada (baseada em documentos)")
                return {**state, "is_valid": True, "hallucination_check": "approved"}
            elif hasattr(grade, 'is_grounded'):
                logger.warning(f"⚠️ Possível alucinação detectada: {getattr(grade, 'issues', 'Sem detalhes')}")
                return {**state, "is_valid": False, "hallucination_check": "rejected"}
            else:
                # Fallback: tentar acessar como dict
                if isinstance(grade, dict) and grade.get('is_grounded') == "sim":
                    logger.info("✅ Resposta validada (baseada em documentos)")
                    return {**state, "is_valid": True, "hallucination_check": "approved"}
                else:
                    logger.warning("⚠️ Formato de resposta inesperado do validador")
                    return {**state, "is_valid": False, "hallucination_check": "format_error"}
        
        except Exception as e:
            logger.error(f"❌ Erro na validação de resposta: {str(e)}")
            # Em caso de erro, assumir que é válida para não bloquear respostas médicas
            logger.warning("⚠️ Erro na validação - assumindo resposta como válida por segurança")
            return {**state, "is_valid": True, "hallucination_check": "error_assumed_valid"}