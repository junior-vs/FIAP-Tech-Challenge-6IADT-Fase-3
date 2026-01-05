"""
Módulo: src/use_cases/nodes.py
Descrição: Implementação dos nós do grafo (Passos da execução).
Motivo da alteração: 
- Alteração dos Prompts para persona "Assistente Médico".
- Uso das chaves do novo AgentState (medical_question, is_safe).
- Inclusão de instruções de segurança (não prescrever sem validação).
"""

import re
from typing import List, Dict, Any
from langchain_core.documents import Document
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from src.domain.state import AgentState
from src.domain.guardrails import GuardrailsGrade, HallucinationGrade, DocumentGrade, GuardrailsValidationResult
from src.domain.guardrails import GuardrailsValidator
from src.infrastructure.llm_factory import LLMFactory
from src.infrastructure.vector_store import VectorStoreRepository
from src.utils.logging import logger

class RAGNodes:
    """
    Nós do grafo RAG para processamento de perguntas médicas.
    Implementa validação, recuperação, classificação e geração de respostas.
    """
    
    def __init__(self, retriever, llm):
        self.retriever = retriever
        self.llm = llm
        self.local_guardrails = GuardrailsValidator()

        self._prescription_patterns = [
            r"\bprescrev\w+\b",
            r"\breceit\w+\b",
            r"\bposolog\w+\b",
            r"\bdose\b",
            r"\bmg\b",
            r"\bml\b",
            r"\bvia\s+(oral|iv|im|sc|subcut)\w*\b",
            r"\bquantas\s+vezes\b",
            r"\ba\s+cada\s+\d+\s*(h|horas)\b",
        ]

        # Chain para validação de guardrails com prompt aprimorado
        self.guardrails_prompt = ChatPromptTemplate.from_messages([
            (
                "system",
                """Você é um classificador especializado em identificar perguntas médicas e clínicas.

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
- "inválida" se a pergunta está claramente fora do escopo médico""",
            ),
            ("human", "Pergunta: {question}"),
        ])

        self.guardrails_chain = self.guardrails_prompt | self.llm | StrOutputParser()

        # Chain para validação estruturada (backup)
        self.structured_guardrails_prompt = ChatPromptTemplate.from_messages([
            (
                "system",
                """Você é um auditor de conformidade médica. Analise se a pergunta está relacionada ao contexto médico/saúde.

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

Responda no formato JSON especificado.""",
            ),
            ("human", "Pergunta: {question}"),
        ])

        self.structured_guardrails_chain = (
            self.structured_guardrails_prompt | self.llm.with_structured_output(GuardrailsGrade)
        )

        # Chain para classificação de documentos
        self.grader_prompt = ChatPromptTemplate.from_messages([
            (
                "system",
                """Você é um classificador que avalia se um documento recuperado é relevante para uma pergunta médica.

Analise o conteúdo do documento e determine se ele contém informações úteis para responder à pergunta.

Responda no formato JSON especificado.""",
            ),
            ("human", "Pergunta: {question}\n\nDocumento: {document}"),
        ])

        self.retrieval_grader = self.grader_prompt | self.llm.with_structured_output(DocumentGrade)

        # Chain para geração de respostas
        self.rag_prompt = ChatPromptTemplate.from_messages([
            (
                "system",
                """Você é um assistente médico especializado que fornece informações baseadas em protocolos clínicos.

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

IMPORTANTE: Esta é uma ferramenta de apoio à decisão médica. Sempre recomende consulta com profissional de saúde para decisões clínicas.""",
            ),
            ("human", "Pergunta: {question}\n\nContexto dos protocolos:\n{context}"),
        ])

        self.rag_chain = self.rag_prompt | self.llm | StrOutputParser()

        # Chain para detecção de alucinações
        self.hallucination_prompt = ChatPromptTemplate.from_messages([
            (
                "system",
                """Você é um verificador que determina se uma resposta do LLM está baseada nos documentos fornecidos.

Analise se a resposta contém APENAS informações presentes nos documentos ou se há conteúdo adicional não fundamentado.

Responda no formato JSON especificado com:
- is_grounded: "sim" se a resposta está totalmente baseada nos documentos, "não" se contém informações extras
- confidence: seu nível de confiança na avaliação
- issues: problemas específicos encontrados (se houver)""",
            ),
            ("human", "Documentos: {documents}\n\nResposta do LLM: {generation}"),
        ])

        self.hallucination_grader = (
            self.hallucination_prompt | self.llm.with_structured_output(HallucinationGrade)
        )

    def _needs_human_validation(self, question: str) -> bool:
        q = question.lower()
        return any(re.search(p, q) for p in self._prescription_patterns)

    def _sanitize_generation_for_grounding_check(self, generation: str) -> str:
        """Remove trechos padronizados que não precisam estar nas fontes.

        A checagem de groundedness deve focar no conteúdo factual/clínico.
        Disclaimers, avisos de segurança e a lista de fontes podem causar falso
        positivo, já que não aparecem literalmente no corpo dos documentos.
        """

        if not generation:
            return ""

        text = generation

        # 1) Remover a seção "Fontes:" anexada pelo sistema.
        # Mantém apenas o que vem antes das fontes.
        text = re.split(r"\n\s*Fontes\s*:\s*\n", text, maxsplit=1, flags=re.IGNORECASE)[0]

        # 2) Remover avisos padronizados (não substitui profissional, etc.).
        disclaimer_patterns = [
            r"\n?\s*⚠️\s*Observação:.*$",
            r"\n?\s*Observa(c|ç)ão:.*$",
            r"\n?\s*IMPORTANTE:.*$",
            r"\n?\s*Esta\s+é\s+uma\s+ferramenta\s+de\s+apoio\s+à\s+decisão\s+médica\..*$",
            r"\n?\s*Isso\s+n(ã|a)o\s+substitui\s+.*$",
            r"\n?\s*Sempre\s+recomende\s+consulta\s+.*$",
            r"\n?\s*Procure\s+(um|uma)\s+profissional\s+de\s+sa(ú|u)de\..*$",
        ]
        for pattern in disclaimer_patterns:
            text = re.sub(pattern, "", text, flags=re.IGNORECASE | re.DOTALL)

        return text.strip()

    def guardrails(self, state: AgentState) -> AgentState:
        """
        Valida se a pergunta é apropriada para o contexto médico.
        Implementa verificação de segurança e relevância com análise semântica.
        """
        question = state["medical_question"]
        request_id = state.get("request_id", "-")
        log = logger.bind(request_id=request_id)
        log.info("Segurança: avaliando pergunta")

        # 1) Validações locais determinísticas (PII / relevância mínima)
        local_result = self.local_guardrails.validate_with_result(question)
        if not local_result.is_valid:
            log.warning(f"Segurança: bloqueada na validação local ({local_result.reason})")
            return {
                **state,
                "is_safe": False,
                "risk_level": "alto",
                "safety_reason": local_result.reason,
                "generation": local_result.reason or "Pergunta rejeitada por validação de segurança.",
            }

        # 2) Sinalizar perguntas que pedem prescrição/conduta (limite de atuação)
        requires_human_validation = self._needs_human_validation(question)
        if requires_human_validation:
            log.warning("Segurança: pedido de prescrição/posologia (exige validação humana)")
            state = {
                **state,
                "requires_human_validation": True,
                "risk_level": "médio",
                "safety_reason": "Pergunta solicita prescrição/posologia; requer validação humana.",
            }
        
        try:
            # Primeira tentativa com chain simples
            try:
                result = self.guardrails_chain.invoke({"question": question})
                result_clean = result.strip().lower()
                
                # Análise mais flexível do resultado
                is_valid = any(term in result_clean for term in ['válida', 'valid', 'sim', 'yes', 'aceita', 'accept'])
                is_invalid = any(term in result_clean for term in ['inválida', 'invalid', 'não', 'no', 'rejeita', 'reject'])
                
                if is_valid and not is_invalid:
                    log.info("Segurança: aprovada pelo classificador")
                    return {**state, "is_safe": True, "risk_level": "baixo"}
                elif is_invalid and not is_valid:
                    log.warning("Segurança: bloqueada pelo classificador (fora do escopo de saúde)")
                    return {
                        **state, 
                        "is_safe": False, 
                        "risk_level": "alto",
                        "safety_reason": "Pergunta fora do escopo médico.",
                        "generation": "Desculpe, mas essa pergunta está fora do escopo médico que posso ajudar. Por favor, faça uma pergunta relacionada à saúde ou medicina."
                    }
                else:
                    # Resultado ambíguo, usar chain estruturada como backup
                    log.info("Segurança: resultado ambíguo; usando validação estruturada")
                    raise Exception("Resultado ambíguo")
                    
            except Exception as e:
                log.info(f"Segurança: fallback para validação estruturada ({str(e)})")
                
                # Usar chain estruturada como backup
                structured_result = self.structured_guardrails_chain.invoke({"question": question})
                
                # Verificar se a resposta está fundamentada nos documentos
                if hasattr(structured_result, 'is_safe') and structured_result.is_safe == "sim":
                    log.info("Segurança: aprovada (validação estruturada)")
                    return {
                        **state, 
                        "is_safe": True, 
                        "risk_level": getattr(structured_result, 'risk_level', 'baixo')
                    }
                elif hasattr(structured_result, 'is_safe'):
                    log.warning("Segurança: bloqueada (validação estruturada)")
                    return {
                        **state, 
                        "is_safe": False, 
                        "risk_level": getattr(structured_result, 'risk_level', 'alto'),
                        "safety_reason": "Pergunta fora do escopo médico.",
                        "generation": "Desculpe, mas essa pergunta está fora do escopo médico que posso ajudar. Por favor, faça uma pergunta relacionada à saúde ou medicina."
                    }
                else:
                    # Fallback: tentar acessar como dict
                    if isinstance(structured_result, dict) and structured_result.get('is_safe') == "sim":
                        log.info("Segurança: aprovada (validação estruturada)")
                        return {
                            **state, 
                            "is_safe": True, 
                            "risk_level": structured_result.get('risk_level', 'baixo')
                        }
                    else:
                        log.warning("Segurança: não consegui interpretar o resultado da validação")
                        return {
                            **state, 
                            "is_safe": False, 
                            "risk_level": "alto",
                            "safety_reason": "Erro na validação da pergunta.",
                            "generation": "Erro na validação da pergunta. Por favor, tente novamente."
                        }
        
        except Exception as e:
            log.error(f"Segurança: erro inesperado na validação ({str(e)})")
            # Em caso de erro, assumir que é seguro para não bloquear perguntas médicas válidas
            log.warning("Segurança: falhou; vou assumir válida para não bloquear perguntas médicas")
            return {**state, "is_safe": True, "risk_level": "baixo"}

    def retrieve(self, state: AgentState) -> AgentState:
        """
        Recupera documentos relevantes usando busca semântica por vetor.
        """
        question = state["medical_question"]
        log = logger.bind(request_id=state.get("request_id", "-"))
        log.info("Busca: consultando a base de protocolos")
        
        try:
            documents = self.retriever.invoke(question)
            log.info(f"Busca: encontrei {len(documents)} documentos")
            
            return {**state, "documents": documents}
        
        except Exception as e:
            log.error(f"Busca: erro ao recuperar documentos ({str(e)})")
            return {**state, "documents": []}

    def grade_documents(self, state: AgentState) -> AgentState:
        """
        Classifica documentos recuperados quanto à relevância para a pergunta.
        """
        question = state["medical_question"]
        documents = state["documents"]
        
        log = logger.bind(request_id=state.get("request_id", "-"))
        log.info(f"Relevância: avaliando {len(documents)} documentos")
        
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
                    log.warning(f"Relevância: falha ao avaliar um documento; mantendo por segurança ({str(e)})")
                    # Em caso de erro, manter o documento
                    filtered_docs.append(doc)
            
            log.info(f"Relevância: mantive {len(filtered_docs)}/{len(documents)} documentos")
            
            return {**state, "documents": filtered_docs}
        
        except Exception as e:
            log.error(f"Relevância: erro ao classificar documentos ({str(e)})")
            return state

    def generate(self, state: AgentState) -> AgentState:
        """
        Gera resposta baseada nos documentos recuperados e na pergunta.
        """
        question = state["medical_question"]
        documents = state["documents"]
        
        log = logger.bind(request_id=state.get("request_id", "-"))
        log.info("Resposta: gerando com base nos documentos")
        
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

            # Explainability: anexar fontes usadas, mesmo se o LLM não citar corretamente.
            sources = []
            for doc in documents:
                src = doc.metadata.get("source") if hasattr(doc, "metadata") else None
                if src and src not in sources:
                    sources.append(src)
            if sources:
                generation = (
                    generation.strip()
                    + "\n\nFontes:\n"
                    + "\n".join(f"- {s}" for s in sources[:10])
                )

            # Limite de atuação: reforçar validação humana quando aplicável.
            if state.get("requires_human_validation"):
                generation = (
                    generation.strip()
                    + "\n\n⚠️ Observação: esta resposta não substitui validação humana; não prescrevo posologia."
                )
            
            log.info("Resposta: pronta")
            
            return {**state, "generation": generation}
        
        except Exception as e:
            log.error(f"Resposta: erro ao gerar ({str(e)})")
            return {**state, "generation": "Desculpe, ocorreu um erro ao gerar a resposta. Tente novamente."}

    def validate_response(self, state: AgentState) -> AgentState:
        """
        Valida se a resposta gerada é baseada nos documentos fornecidos.
        Detecta possíveis alucinações do modelo.
        """
        generation = state["generation"]
        documents = state["documents"]
        
        log = logger.bind(request_id=state.get("request_id", "-"))
        log.info("Confiabilidade: checando se a resposta está sustentada pelas fontes")
        
        try:
            # Preparar contexto dos documentos para verificação
            docs_content = "\n".join([doc.page_content for doc in documents])

            # Focar a checagem no conteúdo factual; ignorar trechos padronizados.
            generation_for_check = self._sanitize_generation_for_grounding_check(generation)
            
            # Verificar se há alucinação usando chain estruturada
            grade = self.hallucination_grader.invoke({
                "documents": docs_content,
                "generation": generation_for_check,
            })
            
            # Verificar se a resposta está fundamentada nos documentos
            if hasattr(grade, 'is_grounded') and grade.is_grounded == "sim":
                log.info("Confiabilidade: ok (baseada nas fontes)")
                return {**state, "is_valid": True, "hallucination_check": "approved"}
            elif hasattr(grade, 'is_grounded'):
                log.warning(f"Confiabilidade: possível conteúdo não suportado ({getattr(grade, 'issues', 'Sem detalhes')})")
                return {**state, "is_valid": False, "hallucination_check": "rejected"}
            else:
                # Fallback: tentar acessar como dict
                if isinstance(grade, dict) and grade.get('is_grounded') == "sim":
                    log.info("Confiabilidade: ok (baseada nas fontes)")
                    return {**state, "is_valid": True, "hallucination_check": "approved"}
                else:
                    log.warning("Confiabilidade: não consegui interpretar o resultado da checagem")
                    return {**state, "is_valid": False, "hallucination_check": "format_error"}
        
        except Exception as e:
            log.error(f"Confiabilidade: erro na checagem ({str(e)})")
            # Em caso de erro, assumir que é válida para não bloquear respostas médicas
            log.warning("Confiabilidade: falhou; vou assumir válida para não bloquear respostas úteis")
            return {**state, "is_valid": True, "hallucination_check": "error_assumed_valid"}