#!/usr/bin/env python3
"""
Assistente Médico Virtual - Interface de Chat Interativa

Este script fornece uma interface de linha de comando para um assistente de IA de suporte
a decisões médicas que usa RAG (Retrieval-Augmented Generation) com bases de conhecimento
médico para responder perguntas relacionadas à saúde de forma segura e precisa.

O assistente é construído usando:
- LangChain: Para orquestração de fluxos de trabalho de IA
- LangGraph: Para criação de máquinas de estado e grafos de decisão
- Google Gemini: Como o modelo de linguagem principal
- ChromaDB: Para armazenamento e recuperação de documentos baseada em vetores
"""

import sys
from pathlib import Path
import uuid
from typing import Any, Dict

# === CONFIGURAÇÃO DO PROJETO ===
# IMPORTANTE: Configurar o caminho ANTES de qualquer importação de módulos src
# Adiciona o diretório raiz do projeto ao caminho de busca de módulos do Python
# Isso permite importar módulos usando importações absolutas como 'src.module'
project_root = Path(__file__).parent.parent.resolve()
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

# === IMPORTAÇÕES ===
# Agora podemos importar os módulos src após configurar o caminho
from loguru import logger

from src.config import settings
from src.domain.state import AgentState
from src.use_cases.graph import GraphBuilder
from src.utils.logging import setup_logging

# === CONSTANTES DE CONFIGURAÇÃO ===
# Constantes de exibição da aplicação para melhor manutenibilidade
APP_NAME = "Assistente Médico Virtual"
APP_DESCRIPTION = "Sistema de suporte a decisões clínicas baseado em protocolos internos"
TECH_STACK = "Desenvolvido com LangChain + LangGraph + Google Gemini"
DIVIDER_LENGTH = 70

# Comandos de saída que os usuários podem digitar para encerrar a aplicação
EXIT_COMMANDS = ("sair", "exit", "quit", "q")


def create_initial_agent_state(user_question: str) -> AgentState:
    """
    Cria o estado inicial para o pipeline de processamento do agente de IA.
    
    Esta função configura a estrutura de dados que fluirá através
    de todo o fluxo de trabalho RAG (Retrieval-Augmented Generation).
    
    Args:
        user_question (str): A pergunta médica do usuário
        
    Returns:
        AgentState: Dicionário de estado inicial com todos os campos necessários
        
    Por que isso é importante:
    - O AgentState atua como uma memória compartilhada entre as etapas de processamento
    - Cada etapa do pipeline pode ler e modificar este estado
    - Isso garante consistência de dados em todo o fluxo de trabalho
    """
    return {
        "request_id": str(uuid.uuid4()),
        # Entrada principal do usuário
        "medical_question": user_question,
        
        # Flags de segurança e validação
        "is_safe": True,          # Se a pergunta passou pelas verificações de segurança
        "is_valid": True,         # Se a resposta foi validada
        "risk_level": "low",      # Nível de avaliação de risco
        "requires_human_validation": False,
        "safety_reason": None,
        
        # Resultados da recuperação de documentos
        "documents": [],          # Protocolos médicos recuperados da base de dados
        
        # Geração de resposta da IA
        "generation": "",         # A resposta gerada pela IA
        "hallucination_check": "", # Validação contra documentos fonte
        
        # Contexto e histórico
        "context_data": "",       # Contexto adicional se necessário
        "chat_history": [],       # Turnos de conversa anteriores
        "loop_count": 0,         # Contador de iterações de processamento
    }


def display_welcome_message() -> None:
    """
    Exibe a tela de boas-vindas da aplicação e instruções.
    
    Isso cria uma interface profissional com tema médico que:
    - Identifica claramente o propósito da aplicação
    - Mostra a pilha de tecnologia para transparência
    - Fornece instruções claras de uso
    """
    print("\n" + "=" * DIVIDER_LENGTH)
    print(f"🏥 {APP_NAME}")
    print("=" * DIVIDER_LENGTH)
    print(f"\n📋 {APP_DESCRIPTION}")
    print(f"{TECH_STACK}\n")


def display_chat_instructions() -> None:
    """Exibe instruções para a sessão de chat interativa."""
    print("-" * DIVIDER_LENGTH)
    print("💬 Digite suas dúvidas clínicas (ou 'sair' para encerrar)\n")


def get_user_input() -> str:
    """
    Obtém e valida a entrada do usuário da linha de comando.
    
    Returns:
        str: A entrada do usuário, sem espaços em branco
        
    Por que removemos espaços em branco:
    - Previne o processamento de entradas vazias ou apenas com espaços
    - Garante formato consistente de entrada para processamento posterior
    """
    return input("👨‍⚕️  Você: ").strip()


def should_exit_application(user_input: str) -> bool:
    """
    Verifica se o usuário quer sair da aplicação.
    
    Args:
        user_input (str): A entrada do usuário para verificar
        
    Returns:
        bool: True se o usuário quer sair, False caso contrário
        
    Esta função fornece flexibilidade aceitando múltiplos comandos de saída
    em diferentes idiomas (Português, Inglês) para melhor experiência do usuário.
    """
    return user_input.lower() in EXIT_COMMANDS


def is_empty_input(user_input: str) -> bool:
    """
    Verifica se o usuário forneceu entrada vazia.
    
    Args:
        user_input (str): A entrada do usuário para validar
        
    Returns:
        bool: True se a entrada estiver vazia, False caso contrário
    """
    return not user_input


def process_medical_question(app, user_question: str) -> Dict[str, Any]:
    """
    Processa uma pergunta médica através do pipeline de IA.
    
    Args:
        app: A aplicação LangGraph compilada
        user_question (str): A pergunta médica para processar
        
    Returns:
        Dict[str, Any]: O estado final após processamento através de todas as etapas do pipeline
        
    Etapas do Pipeline:
    1. Guardrails: Verifica se a pergunta é segura e medicamente relevante
    2. Recuperação: Encontra documentos relevantes da base de conhecimento
    3. Classificação: Filtra documentos por relevância
    4. Geração: Cria resposta da IA baseada nos documentos recuperados
    5. Validação: Verifica resposta por precisão e alucinações
    """
    logger.debug(f"Processando pergunta: {user_question[:60]}...")
    print("\n🔍 Processando pergunta...\n")
    
    # Cria estado inicial para o pipeline de processamento
    initial_state = create_initial_agent_state(user_question)
    
    # Executa o pipeline RAG completo
    # Isso passará o estado através de todos os nós de processamento em sequência
    # O modelo usa temperatura 0.0 para respostas determinísticas e consistentes
    final_state = app.invoke(initial_state)
    
    return final_state


def display_response(result: Dict[str, Any]) -> None:
    """
    Exibe a resposta da IA para o usuário com formatação apropriada.
    
    Args:
        result (Dict[str, Any]): O estado final do pipeline de IA
        
    Esta função trata diferentes cenários de resposta:
    - Perguntas inseguras (rejeitadas pelos guardrails)
    - Respostas inválidas (falharam na validação)
    - Respostas bem-sucedidas com citações de fonte
    """
    # Verifica se a pergunta foi rejeitada pelos guardrails de segurança
    # Os guardrails validam se a pergunta é medicamente relevante e não contém PII
    if result.get("is_safe") is False:
        logger.warning("Pergunta rejeitada pelos guardrails")
        fallback_message = "Pergunta fora do escopo médico."
        response = result.get("generation", fallback_message)
        print(f"⚠️  Assistente: {response}\n")
        return
    
    # Verifica se a resposta falhou na validação (ex: alucinação detectada)
    if result.get("is_valid") is False:
        logger.warning("Resposta rejeitada devido a falha na validação")
        fallback_message = "Não foi possível gerar uma resposta confiável."
        response = result.get("generation", fallback_message)
        print(f"⚠️  Assistente: {response}\n")
        return
    
    # Exibe resposta bem-sucedida
    logger.info("✅ Resposta gerada com sucesso")
    fallback_message = "Desculpe, não consegui processar a pergunta."
    response = result.get("generation", fallback_message)
    print(f"🤖 Assistente: {response}\n")


def display_sources(result: Dict[str, Any]) -> None:
    """
    Exibe os protocolos médicos/fontes consultados para a resposta.
    
    Args:
        result (Dict[str, Any]): O estado final contendo documentos recuperados
        
    Esta função fornece transparência mostrando aos usuários quais
    protocolos médicos foram consultados, permitindo verificar as fontes de informação.
    """
    documents = result.get("documents", [])
    if not documents:
        return
    
    print("📚 Protocolos consultados:")
    # Limita às primeiras 3 fontes para manter a exibição limpa
    for document in documents[:3]:
        source_name = document.metadata.get("source", "Fonte desconhecida")
        print(f"   • {source_name}")
    
    # Mostra contagem se mais fontes foram usadas
    if len(documents) > 3:
        remaining_count = len(documents) - 3
        print(f"   ... e mais {remaining_count} protocolos")
    
    print()  # Adiciona espaçamento após as fontes


def handle_chat_loop(app) -> None:
    """
    Gerencia o loop principal de chat interativo com o usuário.
    
    Args:
        app: A aplicação LangGraph compilada para processar perguntas
        
    Esta função gerencia todo o ciclo de vida da interação do usuário:
    - Obtendo entrada do usuário
    - Processando perguntas médicas
    - Exibindo respostas e fontes
    - Tratando erros graciosamente
    - Gerenciando saída da aplicação
    """
    display_chat_instructions()
    
    while True:
        try:
            # Obtém entrada do usuário
            user_input = get_user_input()
            
            # Verifica comandos de saída
            if should_exit_application(user_input):
                print("\n👋 Encerrando assistente médico. Até logo!\n")
                logger.info("🛑 Usuário encerrou a sessão")
                break
            
            # Valida se a entrada não está vazia
            if is_empty_input(user_input):
                print("⚠️  Digite uma pergunta válida\n")
                continue
            
            # Processa a pergunta médica através do pipeline de IA
            result = process_medical_question(app, user_input)
            
            # Exibe resposta e fontes para o usuário
            display_response(result)
            display_sources(result)
            
        except KeyboardInterrupt:
            # Trata Ctrl+C graciosamente
            print("\n\n👋 Interrupção do usuário. Encerrando...\n")
            logger.info("🛑 Interrompido por Ctrl+C")
            break
            
        except Exception as error:
            # Trata erros inesperados durante o processamento de perguntas
            logger.error(f"❌ Erro ao processar pergunta: {error}", exc_info=True)
            print(f"❌ Erro técnico: {error}\n")


def initialize_ai_system() -> object:
    """
    Inicializa o sistema de IA construindo o grafo de processamento.
    
    Returns:
        object: A aplicação LangGraph compilada pronta para uso
        
    Raises:
        Exception: Se a inicialização falhar
        
    Esta função configura todo o pipeline de IA incluindo:
    - Carregamento de modelos de linguagem
    - Conexão com base de dados vetorial
    - Construção do grafo de processamento
    - Compilação do fluxo de trabalho
    """
    logger.info("🔨 Inicializando grafo de orquestração...")
    
    # Cria e configura o construtor do grafo
    graph_builder = GraphBuilder()
    
    # Constrói e compila o grafo de processamento
    # Isso cria o fluxo de trabalho que processará perguntas dos usuários
    compiled_app = graph_builder.build()
    
    logger.info("✅ Grafo inicializado com sucesso\n")
    return compiled_app


def main() -> None:
    """
    Ponto de entrada principal da aplicação.
    
    Esta função orquestra todo o ciclo de vida da aplicação:
    1. Configura sistema de logging
    2. Exibe mensagem de boas-vindas
    3. Inicializa componentes do sistema de IA
    4. Inicia loop de chat interativo
    5. Trata quaisquer erros críticos
    
    A função é projetada com tratamento adequado de erros para garantir
    uma boa experiência do usuário mesmo quando algo dá errado.
    """
    # Inicializa sistema de logging para depuração e monitoramento
    setup_logging(level=settings.log_level)
    
    # Exibe mensagem de boas-vindas e informações da aplicação
    display_welcome_message()
    
    try:
        # Inicializa o sistema de IA (LLM, base vetorial, grafo de processamento)
        ai_app = initialize_ai_system()
        
        # Inicia a sessão de chat interativa com o usuário
        handle_chat_loop(ai_app)
        
    except Exception as critical_error:
        # Trata erros críticos de inicialização
        logger.critical(f"❌ Erro crítico de inicialização: {critical_error}", exc_info=True)
        print(f"\n❌ ERRO CRÍTICO: {critical_error}")
        print("Verifique os logs em logs/assistente-medico-errors.log para mais detalhes.\n")
        sys.exit(1)


# === PONTO DE ENTRADA DA APLICAÇÃO ===
# Isso garante que main() só execute quando o script for executado diretamente,
# não quando importado como módulo
if __name__ == "__main__":
    main()