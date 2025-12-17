#!/usr/bin/env python3
"""
Interface CLI para o Assistente Médico Virtual.
Executa a aplicação de chat interativo.
"""

import sys
from pathlib import Path

# Adicionar diretório raiz do projeto ao Python path
project_root = Path(__file__).parent.parent.resolve()
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

from loguru import logger
from src.utils.logging import setup_logging
from src.domain.state import AgentState
from src.use_cases.graph import GraphBuilder


def main():
    """Inicia a interface CLI do assistente médico."""
    setup_logging(level="INFO")
    
    print("\n" + "=" * 70)
    print("🏥 MACHADO ORÁCULO - Assistente Médico Virtual")
    print("=" * 70)
    print("\n📋 Sistema de suporte a decisões clínicas baseado em protocolos internos")
    print("Desenvolvido com LangChain + LangGraph + Google Gemini\n")
    
    try:
        # Construir grafo de orquestração
        logger.info("🔨 Inicializando grafo de orquestração...")
        graph_builder = GraphBuilder()
        app = graph_builder.build()
        logger.info("✅ Grafo inicializado com sucesso\n")
        
        # Loop interativo de chat
        print("-" * 70)
        print("💬 Digite suas dúvidas clínicas (ou 'sair' para encerrar)\n")
        
        while True:
            try:
                # Ler pergunta do usuário
                question = input("👨‍⚕️  Você: ").strip()
                
                if question.lower() in ("sair", "exit", "quit", "q"):
                    print("\n👋 Encerrando assistente médico. Até logo!\n")
                    logger.info("🛑 Usuário encerrou a sessão")
                    break
                
                if not question:
                    print("⚠️  Digite uma pergunta válida\n")
                    continue
                
                # Processar pergunta através do grafo
                logger.debug(f"Processando pergunta: {question[:60]}...")
                print("\n🔍 Processando pergunta...\n")
                
                initial_state: AgentState = {
                    "medical_question": question,
                    "is_safe": True,
                    "documents": [],
                    "generation": "",
                    "hallucination_check": "",
                }
                
                result = app.invoke(initial_state)
                
                # Exibir resposta
                if result.get("is_safe") is False:
                    logger.warning("Pergunta rejeitada pelos guardrails")
                    print(f"⚠️  Assistente: {result.get('generation', 'Pergunta fora do escopo médico.')}\n")
                else:
                    logger.info("✅ Resposta gerada com sucesso")
                    print(f"🤖 Assistente: {result.get('generation', 'Desculpe, não consegui processar a pergunta.')}\n")
                
                # Mostrar fontes (se disponível)
                if result.get("documents"):
                    print("📚 Protocolos consultados:")
                    for doc in result["documents"][:3]:
                        source = doc.metadata.get("source", "Desconhecido")
                        print(f"   • {source}")
                    print()
                
            except KeyboardInterrupt:
                print("\n\n👋 Interrupção do usuário. Encerrando...\n")
                logger.info("🛑 Interrupção por Ctrl+C")
                break
            except Exception as e:
                logger.error(f"❌ Erro ao processar pergunta: {e}", exc_info=True)
                print(f"❌ Erro técnico: {e}\n")
    
    except Exception as e:
        logger.critical(f"❌ Erro crítico ao inicializar: {e}", exc_info=True)
        print(f"\n❌ ERRO CRÍTICO: {e}")
        print("Verifique os logs em logs/machado-oraculo-errors.log para mais detalhes.\n")
        sys.exit(1)


if __name__ == "__main__":
    main()