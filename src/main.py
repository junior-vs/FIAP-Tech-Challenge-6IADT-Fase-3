#!/usr/bin/env python3
"""Interface CLI para o Assistente Médico Virtual."""

import sys
from pathlib import Path

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
        logger.info("🔨 Inicializando grafo de orquestração...")
        graph_builder = GraphBuilder()
        app = graph_builder.build()
        logger.info("✅ Grafo inicializado com sucesso\n")
        
        print("-" * 70)
        print("💬 Digite suas dúvidas clínicas (ou 'sair' para encerrar)\n")
        
        while True:
            try:
                question = input("👨‍⚕️  Você: ").strip()
                
                if question.lower() in ("sair", "exit", "quit", "q"):
                    print("\n👋 Encerrando assistente médico. Até logo!\n")
                    logger.info("🛑 Usuário encerrou a sessão")
                    break
                
                if not question:
                    print("⚠️  Digite uma pergunta válida\n")
                    continue
                
                logger.debug(f"Processando pergunta: {question[:60]}...")
                print("\n🔍 Processando pergunta...\n")
                
                initial_state: AgentState = {
                    "medical_question": question,
                    "is_safe": True,
                    "documents": [],
                    "generation": "",
                    "hallucination_check": "",
                } # type: ignore
                
                result = app.invoke(initial_state)
                
                # ✅ NOVO: Mostrar status de validação
                hallucination_status = result.get("hallucination_check", "")
                
                if result.get("is_safe") is False:
                    logger.warning("Pergunta rejeitada pelos guardrails")
                    print(f"⚠️  Assistente: {result.get('generation', 'Pergunta fora do escopo médico.')}\n")
                else:
                    # Mostrar status da validação de alucinação
                    if hallucination_status == "valid":
                        status_emoji = "✅"
                        status_msg = "[Validado com semântica]"
                    elif hallucination_status == "valid_keywords":
                        status_emoji = "✅"
                        status_msg = "[Validado com keywords]"
                    elif hallucination_status == "valid_rejection":
                        status_emoji = "ℹ️"
                        status_msg = "[Rejeição apropriada]"
                    elif hallucination_status == "possible_hallucination":
                        status_emoji = "⚠️"
                        status_msg = "[Aviso: possível alucinação]"
                    elif hallucination_status == "no_docs_available":
                        status_emoji = "ℹ️"
                        status_msg = "[Sem docs para validar]"
                    else:
                        status_emoji = "❓"
                        status_msg = ""
                    
                    logger.info(f"Resposta gerada e validada: {hallucination_status}")
                    
                    print(f"🤖 Assistente: {result.get('generation', 'Desculpe, não consegui processar.')}\n")
                    
                    if status_msg:
                        print(f"{status_emoji} {status_msg}\n")
                
                # Mostrar fontes
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