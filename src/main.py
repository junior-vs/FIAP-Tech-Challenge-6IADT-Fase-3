"""
Módulo: src/main.py
Descrição: Ponto de entrada (Entry Point) da aplicação CLI.
Motivo da alteração: Adaptação para interface de Chat Médico, gerenciamento de histórico 
e exibição de alertas de segurança/alucinação.
"""

import sys
import uuid
from src.use_cases.graph import GraphBuilder
from src.utils.logging import get_logger

# Configuração de Logs (pode ajustar para DEBUG se quiser ver o "pensamento" do robô)
logger = get_logger()

def print_medical_disclaimer():
    """Exibe o aviso legal obrigatório ao iniciar o sistema."""
    print("\n" + "="*60)
    print("🏥  ASSISTENTE VIRTUAL MÉDICO - SISTEMA DE APOIO À DECISÃO")
    print("="*60)
    print("⚠️  AVISO IMPORTANTE:")
    print("Este sistema utiliza Inteligência Artificial baseada em protocolos internos.")
    print("NÃO substitui o julgamento clínico profissional.")
    print("Sempre valide as sugestões antes de aplicar qualquer conduta.")
    print("="*60 + "\n")

def main():
    # 1. Exibe o Disclaimer
    print_medical_disclaimer()

    # 2. Inicializa o Sistema
    print("⏳ Inicializando base de conhecimento e modelos... (Aguarde)")
    try:
        app_graph = GraphBuilder().build()
        print("✅ Sistema pronto! Base de protocolos carregada.")
    except Exception as e:
        print(f"❌ Erro fatal ao iniciar o sistema: {e}")
        sys.exit(1)

    # Configuração da Sessão
    # O thread_id é usado pelo LangGraph para persistir estado se usarmos checkpointer (futuro)
    thread_id = str(uuid.uuid4())
    print(f"🆔 ID da Sessão: {thread_id}")
    print("💡 Digite 'sair' para encerrar ou 'limpar' para reiniciar o histórico.\n")

    # Histórico local de conversa (para manter o contexto durante a execução)
    chat_history = []

    # 3. Loop de Interação (Chat)
    while True:
        try:
            user_input = input("👨‍⚕️  Você (Médico): ").strip()
            
            if not user_input:
                continue
            
            if user_input.lower() in ["sair", "exit", "quit"]:
                print("👋 Encerrando plantão. Até logo!")
                break
            
            if user_input.lower() == "limpar":
                chat_history = []
                print("🧹 Histórico limpo.")
                continue

            # Prepara o estado inicial para o Grafo
            initial_state = {
                "medical_question": user_input,
                "chat_history": chat_history,
                "is_safe": True,     # Assume seguro até o Guardrail verificar
                "loop_count": 0      # Contador para evitar loops infinitos (se houver re-escrita)
            }

            print("🤖 Processando...", end="\r")

            # Executa o Grafo!
            # O stream_mode="values" retorna o estado final após todos os passos
            result = app_graph.invoke(initial_state)

            # Extrai a resposta final
            generation = result.get("generation")
            
            # Atualiza histórico com a nova interação
            chat_history.append(("user", user_input))
            chat_history.append(("assistant", generation))

            # Exibe a resposta formatada
            print(f"\n💊 Assistente: {generation}\n")
            print("-" * 60)

        except KeyboardInterrupt:
            print("\n\n👋 Interrupção detectada. Encerrando...")
            break
        except Exception as e:
            logger.error(f"Erro durante o processamento: {e}")
            print(f"\n❌ Ocorreu um erro ao processar sua solicitação: {e}")

if __name__ == "__main__":
    main()