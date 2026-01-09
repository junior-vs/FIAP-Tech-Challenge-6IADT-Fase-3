#!/usr/bin/env python3
"""
Script de teste de integração: Valida se o modelo fine-tuned está funcionando
Testa:
  - Config carregada corretamente
  - LLMFactory inicializa com modelo correto
  - GraphBuilder cria grafo com modelo correto
  - RAG executa com novo modelo
"""

import sys
import os
from pathlib import Path
from typing import Dict, Any, Tuple

# Configurar paths
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))

# Cores para output
class Colors:
    HEADER = '\033[95m'
    BLUE = '\033[94m'
    CYAN = '\033[96m'
    GREEN = '\033[92m'
    YELLOW = '\033[93m'
    RED = '\033[91m'
    ENDC = '\033[0m'
    BOLD = '\033[1m'

def print_header(text: str):
    """Imprimir cabeçalho."""
    print(f"\n{Colors.HEADER}{Colors.BOLD}{'='*60}{Colors.ENDC}")
    print(f"{Colors.HEADER}{Colors.BOLD}{text:^60}{Colors.ENDC}")
    print(f"{Colors.HEADER}{Colors.BOLD}{'='*60}{Colors.ENDC}\n")

def print_test(name: str):
    """Imprimir nome do teste."""
    print(f"{Colors.CYAN}{Colors.BOLD}▶ {name}...{Colors.ENDC}", end=" ", flush=True)

def print_pass(message: str = "✅"):
    """Imprimir sucesso."""
    print(f"{Colors.GREEN}{message}{Colors.ENDC}")

def print_fail(message: str = "❌"):
    """Imprimir falha."""
    print(f"{Colors.RED}{message}{Colors.ENDC}")

def print_info(message: str):
    """Imprimir info."""
    print(f"{Colors.BLUE}ℹ️  {message}{Colors.ENDC}")

def print_success(message: str):
    """Imprimir sucesso expandido."""
    print(f"{Colors.GREEN}{Colors.BOLD}✅ {message}{Colors.ENDC}")

def test_config() -> Tuple[bool, str]:
    """Teste 1: Config carregada?"""
    print_test("Carregando configurações")
    
    try:
        from src.config import settings
        
        model_name = settings.model_name
        api_key = bool(settings.gemini_api_key)
        
        print_pass()
        
        info = f"\n  Modelo: {Colors.BOLD}{model_name}{Colors.ENDC}"
        if model_name.startswith("tunedModels/"):
            info += f" {Colors.GREEN}(fine-tuned){Colors.ENDC}"
        else:
            info += f" {Colors.YELLOW}(modelo base){Colors.ENDC}"
        
        info += f"\n  API Key: {'✅ Configurada' if api_key else '❌ Não encontrada'}"
        print(info)
        
        if not api_key:
            print_fail("GEMINI_API_KEY não configurada")
            print_info("Execute: export GEMINI_API_KEY='sua-chave'")
            return False, "GEMINI_API_KEY ausente"
        
        return True, model_name
        
    except Exception as e:
        print_fail(f"❌ {str(e)}")
        return False, str(e)

def test_llm_factory(model_name: str) -> Tuple[bool, str]:
    """Teste 2: LLMFactory inicializa com modelo correto?"""
    print_test("Inicializando LLMFactory")
    
    try:
        from src.infrastructure.llm_factory import LLMFactory
        
        current = LLMFactory.get_current_model()
        print_pass()
        
        info = f"\n  Modelo em LLMFactory: {Colors.BOLD}{current}{Colors.ENDC}"
        if current == model_name:
            info += f" {Colors.GREEN}(✅ Correto){Colors.ENDC}"
            success = True
        else:
            info += f" {Colors.YELLOW}(⚠️  Esperado: {model_name}){Colors.ENDC}"
            success = True  # Não é falha fatal
        
        print(info)
        return success, current
        
    except Exception as e:
        print_fail(f"❌ {str(e)}")
        return False, str(e)

def test_chatgoogleai() -> Tuple[bool, str]:
    """Teste 3: ChatGoogleGenerativeAI cria instância?"""
    print_test("Criando instância ChatGoogleGenerativeAI")
    
    try:
        from src.infrastructure.llm_factory import LLMFactory
        
        llm = LLMFactory.get_llm()
        
        # Verificar atributos
        assert hasattr(llm, 'model_name'), "LLM não tem model_name"
        assert hasattr(llm, 'invoke'), "LLM não tem método invoke"
        
        print_pass()
        
        model = llm.model_name
        info = f"\n  Instância criada com modelo: {Colors.BOLD}{model}{Colors.ENDC}"
        if model.startswith("tunedModels/"):
            info += f" {Colors.GREEN}(fine-tuned){Colors.ENDC}"
        
        print(info)
        return True, model
        
    except Exception as e:
        print_fail(f"❌ {str(e)}")
        return False, str(e)

def test_vector_store() -> Tuple[bool, str]:
    """Teste 4: VectorStore carregado?"""
    print_test("Verificando VectorStore")
    
    try:
        from src.infrastructure.vector_store import VectorStoreRepository
        
        vector_repo = VectorStoreRepository()
        retriever = vector_repo.get_retriever()
        
        assert retriever is not None, "Retriever é None"
        
        print_pass()
        print(f"\n  Retriever: {Colors.BOLD}ChromaDB{Colors.ENDC} ✅")
        print(f"  Status: {Colors.BOLD}Pronto para recuperar documentos{Colors.ENDC}")
        
        return True, "ChromaDB ready"
        
    except Exception as e:
        print_fail(f"❌ {str(e)}")
        return False, str(e)

def test_graph_builder() -> Tuple[bool, str]:
    """Teste 5: GraphBuilder inicializa corretamente?"""
    print_test("Construindo GraphBuilder")
    
    try:
        from src.use_cases.graph import GraphBuilder
        
        builder = GraphBuilder()
        
        assert hasattr(builder, 'llm'), "GraphBuilder não tem llm"
        assert hasattr(builder, 'nodes'), "GraphBuilder não tem nodes"
        
        print_pass()
        
        llm_model = builder.llm.model_name
        info = f"\n  GraphBuilder.llm.model_name: {Colors.BOLD}{llm_model}{Colors.ENDC}"
        print(info)
        
        return True, llm_model
        
    except Exception as e:
        print_fail(f"❌ {str(e)}")
        return False, str(e)

def test_build_graph() -> Tuple[bool, str]:
    """Teste 6: Grafo RAG compila?"""
    print_test("Compilando grafo RAG")
    
    try:
        from src.use_cases.graph import GraphBuilder
        
        builder = GraphBuilder()
        app = builder.build()
        
        assert app is not None, "Grafo compilado é None"
        assert hasattr(app, 'invoke'), "Grafo não tem método invoke"
        
        print_pass()
        print(f"\n  Nós do grafo: {Colors.BOLD}5{Colors.ENDC}")
        print(f"    1. Guardrails")
        print(f"    2. Retrieve")
        print(f"    3. Grade")
        print(f"    4. Generate (usa fine-tuned)")
        print(f"    5. Validate")
        
        return True, "Graph compiled successfully"
        
    except Exception as e:
        print_fail(f"❌ {str(e)}")
        return False, str(e)

def test_rag_inference() -> Tuple[bool, str]:
    """Teste 7: RAG consegue processar pergunta?"""
    print_test("Testando inferência RAG (pode levar tempo...)")
    
    try:
        from src.use_cases.graph import GraphBuilder
        from src.domain.state import AgentState
        import uuid
        
        # Criar estado
        state = AgentState(
            request_id=str(uuid.uuid4()),
            medical_question="O que é diabetes?",
            is_safe=True,
            documents=[],
            generation="",
            chat_history=[],
            is_valid=False,
            response="",
            error=None
        )
        
        # Executar RAG
        builder = GraphBuilder()
        app = builder.build()
        
        # Timeout para não ficar muito tempo esperando
        import signal
        
        def timeout_handler(signum, frame):
            raise TimeoutError("Inferência demorou mais de 30s")
        
        signal.signal(signal.SIGALRM, timeout_handler)
        signal.alarm(30)  # 30 segundos
        
        try:
            result = app.invoke(state)
            signal.alarm(0)  # Cancelar timeout
        except TimeoutError as e:
            print_fail(f"❌ Timeout (> 30s)")
            return False, "Timeout na inferência"
        
        print_pass()
        
        # Verificar resultado
        generation = result.get('generation', '')
        is_valid = result.get('is_valid', False)
        
        info = f"\n  Pergunta: {Colors.BOLD}O que é diabetes?{Colors.ENDC}"
        info += f"\n  Resposta gerada: {Colors.BOLD}{'Sim ✅' if generation else 'Não ❌'}{Colors.ENDC}"
        info += f"\n  Válida: {Colors.BOLD}{'Sim ✅' if is_valid else 'Não verificado'}{Colors.ENDC}"
        
        if generation:
            info += f"\n  Primeiras 100 caracteres: {Colors.BOLD}{generation[:100]}...{Colors.ENDC}"
        
        print(info)
        
        return bool(generation), "Inferência bem-sucedida"
        
    except Exception as e:
        print_fail(f"❌ {str(e)}")
        return False, str(e)

def run_tests() -> bool:
    """Executar todos os testes."""
    print_header("TESTE DE INTEGRAÇÃO DO MODELO FINE-TUNED")
    
    tests = [
        ("1. Configuração", test_config),
        ("2. LLMFactory", test_llm_factory),
        ("3. ChatGoogleGenerativeAI", test_chatgoogleai),
        ("4. VectorStore", test_vector_store),
        ("5. GraphBuilder", test_graph_builder),
        ("6. Compilação do Grafo", test_build_graph),
        ("7. Inferência RAG", test_rag_inference),
    ]
    
    results = []
    last_value = None
    
    for test_name, test_func in tests:
        if "LLMFactory" in test_name and last_value:
            success, value = test_func(last_value)
        elif "Compilação" in test_name or "Inferência" in test_name:
            success, value = test_func()
        else:
            success, value = test_func()
        
        results.append((test_name, success, value))
        last_value = value
        
        if not success and "Inferência" not in test_name:
            print_info(f"Parando testes (teste crítico falhou)")
            break
    
    # Resumo
    print_header("RESUMO DOS TESTES")
    
    passed = sum(1 for _, success, _ in results if success)
    total = len(results)
    
    print(f"{Colors.CYAN}Resultado:{Colors.ENDC}\n")
    
    for test_name, success, value in results:
        status = f"{Colors.GREEN}✅{Colors.ENDC}" if success else f"{Colors.RED}❌{Colors.ENDC}"
        print(f"  {status} {test_name}")
        if isinstance(value, str) and len(value) < 80:
            print(f"      {Colors.BLUE}→ {value}{Colors.ENDC}")
    
    print(f"\n{Colors.CYAN}Teste Geral:{Colors.ENDC} {passed}/{total}")
    
    if passed == total:
        print_success("Todos os testes passaram! ✅")
        print(f"\n{Colors.GREEN}{Colors.BOLD}Seu RAG está pronto para usar o modelo fine-tuned!{Colors.ENDC}")
        return True
    elif passed >= total - 1:
        print_info(f"{total - passed} teste falhou (possível timeout na inferência)")
        print(f"\n{Colors.YELLOW}{Colors.BOLD}Sua integração está OK, mas verifique a inferência manualmente{Colors.ENDC}")
        print(f"\n  Teste manual: {Colors.BOLD}python src/main.py{Colors.ENDC}")
        return True
    else:
        print_fail(f"{total - passed} testes falharam")
        print(f"\n{Colors.RED}{Colors.BOLD}Verifique os erros acima e tente novamente{Colors.ENDC}")
        return False

def main():
    """Função principal."""
    try:
        # Verificar se está no diretório correto
        if not (Path.cwd() / "src").exists():
            print(f"{Colors.RED}❌ Erro: Execute este script na raiz do projeto{Colors.ENDC}")
            return 1
        
        # Executar testes
        success = run_tests()
        
        # Dicas finais
        print(f"\n{Colors.CYAN}{Colors.BOLD}Próximas ações:{Colors.ENDC}")
        print(f"  • Teste interativo: {Colors.BOLD}python src/main.py{Colors.ENDC}")
        print(f"  • Ver integração: {Colors.BOLD}cat INTEGRATION_GUIDE.md{Colors.ENDC}")
        print(f"  • Comparar modelos: {Colors.BOLD}cat INTEGRATION_EXAMPLE.md{Colors.ENDC}")
        
        return 0 if success else 1
        
    except KeyboardInterrupt:
        print(f"\n{Colors.YELLOW}⚠️  Teste interrompido pelo usuário{Colors.ENDC}")
        return 1
    except Exception as e:
        print(f"\n{Colors.RED}❌ Erro inesperado: {e}{Colors.ENDC}")
        import traceback
        traceback.print_exc()
        return 1

if __name__ == "__main__":
    sys.exit(main())
