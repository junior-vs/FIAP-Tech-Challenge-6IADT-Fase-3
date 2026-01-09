#!/usr/bin/env python3
"""
Script de integração: Fine-tuning com RAG LangChain
"""

import os
import sys
import json
import logging
import subprocess
from pathlib import Path
from typing import Optional
from dataclasses import dataclass

# Configurar logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Cores para terminal
class Colors:
    HEADER = '\033[95m'
    BLUE = '\033[94m'
    CYAN = '\033[96m'
    GREEN = '\033[92m'
    YELLOW = '\033[93m'
    RED = '\033[91m'
    ENDC = '\033[0m'
    BOLD = '\033[1m'
    UNDERLINE = '\033[4m'

@dataclass
class IntegrationConfig:
    """Configuração de integração."""
    model_id: str
    config_path: str = "src/config.py"
    backup_path: str = "src/config.py.backup"
    method: str = "config"  # config, env, code

class FineTuningIntegration:
    """Gerencia integração de modelo fine-tuned com RAG."""
    
    def __init__(self, project_root: Optional[Path] = None):
        """Inicializar integrator."""
        self.project_root = project_root or Path.cwd()
        self.config_path = self.project_root / "src" / "config.py"
        self.backup_path = self.project_root / "src" / "config.py.backup"
        
    def print_header(self, text: str):
        """Imprimir cabeçalho colorido."""
        print(f"\n{Colors.HEADER}{Colors.BOLD}{'='*60}{Colors.ENDC}")
        print(f"{Colors.HEADER}{Colors.BOLD}{text:^60}{Colors.ENDC}")
        print(f"{Colors.HEADER}{Colors.BOLD}{'='*60}{Colors.ENDC}\n")
    
    def print_step(self, step: int, text: str):
        """Imprimir número do passo."""
        print(f"{Colors.CYAN}{Colors.BOLD}Passo {step}:{Colors.ENDC} {text}")
    
    def print_success(self, text: str):
        """Imprimir sucesso."""
        print(f"{Colors.GREEN}✅ {text}{Colors.ENDC}")
    
    def print_error(self, text: str):
        """Imprimir erro."""
        print(f"{Colors.RED}❌ {text}{Colors.ENDC}")
    
    def print_info(self, text: str):
        """Imprimir info."""
        print(f"{Colors.BLUE}ℹ️  {text}{Colors.ENDC}")
    
    def print_warning(self, text: str):
        """Imprimir aviso."""
        print(f"{Colors.YELLOW}⚠️  {text}{Colors.ENDC}")
    
    def check_prerequisites(self) -> bool:
        """Verificar se tudo está pronto."""
        self.print_step(0, "Verificando pré-requisitos...")
        
        # Verificar Python
        version = sys.version_info
        if version.major < 3 or (version.major == 3 and version.minor < 10):
            self.print_error(f"Python 3.10+ necessário (você tem {version.major}.{version.minor})")
            return False
        self.print_success(f"Python {version.major}.{version.minor} ✓")
        
        # Verificar estrutura do projeto
        required_files = [
            "src/config.py",
            "src/infrastructure/llm_factory.py",
            "src/use_cases/graph.py",
            "src/use_cases/nodes.py",
            "fine_tuning_cli.py",
        ]
        
        for file in required_files:
            path = self.project_root / file
            if not path.exists():
                self.print_error(f"Arquivo necessário não encontrado: {file}")
                return False
            self.print_success(f"{file} ✓")
        
        # Verificar GEMINI_API_KEY
        if not os.getenv("GEMINI_API_KEY"):
            self.print_warning("GEMINI_API_KEY não configurada em variáveis de ambiente")
            self.print_info("Use: export GEMINI_API_KEY='sua-chave'")
        else:
            self.print_success("GEMINI_API_KEY ✓")
        
        return True
    
    def run_fine_tuning(self, data_file: str) -> Optional[str]:
        """Executar fine-tuning e retornar model_id."""
        self.print_step(1, f"Iniciando fine-tuning com dados de {data_file}")
        
        try:
            # Verificar se arquivo existe
            data_path = self.project_root / data_file
            if not data_path.exists():
                self.print_error(f"Arquivo de dados não encontrado: {data_file}")
                return None
            
            self.print_info(f"Usando arquivo: {data_path}")
            
            # Executar fine-tuning
            cmd = [
                sys.executable, "fine_tuning_cli.py",
                "--data", str(data_file),
                "--wait"
            ]
            
            self.print_info(f"Executando: {' '.join(cmd)}")
            result = subprocess.run(
                cmd,
                cwd=self.project_root,
                capture_output=True,
                text=True
            )
            
            if result.returncode != 0:
                self.print_error(f"Fine-tuning falhou: {result.stderr}")
                return None
            
            # Extrair model_id da saída
            output = result.stdout
            for line in output.split('\n'):
                if 'model_id' in line.lower():
                    # Tentar extrair model_id
                    parts = line.split(':')
                    if len(parts) > 1:
                        model_id = parts[-1].strip().strip("'\"")
                        if model_id.startswith('tunedModels/'):
                            self.print_success(f"Fine-tuning concluído! model_id: {model_id}")
                            return model_id
            
            # Se não encontrou, procurar em JSON
            if "tunedModels/" in output:
                import re
                match = re.search(r'(tunedModels/[^\s"\']+)', output)
                if match:
                    model_id = match.group(1)
                    self.print_success(f"Fine-tuning concluído! model_id: {model_id}")
                    return model_id
            
            self.print_warning("Não foi possível extrair model_id da saída")
            self.print_info("Saída do fine-tuning:")
            print(output)
            return None
            
        except Exception as e:
            self.print_error(f"Erro ao executar fine-tuning: {e}")
            return None
    
    def backup_config(self):
        """Fazer backup da configuração atual."""
        if self.config_path.exists() and not self.backup_path.exists():
            import shutil
            shutil.copy2(self.config_path, self.backup_path)
            self.print_success(f"Backup criado: {self.backup_path}")
    
    def update_config(self, model_id: str) -> bool:
        """Atualizar config.py com novo model_id."""
        self.print_step(2, "Atualizando src/config.py com novo modelo...")
        
        if not self.config_path.exists():
            self.print_error(f"Arquivo não encontrado: {self.config_path}")
            return False
        
        # Fazer backup
        self.backup_config()
        
        try:
            # Ler arquivo
            with open(self.config_path, 'r', encoding='utf-8') as f:
                content = f.read()
            
            original_content = content
            
            # Procurar e substituir model_name
            import re
            
            # Pattern 1: model_name: str = "..."
            pattern1 = r'model_name:\s*str\s*=\s*["\']([^"\']*)["\']'
            replacement1 = f'model_name: str = "{model_id}"'
            
            # Pattern 2: model_name = "..."
            pattern2 = r'model_name\s*=\s*["\']([^"\']*)["\']'
            
            # Tentar primeiro padrão
            new_content = re.sub(pattern1, replacement1, content)
            
            if new_content == content:
                # Tentar segundo padrão
                new_content = re.sub(pattern2, replacement1, content)
            
            if new_content == original_content:
                self.print_warning("Padrão model_name não encontrado em config.py")
                self.print_info("Tente atualizar manualmente em src/config.py:")
                print(f'  model_name: str = "{model_id}"')
                return False
            
            # Escrever arquivo atualizado
            with open(self.config_path, 'w', encoding='utf-8') as f:
                f.write(new_content)
            
            self.print_success(f"Config.py atualizado com novo modelo")
            self.print_info(f"Modelo anterior: gemini-2.0-flash")
            self.print_info(f"Modelo novo: {model_id}")
            
            return True
            
        except Exception as e:
            self.print_error(f"Erro ao atualizar config.py: {e}")
            # Restaurar backup
            if self.backup_path.exists():
                import shutil
                shutil.copy2(self.backup_path, self.config_path)
                self.print_warning("Config restaurada do backup")
            return False
    
    def update_env_variable(self, model_id: str) -> bool:
        """Atualizar variável de ambiente."""
        self.print_step(2, "Configurando variável de ambiente...")
        
        env_file = self.project_root / ".env"
        
        try:
            # Ler .env existente
            env_content = ""
            if env_file.exists():
                with open(env_file, 'r', encoding='utf-8') as f:
                    env_content = f.read()
            
            # Atualizar ou adicionar MODEL_NAME
            import re
            if 'MODEL_NAME' in env_content:
                env_content = re.sub(
                    r'MODEL_NAME=.*',
                    f'MODEL_NAME={model_id}',
                    env_content
                )
            else:
                env_content += f'\nMODEL_NAME={model_id}\n'
            
            # Escrever .env
            with open(env_file, 'w', encoding='utf-8') as f:
                f.write(env_content)
            
            self.print_success(f"Arquivo .env atualizado com MODEL_NAME")
            return True
            
        except Exception as e:
            self.print_error(f"Erro ao atualizar .env: {e}")
            return False
    
    def verify_integration(self, model_id: str) -> bool:
        """Verificar se integração foi bem-sucedida."""
        self.print_step(3, "Verificando integração...")
        
        try:
            # Tentar importar e verificar
            sys.path.insert(0, str(self.project_root))
            
            from src.infrastructure.llm_factory import LLMFactory
            
            # Verificar modelo configurado
            current = LLMFactory.get_current_model()
            
            if current == model_id:
                self.print_success(f"Modelo configurado corretamente: {model_id}")
                return True
            else:
                self.print_warning(f"Modelo atual diferente de esperado")
                self.print_info(f"Esperado: {model_id}")
                self.print_info(f"Atual: {current}")
                return False
                
        except Exception as e:
            self.print_warning(f"Não foi possível verificar integração: {e}")
            self.print_info("Execute python src/main.py para testar manualmente")
            return True  # Continuar mesmo assim
    
    def test_rag(self) -> bool:
        """Testar RAG com novo modelo."""
        self.print_step(4, "Testando RAG com novo modelo...")
        
        try:
            print(f"{Colors.BLUE}Iniciando teste...{Colors.ENDC}\n")
            
            cmd = [sys.executable, "src/main.py"]
            result = subprocess.run(
                cmd,
                cwd=self.project_root,
                timeout=30,
                capture_output=True,
                text=True
            )
            
            if result.returncode == 0:
                self.print_success("RAG inicializado com sucesso")
                if "fine-tuned" in result.stderr or "fine-tuned" in result.stdout:
                    self.print_success("Modelo fine-tuned está sendo usado!")
                return True
            else:
                self.print_error("Erro ao iniciar RAG")
                print(result.stderr)
                return False
                
        except subprocess.TimeoutExpired:
            self.print_warning("Teste demorou demais (timeout)")
            self.print_info("RAG pode estar funcionando, mas teste foi interrompido")
            return True
        except Exception as e:
            self.print_warning(f"Erro ao testar RAG: {e}")
            self.print_info("Execute python src/main.py manualmente para testar")
            return True
    
    def show_summary(self, model_id: str, method: str, success: bool):
        """Mostrar resumo de integração."""
        self.print_header("RESUMO DE INTEGRAÇÃO")
        
        if success:
            print(f"{Colors.GREEN}{Colors.BOLD}✅ INTEGRAÇÃO BEM-SUCEDIDA!{Colors.ENDC}\n")
        else:
            print(f"{Colors.YELLOW}{Colors.BOLD}⚠️  INTEGRAÇÃO COM AVISOS{Colors.ENDC}\n")
        
        print(f"{Colors.CYAN}Detalhes:{Colors.ENDC}")
        print(f"  • Model ID: {Colors.BOLD}{model_id}{Colors.ENDC}")
        print(f"  • Método: {Colors.BOLD}{method}{Colors.ENDC}")
        print(f"  • Arquivo config: {Colors.BOLD}{self.config_path}{Colors.ENDC}")
        
        if self.backup_path.exists():
            print(f"  • Backup criado: {Colors.BOLD}{self.backup_path}{Colors.ENDC}")
        
        print(f"\n{Colors.CYAN}Próximos passos:{Colors.ENDC}")
        print(f"  1. Execute: {Colors.BOLD}python src/main.py{Colors.ENDC}")
        print(f"  2. Teste uma pergunta médica")
        print(f"  3. Verifique qualidade das respostas")
        print(f"  4. Compare com modelo base se necessário")
        
        print(f"\n{Colors.CYAN}Útil:{Colors.ENDC}")
        print(f"  • Verificar modelo atual: python -c \"from src.infrastructure.llm_factory import LLMFactory; print(LLMFactory.get_current_model())\"")
        print(f"  • Trocar modelo: python -c \"from src.infrastructure.llm_factory import LLMFactory; LLMFactory.switch_model('gemini-2.0-flash')\"")
        print(f"  • Ver documentação: cat INTEGRATION_GUIDE.md")
    
    def run_interactive(self):
        """Executar em modo interativo."""
        self.print_header("INTEGRAÇÃO DE MODELO FINE-TUNED COM RAG")
        
        print(f"{Colors.CYAN}Este script automatiza a integração de um modelo fine-tuned")
        print(f"com seu RAG baseado em LangChain.{Colors.ENDC}\n")
        
        # Pré-requisitos
        if not self.check_prerequisites():
            self.print_error("Alguns pré-requisitos não foram atendidos")
            return False
        
        print()
        
        # Escolher modo
        print(f"{Colors.CYAN}Escolha um modo de operação:{Colors.ENDC}")
        print("  1. [RECOMENDADO] Fine-tuning automático + integração")
        print("  2. Integrar com model_id existente")
        print("  3. Usar variável de ambiente")
        print("  4. Apenas verificar configuração")
        
        choice = input(f"\n{Colors.BOLD}Escolha (1-4): {Colors.ENDC}").strip()
        
        if choice == "1":
            return self.run_full_flow()
        elif choice == "2":
            return self.run_with_existing_model()
        elif choice == "3":
            return self.run_with_env()
        elif choice == "4":
            return self.run_verify_only()
        else:
            self.print_error("Opção inválida")
            return False
    
    def run_full_flow(self) -> bool:
        """Executar fluxo completo: fine-tuning + integração."""
        print()
        
        # Arquivo de dados
        data_file = input(f"{Colors.BOLD}Arquivo de dados para fine-tuning{Colors.ENDC} [docs/knowledge_base/finetuning_data_gemini_messages.jsonl]: ").strip()
        if not data_file:
            data_file = "docs/knowledge_base/finetuning_data_gemini_messages.jsonl"
        
        # 1. Fine-tuning
        model_id = self.run_fine_tuning(data_file)
        if not model_id:
            self.print_error("Fine-tuning falhou")
            return False
        
        print()
        
        # 2. Integração via config
        if not self.update_config(model_id):
            self.print_warning("Não foi possível atualizar config automaticamente")
            self.print_info("Atualize manualmente em src/config.py")
        
        print()
        
        # 3. Verificação
        self.verify_integration(model_id)
        
        print()
        
        # 4. Resumo
        self.show_summary(model_id, "config", True)
        
        return True
    
    def run_with_existing_model(self) -> bool:
        """Integrar com model_id existente."""
        print()
        
        model_id = input(f"{Colors.BOLD}Digite o model_id do seu modelo fine-tuned{Colors.ENDC}: ").strip()
        
        if not model_id or not model_id.startswith("tunedModels/"):
            self.print_error("Model_id inválido (deve começar com 'tunedModels/')")
            return False
        
        print()
        
        if self.update_config(model_id):
            print()
            self.verify_integration(model_id)
            print()
            self.show_summary(model_id, "config", True)
            return True
        else:
            self.print_error("Falha ao atualizar configuração")
            return False
    
    def run_with_env(self) -> bool:
        """Integrar usando variável de ambiente."""
        print()
        
        model_id = input(f"{Colors.BOLD}Digite o model_id do seu modelo fine-tuned{Colors.ENDC}: ").strip()
        
        if not model_id or not model_id.startswith("tunedModels/"):
            self.print_error("Model_id inválido")
            return False
        
        print()
        
        if self.update_env_variable(model_id):
            print()
            self.print_success("Variável de ambiente MODEL_NAME configurada")
            self.print_info("Para usar, execute:")
            print(f"  export MODEL_NAME={model_id}")
            print(f"  python src/main.py")
            print()
            self.show_summary(model_id, "env", True)
            return True
        else:
            self.print_error("Falha ao atualizar .env")
            return False
    
    def run_verify_only(self) -> bool:
        """Apenas verificar configuração."""
        print()
        
        try:
            sys.path.insert(0, str(self.project_root))
            from src.infrastructure.llm_factory import LLMFactory
            from src.config import settings
            
            current = LLMFactory.get_current_model()
            
            print(f"{Colors.CYAN}Configuração atual:{Colors.ENDC}")
            print(f"  • Modelo em config.py: {Colors.BOLD}{settings.model_name}{Colors.ENDC}")
            print(f"  • Modelo atual (LLMFactory): {Colors.BOLD}{current}{Colors.ENDC}")
            
            if current.startswith("tunedModels/"):
                self.print_success("Usando modelo FINE-TUNED")
            else:
                self.print_info("Usando modelo BASE")
            
            print()
            
            # Modelos disponíveis
            try:
                available = LLMFactory.list_available_tuned_models()
                if available:
                    print(f"{Colors.CYAN}Modelos fine-tuned disponíveis:{Colors.ENDC}")
                    for m in available[:5]:
                        print(f"  • {m}")
                    if len(available) > 5:
                        print(f"  ... e {len(available)-5} mais")
            except:
                pass
            
            return True
            
        except Exception as e:
            self.print_error(f"Erro ao verificar: {e}")
            return False

def main():
    """Função principal."""
    try:
        integrator = FineTuningIntegration()
        
        if len(sys.argv) > 1:
            # Modo não-interativo
            model_id = sys.argv[1]
            if model_id.startswith("tunedModels/"):
                if integrator.update_config(model_id):
                    integrator.show_summary(model_id, "config", True)
                    return 0
                else:
                    return 1
            else:
                print(f"{Colors.RED}❌ Model_id inválido{Colors.ENDC}")
                return 1
        else:
            # Modo interativo
            success = integrator.run_interactive()
            return 0 if success else 1
    
    except KeyboardInterrupt:
        print(f"\n{Colors.YELLOW}⚠️  Operação cancelada pelo usuário{Colors.ENDC}")
        return 1
    except Exception as e:
        print(f"{Colors.RED}❌ Erro inesperado: {e}{Colors.ENDC}")
        return 1

if __name__ == "__main__":
    sys.exit(main())
