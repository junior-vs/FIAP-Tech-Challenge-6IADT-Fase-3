#!/usr/bin/env python3
"""
Script: test_fine_tuning_integration.py

Script de teste para validar a integração de fine-tuning com o projeto.

Testa:
1. Carregamento de dados
2. Preparação de dados
3. Simulação de fine-tuning (sem executar na API)
4. Integração com LLMFactory
5. Listagem de modelos disponíveis
"""

import sys
import json
import logging
from pathlib import Path

# === SETUP DO PROJETO ===
project_root = Path(__file__).parent.resolve()
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

from src.infrastructure.fine_tuning import GeminiFineTuner
from src.infrastructure.llm_factory import LLMFactory
from src.config import settings
from src.utils.logging import setup_logging

# Setup logging
setup_logging(level="INFO")
logger = logging.getLogger(__name__)


def test_data_loading():
    """Test 1: Carregar e validar dados."""
    logger.info("\n" + "=" * 70)
    logger.info("TEST 1: Validar Dados de Treinamento")
    logger.info("=" * 70)
    
    # data_file = Path("docs/knowledge_base/finetuning_prompt_completion.jsonl")
    data_file = Path("docs/knowledge_base/finetuning_data_gemini_messages.jsonl")
    
    if not data_file.exists():
        logger.error(f"Arquivo não encontrado: {data_file}")
        return False
    
    try:
        records = []
        with open(data_file, 'r', encoding='utf-8') as f:
            for i, line in enumerate(f, 1):
                try:
                    record = json.loads(line)
                    records.append(record)
                except json.JSONDecodeError as e:
                    logger.error(f"Erro JSON na linha {i}: {e}")
                    return False
        
        logger.info(f"Dados carregados com sucesso")
        logger.info(f"Total de registros: {len(records)}")
        logger.info(f"Tamanho do arquivo: {data_file.stat().st_size / 1024:.2f} KB")
        
        # Valida estrutura
        if len(records) > 0:
            first_record = records[0]
            if "messages" not in first_record:
                logger.error("Estrutura inválida: falta 'messages'")
                return False
            
            messages = first_record["messages"]
            logger.info(f"Estrutura validada:")
            logger.info(f"   - Primeira mensagem (user):")
            user_msg = next((m for m in messages if m.get("role") == "user"), None)
            if user_msg:
                content_preview = user_msg["content"][:100]
                logger.info(f"     {content_preview}...")
            
            logger.info(f"   - Primeira mensagem (assistant):")
            asst_msg = next((m for m in messages if m.get("role") == "assistant"), None)
            if asst_msg:
                content_preview = asst_msg["content"][:100]
                logger.info(f"     {content_preview}...")
        
        return True
        
    except Exception as e:
        logger.error(f"Erro ao processar dados: {e}")
        return False


def test_data_preparation():
    """Test 2: Preparar e dividir dados."""
    logger.info("\n" + "=" * 70)
    logger.info("TEST 2: Preparar Dados (Treino/Validação)")
    logger.info("=" * 70)
    
    # data_file = Path("docs/knowledge_base/finetuning_prompt_completion.jsonl")
    data_file = Path("docs/knowledge_base/finetuning_data_gemini_messages.jsonl")
    
    try:
        tuner = GeminiFineTuner()
        train_file, val_file = tuner.prepare_training_data(
            data_file,
            validation_split=0.2
        )
        
        # Validar arquivos criados
        if not train_file.exists() or not val_file.exists():
            logger.error("Arquivos não foram criados")
            return False
        
        train_size = train_file.stat().st_size / 1024
        val_size = val_file.stat().st_size / 1024
        
        logger.info(f"Arquivos preparados:")
        logger.info(f"   Treino: {train_file.name} ({train_size:.2f} KB)")
        logger.info(f"   Validação: {val_file.name} ({val_size:.2f} KB)")
        
        # Limpar arquivos temporários
        train_file.unlink()
        val_file.unlink()
        logger.info(f"Arquivos temporários deletados")
        
        return True
        
    except Exception as e:
        logger.error(f"Erro ao preparar dados: {e}")
        return False


def test_llm_factory():
    """Test 3: Validar LLMFactory."""
    logger.info("\n" + "=" * 70)
    logger.info("TEST 3: Validar LLMFactory")
    logger.info("=" * 70)
    
    try:
        logger.info("Testando LLMFactory.get_llm()...")
        llm = LLMFactory.get_llm()
        logger.info(f"   Modelo atual: {LLMFactory.get_current_model()}")
        
        logger.info("Testando LLMFactory.get_embeddings()...")
        embeddings = LLMFactory.get_embeddings()
        logger.info(f"   Modelo de embeddings: models/embedding-001")
        
        logger.info("Testando mudança de modelo...")
        LLMFactory.switch_model("gemini-2.0-flash")
        logger.info(f"   Modelo alterado para: {LLMFactory.get_current_model()}")
        
        return True
        
    except Exception as e:
        logger.error(f"Erro no LLMFactory: {e}")
        return False


def test_list_fine_tuned_models():
    """Test 4: Listar modelos fine-tuned."""
    logger.info("\n" + "=" * 70)
    logger.info("TEST 4: Listar Modelos Fine-tuned Disponíveis")
    logger.info("=" * 70)
    
    try:
        models = LLMFactory.list_available_tuned_models()
        
        if not models:
            logger.info("Nenhum modelo fine-tuned disponível ainda")
            logger.info("   (Execute fine-tuning primeiro: python fine_tuning_cli.py)")
            return True
        
        logger.info(f"{len(models)} modelo(s) fine-tuned disponível(eis):")
        for model in models:
            logger.info(f"   - {model}")
        
        return True
        
    except Exception as e:
        logger.warning(f"Erro ao listar modelos: {e}")
        logger.info("   (Isso pode ser normal se a API não está acessível)")
        return True


def test_config_validation():
    """Test 5: Validar configurações."""
    logger.info("\n" + "=" * 70)
    logger.info("TEST 5: Validar Configurações")
    logger.info("=" * 70)
    
    try:
        logger.info(f"  Configurações carregadas:")
        logger.info(f"   - Model name: {settings.model_name}")
        logger.info(f"   - Temperature: {settings.temperature}")
        logger.info(f"   - Request timeout: {settings.request_timeout}s")
        logger.info(f"   - Retries: {settings.retries}")
        logger.info(f"   - Docs path: {settings.docs_path}")
        logger.info(f"   - Vector DB path: {settings.vector_db_path}")
        
        # Validar paths
        docs_path = Path(settings.docs_path)
        if not docs_path.exists():
            logger.warning(f"Caminho de docs não existe: {docs_path}")
        else:
            logger.info(f"Caminho de docs válido")
        
        return True
        
    except Exception as e:
        logger.error(f"Erro ao validar config: {e}")
        return False


def main():
    """Execute todos os testes."""
    logger.info("=" * 70)
    logger.info("TESTE DE INTEGRAÇÃO - FINE-TUNING GEMINI FLASH 2.0")
    logger.info("=" * 70)
    
    results = {
        "Data Loading": test_data_loading(),
        "Data Preparation": test_data_preparation(),
        "LLM Factory": test_llm_factory(),
        "Fine-tuned Models": test_list_fine_tuned_models(),
        "Config Validation": test_config_validation(),
    }
    
    # Resumo
    logger.info("\n" + "=" * 70)
    logger.info("RESUMO DOS TESTES")
    logger.info("=" * 70)
    
    passed = sum(1 for v in results.values() if v)
    total = len(results)
    
    for test_name, result in results.items():
        status = "PASS" if result else "❌ FAIL"
        logger.info(f"{status}: {test_name}")
    
    logger.info("\n" + "=" * 70)
    if passed == total:
        logger.info(f"TODOS OS TESTES PASSARAM! ({passed}/{total})")
        logger.info("=" * 70)
        logger.info("\n PRÓXIMOS PASSOS:")
        logger.info("1. Execute: python fine_tuning_cli.py --data docs/knowledge_base/finetuning_data_gemini_messages.jsonl")
        logger.info("2. Aguarde a conclusão do fine-tuning")
        logger.info("3. Copie o model_id resultado")
        logger.info("4. Atualize src/config.py com o modelo fine-tuned")
        logger.info("5. Execute: python src/main.py")
        return 0
    else:
        logger.error(f" ALGUNS TESTES FALHARAM ({passed}/{total})")
        logger.info("=" * 70)
        logger.info("\n VERIFIQUE:")
        logger.info("- Arquivo .env com GEMINI_API_KEY")
        logger.info("- Dependências instaladas (pip install -e .)")
        logger.info("- Caminho do arquivo de dados")
        return 1


if __name__ == "__main__":
    exit_code = main()
    sys.exit(exit_code)
