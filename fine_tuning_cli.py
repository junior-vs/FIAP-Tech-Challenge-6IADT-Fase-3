#!/usr/bin/env python3
"""
Script: fine_tuning_cli.py

Interface de linha de comando para executar fine-tuning de modelos Gemini.

Exemplos de uso:
    # Fine-tuning com configuração padrão
    python fine_tuning_cli.py --data docs/knowledge_base/finetuning_data_gemini_messages.jsonl
    
    # Fine-tuning com configuração customizada
    python fine_tuning_cli.py \\
        --data docs/knowledge_base/finetuning_data_gemini_messages.jsonl \\
        --job-name meu_assistente_ft \\
        --epochs 8 \\
        --learning-rate 0.8 \\
        --validation-split 0.3
    
    # Apenas preparar dados sem executar
    python fine_tuning_cli.py --data file.jsonl --prepare-only
"""

import sys
import logging
from pathlib import Path
from typing import Optional
import argparse

# === SETUP DO PROJETO ===
project_root = Path(__file__).parent.parent.resolve()
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

from src.infrastructure.fine_tuning import GeminiFineTuner
from src.utils.logging import setup_logging
# Suporte opcional para treino e inferência local (LLaMA/LoRA)
try:
    from src.infrastructure.llama_finetune import train_lora
    from src.infrastructure.local_llama import load_finetuned
except Exception:
    train_lora = None
    load_finetuned = None


def create_parser() -> argparse.ArgumentParser:
    """Cria o parser de argumentos da CLI."""
    parser = argparse.ArgumentParser(
        description="CLI para fine-tuning de modelos Google Gemini",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Exemplos:
  # Executar fine-tuning com padrões
  python fine_tuning_cli.py --data docs/knowledge_base/finetuning_data_gemini_messages.jsonl
  
  # Customizar hiperparâmetros
  python fine_tuning_cli.py --data data.jsonl --epochs 8 --learning-rate 0.8
  
  # Apenas preparar dados
  python fine_tuning_cli.py --data data.jsonl --prepare-only
        """
    )
    
    # Argumentos obrigatórios
    parser.add_argument(
        "--data",
        type=Path,
        required=False,
        default=None,
        help="Caminho do arquivo JSONL com dados de treinamento"
    )
    
    # Argumentos opcionais - Nomeação
    parser.add_argument(
        "--job-name",
        type=str,
        default=None,
        help="Nome do job de fine-tuning (auto-gerado se não fornecido)"
    )
    
    # Argumentos opcionais - Hiperparâmetros
    parser.add_argument(
        "--epochs",
        type=int,
        default=4,
        help="Número de épocas (1-40, padrão: 4)"
    )
    
    parser.add_argument(
        "--learning-rate",
        type=float,
        default=1.0,
        help="Multiplicador da taxa de aprendizado (0.5-2.0, padrão: 1.0)"
    )
    
    parser.add_argument(
        "--validation-split",
        type=float,
        default=0.2,
        help="Proporção dos dados para validação (0.0-1.0, padrão: 0.2)"
    )
    
    # Argumentos opcionais - Comportamento
    parser.add_argument(
        "--prepare-only",
        action="store_true",
        help="Apenas preparar e separar dados sem executar fine-tuning"
    )

    # Opções para fine-tuning local (LLaMA/LoRA)
    parser.add_argument(
        "--local-train",
        action="store_true",
        help="Executar fine-tuning local usando LoRA (transformers/peft)."
    )
    parser.add_argument(
        "--base-model",
        type=str,
        default=None,
        help="Modelo base HF para treino local (ex: 'meta-llama/Llama-2-7b')."
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("models/llama-lora"),
        help="Diretório de saída para artefatos do modelo local"
    )

    # Opção para testar um modelo local já treinado
    parser.add_argument(
        "--test-model-path",
        type=Path,
        default=None,
        help="Caminho para um modelo local salvo para teste de inferência"
    )
    parser.add_argument(
        "--test-prompt",
        type=str,
        default=None,
        help="Prompt de teste para executar no modelo local"
    )
    
    parser.add_argument(
        "--keep-temp-files",
        action="store_true",
        help="Manter arquivos temporários de treino/validação"
    )
    
    parser.add_argument(
        "--log-level",
        type=str,
        default="INFO",
        choices=["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"],
        help="Nível de logging (padrão: INFO)"
    )
    
    parser.add_argument(
        "--verbose",
        "-v",
        action="store_true",
        help="Modo verbose (equivalente a --log-level DEBUG)"
    )
    
    return parser


def validate_arguments(args: argparse.Namespace) -> None:
    """Valida os argumentos fornecidos."""
    
    # Se for teste de modelo local, --data é opcional
    if args.test_model_path:
        return
    
    # Caso contrário, --data é obrigatório
    if not args.data:
        raise ValueError("--data é obrigatório (ou use --test-model-path e --test-prompt para testar modelo salvo)")
    
    # Valida arquivo de entrada
    if not args.data.exists():
        raise FileNotFoundError(f"Arquivo não encontrado: {args.data}")
    
    if args.data.suffix != ".jsonl":
        raise ValueError(f"Arquivo deve ser JSONL, obtido: {args.data.suffix}")
    
    # Valida hiperparâmetros
    if not 1 <= args.epochs <= 40:
        raise ValueError(f"Epochs deve estar entre 1 e 40, obtido: {args.epochs}")
    
    if not 0.5 <= args.learning_rate <= 2.0:
        raise ValueError(
            f"Learning rate deve estar entre 0.5 e 2.0, obtido: {args.learning_rate}"
        )
    
    if not 0.0 <= args.validation_split <= 1.0:
        raise ValueError(
            f"Validation split deve estar entre 0.0 e 1.0, obtido: {args.validation_split}"
        )


def main():
    """Função principal da CLI."""
    
    # Parse dos argumentos
    parser = create_parser()
    args = parser.parse_args()
    
    # Configura logging
    log_level = "DEBUG" if args.verbose else args.log_level
    setup_logging(level=log_level)
    logger = logging.getLogger(__name__)
    
    logger.info("=" * 70)
    logger.info("FINE-TUNING CLI - GOOGLE GEMINI")
    logger.info("=" * 70)
    
    try:
        # Valida argumentos
        logger.info("\n Validando argumentos...")
        validate_arguments(args)
        logger.info(" Argumentos válidos")
        
        # Exibe configuração
        logger.info("\n Configuração:")
        logger.info(f"   Arquivo de dados: {args.data}")
        logger.info(f"   Job name: {args.job_name or '(auto-gerado)'}")
        logger.info(f"   Epochs: {args.epochs}")
        logger.info(f"   Learning rate multiplier: {args.learning_rate}")
        logger.info(f"   Validation split: {args.validation_split}")
        logger.info(f"   Keep temp files: {args.keep_temp_files}")
        
        # Cria o fine-tuner
        logger.info("\n🔧 Inicializando fine-tuner...")
        tuner = GeminiFineTuner()

        # Fluxo local: treinar via LoRA
        if args.local_train:
            if train_lora is None:
                logger.error("Dependências para treino local não instaladas ou módulo não disponível.")
                return 1

            if not args.base_model:
                logger.error("Para --local-train é necessário informar --base-model")
                return 1

            logger.info("Iniciando fine-tuning local (LoRA)...")
            try:
                train_lora(
                    base_model=args.base_model,
                    output_dir=str(args.output_dir),
                    train_jsonl=str(args.data),
                    num_epochs=args.epochs,
                )
                logger.info(f"Treino local concluído. Artefatos em: {args.output_dir}")
                return 0
            except Exception as e:
                logger.error(f"Erro durante treino local: {e}")
                return 1

        # Fluxo de teste de inferência local
        if args.test_model_path and args.test_prompt:
            if load_finetuned is None:
                logger.error("Módulo de inferência local não disponível. Instale 'transformers' e verifique módulos.")
                return 1

            try:
                logger.info(f"Carregando modelo local para teste: {args.test_model_path}")
                local_model = load_finetuned(str(args.test_model_path))
                logger.info("Executando inferência de teste...")
                out = local_model.generate(args.test_prompt)
                logger.info("Resposta do modelo:")
                logger.info(out)
                return 0
            except Exception as e:
                logger.error(f"Erro ao testar modelo local: {e}")
                return 1
        
        # Se apenas preparar dados
        if args.prepare_only:
            logger.info("\n Modo: Apenas preparar dados")
            train_file, val_file = tuner.prepare_training_data(
                args.data,
                validation_split=args.validation_split
            )
            logger.info(f"\n Dados preparados:")
            logger.info(f"   Treino: {train_file}")
            logger.info(f"   Validação: {val_file}")
            return 0
        
        # Executa fine-tuning completo
        logger.info("\nIniciando fine-tuning...")
        result = tuner.run_fine_tuning(
            data_file=args.data,
            job_name=args.job_name,
            epochs=args.epochs,
            learning_rate_multiplier=args.learning_rate,
            validation_split=args.validation_split,
            auto_delete_files=not args.keep_temp_files,
        )
        
        # Exibe resultados
        logger.info(f"\n RESULTADO DO FINE-TUNING:")
        logger.info("=" * 70)
        for key, value in result.items():
            logger.info(f"   {key}: {value}")
        logger.info("=" * 70)
        
        # Dicas finais
        logger.info(f"\n PRÓXIMOS PASSOS:")
        if result.get("model_id"):
            logger.info(f"   1. Modelo fine-tuned criado: {result['model_id']}")
            logger.info(f"   2. Atualize o config.py com:")
            logger.info(f"      model_name = \"{result['model_id']}\"")
            logger.info(f"   3. Teste o modelo executando: python src/main.py")
        
        return 0
        
    except FileNotFoundError as e:
        logger.error(f"Erro: {e}")
        return 1
    
    except ValueError as e:
        logger.error(f"Erro de validação: {e}")
        return 1
    
    except Exception as e:
        logger.error(f"Erro inesperado: {e}")
        logger.exception("Traceback completo:")
        return 1


if __name__ == "__main__":
    exit_code = main()
    sys.exit(exit_code)
