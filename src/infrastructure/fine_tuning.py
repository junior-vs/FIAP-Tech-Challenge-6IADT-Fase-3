"""
Módulo: src/infrastructure/fine_tuning.py

Fine-tuning de modelos Google Gemini utilizando a API de fine-tuning do Google.

Este módulo fornece funcionalidades para fazer fine-tuning de modelos Gemini
com dados no formato JSONL contendo pares de mensagens (user/assistant).
"""

import json
import logging
import time
from pathlib import Path
from typing import Optional, Dict, Any, List
import google.generativeai as genai

from src.config import settings

logger = logging.getLogger(__name__)


class GeminiFineTuner:
    """
    Classe para fazer fine-tuning de modelos Gemini.
    
    Utiliza a API de fine-tuning do Google Generative AI para treinar modelos
    com dados customizados em formato JSONL.
    """
    
    def __init__(self, api_key: Optional[str] = None):
        """
        Inicializa o fine-tuner com a API key.
        
        Args:
            api_key: Chave da API do Google. Se não fornecida, usa settings.gemini_api_key
        """
        self.api_key = api_key or settings.gemini_api_key
        genai.configure(api_key=self.api_key)
        self.base_model = settings.model_name
        logger.info(f"GeminiFineTuner inicializado com modelo base: {self.base_model}")
    
    def prepare_training_data(
        self,
        jsonl_file: Path,
        validation_split: float = 0.2
    ) -> tuple[Path, Path]:
        """
        Prepara dados de treinamento separando em conjuntos de treinamento e validação.
        
        Args:
            jsonl_file: Caminho do arquivo JSONL com os dados
            validation_split: Proporção dos dados para validação (0.0-1.0)
            
        Returns:
            Tupla com caminhos dos arquivos de treinamento e validação
            
        Raises:
            FileNotFoundError: Se o arquivo não existe
            ValueError: Se validation_split está fora do intervalo válido
        """
        if not jsonl_file.exists():
            raise FileNotFoundError(f"Arquivo não encontrado: {jsonl_file}")
        
        if not 0.0 <= validation_split <= 1.0:
            raise ValueError("validation_split deve estar entre 0.0 e 1.0")
        
        # Lê todos os dados
        records = []
        try:
            with open(jsonl_file, 'r', encoding='utf-8') as f:
                for line_num, line in enumerate(f, 1):
                    if line.strip():
                        try:
                            record = json.loads(line)
                            records.append(record)
                        except json.JSONDecodeError as e:
                            logger.warning(f"Erro ao decodificar linha {line_num}: {e}")
                            continue
        except Exception as e:
            logger.error(f"Erro ao ler arquivo {jsonl_file}: {e}")
            raise
        
        logger.info(f"Carregados {len(records)} registros do arquivo")
        
        if len(records) == 0:
            raise ValueError("Nenhum registro válido encontrado no arquivo")
        
        # Calcula ponto de divisão
        split_idx = int(len(records) * (1 - validation_split))
        train_records = records[:split_idx]
        val_records = records[split_idx:]
        
        # Cria arquivos separados
        output_dir = jsonl_file.parent
        train_file = output_dir / f"{jsonl_file.stem}_train.jsonl"
        val_file = output_dir / f"{jsonl_file.stem}_val.jsonl"
        
        # Escreve arquivo de treinamento
        with open(train_file, 'w', encoding='utf-8') as f:
            for record in train_records:
                f.write(json.dumps(record) + '\n')
        logger.info(f"Arquivo de treinamento: {train_file} ({len(train_records)} registros)")
        
        # Escreve arquivo de validação
        with open(val_file, 'w', encoding='utf-8') as f:
            for record in val_records:
                f.write(json.dumps(record) + '\n')
        logger.info(f"Arquivo de validação: {val_file} ({len(val_records)} registros)")
        
        return train_file, val_file
    
    def upload_training_file(self, file_path: Path, display_name: str) -> str:
        """
        Faz upload do arquivo de treinamento para o Google AI.
        
        Args:
            file_path: Caminho do arquivo JSONL
            display_name: Nome para exibição do arquivo
            
        Returns:
            ID do arquivo no Google AI
        """
        logger.info(f"Fazendo upload do arquivo: {file_path}")
        
        try:
            # Determina o mimetype correto para JSONL
            # A Gemini API requer mimetype explícito para arquivos JSONL
            mime_type = "application/json"
            
            if str(file_path).endswith('.jsonl'):
                mime_type = "application/json"  # JSONL é tratado como JSON
            
            # Faz upload do arquivo com mimetype explícito
            response = genai.upload_file(
                path=str(file_path),
                display_name=display_name,
                mime_type=mime_type,
            )
            
            file_id = response.name
            logger.info(f"Upload concluído. ID do arquivo: {file_id}")
            return file_id
            
        except Exception as e:
            logger.error(f"Erro ao fazer upload: {e}")
            raise
    
    def create_fine_tuning_job(
        self,
        training_file_id: str,
        validation_file_id: Optional[str] = None,
        job_name: Optional[str] = None,
        epochs: int = 4,
        learning_rate_multiplier: float = 1.0,
    ) -> str:
        """
        Cria um job de fine-tuning no Google AI.
        
        Args:
            training_file_id: ID do arquivo de treinamento uploadado
            validation_file_id: ID do arquivo de validação (opcional)
            job_name: Nome do job (será auto-gerado se não fornecido)
            epochs: Número de épocas para treinamento (1-40)
            learning_rate_multiplier: Multiplicador da taxa de aprendizado (0.5-2.0)
            
        Returns:
            ID do job de fine-tuning
        """
        if not 1 <= epochs <= 40:
            raise ValueError("epochs deve estar entre 1 e 40")
        
        if not 0.5 <= learning_rate_multiplier <= 2.0:
            raise ValueError("learning_rate_multiplier deve estar entre 0.5 e 2.0")
        
        job_name = job_name or f"medical_assistant_finetuning_{int(time.time())}"
        
        logger.info(f"Criando job de fine-tuning: {job_name}")
        logger.info(f"Modelo base: {self.base_model}")
        logger.info(f"Épocas: {epochs}")
        logger.info(f"Learning rate multiplier: {learning_rate_multiplier}")
        
        try:
            # Cria o job de fine-tuning
            # A API do Gemini espera um File object, não uma string ID
            logger.info("Enviando solicitação para criar modelo fine-tuned...")
            logger.info(f"Arquivo de treinamento: {training_file_id}")
            
            # Obtém o objeto File da API
            training_file = genai.get_file(training_file_id)
            logger.info(f"File object obtido: {training_file.name}")
            logger.info(f"File URI: {training_file.uri}")
            
            response = genai.create_tuned_model(
                source_model=self.base_model,
                training_data=training_file.uri,  # ← URI string direto
                id=job_name,
                display_name=job_name,
            )
            
            job_id = response.name
            logger.info(f"Job criado com ID: {job_id}")
            return job_id
            
        except Exception as e:
            logger.error(f"Erro ao criar job de fine-tuning: {e}")
            raise
    
    def wait_for_completion(
        self,
        job_id: str,
        check_interval: int = 30,
        max_wait_time: int = 3600,
    ) -> Dict[str, Any]:
        """
        Aguarda a conclusão de um job de fine-tuning.
        
        Args:
            job_id: ID do job de fine-tuning
            check_interval: Intervalo em segundos para verificar o status
            max_wait_time: Tempo máximo de espera em segundos
            
        Returns:
            Dicionário com informações do job concluído
            
        Raises:
            TimeoutError: Se o job não completar dentro de max_wait_time
            RuntimeError: Se o job falhar
        """
        logger.info(f"Aguardando conclusão do job: {job_id}")
        
        start_time = time.time()
        
        while True:
            elapsed = time.time() - start_time
            
            if elapsed > max_wait_time:
                raise TimeoutError(
                    f"Job não completou em {max_wait_time} segundos"
                )
            
            try:
                job = genai.get_tuned_model(job_id)
                
                # Log de progresso
                if hasattr(job, 'state'):
                    logger.info(f"   Status: {job.state}")
                
                # Verifica se completou
                if job.state == "SUCCEEDED":
                    logger.info("Fine-tuning concluído com sucesso!")
                    return self._extract_job_info(job)
                
                elif job.state == "FAILED":
                    error_msg = getattr(job, 'error', 'Erro desconhecido')
                    raise RuntimeError(f"Job falhou: {error_msg}")
                
                elif job.state == "CANCELLED":
                    raise RuntimeError("Job foi cancelado")
                
                # Continua aguardando
                logger.debug(f"Aguardando... ({int(elapsed)}s)")
                time.sleep(check_interval)
                
            except Exception as e:
                if isinstance(e, (TimeoutError, RuntimeError)):
                    raise
                logger.error(f"Erro ao verificar status do job: {e}")
                raise
    
    def _extract_job_info(self, job: Any) -> Dict[str, Any]:
        """
        Extrai informações importantes do job.
        
        Args:
            job: Objeto do job do Gemini
            
        Returns:
            Dicionário com informações do job
        """
        info = {
            "job_id": job.name,
            "model_id": getattr(job, "tuned_model_name", None),
            "state": getattr(job, "state", None),
            "create_time": getattr(job, "create_time", None),
            "update_time": getattr(job, "update_time", None),
        }
        
        # Tenta extrair métricas de treinamento
        if hasattr(job, "training_stats") and job.training_stats:
            info["training_stats"] = {
                "total_steps": getattr(job.training_stats, "total_steps", None),
                "total_billable_characters": getattr(
                    job.training_stats, "total_billable_characters", None
                ),
            }
        
        return info
    
    def run_fine_tuning(
        self,
        data_file: Path,
        job_name: Optional[str] = None,
        epochs: int = 4,
        learning_rate_multiplier: float = 1.0,
        validation_split: float = 0.2,
        auto_delete_files: bool = True,
    ) -> Dict[str, Any]:
        """
        Executa o pipeline completo de fine-tuning.
        
        Este método encadeia todas as etapas:
        1. Prepara os dados (treino/validação)
        2. Faz upload dos arquivos
        3. Cria o job de fine-tuning
        4. Aguarda a conclusão
        
        Args:
            data_file: Arquivo JSONL com os dados de treinamento
            job_name: Nome do job (auto-gerado se não fornecido)
            epochs: Número de épocas
            learning_rate_multiplier: Multiplicador da taxa de aprendizado
            validation_split: Proporção dos dados para validação
            auto_delete_files: Se True, deleta arquivos locais após conclusão
            
        Returns:
            Dicionário com informações do job concluído
        """
        logger.info("=" * 60)
        logger.info("INICIANDO PIPELINE DE FINE-TUNING")
        logger.info("=" * 60)
        
        try:
            # 1. Prepara dados
            logger.info("\n Preparando dados...")
            train_file, val_file = self.prepare_training_data(
                data_file, validation_split
            )
            
            # 2. Faz upload
            logger.info("\n Fazendo upload dos arquivos...")
            train_file_id = self.upload_training_file(
                train_file,
                display_name=f"training_{data_file.stem}"
            )
            
            val_file_id = self.upload_training_file(
                val_file,
                display_name=f"validation_{data_file.stem}"
            )
            
            # 3. Cria job
            logger.info("\n Criando job de fine-tuning...")
            job_id = self.create_fine_tuning_job(
                training_file_id=train_file_id,
                validation_file_id=val_file_id,
                job_name=job_name,
                epochs=epochs,
                learning_rate_multiplier=learning_rate_multiplier,
            )
            
            # 4. Aguarda conclusão
            logger.info("\n Aguardando conclusão...")
            result = self.wait_for_completion(job_id)
            
            # Deleta arquivos locais se solicitado
            if auto_delete_files:
                logger.info("\n Limpando arquivos temporários...")
                for file_path in [train_file, val_file]:
                    try:
                        file_path.unlink()
                        logger.debug(f"   Deletado: {file_path}")
                    except Exception as e:
                        logger.warning(f"   Erro ao deletar {file_path}: {e}")
            
            logger.info("\n" + "=" * 60)
            logger.info("FINE-TUNING CONCLUÍDO COM SUCESSO!")
            logger.info("=" * 60)
            
            return result
            
        except Exception as e:
            logger.error("\n" + "=" * 60)
            logger.error("ERRO DURANTE FINE-TUNING")
            logger.error("=" * 60)
            logger.error(str(e))
            raise


def main():
    """
    Função principal para teste rápido do módulo.
    """
    import sys
    from pathlib import Path
    
    # Configura logging
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    
    # Caminho do arquivo de dados
    # data_file = Path("docs/knowledge_base/finetuning_prompt_completion.jsonl")
    data_file = Path("docs/knowledge_base/finetuning_data_gemini_messages.jsonl")
    
    if not data_file.exists():
        logger.error(f"Arquivo não encontrado: {data_file}")
        sys.exit(1)
    
    # Cria o fine-tuner
    tuner = GeminiFineTuner()
    
    # Executa o fine-tuning
    try:
        result = tuner.run_fine_tuning(
            data_file=data_file,
            job_name="medical_assistant_ft",
            epochs=4,
            learning_rate_multiplier=1.0,
            validation_split=0.2,
            auto_delete_files=True,
        )
        
        logger.info(f"Resultado do fine-tuning:")
        for key, value in result.items():
            logger.info(f"   {key}: {value}")
            
    except Exception as e:
        logger.error(f"Erro: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()
