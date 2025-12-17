"""
Configuração centralizada de logging estruturado para o sistema médico.
Usa loguru para logs com contexto e estrutura.
"""

import sys
import logging
from pathlib import Path
from loguru import logger


def setup_logging(level: str = "INFO") -> None:
    """
    Configura logging estruturado para a aplicação médica.
    
    WHEN [aplicação inicia]
    THE SYSTEM SHALL [configurar logging com formato estruturado e níveis apropriados]
    
    Args:
        level: Nível de logging (DEBUG, INFO, WARNING, ERROR, CRITICAL)
    
    Accessibility Note: Logs use clear, plain language messages for clarity.
    """
    
    # Remover handler padrão do loguru
    logger.remove()
    
    # Criar diretório de logs se não existir
    log_dir = Path("logs")
    log_dir.mkdir(exist_ok=True)
    
    # Formato estruturado para logs em console
    console_format = (
        "<level>{level: <8}</level> | "
        "<cyan>{name}</cyan>:<cyan>{function}</cyan>:<cyan>{line}</cyan> - "
        "<level>{message}</level>"
    )
    
    # Formato estruturado para logs em arquivo
    file_format = (
        "{time:YYYY-MM-DD HH:mm:ss} | "
        "{level: <8} | "
        "{name}:{function}:{line} - "
        "{message}"
    )
    
    # Handler para console (stdout)
    logger.add(
        sys.stdout,
        format=console_format,
        level=level,
        colorize=True,
        backtrace=True,
        diagnose=True,
    )
    
    # Handler para arquivo de log (rotação diária)
    logger.add(
        log_dir / "machado-oraculo.log",
        format=file_format,
        level=level,
        rotation="00:00",  # Rotação diária à meia-noite
        retention="7 days",  # Manter últimos 7 dias
        compression="zip",  # Comprimir logs antigos
        backtrace=True,
        diagnose=False,  # Desabilitar diagnóstico em arquivo para economizar espaço
    )
    
    # Handler para erros (arquivo separado)
    logger.add(
        log_dir / "machado-oraculo-errors.log",
        format=file_format,
        level="ERROR",
        rotation="00:00",
        retention="30 days",
        compression="zip",
        backtrace=True,
    )
    
    # Configurar logging padrão do Python para usar loguru
    logging.basicConfig(handlers=[InterceptHandler()], level=0)
    
    logger.info("✅ Sistema de logging inicializado com sucesso")


class InterceptHandler(logging.Handler):
    """
    Interceptor que redireciona logs padrão do Python para loguru.
    Permite que bibliotecas que usam logging padrão sejam capturadas.
    """
    
    def emit(self, record: logging.LogRecord) -> None:
        try:
            level = logger.level(record.levelname).name
        except ValueError:
            level = record.levelno
        
        logger.log(level, record.getMessage())


def get_logger(name: str) -> logging.Logger:
    """
    Obtém logger configurado para um módulo específico.
    
    WHEN [módulo solicita logger]
    THE SYSTEM SHALL [retornar logger com contexto do módulo]
    
    Args:
        name: Nome do módulo (geralmente __name__)
        
    Returns:
        logging.Logger: Logger configurado
    """
    return logging.getLogger(name)

