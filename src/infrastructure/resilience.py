"""
Estratégias de resiliência para chamadas ao LLM externo.
Implementa retry com backoff exponencial e circuit breaker.
"""

from tenacity import (
    retry,
    stop_after_attempt,
    wait_exponential,
    retry_if_exception_type,
)
from pybreaker import CircuitBreaker as PyCircuitBreaker
import logging

logger = logging.getLogger(__name__)

@retry(
    stop=stop_after_attempt(3),
    wait=wait_exponential(multiplier=1, min=2, max=10),
    retry=retry_if_exception_type(TimeoutError),
    before_sleep=lambda retry_state: logger.warning(
        f"Retentativa {retry_state.attempt_number} após erro de timeout"
    )
)
def call_llm_with_retry(chain, inputs: dict):
    """Chama LLM com retry automático."""
    try:
        return chain.invoke(inputs)
    except TimeoutError:
        logger.error("Timeout ao chamar LLM")
        raise


# Circuit breaker para proteção contra falhas cascata
_llm_circuit_breaker = PyCircuitBreaker(
    fail_max=5,
    reset_timeout=60,
    name="llm_service"
)


@_llm_circuit_breaker
def call_llm_with_circuit_breaker(chain, inputs: dict):
    """Chama LLM com proteção de circuit breaker."""
    return chain.invoke(inputs)