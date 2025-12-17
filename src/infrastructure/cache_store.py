"""
Cache de respostas e documentos recuperados.
Melhora latência em perguntas frequentes.
"""

import json
from functools import lru_cache
from typing import List, Dict, Tuple
from hashlib import md5
import redis
from langchain_core.documents import Document

class ResponseCache:
    """Cache distribuído para respostas médicas."""
    
    def __init__(self, redis_url: str = "redis://localhost:6379"):
        self.redis_client = redis.from_url(redis_url)
        self.ttl = 3600  # 1 hora para respostas clínicas
    
    def _hash_query(self, question: str) -> str:
        """Gera hash determinístico da pergunta."""
        return md5(question.lower().encode()).hexdigest()
    
    def get_cached_response(self, question: str) -> Dict | None:
        """Recupera resposta em cache."""
        cache_key = f"medical_response:{self._hash_query(question)}"
        cached = self.redis_client.get(cache_key)
        return json.loads(cached) if cached else None
    
    def cache_response(self, question: str, documents: List[Document], generation: str) -> None:
        """Armazena resposta em cache."""
        cache_key = f"medical_response:{self._hash_query(question)}"
        cache_data = {
            "documents": [d.page_content for d in documents],
            "generation": generation
        }
        self.redis_client.setex(cache_key, self.ttl, json.dumps(cache_data))
    
    def invalidate(self, pattern: str = "*") -> int:
        """Invalida cache por padrão (ex: após atualização de protocolos)."""
        keys = self.redis_client.keys(f"medical_response:{pattern}")
        if keys:
            return self.redis_client.delete(*keys)
        return 0