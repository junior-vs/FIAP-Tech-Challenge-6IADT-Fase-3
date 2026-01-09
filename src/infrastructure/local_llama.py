"""
Adaptador simples para carregar um modelo fine-tuned local (transformers)
e expor uma função de geração mínima para integrar com RAG.

Este adaptador fornece uma API leve:
- `LocalLlama.from_pretrained(path)` -> instancia
- `generate(prompt, max_new_tokens=256)` -> str

Nota: projetado para integração incremental; não substitui a integração
completa do `langchain`/`langgraph`. Use como fallback local.
"""
from __future__ import annotations

import logging
from typing import Optional
from pathlib import Path

logger = logging.getLogger(__name__)


class LocalLlama:
    """Wrapper mínimo em torno de `transformers` para geração causal."""

    is_local_llama = True

    def __init__(self, model, tokenizer, device: Optional[str] = None):
        self.model = model
        self.tokenizer = tokenizer
        self.device = device

    @classmethod
    def from_pretrained(cls, path: str, device: Optional[str] = None):
        try:
            from transformers import AutoModelForCausalLM, AutoTokenizer, pipeline
        except Exception as e:
            raise RuntimeError("Instale 'transformers' para usar LocalLlama")

        model = AutoModelForCausalLM.from_pretrained(path, device_map="auto")
        tokenizer = AutoTokenizer.from_pretrained(path, use_fast=True)

        return cls(model=model, tokenizer=tokenizer, device=device)

    def generate(self, prompt: str, max_new_tokens: int = 256) -> str:
        try:
            import torch
            from transformers import TextGenerationPipeline
        except Exception:
            raise RuntimeError("Instale 'transformers' para usar LocalLlama")

        # Usa pipeline para conveniência - não especifica device quando o modelo foi carregado com accelerate
        try:
            pipe = TextGenerationPipeline(model=self.model, tokenizer=self.tokenizer)
        except (ValueError, RuntimeError):
            # Fallback: try with device specification
            pipe = TextGenerationPipeline(model=self.model, tokenizer=self.tokenizer, device=-1)
        
        out = pipe(prompt, max_new_tokens=max_new_tokens, do_sample=False)
        # pipeline retorna uma lista de dicts
        return out[0]["generated_text"] if out else ""


def load_finetuned(path: str) -> LocalLlama:
    """Carrega um modelo fine-tuned salvo em `path` e retorna um LocalLlama."""
    if not Path(path).exists():
        raise FileNotFoundError(f"Modelo não encontrado em: {path}")
    return LocalLlama.from_pretrained(path)
