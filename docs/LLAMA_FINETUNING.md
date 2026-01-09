LLAMA FINETUNING - Guia rápido
=============================

Resumo
------
Este projeto adiciona um módulo mínimo para treinar adaptadores LoRA (PEFT)
para modelos LLaMA/HF e um adaptador simples de inferência local.

Pré-requisitos (recomendado)
----------------------------
- GPU com CUDA
- Python 3.10+
- Instalar pacotes:

```bash
pip install transformers datasets accelerate peft bitsandbytes
```

Treinamento
-----------
Exemplo mínimo para treinar com o dataset JSONL já presente em `docs/knowledge_base`:

```python
from src.infrastructure.llama_finetune import train_lora

train_lora(
    base_model='meta-llama/Llama-2-7b',
    output_dir='models/llama-medical-lora',
    train_jsonl='docs/knowledge_base/finetuning_instruction_response.jsonl',
    num_epochs=3,
)
```

Inferência local
----------------
Após treinar, carregue o adaptador local:

```python
from src.infrastructure.local_llama import load_finetuned

model = load_finetuned('models/llama-medical-lora')
resp = model.generate('Explique o que é AMD em uma frase:')
print(resp)
```

Integrando com RAG
-------------------
No repositório a integração principal é feita via `src.infrastructure.llm_factory`.
Para integrar o modelo local ao pipeline RAG você pode:

1. Treinar o LoRA e salvar em `models/llama-medical-lora`
2. Modificar `src/config.py` e `src/infrastructure/llm_factory.py` para detectar um
   `settings.model_name` começando com `local-llama:` e carregar `LocalLlama` via `local_llama.load_finetuned`
3. Adaptar `src/use_cases/nodes.py` para usar o adaptador local como fallback quando o LLM
   não for compatível com a composição de `langchain_core` (ex.: usar chamadas diretas a `model.generate`)

Notas finais
-----------
- Esta integração fornece um ponto de partida. Ajustes são necessários conforme o formato de prompt
  usado para treinar e as exigências de chains/structured outputs no pipeline atual.
