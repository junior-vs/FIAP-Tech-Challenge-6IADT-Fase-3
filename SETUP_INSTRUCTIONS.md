# Instruções de Setup, Preprocess e Fine-Tuning

## Configuração do Arquivo .env

Crie um arquivo `.env` na raiz do projeto com as seguintes variáveis:

### Variáveis Obrigatórias:
- **GEMINI_API_KEY**: Obtenha em https://aistudio.google.com/apikey
- **MODEL_NAME**: Modelo base para fine-tuning (padrão: `MODEL_NAME=local-llama:models/llama-medical-lora`)

---

## Preparação do Ambiente

### 1. Criar Virtual Environment
```bash
# Windows
python -m venv .venv
.\.venv\Scripts\activate

# Linux/Mac
python3 -m venv .venv
source .venv/bin/activate
```

### 2. Instalar Dependências
```bash
# Instalar do requirements.txt
pip install -r requirements.txt

# Ou do pyproject.toml
pip install -e .
```

### 3. Estrutura de Diretórios Esperada
```
projeto/
├── .env                          # Configurações (NÃO versionado)
├── requirements.txt              # Dependências Python
├── docs/
│   └── knowledge_base/
│       ├── 7_SeniorHealth_QA/    # XMLs brutos para processamento
│       ├── finetuning_data.jsonl # Dados para fine-tuning (auto-gerado)
│       └── *.jsonl               # Outros formatos de dados
├── data/
│   └── chroma_db/                # Banco de dados vetorial
└── models/
    └── llama-medical-lora/       # Adaptadores LoRA (se usado)
```

---

## Preprocess de Dados

### Objetivo
Transformar dados brutos (XML, texto) em formato JSONL otimizado para fine-tuning.

### 1. Gerar Dados de Amostra (Teste)
```bash
python script_create_sample_data.py
```
Gera `sample_finetuning_data.jsonl` com 100 exemplos médicos de teste.

### 2. Processar Dados Reais da Knowledge Base

#### Opção A: Desde XMLs (7_SeniorHealth_QA)
```bash
# Os XMLs já estão em docs/knowledge_base/7_SeniorHealth_QA/
# Eles serão processados automaticamente quando usar fine_tuning_cli.py
```

#### Opção B: Criar Dados Customizados
Crie um arquivo JSONL em `docs/knowledge_base/` com o seguinte formato:

**Para fine-tuning com Gemini (conversation format):**
execute o script convert_xml_to_gemini_format.py no caminho src/infrastructure/preprocess/
formato do arquivo que será gerado:
```json
{
  "contents": [
    {"role": "user", "parts": [{"text": "Pergunta sobre saúde"}]},
    {"role": "assistant", "parts": [{"text": "Resposta completa"}]}
  ]
}
```


**Para fine-tuning com LLaMA (instruction format):**
execute o script convert_messages_to_instruction_response.py no caminho src/infrastructure/preprocess/
formato do arquivo que será gerado:
```json
{"instruction": "Pergunta sobre saúde", "response": "Resposta completa"}
```


### 3. Validar Integridade dos Dados
```python
from pathlib import Path
import json

jsonl_path = "docs/knowledge_base/seu_arquivo.jsonl"
with open(jsonl_path) as f:
    for i, line in enumerate(f, 1):
        try:
            json.loads(line)
        except json.JSONDecodeError as e:
            print(f"Erro na linha {i}: {e}")

print(f"✓ {i} registros válidos")
```

---

## Fine-Tuning LLaMA (Local)
executar fine_tuning_cli.py

### Pré-requisitos
```bash
# Instalar dependências adicionais para LLaMA
pip install transformers datasets accelerate peft bitsandbytes

# GPU NVIDIA (recomendado)
# Verificar: nvidia-smi
```

### 1. Treinar Adaptador LoRA
```python
from src.infrastructure.llama_finetune import train_lora

train_lora(
    base_model='meta-llama/Llama-2-7b-hf',  # ou 'meta-llama/Llama-2-13b-hf'
    output_dir='models/llama-medical-lora',
    train_jsonl='docs/knowledge_base/finetuning_instruction_response.jsonl',
    num_epochs=3,
    per_device_train_batch_size=1,
    learning_rate=2e-4,
    lora_r=8,
    lora_alpha=16,
    lora_dropout=0.05,
    fp16=True,
    max_seq_length=1024,
)
```

### 2. Parâmetros de Treinamento
| Parâmetro | Recomendação | Descrição |
|-----------|-------------|-----------|
| `num_epochs` | 3-5 | Número de épocas de treinamento |
| `per_device_train_batch_size` | 1-4 | Batch size (reduzir se OOM) |
| `learning_rate` | 1e-4 a 5e-4 | Taxa de aprendizado |
| `lora_r` | 8-16 | Rank do adaptador LoRA |
| `lora_alpha` | 16-32 | Alpha do LoRA (2x o rank) |
| `fp16` | True | Usar float16 para economizar memória |
| `max_seq_length` | 512-2048 | Tamanho máximo de sequência |

### 3. Usar Modelo Treinado em Inferência
```python
from src.infrastructure.local_llama import load_finetuned

# Carregar modelo
model = load_finetuned('models/llama-medical-lora')

# Gerar resposta
prompt = "Qual é o tratamento para hipertensão?"
response = model.generate(prompt, max_length=500)
print(response)
```

---

## Fluxo Completo: Do Início ao Fim

### Setup Inicial
```bash
# 1. Clone e navegue
cd projeto

# 2. Criar venv
python -m venv .venv
.\.venv\Scripts\activate  # Windows

# 3. Instalar dependências
pip install -r requirements.txt

# 4. Configurar .env
# (editar arquivo .env conforme seção acima)

# 5. Efetuar fine-tuning
```

### Pós Fine-Tuning
```bash
# 8. Atualizar .env com novo model_name
# MODEL_NAME=tunedModels/medical-assistant-ft-abc123xyz

# 9. Executar RAG pipeline
python src/main.py

# 10. Testes
pytest tests/ -v
```

---

## Suporte

Para problemas ou dúvidas:
1. Verifique logs: `logs/` (se habilitado)
2. Veja arquivo `.env` está correto
3. Execute testes: `pytest tests/ -v`
4. Consulte documentação específica em `docs/`
