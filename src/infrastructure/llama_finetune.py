"""
Módulo de fine-tuning para modelos LLaMA usando LoRA/PEFT.

Observações:
- Este script usa `transformers`, `datasets` e `peft` (LoRA).
- Requisitos e execução estão descritos em docs/LLAMA_FINETUNING.md.

Funções principais:
- prepare_dataset_from_jsonl(path) -> datasets.Dataset
- train_lora(...)
"""
from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Optional

logger = logging.getLogger(__name__)


def prepare_dataset_from_jsonl(jsonl_path: str, instruction_field: str = "instruction", response_field: str = "response"):
    """Carrega um arquivo JSONL (formato de fine-tuning) e retorna lista de pares.

    O arquivo em `docs/knowledge_base` já contém exemplos no formato usado.
    Retorna lista de dicts com chaves `prompt` e `completion`.
    """
    path = Path(jsonl_path)
    if not path.exists():
        raise FileNotFoundError(f"Arquivo não encontrado: {jsonl_path}")

    data = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            obj = json.loads(line)
            instruction = obj.get(instruction_field, "")
            response = obj.get(response_field, "")
            if instruction and response:
                # Monta prompt simples (instrução + contexto opcional)
                prompt = f"### Instrução:\n{instruction}\n\n### Resposta:\n"
                completion = f" {response.strip()}"
                data.append({"prompt": prompt, "completion": completion})

    logger.info(f"Dataset carregado: {len(data)} registros de {jsonl_path}")
    return data


def train_lora(
    base_model: str,
    output_dir: str,
    train_jsonl: str,
    num_epochs: int = 3,
    per_device_train_batch_size: int = 1,
    learning_rate: float = 2e-4,
    lora_r: int = 8,
    lora_alpha: int = 16,
    lora_dropout: float = 0.05,
    fp16: bool = True,
    use_8bit: bool = True,
    max_seq_length: int = 1024,
):
    """Treina um adaptador LoRA a partir de um modelo base.

    Nota: Esta função tenta importar as bibliotecas necessárias e
    levanta um erro com dica de instalação quando ausentes.
    """
    try:
        from datasets import Dataset
        from transformers import AutoTokenizer, TrainingArguments, Trainer
        import transformers
        from peft import get_peft_model, LoraConfig, TaskType, prepare_model_for_kbit_training
        import torch
    except Exception as e:
        raise RuntimeError(
            "Dependências faltando. Instale: transformers datasets accelerate peft bitsandbytes (opcional)."
        )

    # Carrega dados
    examples = prepare_dataset_from_jsonl(train_jsonl)
    if not examples:
        raise RuntimeError("Nenhum exemplo encontrado para treinamento")

    ds = Dataset.from_list(examples)

    # Tokenizer
    tokenizer = AutoTokenizer.from_pretrained(base_model, use_fast=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    def tokenize_fn(ex):
        inp = ex["prompt"]
        tgt = ex["completion"]
        full = inp + tgt
        tokenized = tokenizer(
            full,
            truncation=True,
            padding="max_length",
            max_length=max_seq_length,
            return_tensors=None,
        )
        return tokenized

    tokenized = ds.map(tokenize_fn, batched=False, remove_columns=ds.column_names)

    # Model com quantização 8-bit para economizar memória
    quantization_config = None
    if use_8bit:
        try:
            from transformers import BitsAndBytesConfig
            quantization_config = BitsAndBytesConfig(
                load_in_8bit=True,
                bnb_8bit_compute_dtype=torch.float16,
                bnb_8bit_use_double_quant=True,
                bnb_8bit_quant_type="nf8",
            )
        except Exception:
            logger.info("BitsAndBytesConfig não disponível; tentando sem quantização")

    if quantization_config:
        model = transformers.AutoModelForCausalLM.from_pretrained(
            base_model,
            quantization_config=quantization_config,
            device_map="auto",
            low_cpu_mem_usage=True,
        )
    else:
        model = transformers.AutoModelForCausalLM.from_pretrained(
            base_model,
            device_map="auto",
            torch_dtype=torch.float16 if fp16 and torch.cuda.is_available() else torch.float32,
            low_cpu_mem_usage=True,
        )

    # Preparar modelo para LoRA
    model = prepare_model_for_kbit_training(model) if use_8bit else model
    model.config.use_cache = False  # Desabilita cache para economizar memória
    model.gradient_checkpointing_enable()  # Gradient checkpointing economiza memória

    peft_config = LoraConfig(
        task_type=TaskType.CAUSAL_LM,
        inference_mode=False,
        r=lora_r,
        lora_alpha=lora_alpha,
        lora_dropout=lora_dropout,
        target_modules=["q_proj", "v_proj"],  # Reduz para os módulos principais
    )

    model = get_peft_model(model, peft_config)

    # Treinamento com otimizações para GPU pequena
    training_args = TrainingArguments(
        output_dir=output_dir,
        per_device_train_batch_size=per_device_train_batch_size,
        num_train_epochs=num_epochs,
        learning_rate=learning_rate,
        fp16=fp16 and torch.cuda.is_available(),
        logging_steps=10,
        save_total_limit=2,
        remove_unused_columns=False,
        gradient_accumulation_steps=4,  # Simula batch maior sem usar mais memória
        optim="paged_adamw_8bit" if use_8bit else "adamw_torch",
    )

    from transformers import DataCollatorForLanguageModeling
    data_collator = DataCollatorForLanguageModeling(tokenizer=tokenizer, mlm=False)

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=tokenized,
        data_collator=data_collator,
    )

    logger.info("Iniciando treinamento LoRA... (pode demorar dependendo do hardware)")
    trainer.train()

    # Salvar
    Path(output_dir).mkdir(parents=True, exist_ok=True)
    model.save_pretrained(output_dir)
    tokenizer.save_pretrained(output_dir)

    logger.info(f"Treinamento concluído. Artefatos gravados em: {output_dir}")


if __name__ == "__main__":
    print("Este arquivo expõe funções de treino. Use como módulo ou veja docs/LLAMA_FINETUNING.md")
