#!/usr/bin/env python3
"""
Script para testar acesso e permissões da API Gemini
"""
import google.generativeai as genai
from src.config import settings

print("=" * 80)
print("TESTE DE ACESSO À API GEMINI")
print("=" * 80)

# Configura API
api_key = settings.gemini_api_key
genai.configure(api_key=api_key)
print(f"✓ API configurada com chave: {api_key[:20]}...")

# Lista modelos disponíveis
try:
    print("\n📋 Modelos disponíveis:")
    models = genai.list_models()
    for model in models:
        print(f"  - {model.name}")
except Exception as e:
    print(f"✗ Erro ao listar modelos: {e}")

# Testa modelo padrão
try:
    print(f"\n🧪 Testando modelo: {settings.model_name}")
    model = genai.GenerativeModel(settings.model_name)
    response = model.generate_content("Diga 'ok'")
    print(f"✓ Modelo funciona! Resposta: {response.text[:50]}...")
except Exception as e:
    print(f"✗ Erro ao testar modelo: {e}")

# Testa gemini-1.5-pro (melhor para fine-tuning)
try:
    print(f"\n🧪 Testando modelo: gemini-1.5-pro")
    model = genai.GenerativeModel("gemini-1.5-pro")
    response = model.generate_content("Diga 'ok'")
    print(f"✓ gemini-1.5-pro funciona! Resposta: {response.text[:50]}...")
except Exception as e:
    print(f"✗ Erro ao testar gemini-1.5-pro: {e}")

# Testa permissão de upload de arquivos
try:
    print(f"\n📤 Testando upload de arquivo...")
    test_file = genai.upload_file(
        path="docs/knowledge_base/finetuning_data_gemini_messages.jsonl",
        mime_type="application/json"
    )
    print(f"✓ Upload bem-sucedido! File ID: {test_file.name}")
    genai.delete_file(test_file.name)
    print(f"✓ Arquivo deletado")
except Exception as e:
    print(f"✗ Erro ao fazer upload: {e}")

print("\n" + "=" * 80)
