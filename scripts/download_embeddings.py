#!/usr/bin/env python3
"""
Script para baixar o modelo de embeddings necessário.
Evita problemas de cache corrompido ou ausente.
"""
import os
import sys
from pathlib import Path

# Desabilitar symlinks warnings no Windows
os.environ["HF_HUB_DISABLE_SYMLINKS_WARNING"] = "1"

try:
    from sentence_transformers import SentenceTransformer
except ImportError:
    print("❌ sentence-transformers não instalado.")
    print("   Execute: pip install sentence-transformers")
    sys.exit(1)

def main():
    print(f"📥 Baixando modelo de embeddings...")
    print(f"   Modelo: sentence-transformers/all-MiniLM-L6-v2")

    try:
        # Usar SentenceTransformer diretamente (mais robusto)
        model = SentenceTransformer(
            "all-MiniLM-L6-v2",
            device="cpu"
        )
        print("✅ Modelo de embeddings baixado com sucesso!")
        
        # Test embedding
        test_embed = model.encode("teste")
        print(f"✅ Teste de embedding OK (dimensão: {len(test_embed)})")
    except Exception as e:
        print(f"❌ Erro ao baixar embeddings: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)

if __name__ == "__main__":
    main()


