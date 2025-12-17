# 🔧 Configuração do Projeto

## 📋 Visão Geral

Este projeto segue o padrão **Single Source of Truth (SSOT)** para configuração:

- **`.env`** = Valores em runtime (ambiente-específico)
- **`src/config.py`** = Schema e tipos (definição de contrato)

### Diagrama de Hierarquia

```
┌─────────────────────────────────┐
│ .env (Runtime Values)           │
│ - GEMINI_API_KEY                │
│ - MODEL_NAME                    │
│ - TEMPERATURE                   │
└──────────┬──────────────────────┘
           │
           ↓ Pydantic BaseSettings
┌──────────────────────────────────┐
│ src/config.py (Schema)           │
│ class Settings:                  │
│   - gemini_api_key: str          │
│   - model_name: str              │
│   - temperature: float           │
│   ✅ NO DEFAULTS (Required)      │
└──────────┬──────────────────────┘
           │
           ↓ Instantiation
┌──────────────────────────────────┐
│ settings = Settings()            │
│ settings.model_name              │
│ settings.gemini_api_key          │
└──────────────────────────────────┘
```

---

## ✅ Variáveis Obrigatórias

Todas as seguintes variáveis **DEVEM** estar presentes no `.env`:

| Variável | Tipo | Exemplo | Descrição |
|----------|------|---------|-----------|
| `GEMINI_API_KEY` | string | `AIzaSyA...` | Chave da API Google Gemini |
| `MODEL_NAME` | string | `gemini-2.0-flash` | Modelo Generativo a usar |
| `TEMPERATURE` | float | `0.0` | Determinismo (0=deterministico, 1=criativo) |

### ❌ O que NÃO deve estar em `.env`

Campos legados que causavam erro:
- ~~`FAISS_INDEX_PATH`~~ (substituído por Chroma)
- ~~`STORAGE_PATH`~~ (legado)
- ~~`BOOK_URL`~~ (legado - Machado de Assis)

---

## 🚀 Configuração Rápida

### 1. Criar `.env` Local

```bash
# Copiar template
cp .env.example .env  # Se existir

# Ou editar manualmente
cat > .env << 'EOF'
GEMINI_API_KEY=AIzaSyA...YOUR_KEY_HERE...
MODEL_NAME=gemini-2.0-flash
TEMPERATURE=0.0
EOF
```

### 2. Validar Configuração

```bash
# Teste rápido
.venv/bin/python -c "from src.config import settings; print(f'✅ {settings.model_name}')"

# Output esperado:
# ✅ gemini-2.0-flash
```

### 3. Rodar Sistema

```bash
# Initialize (cria vector store)
.venv/bin/python initialize.py

# Run CLI
.venv/bin/python src/main.py
```

---

## 🔄 Fluxo de Carregamento

```python
# 1. Pydantic lê .env
# (via BaseSettings com env_file=".env")

# 2. Mapeia variáveis (case-insensitive)
MODEL_NAME       →  model_name
GEMINI_API_KEY   →  gemini_api_key

# 3. Valida tipos
model_name: str  # "gemini-2.0-flash" ✅

# 4. Cria instance singleton
settings = Settings()

# 5. Acessar em código
from src.config import settings
print(settings.model_name)  # "gemini-2.0-flash"
```

---

## 💡 Por Ambiente

### 🔸 Development (dev)

```env
# .env.local (nunca commitar!)
GEMINI_API_KEY=AIzaSyA...DEV...
MODEL_NAME=gemini-1.5-flash        # Rápido, barato
TEMPERATURE=0.0
```

**Vantagem:** Iteração rápida, baixo custo

### 🔹 Staging

```env
# .env.staging
GEMINI_API_KEY=AIzaSyA...STAGING...
MODEL_NAME=gemini-1.5-pro          # Mais preciso
TEMPERATURE=0.0
```

**Vantagem:** Qualidade próxima de produção

### 🔴 Production

```env
# .env.prod (CI/CD secrets)
GEMINI_API_KEY=AIzaSyA...PROD...
MODEL_NAME=gemini-2.0-flash        # Balance: rápido + preciso
TEMPERATURE=0.0
```

**Vantagem:** Melhor relação custo/qualidade

---

## ❓ FAQ

### P: Posso usar variáveis de ambiente do SO?

**R:** Sim! Pydantic lê do SO se .env não existir.

```bash
export MODEL_NAME=gemini-2.0-flash
.venv/bin/python src/main.py  # ✅ Funciona
```

### P: Posso ter defaults em config.py?

**R:** Não recomendado. Força documentar valores no .env.

**Antes (❌ Ruim):**
```python
model_name: str = "gemini-2.0-flash"  # Default escondido
```

**Depois (✅ Bom):**
```python
model_name: str  # Obrigatório, deve estar em .env
```

### P: Como mudar modelo sem redeploy?

**R:** Apenas atualize `.env` e reinicie:

```bash
# Produção: trocar .env
sed -i 's/gemini-2.0-flash/gemini-1.5-pro/g' .env

# Reiniciar app
systemctl restart machado-oraculo
# ou
docker restart oraculo-container
```

---

## 🔐 Segurança

### ✅ Fazer

- Adicionar `.env` ao `.gitignore`
- Usar CI/CD secrets para produção
- Rotacionar `GEMINI_API_KEY` regularmente
- Usar `extra="forbid"` em Settings (rejeita campos desconhecidos)

### ❌ Não Fazer

- Commitar `.env` no Git
- Deixar `GEMINI_API_KEY` em logging
- Hardcodar valores em código
- Deixar `.env` sem controle de acesso (chmod 600)

---

## 📚 Referências

- [Pydantic Settings](https://docs.pydantic.dev/latest/concepts/pydantic_settings/)
- [12-Factor App - Config](https://12factor.net/config)
- [Google Cloud - Managing API Keys](https://cloud.google.com/docs/authentication/api-keys)

