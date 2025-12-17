# Machado Oráculo - Assistente Virtual Médico

## 🏥 Visão Geral

**Machado Oráculo** é um assistente virtual médico inteligente, baseado em **Retrieval-Augmented Generation (RAG)**, treinado com protocolos internos do hospital. O sistema auxilia profissionais de saúde com:

- 💊 **Sugestões de condutas clínicas** baseadas em protocolos validados
- 🔍 **Respostas a dúvidas médicas** com citação de fontes
- 📋 **Procedimentos recomendados** com base nos protocolos internos
- 🛡️ **Validação de segurança** contra alucinações e dados sensíveis

## 🎯 Características Principais

- ✅ **LLM Determinístico**: Google Gemini 1.5 Flash com temperatura 0.0 para respostas consistentes
- 📚 **RAG Baseado em XMLs**: Protocolos estruturados em `docs/knowledge_base/`
- 🔐 **Guardrails de Segurança**: Validação de PII e classificação de risco clínico
- 🛡️ **Detecção de Alucinações**: Verifica se a resposta está fundamentada nos protocolos
- 💾 **Vetorização com ChromaDB**: Persistência automática de embeddings
- 🔄 **Workflow Inteligente**: Grafo LangGraph de 5 nós com roteamento condicional

## 🚀 Quick Start

### Instalação

```bash
# Clonar repositório
git clone https://github.com/seu-hospital/machado-oraculo.git
cd machado-oraculo

# Instalar dependências
uv sync

# Configurar variáveis de ambiente
cp .env.example .env
# Editar .env e adicionar GEMINI_API_KEY
```

### Inicializar Sistema

```bash
# Preparar vectorstore com protocolos
python initialize.py

# Iniciar assistente
python src/main.py
```

### Uso

```bash
👨‍⚕️  Você (Médico): Qual é o protocolo de tratamento para sepse em idosos?

🤖 Assistente: [Resposta com citação de protocolos]

💡 Digite 'sair' para encerrar ou 'limpar' para reiniciar histórico.
```

## 📁 Estrutura do Projeto

```
machado-oraculo/
├── src/
│   ├── domain/
│   │   ├── state.py                # AgentState para fluxo de dados
│   │   └── guardrails_check.py     # Validação e modelos Pydantic
│   ├── infrastructure/
│   │   ├── llm_factory.py          # Factory pattern para LLM/Embeddings
│   │   ├── vector_store.py         # Gerencimento de ChromaDB
│   │   └── preprocess/             # Scripts de preparação (legado)
│   ├── use_cases/
│   │   ├── graph.py                # Orquestração LangGraph
│   │   └── nodes.py                # Implementação dos 5 nós
│   ├── utils/
│   │   └── logging.py              # Configuração de logs
│   ├── config.py                   # Gerenciamento de configurações
│   └── main.py                     # Entry point CLI
├── docs/
│   ├── data/knowledge_base/
│   │   ├── 7_SeniorHealth_QA/      # Protocolos médicos (XMLs)
│   │   ├── ori_pqal/               # Dataset legado
│   │   └── CATALOG.md              # Índice da base de conhecimento
│   └── spec/                       # Especificações técnicas
├── tests/
│   └── unit/                       # Testes unitários
├── vectorstore/
│   └── chroma_db/                  # Persistência de embeddings
├── initialize.py                   # Setup inicial
├── pyproject.toml                  # Dependências
└── README.md                       # Este arquivo
```

## 🔧 Arquitetura

### Fluxo de Execução (5 Nós)

```
1. GUARDRAILS
   ↓ Validação de PII, relevância médica, risco
   ├─ Seguro? → Continua
   └─ Inseguro? → Encerra com recusa

2. RETRIEVE
   ↓ Busca vetorial em ChromaDB
   └─ Retorna documentos relevantes

3. GRADE
   ↓ Classifica relevância dos documentos
   └─ Filtra documentos inúteis

4. GENERATE
   ↓ LLM gera resposta com contexto
   └─ Inclui citações de protocolos

5. VALIDATE
   ↓ Detecção de alucinações
   ├─ Válido? → Retorna resposta
   └─ Alucinação? → Rejeita e avisa
```

### Estado de Fluxo (AgentState)

```python
{
    "medical_question": str,        # Pergunta do médico
    "context_data": Optional[str],  # Info anonimizada do paciente
    "documents": List[str],         # Protocolos recuperados
    "generation": str,              # Resposta gerada
    "is_safe": bool,                # Flag de segurança
    "risk_level": str,              # "informativo" | "emergencia" | ...
}
```

## 📚 Adicionando Protocolos Médicos

### 1. Preparar Protocolo XML

Salvar em `docs/knowledge_base/7_SeniorHealth_QA/`

```xml
<?xml version="1.0" encoding="UTF-8"?>
<Protocol>
  <Metadata>
    <ID>0000101</ID>
    <Title>Protocolo de Tratamento de Sepse em Idosos</Title>
    <Version>1.0</Version>
    <LastUpdated>2025-01-15</LastUpdated>
    <Authority>Hospital Universitário</Authority>
  </Metadata>
  <Content>
    <Section>
      <Name>Diagnóstico</Name>
      <Text>Critérios de qSOFA: ...</Text>
    </Section>
  </Content>
</Protocol>
```

### 2. Reiniciar Sistema

O `VectorStoreRepository` recarrega automaticamente na próxima execução:

```bash
# Vectorstore é invalidado, novos embeddings criados
python src/main.py
```

## 🧪 Testes

```bash
# Executar testes unitários
pytest tests/unit/ -v

# Com cobertura
pytest tests/unit/ --cov=src --cov-report=html
```

## 🔐 Segurança

- ✅ **Zero Temperature**: LLM determinístico, sem alucinações por aleatoriedade
- ✅ **Guardrails**: Validação de input contra PII (CPF, email, etc.)
- ✅ **Alucinação Check**: Verifica se resposta está nos protocolos
- ✅ **HTTPS Only**: Todas as chamadas externas usam HTTPS
- ⚠️ **Dados Sensíveis**: Implementar anonymizer com Presidio (TODO)

## 📊 Monitoramento

### Logs Estruturados

```bash
# Logs em tempo real
tail -f logs/machado.log

# Filtrar por nível
grep "ERROR\|WARNING" logs/machado.log
```

### Métricas

```python
# Adicionar tracking em produção
from opentelemetry import metrics
provider = MeterProvider()
meter = provider.get_meter(__name__)
```

## 🚀 Deployment

### Docker

```dockerfile
FROM python:3.10-slim
WORKDIR /app
COPY pyproject.toml uv.lock ./
RUN pip install uv && uv sync --no-dev
COPY . .
CMD ["python", "src/main.py"]
```

```bash
docker build -t machado-oraculo .
docker run -e GEMINI_API_KEY=xxx machado-oraculo
```

### Variáveis de Ambiente

| Variável | Descrição | Padrão |
|----------|-----------|--------|
| `GEMINI_API_KEY` | Chave da API Google Gemini | **Obrigatória** |
| `MODEL_NAME` | Modelo Gemini a usar | `gemini-1.5-flash` |
| `TEMPERATURE` | Temperatura do LLM | `0.0` |
| `DOCS_PATH` | Caminho para protocolos | `docs/knowledge_base` |
| `VECTOR_DB_PATH` | Caminho para ChromaDB | `data/chroma_db` |

## 📝 Changelog

### v0.1.0 (2025-01-15)
- ✅ Implementação de RAG com 5 nós
- ✅ Validação de segurança (guardrails)
- ✅ Detecção de alucinações
- ✅ CLI interativa com histórico
- ⚠️ Anonymizer (em desenvolvimento)
- ⚠️ Cache Redis (em desenvolvimento)

## 🤝 Contribuindo

1. Fork o repositório
2. Crie feature branch (`git checkout -b feature/MinhaFeature`)
3. Commit mudanças (`git commit -am 'Adiciona MinhaFeature'`)
4. Push para branch (`git push origin feature/MinhaFeature`)
5. Abra Pull Request

## 📄 Licença

Proprietary - Hospital Universitário

## 📞 Suporte

- 📧 Email: dev-team@hospital.com
- 💬 Slack: #machado-oraculo-dev
- 📚 Wiki: [Documentação Interna](https://wiki.hospital.com/machado-oraculo)

---

**Desenvolvido com ❤️ para auxiliar profissionais de saúde**
