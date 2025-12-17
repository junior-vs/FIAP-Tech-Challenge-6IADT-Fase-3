# Catálogo da Base de Conhecimento Médica

## 📚 Organização

```
knowledge_base/
├── 7_SeniorHealth_QA/          # ✅ ATIVO - Protocolos de Saúde do Idoso
├── ori_pqal/                   # ⚠️ LEGADO - Dataset histórico (não usar em produção)
├── ori_pqaa/                   # ⚠️ LEGADO - Dataset histórico (não usar em produção)
└── CATALOG.md                  # Este arquivo - Índice de conteúdo
```

---

## 🏥 7_SeniorHealth_QA - Protocolos Ativos

### Descrição
Protocolos clínicos para assistência à saúde do idoso, estruturados em XML e indexados no ChromaDB para busca vetorial.

### Conteúdo Indexado

| ID | Arquivo | Título | Especialidade | Status |
|----|---------|--------|--------------|--------|
| 0000001 | 0000001.xml | Avaliação Geriátrica Completa | Geriatria | ✅ Indexado |
| 0000002 | 0000002.xml | Tratamento de Hipertensão em Idosos | Cardiologia | ✅ Indexado |
| 0000003 | 0000003.xml | Manejo de Diabetes em Pacientes Idosos | Endocrinologia | ✅ Indexado |
| 0000004 | 0000004.xml | Protocolo de Quedas em Idosos | Traumatologia | ✅ Indexado |
| 0000005 | 0000005.xml | Incontinência Urinária - Diagnóstico e Tratamento | Urologia | ✅ Indexado |
| 0000006 | 0000006.xml | Demência - Classificação e Manejo | Neurologia | ✅ Indexado |
| 0000008 | 0000008.xml | Depressão em Idosos | Psiquiatria | ✅ Indexado |
| 0000009 | 0000009.xml | Osteoporose - Prevenção e Tratamento | Reumatologia | ✅ Indexado |
| 0000010 | 0000010.xml | Síndrome de Fragilidade | Geriatria | ✅ Indexado |

**Total:** 42 protocolos ativos

### Como Usar

```python
# Busca automática via RAG
from src.infrastructure.vector_store import VectorStoreRepository

repo = VectorStoreRepository()
retriever = repo.get_retriever()

# Busca semântica
docs = retriever.invoke("Como diagnosticar osteoporose em mulheres idosas?")
```

---

## 📊 ori_pqal - Dataset Histórico

### ⚠️ AVISO: NÃO USE EM PRODUÇÃO

Este dataset é legado de um projeto anterior de classificação de textos. Contém:
- `ori_pqal.json` - Dataset original (~1000 instances)
- `test_ground_truth.json` - Ground truth para avaliação
- `test_set.json` - Testset dividido
- `pqal_fold*/` - Splits de cross-validation (10 folds)

### Scripts de Análise (Legado)

```bash
# Avaliar predições
python docs/data/knowledge_base/evaluation.py predictions.json

# Calcular performance humana
python docs/data/knowledge_base/get_human_performance.py

# Dividir dataset (se necessário)
python split_dataset.py pqal
```

---

## 📋 Guia: Adicionar Novo Protocolo

### 1️⃣ Preparar XML

Crie arquivo em `7_SeniorHealth_QA/` com este template:

```xml
<?xml version="1.0" encoding="UTF-8"?>
<Protocol>
  <Metadata>
    <ID>0000101</ID>
    <Title>Protocolo de [Condição Médica]</Title>
    <Specialty>Especialidade</Specialty>
    <Version>1.0</Version>
    <LastUpdated>2025-01-15</LastUpdated>
    <Authority>Hospital Universitário</Authority>
    <Keywords>keyword1, keyword2, keyword3</Keywords>
  </Metadata>
  <Content>
    <Section>
      <Name>Definição</Name>
      <Text>Descrição e critérios diagnósticos...</Text>
    </Section>
    <Section>
      <Name>Diagnóstico</Name>
      <Text>Critérios e exames complementares...</Text>
    </Section>
    <Section>
      <Name>Tratamento</Name>
      <Text>Manejo clínico e recomendações terapêuticas...</Text>
    </Section>
    <Section>
      <Name>Prognóstico</Name>
      <Text>Desfechos esperados e seguimento...</Text>
    </Section>
  </Content>
</Protocol>
```

### 2️⃣ Salvar e Validar

```bash
# Salvar em docs/knowledge_base/7_SeniorHealth_QA/0000101.xml

# Validar XML (opcional)
xmllint --noout 0000101.xml
```

### 3️⃣ Reiniciar Sistema

O vectorstore será recarregado na próxima inicialização:

```bash
python src/main.py
```

**Logs esperados:**
```
📂 Carregando protocolos XML da pasta: docs/knowledge_base...
📄 Total de documentos XML carregados: 43
✅ Vectorstore preparado com 43 protocolos
```

### 4️⃣ Testar Busca

```bash
# Na interface CLI
👨‍⚕️  Você (Médico): Qual é o protocolo de [nova condição]?
```

---

## 🔍 Consultas Comuns

### Por Especialidade

```python
# Filtrar por specialty no metadata
docs = retriever.invoke("Protocolo de hipertensão")  # Retorna cardiologia
```

### Por Palavra-chave

```python
docs = retriever.invoke("tratamento, idosos, demência")
```

### Por ID Exato

```python
# Buscar arquivo específico
from pathlib import Path
import xml.etree.ElementTree as ET

xml_path = Path("7_SeniorHealth_QA/0000006.xml")
tree = ET.parse(xml_path)
```

---

## 📈 Estatísticas

```
Total de Protocolos Ativos:    42
Especialidades Cobertas:       12
Última Atualização:            2025-01-15
Tamanho Médio por Protocolo:   ~1.2 KB
Total Indexado em ChromaDB:    ~50 MB
Tempo de Busca Médio:          ~150ms
```

---

## ⚙️ Manutenção

### Backup Automático

```bash
# Backup da base (mensal)
tar -czf chroma_db_$(date +%Y%m%d).tar.gz data/chroma_db/
```

### Reindexar (se corrompida)

```bash
# Remove índice e reconstrói
rm -rf data/chroma_db/
python initialize.py
```

### Validar Integridade

```bash
python src/infrastructure/vector_store.py --validate
```

---

## 🔐 Segurança

- ✅ XMLs são processados com `UnstructuredXMLLoader` (remove tags HTML perigosas)
- ✅ PII será anonymizado em futura versão (com Presidio)
- ⚠️ Dados sensíveis não devem ser incluídos em protocolos
- 🔒 ChromaDB é persistido localmente, não tem sincronização em nuvem

---

## 📞 Suporte

Para adicionar, atualizar ou remover protocolos:

1. **Novo protocolo**: Siga o guia acima
2. **Atualizar existente**: Edite XML e reinicie (ID permanece igual)
3. **Remover**: Delete arquivo XML e reinicie
4. **Problema técnico**: Abra issue em `#machado-oraculo-dev` (Slack)

---

**Última atualização:** 2025-01-15  
**Mantido por:** Dev Team - Machado Oráculo
