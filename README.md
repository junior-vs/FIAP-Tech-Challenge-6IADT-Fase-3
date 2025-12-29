# Assistente Médico Virtual

Um tutorial completo para construir um assistente de IA médica usando LangChain, LangGraph e Google Gemini. Este projeto demonstra como criar um sistema de suporte a decisões clínicas com validações de segurança e busca baseada em vetores.

## 📋 Visão Geral do Projeto

### O que este projeto faz:

Este assistente médico virtual fornece uma interface de chat interativa onde profissionais de saúde podem fazer perguntas médicas e receber respostas baseadas em evidências, extraídas de protocolos médicos validados. O sistema garante segurança e precisão através de múltiplas camadas de validação.

### Por que é útil para aprender:

- **Pipeline RAG completo**: Implementação prática de Retrieval-Augmented Generation
- **Validações de segurança**: Sistema robusto de guardrails para aplicações médicas
- **Arquitetura modular**: Código bem estruturado seguindo boas práticas do Python
- **Gerenciamento de estado**: Uso do LangGraph para fluxos de trabalho complexos
- **Busca semântica**: Integração com base vetorial (ChromaDB) para recuperação inteligente

### Tecnologias principais:

- **LangChain**: Orquestração de fluxos de trabalho de IA
- **LangGraph**: Máquinas de estado para decisões complexas  
- **Google Gemini**: Modelo de linguagem para geração e compreensão
- **ChromaDB**: Armazenamento e busca vetorial de documentos médicos
- **Python 3.8+**: Linguagem principal com práticas modernas

## 🚀 Instruções de Configuração

### Pré-requisitos

Antes de começar, certifique-se de ter instalado:

1. **Python 3.8+** em seu sistema
2. **Chave da API Google Gemini** - Obtenha uma em [Google AI Studio](https://makersuite.google.com/app/apikey)
3. **Git** para controle de versão

### Passos de Instalação

1. **Clone o repositório:**
```bash
git clone [URL-DO-REPOSITORIO]
cd FIAP-Tech-Challenge-6IADT-Fase-3
```

2. **Crie um ambiente virtual:**
```bash
python -m venv .venv

# No Linux/Mac:
source .venv/bin/activate

# No Windows:
.venv\Scripts\activate
```

3. **Instale as dependências:**
```bash
pip install -r requirements.txt
```

4. **Configure as variáveis de ambiente:**

Crie um arquivo `.env` na raiz do projeto:
```bash
# Sua chave da API Google Gemini (obrigatório)
GEMINI_API_KEY=sua_chave_aqui

# Modelo a ser usado (opcional - padrão: gemini-1.5-flash)
MODEL_NAME=gemini-1.5-flash

# Configurações de logging (opcional)
LOG_LEVEL=INFO
```

5. **Execute o assistente:**
```bash
python src/main.py
```

## 🔍 Como Funciona

### Arquitetura do Sistema

O assistente funciona através de um pipeline de 5 etapas implementado como um grafo de estados:

```
Pergunta do Usuário
        ↓
1. 🛡️  Guardrails (Validação de Segurança)
        ↓
2. 📚 Retrieve (Busca de Documentos)
        ↓
3. ⭐ Grade (Classificação de Relevância)
        ↓
4. 🤖 Generate (Geração de Resposta)
        ↓
5. ✅ Validate (Validação Anti-Alucinação)
        ↓
   Resposta Final
```

### Componentes Principais

#### 1. **Estado do Agente (`AgentState`)**
```python
def create_initial_agent_state(user_question: str) -> AgentState:
    """
    Cria o estado inicial que flui através de todo o pipeline.
    Este estado atua como memória compartilhada entre as etapas.
    """
    return {
        "medical_question": user_question,    # Pergunta original
        "is_safe": True,                      # Flag de segurança
        "documents": [],                      # Protocolos encontrados
        "generation": "",                     # Resposta da IA
        # ... outros campos
    }
```

#### 2. **Validação de Segurança (Guardrails)**
- Verifica se a pergunta é medicamente relevante
- Detecta informações pessoais identificáveis (PII)
- Avalia o nível de risco da pergunta
- Rejeita perguntas fora do escopo médico

#### 3. **Busca Semântica**
```python
# O sistema converte a pergunta em vetores e busca documentos similares
documents = retriever.invoke(user_question)
# Usa embeddings do Google para encontrar protocolos relevantes
```

#### 4. **Geração de Resposta**
- Usa Google Gemini com temperatura 0.0 (determinística)
- Baseia respostas apenas nos documentos recuperados
- Inclui citações dos protocolos consultados

#### 5. **Validação Anti-Alucinação**
- Compara a resposta gerada com os documentos fonte
- Rejeita respostas que não são suportadas pelos protocolos
- Garante precisão factual das informações

### Fluxo de Processamento Detalhado

#### **Função `process_medical_question()`**
```python
def process_medical_question(app, user_question: str) -> Dict[str, Any]:
    """
    Etapas do Pipeline:
    1. Guardrails: Verifica segurança e relevância médica
    2. Recuperação: Encontra documentos relevantes
    3. Classificação: Filtra por relevância
    4. Geração: Cria resposta baseada nos documentos
    5. Validação: Verifica precisão contra fontes
    """
    initial_state = create_initial_agent_state(user_question)
    final_state = app.invoke(initial_state)  # Executa todo o pipeline
    return final_state
```

#### **Gerenciamento da Interface (`handle_chat_loop()`)**
```python
def handle_chat_loop(app) -> None:
    """
    Loop principal de interação:
    - Obtém entrada do usuário
    - Processa através do pipeline de IA
    - Exibe resposta e fontes consultadas
    - Trata erros graciosamente
    """
    while True:
        user_input = get_user_input()
        if should_exit_application(user_input):
            break
        
        result = process_medical_question(app, user_input)
        display_response(result)
        display_sources(result)
```

## 💻 Exemplo de Uso

### Executando o Assistente

```bash
# Active o ambiente virtual
source .venv/bin/activate

# Execute o programa principal
python src/main.py
```

### Sessão de Exemplo

```
======================================================================
🏥  Assistente Médico Virtual
======================================================================

📋 Sistema de suporte a decisões clínicas baseado em protocolos internos
Desenvolvido com LangChain + LangGraph + Google Gemini

----------------------------------------------------------------------
💬 Digite suas dúvidas clínicas (ou 'sair' para encerrar)

👨‍⚕️  Você: Quais são as indicações para prescrição de antibióticos em infecções respiratórias?

🔍 Processando pergunta...

🤖 Assistente: Com base nos protocolos consultados, os antibióticos são indicados 
em infecções respiratórias nas seguintes situações:

1. **Pneumonia bacteriana confirmada**: Presença de infiltrado pulmonar em 
   radiografia de tórax associado a sintomas como febre, tosse produtiva e 
   leucocitose.

2. **Sinusite bacteriana aguda**: Quando há sintomas persistentes por mais de 
   10 dias ou piora após melhora inicial.

3. **Faringite estreptocócica**: Confirmada por teste rápido ou cultura positiva
   para Streptococcus pyogenes.

É importante evitar o uso em infecções virais, que representam a maioria dos 
casos de infecções respiratórias superiores.

📚 Protocolos consultados:
   • protocolo_antibioticos_respiratorios.xml
   • diretrizes_pneumonia_ambulatorial.xml
   • manual_prescricao_racional.xml

👨‍⚕️  Você: sair

👋 Encerrando assistente médico. Até logo!
```

## 🎯 Saída de Exemplo

### Resposta Bem-Sucedida
```
🤖 Assistente: [Resposta médica baseada em evidências]

📚 Protocolos consultados:
   • protocolo_cardiologia_2023.xml
   • diretrizes_hipertensao.xml
   ... e mais 2 protocolos
```

### Pergunta Rejeitada por Segurança
```
⚠️  Assistente: Pergunta fora do escopo médico.
```

### Falha na Validação
```
⚠️  Assistente: Não foi possível gerar uma resposta confiável.
```

## 📁 Estrutura do Projeto

```
src/
├── main.py                 # 🎯 Script principal - Interface de chat
├── config.py              # ⚙️  Configurações e variáveis de ambiente
├── domain/
│   ├── state.py           # 📊 Definição do estado do agente
│   └── guardrails.py      # 🛡️  Modelos de validação Pydantic
├── use_cases/
│   ├── graph.py          # 🔧 Construção do grafo LangGraph
│   └── nodes.py          # 🔗 Implementação dos nós de processamento
├── infrastructure/
│   ├── llm_factory.py    # 🤖 Factory para modelos de linguagem
│   └── vector_store.py   # 📚 Repositório da base vetorial
└── utils/
    └── logging.py        # 📝 Configuração de logs

docs/knowledge_base/       # 📖 Protocolos médicos em XML
data/chroma_db/           # 💾 Base de dados vetorial persistente
```

## 🎓 Conceitos Aprendidos

### 1. **RAG (Retrieval-Augmented Generation)**
- Como combinar busca semântica com geração de linguagem
- Implementação de pipeline de recuperação de documentos
- Integração de embeddings e LLMs

### 2. **LangGraph para Fluxos Complexos**
- Criação de máquinas de estado com múltiplos nós
- Gerenciamento de estado compartilhado entre etapas
- Roteamento condicional baseado em resultados

### 3. **Validações de Segurança em IA**
- Implementação de guardrails para aplicações críticas
- Detecção e prevenção de alucinações
- Validação de entrada e saída

### 4. **Busca Vetorial**
- Uso do ChromaDB para armazenamento persistente
- Conversão de texto em embeddings
- Busca por similaridade semântica

### 5. **Boas Práticas de Python**
- Estrutura modular e separação de responsabilidades
- Type hints e documentação clara
- Tratamento robusto de erros
- Configuração através de variáveis de ambiente

## 🚀 Próximos Passos

Para expandir este projeto, considere:

1. **Interface Web**: Criar uma interface React ou Streamlit
2. **Mais Validadores**: Adicionar validações específicas por especialidade
3. **Cache Inteligente**: Implementar cache de respostas frequentes
4. **Métricas**: Adicionar monitoramento de qualidade das respostas
5. **API REST**: Transformar em serviço web com FastAPI

## 🤝 Contribuições

Este é um projeto educacional. Sinta-se à vontade para:
- Fazer fork e experimentar
- Sugerir melhorias
- Reportar issues
- Compartilhar casos de uso interessantes

---

**⚠️ Aviso Importante**: Este assistente é apenas para fins educacionais e de pesquisa. Não deve ser usado para diagnósticos reais sem supervisão médica apropriada.