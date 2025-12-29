# 🏥  Assistente Médico Virtual

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
```env
GEMINI_API_KEY=your_gemini_api_key_here
MODEL_NAME=gemini-2.0-flash
TEMPERATURE=0.0
CHUNK_SIZE=1000
```

 5. **Initialize the knowledge base:**
 The system will automatically load medical protocols from `docs/knowledge_base/7_SeniorHealth_QA/` on first run.
 
 ### Executando a Aplicação
 
 ```bash
 python src/main.py
 ```
 
 ## 🔧 How It Works
The Medical AI Assistant follows a sophisticated RAG (Retrieval-Augmented Generation) workflow designed specifically for medical applications:

### 1. **Input Processing & Safety Validation**
```python
def create_initial_agent_state(user_question: str) -> AgentState:
    """Creates the data structure that flows through the entire pipeline"""
```
- **User Input**: Healthcare professional types a medical question
- **Guardrails Check**: System validates the question is medical-relevant and safe
- **Language Detection**: Automatically handles multiple languages

### 2. **Knowledge Retrieval**
```python
def process_medical_question(app, user_question: str) -> Dict[str, Any]:
    """Processes question through the complete AI pipeline"""
```
- **Vector Search**: Converts question to embeddings and searches medical protocol database
- **Semantic Matching**: Finds relevant documents based on meaning, not just keywords
- **Document Filtering**: Grades retrieved documents for relevance

### 3. **Response Generation**
- **Context Assembly**: Combines relevant medical protocols as context
- **AI Generation**: Google Gemini generates evidence-based response
- **Source Citation**: Automatically includes protocol references

### 4. **Validation & Safety**
```python
def display_response(result: Dict[str, Any]) -> None:
    """Displays response with appropriate safety checks"""
```
- **Hallucination Detection**: Verifies response is grounded in source documents
- **Safety Verification**: Ensures response meets medical safety standards
- **Quality Control**: Rejects responses that don't meet validation criteria

### 5. **User Interface**
```python
def handle_chat_loop(app) -> None:
    """Manages the interactive conversation with healthcare professional"""
```
- **Professional Interface**: Clean, medical-themed command-line interface
- **Source Transparency**: Shows which protocols were consulted
- **Error Handling**: Graceful handling of technical issues

## 💡 Example Usage

### Starting the Application
```bash
$ python src/main.py

======================================================================
🏥 Assistente Médico Virtual
======================================================================

📋 Sistema de suporte a decisões clínicas baseado em protocolos internos
Desenvolvido com LangChain + LangGraph + Google Gemini

----------------------------------------------------------------------
💬 Digite suas dúvidas clínicas (ou 'sair' para encerrar)

👨‍⚕️  Você: 
```

### Sample Conversation
```
👨‍⚕️  Você: What are the treatments for diabetic retinopathy?

🔍 Processando pergunta...

🤖 Assistente: Based on the medical protocols, treatments for diabetic retinopathy include:

1. **Anti-VEGF medications**: Injected directly into the eye to reduce fluid leakage and interfere with new blood vessel growth in the retina.

2. **Laser therapy**: 
   - Focal laser treatment for macular edema
   - Scatter laser treatment for proliferative retinopathy
   - Usually completed in one session with possible additional treatments

3. **Combined approach**: Anti-VEGF injections with laser therapy often provide better results than laser treatment alone.

The treatment choice depends on the stage and type of diabetic retinopathy. Regular dilated eye exams are essential for monitoring progression.

📚 Protocolos consultados:
   • 0000016.xml
   • 0000015.xml

👨‍⚕️  Você: Quais são os fatores de risco para diabetes?

🔍 Processando pergunta...

🤖 Assistente: Os principais fatores de risco para diabetes tipo 2 incluem:

**Fatores não modificáveis:**
- Idade acima de 45 anos
- Histórico familiar de diabetes
- Etnia (maior risco em afro-americanos, hispânicos, nativos americanos)

**Fatores modificáveis:**
- Sobrepeso e obesidade (IMC ≥ 25)
- Sedentarismo e falta de atividade física regular
- Hipertensão arterial (≥ 140/90 mmHg)
- HDL baixo (< 35 mg/dL) ou triglicerídeos altos (≥ 250 mg/dL)
- Histórico de diabetes gestacional
- Síndrome dos ovários policísticos

A prevenção foca na modificação dos fatores de risco controláveis através de mudanças no estilo de vida.

📚 Protocolos consultados:
   • 0000015.xml

👨‍⚕️  Você: sair

👋 Encerrando assistente médico. Até logo!
```

## 📊 Sample Output Features

### Safety Features
- ✅ **Input Validation**: Rejects non-medical questions
- ⚠️ **Hallucination Detection**: Prevents AI-generated misinformation
- 🛡️ **PII Protection**: Blocks personal information in questions
- 📋 **Source Citation**: Always shows medical protocol sources

### Response Quality
- **Evidence-based**: All responses grounded in medical protocols
- **Professional tone**: Appropriate for healthcare settings
- **Structured format**: Clear, organized information presentation
- **Multilingual**: Responds in the same language as the question

### Technical Reliability
- **Error Handling**: Graceful degradation when issues occur
- **Logging**: Comprehensive logging for debugging and monitoring
- **Performance**: Optimized for quick response times
- **Scalability**: Designed to handle multiple concurrent users

## 🏗️ Project Architecture

```
src/
├── main.py                 # Main application entry point (refactored)
├── domain/
│   ├── state.py           # Data structures for the AI pipeline
│   └── guardrails.py      # Safety validation models
├── use_cases/
│   ├── nodes.py           # RAG processing nodes
│   └── graph.py           # LangGraph workflow orchestration
├── infrastructure/
│   ├── llm_factory.py     # Language model initialization
│   └── vector_store.py    # ChromaDB vector database
└── utils/
    └── logging.py         # Logging configuration

docs/knowledge_base/       # Medical protocol documents
data/chroma_db/           # Vector database storage
logs/                     # Application logs
```

## 🔍 Code Quality Features

### Python Best Practices Applied
- **PEP 8 Compliance**: Consistent coding style throughout
- **Type Hints**: Full type annotation for better code clarity
- **Docstrings**: Comprehensive documentation for all functions
- **Error Handling**: Robust exception handling with user-friendly messages
- **Separation of Concerns**: Clean architecture with distinct responsibilities

### Beginner-Friendly Design
- **Instructional Comments**: Explains not just what, but why
- **Modular Functions**: Small, focused functions with single responsibilities
- **Clear Variable Names**: Self-documenting code with descriptive naming
- **Consistent Patterns**: Repeatable patterns for similar operations
- **Safety First**: Defensive programming practices throughout

## 🤝 Contributing

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/amazing-feature`)
3. Commit your changes (`git commit -m 'Add amazing feature'`)
4. Push to the branch (`git push origin feature/amazing-feature`)
5. Open a Pull Request

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## ⚠️ Medical Disclaimer

This AI assistant is designed for healthcare professional decision support only. It should never be used for direct patient diagnosis or treatment decisions without proper medical oversight. Always consult with qualified healthcare providers for medical decisions.

---

**Built with ❤️ for healthcare professionals**

*Combining cutting-edge AI with medical expertise to support better patient outcomes.*
