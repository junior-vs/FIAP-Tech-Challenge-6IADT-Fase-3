# Contribuindo para Machado Oráculo

Obrigado por considerar contribuir para o **Machado Oráculo**! Este documento fornece diretrizes e informações sobre como contribuir.

## 📋 Índice

- [Código de Conduta](#código-de-conduta)
- [Como Reportar Bugs](#como-reportar-bugs)
- [Sugestões de Melhorias](#sugestões-de-melhorias)
- [Pull Requests](#pull-requests)
- [Guia de Estilo](#guia-de-estilo)
- [Processo de Desenvolvimento](#processo-de-desenvolvimento)

---

## Código de Conduta

Este projeto adere a um Código de Conduta baseado em respeito e inclusão. Esperamos que todos os contribuidores:

- Sejam respeitosos e colaborativos
- Considerem diferentes perspectivas
- Reportem comportamentos inadequados

---

## Como Reportar Bugs

### Antes de Submeter um Bug

1. **Verifique se já foi reportado**: Procure em Issues abertas/fechadas
2. **Teste com a versão mais recente**: Seu bug pode já estar corrigido
3. **Isole o problema**: Forneça passos reprodutíveis mínimos

### Como Submeter um Bug

Quando reportar um bug, inclua:

```markdown
**Descrição do Bug**
Explicação clara e concisa do problema.

**Passos para Reproduzir**
1. ...
2. ...
3. ...

**Comportamento Esperado**
O que deveria acontecer.

**Comportamento Real**
O que na verdade aconteceu.

**Ambiente**
- OS: [ex: Linux, macOS, Windows]
- Python: [ex: 3.10, 3.11]
- Versão do Projeto: [ex: 0.1.0]

**Logs/Screenshots**
Se aplicável, inclua logs ou capturas de tela.
```

---

## Sugestões de Melhorias

### Antes de Submeter

- Verifique se a sugestão já existe
- Considere se é aplicável ao projeto
- Forneça exemplos de casos de uso

### Como Submeter

```markdown
**Descrição da Melhoria**
Explicação clara da funcionalidade proposta.

**Motivação**
Por que isso seria útil?

**Exemplos**
Como seria usada?

**Implementação**
Ideias sobre como implementar (opcional).
```

---

## Pull Requests

### Antes de Começar

1. Fork o repositório
2. Crie um branch descritivo: `git checkout -b feature/seu-recurso`
3. Faça commits atômicos com mensagens claras
4. Escreva ou atualize testes
5. Verifique cobertura: `pytest --cov=src`

### Processo

1. **Desenvolva** sua feature ou fix
2. **Teste** localmente:
   ```bash
   uv sync
   pytest tests/ -v
   ```
3. **Formato**: Garanta aderência ao style guide
4. **Mensagem de Commit**: Use padrão convencional
   ```
   feat(guardrails): melhorar detecção de PII
   fix(llm): retry em timeout
   docs(readme): atualizar instruções
   ```
5. **Push** seu branch
6. **Abra Pull Request** com descrição completa

### Template de PR

```markdown
## Descrição
Breve descrição das mudanças.

## Tipo de Mudança
- [ ] Bug fix
- [ ] Nova feature
- [ ] Documentação
- [ ] Refatoração

## Checklist
- [ ] Código segue style guide
- [ ] Testes adicionados/atualizados
- [ ] Cobertura >= 50%
- [ ] Documentação atualizada
- [ ] Sem quebra de compatibilidade

## Issues Relacionadas
Fixes #123
Relates to #456
```

---

## Guia de Estilo

### Python

Seguimos PEP 8 com algumas extensões:

```python
# ✅ BOM
def process_medical_question(question: str) -> dict:
    """Processa pergunta médica com validação.
    
    Args:
        question: Pergunta do usuário
        
    Returns:
        Dicionário com resultado do processamento
    """
    logger.info(f"Processando: {question[:50]}...")
    
    try:
        result = validate_and_process(question)
        return result
    except ValueError as e:
        logger.error(f"Validação falhou: {e}")
        raise


# ❌ RUIM
def process_q(q):
    r = validate(q)
    return r
```

### Documentação

- **Docstrings**: Use Google-style docstrings
- **Comentários**: Explique "por quê", não "o quê"
- **Type hints**: Sempre use type hints

### Estrutura de Commit

```
<tipo>(<escopo>): <mensagem breve>

<corpo detalhado, se necessário>

<referências: Fixes #123>
```

**Tipos**: feat, fix, docs, style, refactor, perf, test, chore

---

## Processo de Desenvolvimento

### Fluxo

1. **Análise**: Entenda o problema/feature
2. **Design**: Planeje a abordagem (para PRs grandes)
3. **Implementação**: Código com testes
4. **Validação**: Testes passam, cobertura OK
5. **Review**: Feedback e ajustes
6. **Merge**: Incorporado ao main

### Dicas

- **PRs Pequenas**: Mais fáceis de revisar
- **Testes Primeiro**: Considere TDD
- **Documentação**: Atualize docs ao mudar code
- **Performance**: Considere impacto em produção

---

## Ambiente de Desenvolvimento

### Setup

```bash
# Clone e instale
git clone https://github.com/seu-hospital/machado-oraculo
cd machado-oraculo
uv sync

# Configure git hooks (opcional)
pre-commit install
```

### Testes

```bash
# Testes unitários
pytest tests/unit/ -v

# Integração
pytest tests/integration/ -v

# Com cobertura
pytest --cov=src --cov-report=html

# Específico
pytest tests/unit/test_nodes.py::test_guardrails -v
```

### Lint & Format

```bash
# Verificar estilo
flake8 src/ tests/

# Formatar código
black src/ tests/

# Type check
mypy src/
```

---

## Melhorias Prioritárias

Veja [ROADMAP.md](./ROADMAP.md) para features planejadas:

- 🔴 **P0**: Anonymizer com Presidio
- 🟡 **P1**: Cache Redis
- 🟢 **P2**: Logging estruturado avançado

---

## Perguntas?

- 📧 dev-team@hospital.com
- 💬 Slack: #machado-oraculo-dev
- 📚 Wiki: https://wiki.hospital.com/machado-oraculo

---

**Obrigado por contribuir! 🙏**
