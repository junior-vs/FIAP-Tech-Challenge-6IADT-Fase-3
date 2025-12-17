# Base de Conhecimento Médica - Machado Oráculo

## Organização

```
docs/knowledge_base/
├── 7_SeniorHealth_QA/          # Protocolos de Saúde do Idoso (ATIVO)
│   ├── cardiac_protocols.xml   # Protocolos cardíacos
│   ├── geriatric_care.xml      # Cuidados geriátricos
│   └── ...
├── ori_pqal/                   # Dataset legado (LEGADO - não usar)
├── ori_pqaa/                   # Dataset legado (LEGADO - não usar)
└── CATALOG.md                  # Índice de conteúdo
```

## Adicionando Novo Protocolo

1. **Validação:** `python scripts/validate_protocol.py <arquivo>.xml`
2. **Anonymização:** `python scripts/anonymize_protocol.py <arquivo>.xml`
3. **Ingestão:** Sistema recarrega automaticamente na próxima inicialização

## Versionamento

Todos os protocolos devem incluir:
```xml
<Protocol>
  <Metadata>
    <Version>1.0</Version>
    <LastUpdated>2025-01-15</LastUpdated>
    <Authority>Hospital X</Authority>
  </Metadata>
  ...
</Protocol>
```