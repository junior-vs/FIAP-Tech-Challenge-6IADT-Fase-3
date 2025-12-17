"""
Script legado: Calcula performance humana no test set.
Ajustado para novo caminho: docs/data/knowledge_base/

Propósito:
    - Comparar acurácia e F1 score de predições humanas
    - Baseline: sempre prediz "yes"
    - Reasoning-Free: predição sem justificativa
    - Reasoning-Required: predição com justificativa

Uso:
    python docs/data/knowledge_base/get_human_performance.py

NOTA: Este script é legado. O fluxo atual carrega protocolos XML via VectorStoreRepository.
      Use este script apenas se precisar analisar performance de datasets históricos.
"""

import json
from pathlib import Path

from sklearn.metrics import accuracy_score, f1_score

# Constantes para formatação das métricas
ACCURACY_FORMAT = 'Accuracy %f'
MACRO_F1_FORMAT = 'Macro-F1 %f'

# Definir caminho correto
data_path = Path("docs/data/knowledge_base/ori_pqal")
test_set_file = data_path / "test_set.json"

if not test_set_file.exists():
    raise FileNotFoundError(f"Arquivo não encontrado: {test_set_file}")

test_set = json.load(open(test_set_file))

labels = []
r_free = []
r_req = []

for pmid, info in test_set.items():
    labels.append(info['final_decision'])
    r_free.append(info['reasoning_free_pred'])
    r_req.append(info['reasoning_required_pred'])

maj = ['yes' for _ in labels]

print('====Majority Performance====')
print(ACCURACY_FORMAT % accuracy_score(labels, maj))
print(MACRO_F1_FORMAT % f1_score(labels, maj, average='macro'))

print('====Reasoning-Free Human Performance====')
print(ACCURACY_FORMAT % accuracy_score(labels, r_free))
print(MACRO_F1_FORMAT % f1_score(labels, r_free, average='macro'))

print('====Reasoning-Required Human Performance====')
print(ACCURACY_FORMAT % accuracy_score(labels, r_req))
print(MACRO_F1_FORMAT % f1_score(labels, r_req, average='macro'))
