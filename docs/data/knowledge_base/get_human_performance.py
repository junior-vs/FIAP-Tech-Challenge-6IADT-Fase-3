__author__ = 'Qiao Jin'

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
from sklearn.metrics import f1_score, accuracy_score

# Definir caminho correto (agora o script está na pasta de dados)
data_path = Path("ori_pqal")
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
print('Accuracy %f' % accuracy_score(labels, maj))
print('Macro-F1 %f' % f1_score(labels, maj, average='macro'))

print('====Reasoning-Free Human Performance====')
print('Accuracy %f' % accuracy_score(labels, r_free))
print('Macro-F1 %f' % f1_score(labels, r_free, average='macro'))

print('====Reasoning-Required Human Performance====')
print('Accuracy %f' % accuracy_score(labels, r_req))
print('Macro-F1 %f' % f1_score(labels, r_req, average='macro'))
