__author__ = 'Qiao Jin'
"""
Script de avaliação de classificação médica.
Ajustado para o novo contexto: docs/data/knowledge_base/

Uso:
    python src/infrastructure/preprocess/evaluation.py <predictions_file>
    
Exemplo:
    python src/infrastructure/preprocess/evaluation.py predictions.json
"""

import json
import sys
from pathlib import Path

from sklearn.metrics import accuracy_score, f1_score

pred_path = sys.argv[1]

# Ajustar para novo caminho da base de conhecimento
knowledge_base_path = Path("docs/data/knowledge_base")
ground_truth_file = knowledge_base_path / "ori_pqal" / "test_ground_truth.json"
pred_file = Path(pred_path)

# Verificar se os arquivos existem
if not ground_truth_file.exists():
    raise FileNotFoundError(f"Ground truth não encontrado em: {ground_truth_file}")
if not pred_file.exists():
    raise FileNotFoundError(f"Predições não encontradas em: {pred_file}")

ground_truth = json.load(open(ground_truth_file)) 
predictions = json.load(open(pred_file))

assert set(ground_truth) == set(predictions), 'Please predict all and only the instances in the test set.'

pmids = list(ground_truth)
truth = [ground_truth[pmid] for pmid in pmids]
preds = [predictions[pmid] for pmid in pmids]

acc = accuracy_score(truth, preds)
maf = f1_score(truth, preds, average='macro')

print('Accuracy %f' % acc)
print('Macro-F1 %f' % maf)
