__author__ = 'Qiao Jin'

"""
Script de divisão de dataset para validação cruzada (ML).
Ajustado para novo caminho: docs/data/knowledge_base/

Propósito:
    - Dividir dataset em train/dev/test
    - Criar splits de k-fold cross-validation
    - Estratificar por label para manter proporções

Uso:
    python src/infrastructure/preprocess/split_dataset.py pqal
    python src/infrastructure/preprocess/split_dataset.py pqaa

NOTA: Este script é legado. O fluxo atual carrega protocolos XML via VectorStoreRepository.
      Use este script apenas se precisar re-preparar datasets antigos para análise.
"""

from functools import reduce
import json
import math
import os
from pathlib import Path
import random
random.seed(0)
import shutil
import sys

def split(dataset, fold):
    '''
    dataset: dataset dict
    fold: number of splits

    output list of splited datasets

    Split the dataset for each label to ensure label proportion of different subsets are similar
    '''
    add = lambda x: reduce(lambda a, b: a+b, x)
    
    label2pmid = {'yes': [], 'no': [], 'maybe': []}
    for pmid, info in dataset.items():
        label2pmid[info['final_decision']].append(pmid)

    label2pmid = {k: split_label(v, fold) for k, v in label2pmid.items()} # splited

    output = []

    for i in range(fold):
        pmids = add([v[i] for _, v in label2pmid.items()])
        output.append({pmid: dataset[pmid] for pmid in pmids})

    if len(output[-1]) != len(output[0]): # imbalanced: [51, 51, 51, 51, 51, 51, 51, 51, 51, 41]
        # randomly pick one from each to the last
        for i in range(fold-1):
            pmids = list(output[i])
            picked = random.choice(pmids)
            output[-1][picked] = output[i][picked]
            output[i].pop(picked)

    return output

def split_label(pmids, fold):
    '''
    pmids: a list of pmids (of the same label)
    fold: number of splits

    output: list of split lists
    '''
    random.shuffle(pmids)

    num_all = len(pmids)
    num_split = math.ceil(num_all / fold)

    output = []
    for i in range(fold):
        if i == fold - 1:
            output.append(pmids[i*num_split: ])
        else:
            output.append(pmids[i*num_split: (i+1)*num_split])

    return output

def combine_other(cv_sets, fold):
    '''
    combine other cv sets
    '''
    output = {}

    for i in range(10):
        if i != fold:
            for pmid, info in cv_sets[i].items():
                output[pmid] = info

    return output

split_name = sys.argv[1]

# Definir caminho base para dados
data_path = Path("docs/data/knowledge_base")
ori_pqal_file = data_path / "ori_pqal" / "ori_pqal.json"

if split_name == 'pqal':
    # 500 for 10-CV and 500 for test
    if not ori_pqal_file.exists():
        raise FileNotFoundError(f"Arquivo não encontrado: {ori_pqal_file}")
    
    dataset = json.load(open(ori_pqal_file))

    CV_set, testset = split(dataset, 2)
    test_file = data_path / "ori_pqal" / "test_set.json"
    with open(test_file, 'w') as f:
        json.dump(testset, f, indent=4)

    CV_sets = split(CV_set, 10)
    for i in range(10):
        fold_dir = data_path / "ori_pqal" / f"pqal_fold{i}"
        if fold_dir.is_dir():
            shutil.rmtree(fold_dir)
        fold_dir.mkdir(parents=True, exist_ok=True)
        with open(fold_dir / "dev_set.json", 'w') as f:
            json.dump(CV_sets[i], f, indent=4)
        with open(fold_dir / "train_set.json", 'w') as f:
            json.dump(combine_other(CV_sets, i), f, indent=4)

elif split_name == 'pqaa':
    # get 200k for training and rest for dev
    ori_pqaa_file = data_path / "ori_pqaa" / "ori_pqaa.json"
    
    if not ori_pqaa_file.exists():
        raise FileNotFoundError(f"Arquivo não encontrado: {ori_pqaa_file}")
    
    dataset = json.load(open(ori_pqaa_file))
    
    pmids = list(dataset)
    random.shuffle(pmids)

    train_split = {pmid: dataset[pmid] for pmid in pmids[:200000]}
    dev_split = {pmid: dataset[pmid] for pmid in pmids[200000:]}

    train_file = data_path / "ori_pqaa" / "pqaa_train_set.json"
    dev_file = data_path / "ori_pqaa" / "pqaa_dev_set.json"
    
    with open(train_file, 'w') as f:
        json.dump(train_split, f, indent=4)
    with open(dev_file, 'w') as f:
        json.dump(dev_split, f, indent=4)
