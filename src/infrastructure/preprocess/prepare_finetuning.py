"""
Faz a conversão de arquivos XML de QA pairs para JSONL.

Propósito:
    - Transformar dados de QA pairs em XML para um formato JSONL adequado para fine-tuning de modelos de linguagem.
"""

import json
import xml.etree.ElementTree as ET
from pathlib import Path
from typing import List, Dict, Any
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def parse_xml_to_qa_pairs(xml_file: Path) -> List[Dict[str, Any]]:
    """
    Extrai QA pairs do arquivo XML.
    
    Args:
        xml_file: Caminho para o arquivo XML
        
    Returns:
        Lista contendo pares de QA com metadados
    """
    try:
        tree = ET.parse(xml_file)
        root = tree.getroot()
        
        # Extrai metadata do documento
        doc_id = root.get('id')
        source = root.get('source')
        url = root.get('url')
        focus = root.findtext('Focus')
        
        qa_pairs = []
        
        # Extrai todos os elementos QAPair
        qapairs_elem = root.find('QAPairs')
        if qapairs_elem is not None:
            for qapair in qapairs_elem.findall('QAPair'):
                pid = qapair.get('pid')
                
                question_elem = qapair.find('Question')
                answer_elem = qapair.find('Answer')
                
                if question_elem is not None and answer_elem is not None:
                    qa_pair = {
                        'doc_id': doc_id,
                        'pair_id': pid,
                        'focus': focus,
                        'source': source,
                        'url': url,
                        'question': question_elem.text,
                        'question_type': question_elem.get('qtype'),
                        'answer': answer_elem.text,
                    }
                    qa_pairs.append(qa_pair)
        
        return qa_pairs
        
    except Exception as e:
        logger.error(f"Erro convertendo {xml_file}: {str(e)}")
        return []


def convert_xml_to_jsonl(input_dir: Path, output_file: Path) -> int:
    """
    Converte todos os arquivos XML em um diretório para o formato JSONL.
    
    Args:
        input_dir: Diretório contendo os arquivos XML
        output_file: Caminho do arquivo JSONL de saída
        
    Returns:
        Número de registros escritos
    """
    xml_files = sorted(input_dir.glob('*.xml'))
    logger.info(f"Encontrados {len(xml_files)} arquivos XML em {input_dir}")
    
    total_records = 0
    
    with open(output_file, 'w', encoding='utf-8') as jsonl_file:
        for xml_file in xml_files:
            qa_pairs = parse_xml_to_qa_pairs(xml_file)
            
            for qa_pair in qa_pairs:
                # Formato para fine tuning: question -> answer
                finetuning_record = {
                    'messages': [
                        {
                            'role': 'user',
                            'content': f"Context: {qa_pair['focus']}\n\nQuestion: {qa_pair['question']}"
                        },
                        {
                            'role': 'assistant',
                            'content': qa_pair['answer']
                        }
                    ],
                    'metadata': {
                        'doc_id': qa_pair['doc_id'],
                        'pair_id': qa_pair['pair_id'],
                        'focus': qa_pair['focus'],
                        'question_type': qa_pair['question_type'],
                        'source': qa_pair['source'],
                        'url': qa_pair['url']
                    }
                }
                
                jsonl_file.write(json.dumps(finetuning_record, ensure_ascii=False) + '\n')
                total_records += 1
            
            logger.info(f"Processado {xml_file.name}: {len(qa_pairs)} QA pairs")
    
    return total_records


def corrigir_jsonl(entrada, saida):
    with open(entrada, "r", encoding="utf-8") as fin, open(saida, "w", encoding="utf-8") as fout:
        for linha in fin:
            linha = linha.strip()
            if not linha:
                continue

            try:
                dado = json.loads(linha)
            except:
                continue  # descarta linhas quebradas

            fout.write(json.dumps(dado, ensure_ascii=False) + "\n")




if __name__ == '__main__':
    # Define paths
    project_root = Path(__file__).parent.parent.parent
    input_dir = project_root / 'docs' / 'knowledge_base' / '7_SeniorHealth_QA'
    output_dir = project_root / 'docs' / 'knowledge_base'

    # Cria o diretório de saída se não existir
    output_dir.mkdir(parents=True, exist_ok=True)

    # Converte para o formato JSONL (XMLs)
    output_file_finetuning = output_dir / 'finetuning_data_gemini_messages.jsonl'
    records = convert_xml_to_jsonl(input_dir, output_file_finetuning)
    logger.info(f"Created {output_file_finetuning} with {records} records")

    # Faz limpeza no JSONL gerado
    output_file_finetuning_limpo = output_dir / 'dataset_finetuning_limpo.jsonl'
    corrigir_jsonl(output_file_finetuning, output_file_finetuning_limpo)