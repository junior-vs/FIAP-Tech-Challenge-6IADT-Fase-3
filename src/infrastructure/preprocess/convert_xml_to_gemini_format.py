#!/usr/bin/env python3
"""
Script para converter XMLs da pasta 7_SeniorHealth_QA para formato Gemini fine-tuning.

Formato Gemini API para fine-tuning:
[
  {
    "role": "user",
    "content": "Question text with context"
  },
  {
    "role": "model",
    "content": "Answer text"
  }
]

Cada linha do JSONL deve conter um desses formatos.
"""

import json
import xml.etree.ElementTree as ET
from pathlib import Path
import logging

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def extract_qa_from_xml(xml_file: Path) -> list[dict]:
    """
    Extrai pares Question/Answer do arquivo XML.
    
    Args:
        xml_file: Caminho para o arquivo XML
        
    Returns:
        Lista de dicts com question, answer e context
    """
    try:
        tree = ET.parse(xml_file)
        root = tree.getroot()
        
        # Extrai o foco (contexto) do documento
        focus_elem = root.find("Focus")
        context = focus_elem.text if focus_elem is not None else "Medical Information"
        
        qa_pairs = []
        
        # Itera sobre todos os QAPair
        for qa_pair in root.findall(".//QAPair"):
            question_elem = qa_pair.find("Question")
            answer_elem = qa_pair.find("Answer")
            
            if question_elem is not None and answer_elem is not None:
                question_text = question_elem.text or ""
                answer_text = answer_elem.text or ""
                
                if question_text.strip() and answer_text.strip():
                    qa_pairs.append({
                        "question": question_text.strip(),
                        "answer": answer_text.strip(),
                        "context": context
                    })
        
        return qa_pairs
        
    except Exception as e:
        logger.error(f"Erro ao processar {xml_file.name}: {e}")
        return []


def create_gemini_messages(qa_pair: dict) -> list[dict]:
    """
    Converte um Q&A pair para formato de messages do Gemini.
    
    Args:
        qa_pair: Dict com question, answer e context
        
    Returns:
        Lista com [user_message, model_message]
    """
    # Inclui contexto na pergunta
    user_content = f"Context: {qa_pair['context']}\n\nQuestion: {qa_pair['question']}"
    
    return [
        {
            "role": "user",
            "content": user_content
        },
        {
            "role": "model",
            "content": qa_pair['answer']
        }
    ]


def convert_xmls_to_jsonl(
    input_dir: Path,
    output_file: Path,
    format_type: str = "messages"
) -> int:
    """
    Converte todos os XMLs da pasta para JSONL otimizado para Gemini.
    
    Args:
        input_dir: Diretório com arquivos XML
        output_file: Arquivo JSONL de saída
        format_type: "messages" (recomendado) ou "prompt_completion"
        
    Returns:
        Número de linhas geradas
    """
    
    input_path = Path(input_dir)
    if not input_path.exists():
        logger.error(f"Diretório não encontrado: {input_path}")
        return 0
    
    xml_files = sorted(input_path.glob("*.xml"))
    if not xml_files:
        logger.error(f"Nenhum arquivo XML encontrado em {input_path}")
        return 0
    
    logger.info(f"Encontrados {len(xml_files)} arquivos XML")
    
    total_lines = 0
    
    with open(output_file, 'w', encoding='utf-8') as f:
        for xml_file in xml_files:
            qa_pairs = extract_qa_from_xml(xml_file)
            logger.info(f"{xml_file.name}: {len(qa_pairs)} Q&A pairs extraídos")
            
            for qa_pair in qa_pairs:
                if format_type == "messages":
                    # Formato recomendado para Gemini 2.0
                    messages = create_gemini_messages(qa_pair)
                    
                    # Escreve como um único exemplo de conversa
                    line = {
                        "messages": messages
                    }
                    f.write(json.dumps(line, ensure_ascii=False) + '\n')
                    total_lines += 1
                    
                elif format_type == "prompt_completion":
                    # Formato alternativo (menos eficiente)
                    line = {
                        "prompt": f"Context: {qa_pair['context']}\n\nQuestion: {qa_pair['question']}",
                        "completion": qa_pair['answer']
                    }
                    f.write(json.dumps(line, ensure_ascii=False) + '\n')
                    total_lines += 1
    
    logger.info(f"✓ Arquivo gerado: {output_file}")
    logger.info(f"✓ Total de linhas: {total_lines}")
    
    return total_lines


def main():
    """Executa a conversão completa."""
    
    print("=" * 80)
    print("CONVERSOR DE XML PARA GEMINI FINE-TUNING FORMAT")
    print("=" * 80)
    
    # Caminhos
    input_dir = Path("docs/knowledge_base/7_SeniorHealth_QA")
    output_file_messages = Path("docs/finetuning_data_gemini_messages.jsonl")
    
    # Verifica diretório de entrada
    if not input_dir.exists():
        logger.error(f"Diretório não encontrado: {input_dir}")
        return
    
    # Opção 1: Formato "messages" (RECOMENDADO para Gemini 2.0)
    print("\nGerando formato 'messages' (recomendado para Gemini 2.0)...")
    lines = convert_xmls_to_jsonl(
        input_dir=input_dir,
        output_file=output_file_messages,
        format_type="messages"
    )
    
    if lines > 0:
        # Mostra amostra do arquivo gerado
        print(f"\n Sucesso! {lines} exemplos de treinamento gerados")
        print(f"\n Amostra do arquivo ({output_file_messages}):\n")
        
        with open(output_file_messages, 'r', encoding='utf-8') as f:
            for i, line in enumerate(f):
                if i < 2:  # Mostra 2 primeiros exemplos
                    data = json.loads(line)
                    print(f"Exemplo {i+1}:")
                    print(json.dumps(data, indent=2, ensure_ascii=False))
                    print()
                else:
                    break
        
        # Instruções de uso
        print("\n" + "=" * 80)
        print(" PRÓXIMOS PASSOS:")
        print("=" * 80)
        print(f"""
1. Use este arquivo para fine-tuning do Gemini:
   
   python fine_tuning_cli.py --data {output_file_messages} --wait

2. Formato esperado pela API Gemini 2.0:
   - Cada linha é um objeto JSON
   - Campo "messages" contém lista de mensagens [user, model]
   - Alternância entre role="user" e role="model"

3. Validar arquivo (opcional):
   python -c "
import json
with open('{output_file_messages}') as f:
    for i, line in enumerate(f):
        data = json.loads(line)
        assert 'messages' in data, f'Linha {{i}} sem messages'
        assert len(data['messages']) == 2, f'Linha {{i}} sem 2 mensagens'
print('✓ Arquivo validado com sucesso!')
"
        """)
        
    else:
        logger.error("Nenhum exemplo foi gerado")


if __name__ == "__main__":
    main()
