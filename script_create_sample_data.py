#!/usr/bin/env python3
"""
script_create_sample_data.py

Script para criar arquivo JSONL de exemplo para testar fine-tuning.
Útil para validar o pipeline sem usar dados reais.
"""

import json
from pathlib import Path

def create_sample_data(output_file: str = "sample_finetuning_data.jsonl", num_examples: int = 100):
    """
    Cria arquivo JSONL de exemplo para teste.
    
    Args:
        output_file: Caminho do arquivo de saída
        num_examples: Número de exemplos a gerar
    """
    
    # Exemplos de Q&A sobre saúde geral
    qa_examples = [
        ("Qual é a importância da atividade física?", 
         "A atividade física regular é essencial para manter a saúde. Ajuda a fortalecer o coração, melhorar a resistência, prevenir obesidade e reduzir o risco de doenças crônicas como diabetes e hipertensão."),
        
        ("Como prevenir o estresse?", 
         "Para prevenir e gerenciar o estresse, é importante manter uma rotina equilibrada, praticar exercícios físicos, dormir bem, se conectar com pessoas, praticar meditação ou técnicas de relaxamento e buscar ajuda profissional se necessário."),
        
        ("Quais são os sinais de desidratação?",
         "Os sinais de desidratação incluem boca seca, sede intensa, urina escura, fadiga, tontura, dor de cabeça e confusão em casos graves. É importante beber água regularmente, especialmente em dias quentes ou durante atividades físicas."),
        
        ("Como manter uma boa higiene pessoal?",
         "A boa higiene pessoal envolve lavar as mãos regularmente, tomar banho diariamente, escovar os dentes, manter as unhas limpas, usar roupas limpas e manter o ambiente de convivência limpo para evitar infecções."),
        
        ("Qual é a importância do sono?",
         "O sono adequado é fundamental para a recuperação do corpo e da mente. Durante o sono, o corpo realiza reparos celulares, consolida memórias e restaura a energia. A maioria dos adultos precisa de 7-9 horas de sono por noite."),
        
        ("Como identificar uma reação alérgica?",
         "Os sintomas de reação alérgica incluem erupção na pele, coceira, inchaço nos lábios ou garganta, dificuldade respiratória, náusea e dor abdominal. Reações graves requerem atendimento médico de emergência."),
        
        ("Qual é a importância de uma dieta equilibrada?",
         "Uma dieta equilibrada fornece os nutrientes necessários para o funcionamento do corpo. Deve incluir proteínas, carboidratos, gorduras saudáveis, vitaminas e minerais em proporções adequadas."),
        
        ("Como prevenir infecções?",
         "Para prevenir infecções, é importante manter a higiene pessoal, lavar as mãos frequentemente, usar água limpa, cozinhar alimentos adequadamente, manter as feridas limpas e cobrir a boca ao tossir ou espirrar."),
        
        ("Quais são os fatores de risco para doenças do coração?",
         "Os principais fatores de risco incluem pressão alta, colesterol elevado, diabetes, obesidade, falta de atividade física, fumo, estresse e histórico familiar. É importante controlar estes fatores."),
        
        ("Como manter a saúde mental?",
         "Para manter a saúde mental, é importante buscar apoio social, praticar atividades prazerosas, manter rotina, exercitar-se, dormir bem, alimentar-se adequadamente e buscar ajuda profissional quando necessário."),
    ]
    
    # Criar múltiplos exemplos
    records = []
    for i in range(num_examples):
        qa_idx = i % len(qa_examples)
        question, answer = qa_examples[qa_idx]
        
        # Adicionar variação
        if i % 3 == 0:
            question = f"Como lidar com: {question.lower()}"
        elif i % 3 == 1:
            question = f"Explique sobre: {question.lower()}"
        else:
            question = f"Qual é a importância de: {question.lower()[9:] if question.startswith('Qual é') else question.lower()}"
        
        record = {
            "messages": [
                {"role": "user", "content": question},
                {"role": "assistant", "content": answer}
            ]
        }
        records.append(record)
    
    # Escrever arquivo JSONL
    output_path = Path(output_file)
    with open(output_path, 'w', encoding='utf-8') as f:
        for record in records:
            f.write(json.dumps(record) + '\n')
    
    print(f"✅ Arquivo criado: {output_path}")
    print(f"   Total de exemplos: {len(records)}")
    print(f"   Tamanho: {output_path.stat().st_size / 1024:.2f} KB")
    
    # Mostrar amostra
    print(f"\n📝 Amostra do primeiro exemplo:")
    first = json.loads(Path(output_path).read_text().split('\n')[0])
    print(f"   User: {first['messages'][0]['content'][:60]}...")
    print(f"   Assistant: {first['messages'][1]['content'][:60]}...")
    
    return output_path


if __name__ == "__main__":
    import sys
    
    # Número de exemplos a criar (padrão: 100)
    num_examples = int(sys.argv[1]) if len(sys.argv) > 1 else 100
    
    output_file = create_sample_data(
        output_file="sample_finetuning_data.jsonl",
        num_examples=num_examples
    )
    
    print(f"\n✅ Pronto para usar!")
    print(f"   Execute: python fine_tuning_cli.py --data {output_file}")
