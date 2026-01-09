from pathlib import Path
import json


def convert(input_path: Path, output_path: Path) -> int:
    total = 0
    with open(input_path, 'r', encoding='utf-8') as inp, open(output_path, 'w', encoding='utf-8') as out:
        for line in inp:
            line=line.strip()
            if not line:
                continue
            try:
                obj = json.loads(line)
            except Exception:
                continue

            messages = obj.get('messages')
            if messages and isinstance(messages, list):
                user_msgs = [m.get('content','') for m in messages if m.get('role')=='user' and m.get('content')]
                assistant_msgs = [m.get('content','') for m in messages if m.get('role')=='assistant' and m.get('content')]
                instruction = '\n\n'.join(user_msgs) if user_msgs else obj.get('instruction') or obj.get('question')
                response = assistant_msgs[-1] if assistant_msgs else obj.get('response') or obj.get('answer')
            else:
                instruction = obj.get('question') or obj.get('instruction')
                response = obj.get('answer') or obj.get('response')

            if not instruction or not response:
                continue

            # Ensure response begins with space and ends with newline
            if not response.startswith(' '):
                response = ' ' + response
            if not response.endswith('\n'):
                response = response + '\n'

            out.write(json.dumps({'instruction': instruction, 'response': response}, ensure_ascii=False) + '\n')
            total += 1

    return total


if __name__ == '__main__':
    p = Path('docs/knowledge_base/finetuning_data.jsonl')
    out = Path('docs/knowledge_base/finetuning_instruction_response.jsonl')
    n = convert(p, out)
    print(f'Wrote {n} records to {out}')
