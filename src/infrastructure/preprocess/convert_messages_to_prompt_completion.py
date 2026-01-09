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
                prompt = '\n\n'.join(user_msgs) if user_msgs else obj.get('prompt') or obj.get('question')
                completion = assistant_msgs[-1] if assistant_msgs else obj.get('completion') or obj.get('answer')
            else:
                prompt = obj.get('question') or obj.get('prompt')
                completion = obj.get('answer') or obj.get('completion')

            if not prompt or not completion:
                continue

            # Ensure completion begins with space and ends with newline
            if not completion.startswith(' '):
                completion = ' ' + completion
            if not completion.endswith('\n'):
                completion = completion + '\n'

            out.write(json.dumps({'prompt': prompt, 'completion': completion}, ensure_ascii=False) + '\n')
            total += 1

    return total


if __name__ == '__main__':
    p = Path('docs/knowledge_base/finetuning_data.jsonl')
    out = Path('docs/knowledge_base/finetuning_prompt_completion.jsonl')
    n = convert(p, out)
    print(f'Wrote {n} records to {out}')
