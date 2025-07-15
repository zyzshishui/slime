#!/usr/bin/env python3
import json
import re

def clean_prompt(prompt_text):
    """Remove <｜User｜> and <｜Assistant｜><think> from prompt"""
    prompt_text = re.sub(r'<\uFF5CUser\uFF5C>', '', prompt_text)
    prompt_text = re.sub(r'<\uFF5CAssistant\uFF5C><think>', '', prompt_text)
    return prompt_text.strip()

def convert_train_dataset(input_file, output_file):
    print(f"Converting {input_file} to {output_file}")

    with open(input_file, 'r', encoding='utf-8') as f_in, \
         open(output_file, 'w', encoding='utf-8') as f_out:

        for line_num, line in enumerate(f_in, 1):
            try:
                data = json.loads(line.strip())

                cleaned_prompt = clean_prompt(data['prompt'])

                new_data = {
                    "prompt": [{
                        "content": cleaned_prompt,
                        "role": "user"
                    }],
                    "input_output": data.get('input_output', {})
                }

                f_out.write(json.dumps(new_data, ensure_ascii=False) + '\n')

            except json.JSONDecodeError as e:
                print(f"Error parsing line {line_num}: {e}")
            except KeyError as e:
                print(f"Missing key {e} in line {line_num}")

def convert_eval_dataset(input_file, output_file):
    print(f"Converting {input_file} to {output_file}")
    
    with open(input_file, 'r', encoding='utf-8') as f_in, \
         open(output_file, 'w', encoding='utf-8') as f_out:
        
        for line_num, line in enumerate(f_in, 1):
            try:
                data = json.loads(line.strip())

                question_content = data.get('question', '')

                new_data = {
                    "prompt": [{
                        "content": question_content,
                        "role": "user"
                    }],
                    "input_output": data.get('input_output', {})
                }
                
                f_out.write(json.dumps(new_data, ensure_ascii=False) + '\n')
                
            except json.JSONDecodeError as e:
                print(f"Error parsing line {line_num}: {e}")
            except KeyError as e:
                print(f"Missing key {e} in line {line_num}")

def main():
    convert_train_dataset(
        '/root/coding_data/train/train.jsonl',
        '/root/coding_dataset/train.jsonl'
    )

    eval_datasets = [
        ('/root/coding_data/code_benchmark/codeforces/test.jsonl', 
         '/root/coding_dataset/codeforces.jsonl'),
        ('/root/coding_data/code_benchmark/lcb_v5_2410_2502/test.jsonl',
         '/root/coding_dataset/lcb_v5_2410_2502.jsonl'),
        ('/root/coding_data/code_benchmark/code_contest_all/test.jsonl',
         '/root/coding_dataset/code_contest_all.jsonl')
    ]
    
    for input_file, output_file in eval_datasets:
        convert_eval_dataset(input_file, output_file)
    
    print("Conversion completed")

if __name__ == "__main__":
    main() 