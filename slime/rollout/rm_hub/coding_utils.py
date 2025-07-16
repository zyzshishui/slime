import asyncio
import json
import re
import subprocess
import tempfile
import os
from typing import Dict, List, Any, Optional
import concurrent.futures
import time
from functools import partial


def extract_code_from_response(response: str) -> Optional[str]:
    python_code_pattern = r'```python\s*\n(.*?)\n```'
    matches = re.findall(python_code_pattern, response, re.DOTALL)
    if matches:
        code = matches[-1].strip()
        return code
    
    general_code_pattern = r'```\s*\n(.*?)\n```'
    matches = re.findall(general_code_pattern, response, re.DOTALL)
    if matches:
        code = matches[-1].strip()
        return code
    return None


def run_code_with_input(code: str, input_data: str, timeout: int = 10) -> tuple[bool, str, str]:
    try:
        with tempfile.NamedTemporaryFile(mode='w', suffix='.py', delete=False) as f:
            f.write(code)
            temp_file = f.name
        try:
            result = subprocess.run(
                ['python', temp_file],
                input=input_data.encode(),
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                timeout=timeout
            )
            success = result.returncode == 0
            stdout_str = result.stdout.decode()
            stderr_str = result.stderr.decode()
            return success, stdout_str, stderr_str
        except subprocess.TimeoutExpired:
            return False, '', 'Timeout'
    except Exception as e:
        return False, '', str(e)
    finally:
        if 'temp_file' in locals() and os.path.exists(temp_file):
            os.unlink(temp_file)


def normalize_output(output: str) -> str:
    return output.strip().rstrip('\n')


def eval_one(args, code):
    input_data, expected_output = args
    expected_output_norm = normalize_output(expected_output)
    success, actual_output, error = run_code_with_input(code, input_data)
    if success:
        actual_output_norm = normalize_output(actual_output)
        if actual_output_norm == expected_output_norm:
            return 1
    return 0


def evaluate_coding_solution(response: str, label: str, max_workers: int = 4) -> float:
    code = extract_code_from_response(response)
    if not code:
        return 0.0
    try:
        test_data = json.loads(label)
        inputs = test_data.get('inputs', [])
        expected_outputs = test_data.get('outputs', [])
        if not inputs or not expected_outputs:
            return 0.0
        assert len(inputs) == len(expected_outputs), "Number of inputs and expected outputs must match"
        correct = 0
        total = len(inputs)
        num_process = max(1, os.cpu_count() // 8)
        with concurrent.futures.ProcessPoolExecutor(max_workers=num_process) as executor:
            results = list(executor.map(partial(eval_one, code=code), zip(inputs, expected_outputs)))
        score = sum(results) / total if total > 0 else 0.0
        return score
    except json.JSONDecodeError as e:
        print(f"[DEBUG] JSON decode error: {e}")
        return 0.0
    except Exception as e:
        print(f"[DEBUG] Exception in evaluate_coding_solution: {e}")
        return 0.0 