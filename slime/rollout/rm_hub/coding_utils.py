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
import logging

CODE_EVAL_EXECUTOR: Optional[concurrent.futures.ProcessPoolExecutor] = None

def initialize_coding_executor(max_workers: Optional[int] = None):
    global CODE_EVAL_EXECUTOR
    if CODE_EVAL_EXECUTOR is None:
        if max_workers is None:
            max_workers = os.cpu_count()
        logging.info(f"Initializing global ProcessPoolExecutor for coding evaluation with {max_workers} workers.")
        CODE_EVAL_EXECUTOR = concurrent.futures.ProcessPoolExecutor(max_workers=max_workers)
    else:
        logging.warning("Coding executor has already been initialized.")


def shutdown_coding_executor():
    global CODE_EVAL_EXECUTOR
    if CODE_EVAL_EXECUTOR is not None:
        CODE_EVAL_EXECUTOR.shutdown(wait=True)
        CODE_EVAL_EXECUTOR = None


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


def eval_one(code: str, input_data: str, expected_output: str) -> int:
    expected_output_norm = normalize_output(expected_output)
    success, actual_output, error = run_code_with_input(code, input_data)
    if not success:
        print(f"[DEBUG] Error running code: {error}")
        return 0
    actual_output_norm = normalize_output(actual_output)
    if actual_output_norm == expected_output_norm:
        return 1
    return 0


async def evaluate_coding_solution(response: str, label: str, max_workers: int = 4, executor=None) -> float:
    global CODE_EVAL_EXECUTOR
    if CODE_EVAL_EXECUTOR is None:
        initialize_coding_executor()

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

        total_cases = len(inputs)
        loop = asyncio.get_running_loop()
        tasks = [
            loop.run_in_executor(
                CODE_EVAL_EXECUTOR,
                eval_one,
                code,
                input_data, 
                expected_output
            )
            for input_data, expected_output in zip(inputs, expected_outputs)
        ]

        results = await asyncio.gather(*tasks)
        score = sum(results) / total_cases if total_cases > 0 else 0.0
        return score
    except json.JSONDecodeError as e:
        print(f"[DEBUG] JSON decode error: {e}")
        return 0.0
    except Exception as e:
        print(f"[DEBUG] Exception in evaluate_coding_solution: {e}")
        return 0.0 