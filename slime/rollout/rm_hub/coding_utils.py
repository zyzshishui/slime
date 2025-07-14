import asyncio
import json
import re
import subprocess
import tempfile
import os
from typing import Dict, List, Any, Optional


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


async def run_code_with_input(code: str, input_data: str, timeout: int = 10) -> tuple[bool, str, str]:
    try:
        with tempfile.NamedTemporaryFile(mode='w', suffix='.py', delete=False) as f:
            f.write(code)
            temp_file = f.name
        process = await asyncio.create_subprocess_exec(
            'python', temp_file,
            stdin=asyncio.subprocess.PIPE,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE
        )
        
        try:
            stdout, stderr = await asyncio.wait_for(
                process.communicate(input=input_data.encode()), 
                timeout=timeout
            )
            
            success = process.returncode == 0
            stdout_str = stdout.decode()
            stderr_str = stderr.decode()
            return success, stdout_str, stderr_str
            
        except asyncio.TimeoutError:
            process.kill()
            return False, "", "Timeout"
            
    except Exception as e:
        return False, "", str(e)
    finally:
        if 'temp_file' in locals() and os.path.exists(temp_file):
            os.unlink(temp_file)


def normalize_output(output: str) -> str:
    return output.strip().rstrip('\n')


async def evaluate_coding_solution(response: str, label: str) -> float:
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
        total = 0
        for i in range(min_cases):
            input_data = inputs[i]
            expected_output = normalize_output(expected_outputs[i])
            success, actual_output, error = await run_code_with_input(code, input_data)
            
            if success:
                actual_output = normalize_output(actual_output)
                
                if actual_output == expected_output:
                    correct += 1
            
            total += 1
        score = correct / total if total > 0 else 0.0
        
        return score
        
    except json.JSONDecodeError as e:
        return 0.0
    except Exception as e:
        return 0.0 