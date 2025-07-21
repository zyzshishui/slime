import asyncio
import json
import os
import sys
import uuid
import subprocess
import os
import concurrent.futures
import atexit
from .coding_utils import SINGLE_CASE_EXEC_TIMEOUT


POOL_MAX_WORKERS = max(1, os.cpu_count() // 2)
CODE_EVAL_EXECUTOR = concurrent.futures.ProcessPoolExecutor(
    max_workers=POOL_MAX_WORKERS
)
print(f"[CODE_EVAL_EXECUTOR] Process Pool Initialized in PID: {os.getpid()} with {POOL_MAX_WORKERS} workers.", flush=True)


def shutdown_code_executor():
    if CODE_EVAL_EXECUTOR:
        print(f"[CODE_EVAL_EXECUTOR] Shutting down process pool in PID: {os.getpid()}...", flush=True)
        CODE_EVAL_EXECUTOR.shutdown(wait=True)
        print(f"[CODE_EVAL_EXECUTOR] Process pool shut down successfully in PID: {os.getpid()}.", flush=True)


atexit.register(shutdown_code_executor)

    
def evaluate_in_subprocess(response: str, label: str) -> float:
    PROCESS_TIMEOUT = SINGLE_CASE_EXEC_TIMEOUT + 4
    tmp_id = str(uuid.uuid4())
    input_file = f"/tmp/{tmp_id}-input.json"
    output_file = f"/tmp/{tmp_id}-output.json"
    input_data = {"response": response, "label": label}

    try:
        with open(input_file, "w") as f:
            json.dump(input_data, f)

        completed_process = subprocess.run(
            [
                sys.executable,
                "/root/slime_/slime/rollout/rm_hub/coding_utils.py",
                "--input-file", input_file,
                "--output-file", output_file,
                "--tmp-id", tmp_id, 
            ],
            timeout=PROCESS_TIMEOUT,
            capture_output=True,
            text=True
        )

        if completed_process.stderr:
            print(f"[CODE_EVAL_EXECUTOR {tmp_id}] Subprocess stderr:\n{completed_process.stderr}", flush=True)

        if os.path.exists(output_file):
            with open(output_file, 'r') as f:
                result_data = json.load(f)
            return result_data.get('score', 0.0)
        else:
            print(f"[CODE_EVAL_EXECUTOR {tmp_id}] Output file not found. Assuming failure.", flush=True)
            return 0.0

    except subprocess.TimeoutExpired:
        print(f"[CODE_EVAL_EXECUTOR {tmp_id}] Process timed out after {PROCESS_TIMEOUT}s. It was killed by subprocess.run.", flush=True)
        return 0.0
    except Exception as e:
        print(f"[CODE_EVAL_EXECUTOR {tmp_id}] Exception: {e}", flush=True)
        return 0.0
    finally:
        if os.path.exists(input_file):
            os.remove(input_file)
        if os.path.exists(output_file):
            os.remove(output_file)
            

async def evaluate_coding_solution(response: str, label: str) -> float:
    loop = asyncio.get_running_loop()
    
    try:
        score = await loop.run_in_executor(
            CODE_EVAL_EXECUTOR,
            evaluate_in_subprocess,
            response,
            label
        )
        return score
    except Exception as e:
        print(f"[CODE_EVAL_EXECUTOR] Error submitting job to process pool: {e}", flush=True)
        return 0.0