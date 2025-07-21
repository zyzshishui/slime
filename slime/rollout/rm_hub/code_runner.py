import asyncio
import json
import os
import sys
import logging
import uuid

from .coding_utils import SINGLE_CASE_EXEC_TIMEOUT

    
async def evaluate_coding_solution(response: str, label: str) -> float:
    PROCESS_TIMEOUT = SINGLE_CASE_EXEC_TIMEOUT + 4

    tmp_id = str(uuid.uuid4())
    input_file = f"/tmp/{tmp_id}-input.json"
    output_file = f"/tmp/{tmp_id}-output.json"

    input_data = {
        "response": response,
        "label": label,
    }

    process = None
    try:
        with open(input_file, "w") as f:
            json.dump(input_data, f)

        process = await asyncio.create_subprocess_exec(
            sys.executable,
            "/root/slime_/slime/rollout/rm_hub/coding_utils.py",
            "--input-file", input_file,
            "--output-file", output_file,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE
        )

        try:
            stdout, stderr = await asyncio.wait_for(process.communicate(), timeout=PROCESS_TIMEOUT)
            if stderr:
                logging.warning(f"Code runner stderr for {tmp_id}:\n{stderr.decode()}")

        except asyncio.TimeoutError:
            logging.warning(f"Process {process.pid} timed out after {PROCESS_TIMEOUT}s. Terminating.")
            process.terminate()
            await process.wait()
            return 0.0

        if os.path.exists(output_file):
            with open(output_file, 'r') as f:
                result_data = json.load(f)
            return result_data.get('score', 0.0)
        else:
            logging.warning(f"Output file not found for {tmp_id}. Assuming failure.")
            return 0.0

    except Exception as e:
        logging.error(f"Exception in evaluate_coding_solution master process: {e}")
        if process and process.returncode is None:
            process.terminate()
            await process.wait()
        return 0.0
    
    finally:
        if os.path.exists(input_file):
            os.remove(input_file)
        if os.path.exists(output_file):
            os.remove(output_file)