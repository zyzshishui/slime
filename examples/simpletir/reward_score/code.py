from __future__ import annotations

import json
import os
from typing import Any

import numpy as np

from ..text_utils import truncate_content

try:
    if os.getenv("SANDBOX_ENDPOINT"):
        from ..sandbox.local_sandbox import parallel_sandbox
    else:
        from ..sandbox.internal_sandbox import parallel_sandbox
except Exception:  # noqa: BLE001
    parallel_sandbox = None


async def _run_sandbox(codes: list[str], stdin_list: list[str] | None = None):
    if parallel_sandbox is None:
        raise RuntimeError("Sandbox backend is not available for code reward verification.")
    return await parallel_sandbox(codes, stdin_list=stdin_list, num_processes=256)


async def compute_score_async(solution_str: str, ground_truth: Any, extra_info: Any = None):
    if isinstance(extra_info, np.ndarray):
        extra_info = extra_info.item()

    has_code_piece = bool(solution_str and solution_str.strip())
    if not has_code_piece:
        return {
            "score": 0.0,
            "extra_info": {
                "score": 0.0,
                "valid_code": 0,
            },
        }

    if isinstance(ground_truth, str):
        ground_truth = json.loads(ground_truth)

    if isinstance(ground_truth, list):
        code_list = []
        stdin_list = []
        expected_stdout = []

        for test in ground_truth:
            test_type = test.get("testtype")
            if test_type == "stdin":
                code_list.append(solution_str)
                stdin_list.append(test.get("input", ""))
                expected_stdout.append(test.get("output", ""))
            elif test_type == "functional":
                exec_lines = [solution_str, "sol = Solution()"]
                func_name = test["metadata"]["func_name"]
                arg_lines = [
                    ln.strip()
                    for ln in test.get("input", "").strip().split("\\n")
                    if ln.strip()
                ]
                args_str = ", ".join(arg_lines)
                exec_lines.append(f"print(sol.{func_name}({args_str}))")
                code_list.append("\n".join(exec_lines))
                stdin_list.append("")
                expected_stdout.append(test.get("output", ""))
            else:
                raise ValueError(f"Unsupported test type: {test_type}")

        success, stdout_list, stderr_list = await _run_sandbox(code_list, stdin_list=stdin_list)

        for ok, stdout, stderr, expected in zip(success, stdout_list, stderr_list, expected_stdout):
            if not ok:
                return {
                    "score": 0.0,
                    "extra_info": {
                        "score": 0.0,
                        "valid_code": 1,
                        "stderr": truncate_content(stderr) if stderr else "",
                    },
                }
            if stdout.strip() != expected.strip():
                return {
                    "score": 0.0,
                    "extra_info": {
                        "score": 0.0,
                        "valid_code": 1,
                        "stdout": truncate_content(stdout),
                        "expected": expected,
                    },
                }
        return {
            "score": 1.0,
            "extra_info": {
                "score": 1.0,
                "valid_code": 1,
            },
        }

    if isinstance(ground_truth, dict):
        if "functional" in ground_truth:
            code_to_execute = solution_str + "\n" + ground_truth["functional"]
            success, stdout_list, stderr_list = await _run_sandbox([code_to_execute])
            ok = success[0]
            stdout = stdout_list[0]
            stderr = stderr_list[0]
            if not ok or stderr:
                return {
                    "score": 0.0,
                    "extra_info": {
                        "score": 0.0,
                        "valid_code": 1,
                        "stderr": truncate_content(stderr) if stderr else "",
                    },
                }
            return {
                "score": 1.0,
                "extra_info": {
                    "score": 1.0,
                    "valid_code": 1,
                    "stdout": truncate_content(stdout),
                },
            }

        if "inputs" in ground_truth and "outputs" in ground_truth:
            inputs = ground_truth["inputs"]
            outputs = ground_truth["outputs"]
            success, stdout_list, stderr_list = await _run_sandbox(
                [solution_str] * len(inputs),
                stdin_list=inputs,
            )
            for ok, stdout, stderr, expected in zip(success, stdout_list, stderr_list, outputs):
                if not ok or stderr or stdout.strip() != expected.strip():
                    return {
                        "score": 0.0,
                        "extra_info": {
                            "score": 0.0,
                            "valid_code": 1,
                            "stdout": truncate_content(stdout),
                            "expected": expected,
                            "stderr": truncate_content(stderr) if stderr else "",
                        },
                    }
            return {
                "score": 1.0,
                "extra_info": {
                    "score": 1.0,
                    "valid_code": 1,
                },
            }

        raise ValueError(f"Unsupported ground truth dictionary format: {ground_truth}")

    raise ValueError(f"Unsupported ground truth format: {type(ground_truth)}")
