from __future__ import annotations

import asyncio
import os
import sys
from typing import Iterable, List, Tuple

RunCodeRequest = None
RunStatus = None
set_sandbox_endpoint = None
run_code_async = None

try:  # pragma: no cover - optional internal dependency
    from sandbox_fusion import RunCodeRequest, RunStatus, run_code_async, set_sandbox_endpoint
except Exception:  # noqa: BLE001
    RunCodeRequest = None


async def _run_remote(
    codes: List[str],
    stdin_list: List[str] | None,
    num_processes: int,
    compile_timeout: float,
    run_timeout: float,
) -> Tuple[List[bool], List[str], List[str]]:
    semaphore = asyncio.Semaphore(num_processes)
    endpoint = os.getenv("SIMPLETIR_FUSION_ENDPOINT", "https://seed-sandbox.byteintl.net/faas/sandbox/")
    set_sandbox_endpoint(endpoint)

    async def _single(code: str, stdin: str | None) -> dict:
        request = RunCodeRequest(
            code=code,
            stdin=stdin,
            language="python",
            compile_timeout=compile_timeout,
            run_timeout=run_timeout,
        )
        async with semaphore:
            response = await run_code_async(request, client_timeout=30.0, max_attempts=2)
        await asyncio.sleep(2)
        return response.dict()

    if stdin_list is None:
        tasks = [_single(code, None) for code in codes]
    else:
        tasks = [_single(code, stdin) for code, stdin in zip(codes, stdin_list)]

    results = await asyncio.gather(*tasks)
    return (
        [r.get("status") == RunStatus.Success for r in results],
        [r.get("run_result", {}).get("stdout", "") for r in results],
        [r.get("run_result", {}).get("stderr", "") for r in results],
    )


async def _run_local(
    codes: List[str],
    stdin_list: List[str] | None,
    num_processes: int,
    run_timeout: float,
) -> Tuple[List[bool], List[str], List[str]]:
    semaphore = asyncio.Semaphore(max(1, num_processes))

    async def _single(code: str, stdin: str | None):
        encoded_stdin = None if stdin is None else stdin.encode("utf-8")
        async with semaphore:
            proc = await asyncio.create_subprocess_exec(
                sys.executable,
                "-u",
                "-c",
                code,
                stdin=asyncio.subprocess.PIPE,
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE,
            )
            try:
                stdout, stderr = await asyncio.wait_for(proc.communicate(encoded_stdin), timeout=run_timeout)
                timed_out = False
            except asyncio.TimeoutError:
                proc.kill()
                stdout, stderr = await proc.communicate()
                timed_out = True
        success = proc.returncode == 0 and not timed_out
        stdout_text = stdout.decode("utf-8", errors="ignore")
        stderr_text = stderr.decode("utf-8", errors="ignore")
        if timed_out:
            stderr_text = (stderr_text or "") + "\n<timeout>"
        return success, stdout_text, stderr_text

    if stdin_list is None:
        tasks = [_single(code, None) for code in codes]
    else:
        tasks = [_single(code, stdin) for code, stdin in zip(codes, stdin_list)]

    results = await asyncio.gather(*tasks)
    ok_flags: List[bool] = []
    stdouts: List[str] = []
    stderrs: List[str] = []
    for success, stdout_text, stderr_text in results:
        ok_flags.append(success)
        stdouts.append(stdout_text)
        stderrs.append(stderr_text)
    return ok_flags, stdouts, stderrs


async def parallel_sandbox(
    tasks: Iterable[str],
    stdin_list: List[str] | None = None,
    num_processes: int = 200,
    compile_timeout: float = 1.0,
    run_timeout: float = 3.0,
):
    codes = list(tasks)
    if stdin_list is not None:
        assert len(codes) == len(stdin_list), f"len(tasks) ({len(codes)}) != len(stdin_list) ({len(stdin_list)})"

    use_remote = RunCodeRequest is not None and os.getenv("SIMPLETIR_FORCE_LOCAL_SANDBOX") != "1"

    if use_remote:
        return await _run_remote(codes, stdin_list, num_processes, compile_timeout, run_timeout)
    return await _run_local(codes, stdin_list, num_processes, run_timeout)


if __name__ == "__main__":  # pragma: no cover
    code_list = [
        """
def fib(n):
    a, b = 0, 1
    for _ in range(n):
        a, b = b, a + b
    return a

print([fib(i) for i in range(10)])
""",
        """
def fib(n):
    a, b = 0, 1
    for _ in range(n):
        a, b = b, a + b, v
    return a

print([fib(i) for i in range(10)])
""",
        """
name = input("Your name:")
print(f"Hi, {name}!")
""",
    ]
    stdin_list = ["", "", "Alice"]

    for code, stdin in zip(code_list, stdin_list):
        print(f"code: {code}")
        sandbox_success, sandbox_stdout, sandbox_stderr = asyncio.run(
            parallel_sandbox(tasks=[code], stdin_list=[stdin])
        )
        print(f"sandbox_success: {sandbox_success}")
        print(f"sandbox_stdout: {sandbox_stdout}")
        print(f"sandbox_stderr: {sandbox_stderr}")
