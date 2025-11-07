from __future__ import annotations


def truncate_content(text: str, limit: int = 512) -> str:
    if len(text) <= limit:
        return text
    half = limit // 2
    return f"{text[:half]}\n... _truncated_ ...\n{text[-half:]}"
