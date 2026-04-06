from __future__ import annotations

import json
import os
from datetime import datetime
from hashlib import sha256
from typing import Any, Dict, Iterable, List, Optional

DATA_DIR_NAME = "data"
MEMORY_FILENAME = "insight_memory.jsonl"


def _data_dir() -> str:
    base = os.getcwd()
    return os.path.join(base, DATA_DIR_NAME)


def memory_path() -> str:
    return os.path.join(_data_dir(), MEMORY_FILENAME)


def _ensure_data_dir() -> None:
    os.makedirs(_data_dir(), exist_ok=True)


def compute_hash(parts: Iterable[str]) -> str:
    joined = "\n".join([str(x) for x in parts if x is not None])
    return sha256(joined.encode("utf-8")).hexdigest()


def load_memory(limit: int = 500) -> List[Dict[str, Any]]:
    path = memory_path()
    if not os.path.exists(path):
        return []
    entries: List[Dict[str, Any]] = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                entries.append(json.loads(line))
            except json.JSONDecodeError:
                continue
    if limit is not None and len(entries) > int(limit):
        entries = entries[-int(limit) :]
    return entries


def save_memory(entry: Dict[str, Any]) -> None:
    _ensure_data_dir()
    payload = dict(entry)
    payload.setdefault("created_at", datetime.now().isoformat(timespec="seconds"))
    with open(memory_path(), "a", encoding="utf-8") as f:
        f.write(json.dumps(payload, ensure_ascii=False) + "\n")


def select_memory_context(
    entries: List[Dict[str, Any]],
    scope_title: str,
    region: str,
    limit: int = 5,
    exact_hash: Optional[str] = None,
) -> List[Dict[str, Any]]:
    if not entries:
        return []

    scope_title = str(scope_title or "").strip()
    region = str(region or "").strip()

    def _score(e: Dict[str, Any]) -> tuple[int, str]:
        same_scope = 1 if str(e.get("scope_title", "")).strip() == scope_title else 0
        same_region = 1 if str(e.get("region", "")).strip() == region else 0
        exact = 1 if exact_hash and str(e.get("context_hash", "")) == exact_hash else 0
        return (exact * 4 + same_scope * 2 + same_region * 1, str(e.get("created_at", "")))

    ranked = sorted(entries, key=_score, reverse=True)
    out: List[Dict[str, Any]] = []
    seen: set[str] = set()
    for entry in ranked:
        key = str(entry.get("id", "")) or str(entry.get("created_at", ""))
        if key in seen:
            continue
        out.append(entry)
        seen.add(key)
        if len(out) >= int(limit):
            break
    return out


def build_prompt(
    context_title: str,
    context_lines: List[str],
    focus_lines: List[str],
    consecutive_lines: List[str],
    past_summaries: List[str],
    user_note: str = "",
) -> str:
    sections: List[str] = []
    sections.append("당신은 고용지표/지역경제 분석 전문가입니다. 최신 데이터 기반으로 인사이트를 작성하세요.")
    if context_title:
        sections.append(f"[분석 범위]\n- {context_title}")
    if context_lines:
        sections.append("[최신 NEW HISTORY 요약]\n" + "\n".join(context_lines))
    if focus_lines:
        sections.append("[핵심 NEW 포인트]\n" + "\n".join(focus_lines))
    if consecutive_lines:
        sections.append("[연속 변화 요약]\n" + "\n".join(consecutive_lines))
    if past_summaries:
        sections.append("[과거 인사이트 요약(참고용)]\n" + "\n".join(past_summaries))
    if user_note.strip():
        sections.append("[추가 메모]\n" + user_note.strip())
    sections.append(
        "[작성 지침]\n- 최신 변화가 있다면 과거와 비교해 차이를 강조하세요.\n"
        "- 과거 요약은 참고만 하고, 이번 변화를 1순위로 설명하세요.\n"
        "- 3~6개의 불릿으로 간결하게 작성하세요."
    )
    return "\n\n".join(sections)
