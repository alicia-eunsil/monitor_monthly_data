from __future__ import annotations

from typing import Any, Dict, List

import requests

OPENAI_RESPONSES_URL = "https://api.openai.com/v1/responses"


def _extract_output_text(payload: Dict[str, Any]) -> str:
    output = payload.get("output", [])
    if not isinstance(output, list):
        return ""
    texts: List[str] = []
    for item in output:
        if not isinstance(item, dict):
            continue
        content = item.get("content", [])
        if not isinstance(content, list):
            continue
        for chunk in content:
            if not isinstance(chunk, dict):
                continue
            if chunk.get("type") == "output_text":
                text = str(chunk.get("text", "")).strip()
                if text:
                    texts.append(text)
    return "\n".join(texts).strip()


def create_response_text(
    api_key: str,
    prompt: str,
    model: str,
    temperature: float = 0.3,
    max_output_tokens: int = 800,
    timeout: int = 60,
) -> Dict[str, Any]:
    if not api_key or not api_key.strip():
        return {"ok": False, "error": "OPENAI_API_KEY가 설정되지 않았습니다."}
    if not prompt or not str(prompt).strip():
        return {"ok": False, "error": "프롬프트가 비어 있습니다."}

    headers = {
        "Authorization": f"Bearer {api_key.strip()}",
        "Content-Type": "application/json",
    }
    payload = {
        "model": model,
        "input": str(prompt),
        "temperature": float(temperature),
        "max_output_tokens": int(max_output_tokens),
    }
    try:
        resp = requests.post(OPENAI_RESPONSES_URL, headers=headers, json=payload, timeout=timeout)
    except requests.RequestException as exc:
        return {"ok": False, "error": f"요청 실패: {exc}"}

    if resp.status_code >= 400:
        return {"ok": False, "error": f"OpenAI API 오류({resp.status_code}): {resp.text}"}

    try:
        data = resp.json()
    except ValueError:
        return {"ok": False, "error": "응답 파싱 실패"}

    text = _extract_output_text(data)
    if not text:
        return {"ok": False, "error": "응답 텍스트를 추출하지 못했습니다.", "raw": data}

    return {"ok": True, "text": text, "raw": data}
