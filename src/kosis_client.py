from __future__ import annotations

from typing import Any, Dict, List

import requests

from .config import KOSIS_BASE_URL, DatasetConfig


class KosisClient:
    def __init__(self, api_key: str, timeout: int = 30):
        self.api_key = api_key
        self.timeout = timeout

    def fetch(self, config: DatasetConfig, end_prd_de: str) -> List[Dict[str, Any]]:
        params = config.to_params(api_key=self.api_key, end_prd_de=end_prd_de)
        response = requests.get(KOSIS_BASE_URL, params=params, timeout=self.timeout)
        response.raise_for_status()
        payload = response.json()
        if isinstance(payload, dict):
            error = payload.get("err") or payload.get("error")
            if error:
                raise RuntimeError(f"{config.title} 조회 실패: {error}")
            return [payload]
        if not isinstance(payload, list):
            raise RuntimeError(f"{config.title} 응답 형식이 예상과 다릅니다.")
        return payload

