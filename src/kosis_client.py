from __future__ import annotations

from datetime import datetime
from typing import Any, Dict, List

import requests

from .config import KOSIS_BASE_URL, DatasetConfig


class KosisClient:
    def __init__(self, api_key: str, timeout: int = 30):
        self.api_key = api_key
        self.timeout = timeout

    def fetch(self, config: DatasetConfig, end_prd_de: str) -> List[Dict[str, Any]]:
        params = config.to_params(api_key=self.api_key, end_prd_de=end_prd_de)
        return self._fetch_with_split(config, params)

    def _fetch_with_split(self, config: DatasetConfig, params: Dict[str, str]) -> List[Dict[str, Any]]:
        payload = self._request(params)

        if isinstance(payload, list):
            return payload

        if isinstance(payload, dict):
            error = str(payload.get("err") or payload.get("error") or "").strip()
            if not error:
                return [payload]

            # KOSIS error 31 means row limit exceeded; split monthly range and retry.
            if error == "31" and params.get("prdSe") == "M":
                start = params.get("startPrdDe", "")
                end = params.get("endPrdDe", "")
                if start and end and start < end:
                    left_end, right_start = self._split_month_range(start, end)
                    left_params = dict(params)
                    right_params = dict(params)
                    left_params["endPrdDe"] = left_end
                    right_params["startPrdDe"] = right_start
                    return self._fetch_with_split(config, left_params) + self._fetch_with_split(
                        config, right_params
                    )

            raise RuntimeError(f"{config.title} 조회 실패: {error}")

        raise RuntimeError(f"{config.title} 응답 형식이 예상과 다릅니다.")

    def _request(self, params: Dict[str, str]) -> Any:
        response = requests.get(KOSIS_BASE_URL, params=params, timeout=self.timeout)
        response.raise_for_status()
        return response.json()

    @staticmethod
    def _split_month_range(start_yyyymm: str, end_yyyymm: str) -> tuple[str, str]:
        start_idx = KosisClient._month_index(start_yyyymm)
        end_idx = KosisClient._month_index(end_yyyymm)
        if start_idx >= end_idx:
            return start_yyyymm, end_yyyymm
        mid_idx = (start_idx + end_idx) // 2
        left_end = KosisClient._index_to_yyyymm(mid_idx)
        right_start = KosisClient._index_to_yyyymm(mid_idx + 1)
        return left_end, right_start

    @staticmethod
    def _month_index(yyyymm: str) -> int:
        dt = datetime.strptime(yyyymm, "%Y%m")
        return dt.year * 12 + (dt.month - 1)

    @staticmethod
    def _index_to_yyyymm(index: int) -> str:
        year = index // 12
        month = (index % 12) + 1
        return f"{year:04d}{month:02d}"

