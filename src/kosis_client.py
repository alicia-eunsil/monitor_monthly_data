from __future__ import annotations

from datetime import datetime
from typing import Any, Dict, List, Optional

import requests

from .config import KOSIS_BASE_URL, DatasetConfig

# National + 17 provinces/cities (standard region codes commonly used in KOSIS tables)
REGION_OBJL1_CODES = [
    "00",
    "11",
    "21",
    "22",
    "23",
    "24",
    "25",
    "26",
    "29",
    "31",
    "32",
    "33",
    "34",
    "35",
    "36",
    "37",
    "38",
    "39",
]


class KosisClient:
    _MAX_DEBUG_LOGS = 500

    def __init__(self, api_key: str, timeout: int = 30):
        self.api_key = api_key
        self.timeout = timeout
        self._debug_logs: List[str] = []

    def fetch(self, config: DatasetConfig, end_prd_de: str) -> List[Dict[str, Any]]:
        params = config.to_params(api_key=self.api_key, end_prd_de=end_prd_de)
        return self._fetch_with_fallbacks(config, params)

    def fetch_with_debug(self, config: DatasetConfig, end_prd_de: str) -> tuple[List[Dict[str, Any]], List[str]]:
        self._debug_logs = []
        params = config.to_params(api_key=self.api_key, end_prd_de=end_prd_de)
        rows = self._fetch_with_fallbacks(config, params)
        return rows, list(self._debug_logs)

    def _fetch_with_fallbacks(self, config: DatasetConfig, params: Dict[str, str]) -> List[Dict[str, Any]]:
        self._log(f"request {self._param_summary(params)}")
        payload = self._request(params)

        if isinstance(payload, list):
            self._log(f"success rows={len(payload)}")
            return payload

        if isinstance(payload, dict):
            error = str(payload.get("err") or payload.get("error") or "").strip()
            if not error:
                self._log("success single-row payload")
                return [payload]

            # KOSIS err=31: too many rows. Retry with narrower requests.
            if error == "31":
                self._log("err=31 detected; trying split strategies")
                rows = self._try_split_by_period(config, params)
                if rows is not None:
                    self._log(f"resolved by period split rows={len(rows)}")
                    return rows

                rows = self._try_split_by_item(config, params)
                if rows is not None:
                    self._log(f"resolved by item split rows={len(rows)}")
                    return rows

                rows = self._try_split_by_region(config, params)
                if rows is not None:
                    self._log(f"resolved by region split rows={len(rows)}")
                    return rows

            self._log(f"failed err={error}")
            raise RuntimeError(f"{config.title} query failed: {error}")

        self._log(f"failed unexpected_payload_type={type(payload).__name__}")
        raise RuntimeError(f"{config.title} response format is unexpected.")

    def _try_split_by_period(
        self, config: DatasetConfig, params: Dict[str, str]
    ) -> Optional[List[Dict[str, Any]]]:
        if params.get("prdSe") != "M":
            return None
        start = params.get("startPrdDe", "")
        end = params.get("endPrdDe", "")
        if not (start and end and start < end):
            return None

        left_end, right_start = self._split_month_range(start, end)
        self._log(f"period split {start}-{end} -> {start}-{left_end} | {right_start}-{end}")
        left_params = dict(params)
        right_params = dict(params)
        left_params["endPrdDe"] = left_end
        right_params["startPrdDe"] = right_start
        return self._fetch_with_fallbacks(config, left_params) + self._fetch_with_fallbacks(
            config, right_params
        )

    def _try_split_by_item(self, config: DatasetConfig, params: Dict[str, str]) -> Optional[List[Dict[str, Any]]]:
        raw_items = params.get("itmId", "")
        items = [x for x in raw_items.split("+") if x]
        if len(items) <= 1:
            return None

        mid = len(items) // 2
        left_items = items[:mid]
        right_items = items[mid:]
        self._log(
            f"item split count={len(items)} -> left={len(left_items)} right={len(right_items)}"
        )

        left_params = dict(params)
        right_params = dict(params)
        left_params["itmId"] = "+".join(left_items) + "+"
        right_params["itmId"] = "+".join(right_items) + "+"
        return self._fetch_with_fallbacks(config, left_params) + self._fetch_with_fallbacks(
            config, right_params
        )

    def _try_split_by_region(self, config: DatasetConfig, params: Dict[str, str]) -> Optional[List[Dict[str, Any]]]:
        rows = self._try_split_region_dim(config, params, "objL1")
        if rows is not None:
            return rows
        return self._try_split_region_dim(config, params, "objL2")

    def _try_split_region_dim(
        self, config: DatasetConfig, params: Dict[str, str], dim_key: str
    ) -> Optional[List[Dict[str, Any]]]:
        if params.get(dim_key) != "ALL":
            return None

        merged: List[Dict[str, Any]] = []
        success_count = 0
        self._log(f"region split on {dim_key}")
        for code in REGION_OBJL1_CODES:
            region_params = dict(params)
            region_params[dim_key] = code
            try:
                rows = self._fetch_with_fallbacks(config, region_params)
            except RuntimeError:
                continue
            if rows:
                merged.extend(rows)
                success_count += 1
                self._log(f"region {dim_key}={code} rows={len(rows)}")

        if success_count > 0:
            self._log(f"region split success {dim_key} success_count={success_count}")
            return merged
        self._log(f"region split failed {dim_key}")
        return None

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

    def _log(self, message: str) -> None:
        if len(self._debug_logs) < self._MAX_DEBUG_LOGS:
            self._debug_logs.append(message)
        elif len(self._debug_logs) == self._MAX_DEBUG_LOGS:
            self._debug_logs.append("... log truncated ...")

    @staticmethod
    def _param_summary(params: Dict[str, str]) -> str:
        items = [x for x in str(params.get("itmId", "")).split("+") if x]
        return (
            f"prd={params.get('startPrdDe','')}-{params.get('endPrdDe','')}, "
            f"itm={len(items)}, objL1={params.get('objL1','')}, objL2={params.get('objL2','')}"
        )
