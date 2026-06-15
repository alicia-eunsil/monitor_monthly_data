from __future__ import annotations

from datetime import datetime
import re
from typing import Any, Dict, List, Optional

import requests
from requests.adapters import HTTPAdapter
from requests.exceptions import ConnectTimeout, RequestException
from urllib3.util.retry import Retry
import time

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
    _MAX_MONTHS_PER_REQUEST = 60
    _MAX_HALVES_PER_REQUEST = 8
    _MAX_QUARTERS_PER_REQUEST = 20

    def __init__(self, api_key: str, timeout: int = 90):
        self.api_key = api_key
        self.timeout = timeout
        self.connect_timeout = 30
        self._debug_logs: List[str] = []
        self._request_retry_count = 3
        self._session = requests.Session()
        retry = Retry(
            total=2,
            connect=0,
            read=0,
            status=2,
            backoff_factor=0.6,
            status_forcelist=(429, 500, 502, 503, 504),
            allowed_methods=frozenset(["GET"]),
            respect_retry_after_header=True,
            raise_on_status=False,
        )
        adapter = HTTPAdapter(max_retries=retry, pool_connections=2, pool_maxsize=2)
        self._session.mount("https://", adapter)
        self._session.mount("http://", adapter)
        self._session.headers.update(
            {
                "User-Agent": (
                    "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
                    "AppleWebKit/537.36 (KHTML, like Gecko) "
                    "Chrome/123.0.0.0 Safari/537.36"
                ),
                "Accept": "application/json,text/plain,*/*",
                "Referer": "https://kosis.kr/",
                "Connection": "keep-alive",
            }
        )

    def fetch(self, config: DatasetConfig, end_prd_de: str) -> List[Dict[str, Any]]:
        params = config.to_params(api_key=self.api_key, end_prd_de=end_prd_de)
        if self._should_proactively_split_period(params):
            if str(getattr(config, "key", "")).strip() == "activity":
                params = dict(params)
                params.pop("outputFields", None)
                self._log("activity chunked request: dropped outputFields preemptively")
            self._log("large period range detected; using proactive period chunking")
            return self._fetch_chunked_by_period(config, params)
        rows = self._fetch_with_fallbacks(config, params)
        return self._resolve_dt_only_rows(config, params, rows)

    def fetch_with_debug(self, config: DatasetConfig, end_prd_de: str) -> tuple[List[Dict[str, Any]], List[str]]:
        self._debug_logs = []
        rows = self.fetch(config, end_prd_de=end_prd_de)
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
        prd_se = str(params.get("prdSe", "")).upper()
        if prd_se not in {"M", "H", "Q"}:
            return None
        start = params.get("startPrdDe", "")
        end = params.get("endPrdDe", "")
        if not (start and end and start < end):
            return None

        if prd_se == "H":
            left_end, right_start = self._split_halfyear_range(start, end)
        elif prd_se == "Q":
            left_end, right_start = self._split_quarter_range(start, end)
        else:
            left_end, right_start = self._split_month_range(start, end)
        self._log(f"period split {start}-{end} -> {start}-{left_end} | {right_start}-{end}")
        left_params = dict(params)
        right_params = dict(params)
        left_params["endPrdDe"] = left_end
        right_params["startPrdDe"] = right_start
        return self._fetch_with_fallbacks(config, left_params) + self._fetch_with_fallbacks(
            config, right_params
        )

    def _fetch_chunked_by_period(self, config: DatasetConfig, params: Dict[str, str]) -> List[Dict[str, Any]]:
        prd_se = str(params.get("prdSe", "")).upper()
        start = params.get("startPrdDe", "")
        end = params.get("endPrdDe", "")
        ranges = self._chunk_period_ranges(start, end, prd_se)
        merged: List[Dict[str, Any]] = []
        for chunk_start, chunk_end in ranges:
            merged.extend(self._fetch_chunk_range(config, params, chunk_start, chunk_end))
            time.sleep(0.25)
        return merged

    def _fetch_chunk_range(
        self,
        config: DatasetConfig,
        base_params: Dict[str, str],
        chunk_start: str,
        chunk_end: str,
    ) -> List[Dict[str, Any]]:
        chunk_params = dict(base_params)
        chunk_params["startPrdDe"] = chunk_start
        chunk_params["endPrdDe"] = chunk_end
        self._log(f"chunk request {chunk_start}-{chunk_end}")
        try:
            chunk_rows = self._fetch_with_fallbacks(config, chunk_params)
            return self._resolve_dt_only_rows(config, chunk_params, chunk_rows)
        except Exception as exc:  # noqa: BLE001
            prd_se = str(chunk_params.get("prdSe", "")).upper()
            if not self._can_split_period_range(chunk_start, chunk_end, prd_se):
                self._log(f"chunk request failed without further split {chunk_start}-{chunk_end}: {exc}")
                raise
            self._log(f"chunk request failed; splitting smaller {chunk_start}-{chunk_end}: {exc}")
            if prd_se == "H":
                left_end, right_start = self._split_halfyear_range(chunk_start, chunk_end)
            elif prd_se == "Q":
                left_end, right_start = self._split_quarter_range(chunk_start, chunk_end)
            else:
                left_end, right_start = self._split_month_range(chunk_start, chunk_end)
            left_rows = self._fetch_chunk_range(config, base_params, chunk_start, left_end)
            right_rows = self._fetch_chunk_range(config, base_params, right_start, chunk_end)
            return left_rows + right_rows

    def _resolve_dt_only_rows(
        self,
        config: DatasetConfig,
        params: Dict[str, str],
        rows: List[Dict[str, Any]],
    ) -> List[Dict[str, Any]]:
        if not self._is_dt_only_rows(rows):
            return rows
        self._log("dt-only payload detected; trying alternate outputFields")
        for output_fields in [
            "",
            "TBL_ID+TBL_NM+OBJ_ID+OBJ_NM+ITM_ID+ITM_NM+UNIT_NM+PRD_SE+PRD_DE+",
            "TBL_ID+TBL_NM+ITM_ID+ITM_NM+PRD_SE+PRD_DE+DT+OBJ_ID+OBJ_NM+",
        ]:
            alt_params = dict(params)
            if output_fields:
                alt_params["outputFields"] = output_fields
            else:
                alt_params.pop("outputFields", None)
            alt_rows = self._fetch_with_fallbacks(config, alt_params)
            if not self._is_dt_only_rows(alt_rows):
                self._log("dt-only resolved with alternate outputFields")
                return alt_rows
        self._log("dt-only still unresolved after alternate outputFields")
        return rows

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
        last_error: Optional[Exception] = None
        for attempt in range(1, self._request_retry_count + 1):
            try:
                response = self._session.get(
                    KOSIS_BASE_URL,
                    params=params,
                    timeout=(self.connect_timeout, self.timeout),
                )
                response.raise_for_status()
                return response.json()
            except ConnectTimeout as exc:
                last_error = exc
                self._log(
                    f"connect_timeout attempt={attempt}/{self._request_retry_count}: {exc}"
                )
                if attempt < self._request_retry_count:
                    sleep_sec = 1.5 * attempt
                    time.sleep(sleep_sec)
                    continue
                raise
            except RequestException as exc:
                last_error = exc
                self._log(f"http_error attempt={attempt}/{self._request_retry_count}: {exc}")
                if attempt < self._request_retry_count:
                    sleep_sec = 0.8 * (2 ** (attempt - 1))
                    time.sleep(sleep_sec)
                    continue
                raise
            except ValueError as exc:
                last_error = exc
                self._log(f"json_decode_error attempt={attempt}/{self._request_retry_count}: {exc}")
                if attempt < self._request_retry_count:
                    time.sleep(0.5)
                    continue
                raise

        if last_error is not None:
            raise last_error
        raise RuntimeError("request failed without explicit exception")

    def _should_proactively_split_period(self, params: Dict[str, str]) -> bool:
        prd_se = str(params.get("prdSe", "")).upper()
        start = str(params.get("startPrdDe", "")).strip()
        end = str(params.get("endPrdDe", "")).strip()
        if not start or not end:
            return False
        try:
            if prd_se == "M":
                span = self._month_index(end) - self._month_index(start) + 1
                return span > self._MAX_MONTHS_PER_REQUEST
            if prd_se == "H":
                span = self._half_index(end) - self._half_index(start) + 1
                return span > self._MAX_HALVES_PER_REQUEST
            if prd_se == "Q":
                span = self._quarter_index(end) - self._quarter_index(start) + 1
                return span > self._MAX_QUARTERS_PER_REQUEST
        except ValueError:
            return False
        return False

    def _chunk_period_ranges(self, start: str, end: str, prd_se: str) -> List[tuple[str, str]]:
        if str(prd_se).upper() == "H":
            max_units = self._MAX_HALVES_PER_REQUEST
            start_idx = self._half_index(start)
            end_idx = self._half_index(end)
            to_text = self._index_to_yyyyhh
        elif str(prd_se).upper() == "Q":
            max_units = self._MAX_QUARTERS_PER_REQUEST
            start_idx = self._quarter_index(start)
            end_idx = self._quarter_index(end)
            to_text = self._index_to_yyyyqq
        else:
            max_units = self._MAX_MONTHS_PER_REQUEST
            start_idx = self._month_index(start)
            end_idx = self._month_index(end)
            to_text = self._index_to_yyyymm

        ranges: List[tuple[str, str]] = []
        cursor = start_idx
        while cursor <= end_idx:
            chunk_end_idx = min(cursor + max_units - 1, end_idx)
            ranges.append((to_text(cursor), to_text(chunk_end_idx)))
            cursor = chunk_end_idx + 1
        return ranges

    def _can_split_period_range(self, start: str, end: str, prd_se: str) -> bool:
        try:
            if str(prd_se).upper() == "H":
                return self._half_index(start) < self._half_index(end)
            if str(prd_se).upper() == "Q":
                return self._quarter_index(start) < self._quarter_index(end)
            return self._month_index(start) < self._month_index(end)
        except ValueError:
            return False

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
    def _split_halfyear_range(start_yyyyhh: str, end_yyyyhh: str) -> tuple[str, str]:
        start_idx = KosisClient._half_index(start_yyyyhh)
        end_idx = KosisClient._half_index(end_yyyyhh)
        if start_idx >= end_idx:
            return start_yyyyhh, end_yyyyhh
        mid_idx = (start_idx + end_idx) // 2
        left_end = KosisClient._index_to_yyyyhh(mid_idx)
        right_start = KosisClient._index_to_yyyyhh(mid_idx + 1)
        return left_end, right_start

    @staticmethod
    def _split_quarter_range(start_yyyyqq: str, end_yyyyqq: str) -> tuple[str, str]:
        start_idx = KosisClient._quarter_index(start_yyyyqq)
        end_idx = KosisClient._quarter_index(end_yyyyqq)
        if start_idx >= end_idx:
            return start_yyyyqq, end_yyyyqq
        mid_idx = (start_idx + end_idx) // 2
        left_end = KosisClient._index_to_yyyyqq(mid_idx)
        right_start = KosisClient._index_to_yyyyqq(mid_idx + 1)
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

    @staticmethod
    def _half_index(yyyyhh: str) -> int:
        text = str(yyyyhh).strip()
        if len(text) == 6:
            year = int(text[:4])
            half = text[4:6]
            if half == "01":
                return year * 2
            if half == "02":
                return year * 2 + 1
        if len(text) == 5 and text[-1] in {"1", "2"}:
            year = int(text[:4])
            return year * 2 + (int(text[-1]) - 1)
        raise ValueError(f"Unsupported half-year format: {yyyyhh}")

    @staticmethod
    def _index_to_yyyyhh(index: int) -> str:
        year = index // 2
        half = "01" if (index % 2) == 0 else "02"
        return f"{year:04d}{half}"

    @staticmethod
    def _quarter_index(yyyyqq: str) -> int:
        text = str(yyyyqq).strip()
        match = re.match(r"^(\d{4})\D*[Qq]?([1-4])$", text)
        if match:
            year = int(match.group(1))
            quarter = int(match.group(2))
            return year * 4 + (quarter - 1)
        digits = re.sub(r"\D", "", text)
        if len(digits) == 6 and digits[4:6] in {"01", "02", "03", "04"}:
            year = int(digits[:4])
            quarter = int(digits[4:6])
            return year * 4 + (quarter - 1)
        if len(digits) == 5 and digits[-1] in {"1", "2", "3", "4"}:
            year = int(digits[:4])
            quarter = int(digits[-1])
            return year * 4 + (quarter - 1)
        raise ValueError(f"Unsupported quarter format: {yyyyqq}")

    @staticmethod
    def _index_to_yyyyqq(index: int) -> str:
        year = index // 4
        quarter = (index % 4) + 1
        return f"{year:04d}Q{quarter}"

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
            f"itm={len(items)}, objL1={params.get('objL1','')}, objL2={params.get('objL2','')}, "
            f"outputFields={'Y' if 'outputFields' in params else 'N'}"
        )

    @staticmethod
    def _is_dt_only_rows(rows: List[Dict[str, Any]]) -> bool:
        if not rows:
            return False
        sample = rows[0]
        if not isinstance(sample, dict):
            return False
        keys = set(sample.keys())
        return keys == {"DT"}
