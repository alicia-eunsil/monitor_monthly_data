from __future__ import annotations

import re
from typing import Dict, Iterable, Optional, Tuple

import numpy as np
import pandas as pd

from .config import DatasetConfig, TARGET_REGIONS


REGION_PATTERNS = {
    "전국": ["전국"],
    "서울특별시": ["서울특별시", "서울"],
    "부산광역시": ["부산광역시", "부산"],
    "대구광역시": ["대구광역시", "대구"],
    "인천광역시": ["인천광역시", "인천"],
    "광주광역시": ["광주광역시", "광주"],
    "대전광역시": ["대전광역시", "대전"],
    "울산광역시": ["울산광역시", "울산"],
    "세종특별자치시": ["세종특별자치시", "세종"],
    "경기도": ["경기도", "경기"],
    "강원특별자치도": ["강원특별자치도", "강원도", "강원"],
    "충청북도": ["충청북도", "충북"],
    "충청남도": ["충청남도", "충남"],
    "전라북도": ["전라북도", "전북"],
    "전라남도": ["전라남도", "전남"],
    "경상북도": ["경상북도", "경북"],
    "경상남도": ["경상남도", "경남"],
    "제주특별자치도": ["제주특별자치도", "제주"],
}


def _pick_first(columns: Iterable[str], candidates: Iterable[str]) -> Optional[str]:
    col_set = set(columns)
    for name in candidates:
        if name in col_set:
            return name
    return None


def _to_timestamp(value: str) -> pd.Timestamp:
    text = str(value).strip()
    if not text or text in {"None", "nan", "NaN", "null"}:
        return pd.NaT

    # Common monthly formats from KOSIS: YYYYMM, YYYY.MM, YYYY-MM, YYYYMmm
    if len(text) == 6 and text.isdigit():
        return pd.to_datetime(text + "01", format="%Y%m%d", errors="coerce")
    if len(text) == 4 and text.isdigit():
        return pd.to_datetime(text + "0101", format="%Y%m%d", errors="coerce")

    m = re.search(r"^(\d{4})\D*([01]?\d)$", text)
    if m:
        year = int(m.group(1))
        month = int(m.group(2))
        if 1 <= month <= 12:
            return pd.to_datetime(f"{year:04d}{month:02d}01", format="%Y%m%d", errors="coerce")

    digits = re.sub(r"\D", "", text)
    if len(digits) >= 6:
        year = int(digits[:4])
        month = int(digits[4:6])
        if 1 <= month <= 12:
            return pd.to_datetime(f"{year:04d}{month:02d}01", format="%Y%m%d", errors="coerce")

    return pd.NaT


def _to_float(value: object) -> float:
    if value is None:
        return np.nan
    text = str(value).strip().replace(",", "")
    if text in {"", "-", "null", "None"}:
        return np.nan
    try:
        return float(text)
    except ValueError:
        return np.nan


def canonical_region(raw_name: object) -> str:
    text = str(raw_name or "").strip()
    if not text:
        return ""
    if text == "계":
        return "전국"
    compact = re.sub(r"\s+", "", text)
    for canonical, patterns in REGION_PATTERNS.items():
        for p in patterns:
            if p in compact:
                return canonical
    return text


def _dimension_columns(df: pd.DataFrame) -> Tuple[list[str], list[str]]:
    name_cols = sorted(
        [c for c in df.columns if re.fullmatch(r"C\d+_(OBJ_NM|NM)", c)],
        key=lambda x: int(re.search(r"C(\d+)_", x).group(1)),
    )
    code_cols = sorted(
        [c for c in df.columns if re.fullmatch(r"C\d+_(OBJ_CD|CD)", c)],
        key=lambda x: int(re.search(r"C(\d+)_", x).group(1)),
    )
    if not name_cols and "OBJ_NM" in df.columns:
        name_cols = ["OBJ_NM"]
    if not code_cols and "OBJ_ID" in df.columns:
        code_cols = ["OBJ_ID"]
    return name_cols, code_cols


def _dim_no(name_col: str) -> Optional[str]:
    match = re.fullmatch(r"C(\d+)_.*", name_col)
    return match.group(1) if match else None


def _matching_code_col(name_col: str, code_cols: list[str]) -> Optional[str]:
    no = _dim_no(name_col)
    if no is None:
        return "OBJ_ID" if "OBJ_ID" in code_cols else None
    wanted_1 = f"C{no}_OBJ_CD"
    wanted_2 = f"C{no}_CD"
    if wanted_1 in code_cols:
        return wanted_1
    if wanted_2 in code_cols:
        return wanted_2
    return None


def _select_region_column(df: pd.DataFrame, name_cols: list[str]) -> Optional[str]:
    if not name_cols:
        return None
    best_col = name_cols[0]
    best_score = -1
    for col in name_cols:
        score = df[col].map(canonical_region).isin(TARGET_REGIONS).sum()
        if score > best_score:
            best_score = int(score)
            best_col = col
    return best_col


def _select_category_column(
    df: pd.DataFrame,
    name_cols: list[str],
    region_name_col: Optional[str],
) -> Optional[str]:
    candidates = [c for c in name_cols if c != region_name_col]
    if not candidates:
        return None

    def _uniq_count(col: str) -> int:
        series = df[col].astype(str).str.strip()
        return int(series[series != ""].nunique(dropna=True))

    # Prefer value columns like C2_NM over descriptor columns like C2_OBJ_NM.
    value_cols = [c for c in candidates if re.fullmatch(r"C\d+_NM", c)]
    value_cols = [c for c in value_cols if _uniq_count(c) > 1]
    if value_cols:
        return sorted(value_cols, key=lambda c: _uniq_count(c), reverse=True)[0]

    non_obj_cols = [c for c in candidates if not c.endswith("_OBJ_NM")]
    ranked = non_obj_cols or candidates
    return sorted(ranked, key=lambda c: _uniq_count(c), reverse=True)[0]


def normalize_records(config: DatasetConfig, records: list[Dict]) -> pd.DataFrame:
    if not records:
        return pd.DataFrame()

    df = pd.DataFrame(records)
    value_col = _pick_first(df.columns, ["DT", "DATA_VALUE", "VALUE"])
    period_col = _pick_first(df.columns, ["PRD_DE", "PRDDE"])
    indicator_id_col = _pick_first(df.columns, ["ITM_ID", "ITMID"])
    indicator_nm_col = _pick_first(df.columns, ["ITM_NM", "ITMNM"])
    unit_col = _pick_first(df.columns, ["UNIT_NM", "UNITNM"])
    name_dims, code_dims = _dimension_columns(df)

    region_name_col = _select_region_column(df, name_dims)
    region_code_col = _matching_code_col(region_name_col, code_dims) if region_name_col else None
    category_name_col = None
    category_code_col = None
    if config.has_category:
        category_name_col = _select_category_column(df, name_dims, region_name_col)
        if category_name_col:
            category_code_col = _matching_code_col(category_name_col, code_dims)

    out = pd.DataFrame(
        {
            "dataset_key": config.key,
            "dataset_title": config.title,
            "period": df[period_col].astype(str).map(_to_timestamp) if period_col else pd.NaT,
            "value": df[value_col].map(_to_float) if value_col else np.nan,
            "unit": df[unit_col].astype(str).str.strip() if unit_col else "",
            "region_code": df[region_code_col].astype(str).str.strip() if region_code_col else "",
            "region_name": df[region_name_col].astype(str).str.strip() if region_name_col else "",
            "indicator_code": df[indicator_id_col].astype(str).str.strip() if indicator_id_col else "",
            "indicator_name": df[indicator_nm_col].astype(str).str.strip() if indicator_nm_col else "",
            "category_code": (
                df[category_code_col].astype(str).str.strip() if category_code_col else ""
            ),
            "category_name": (
                df[category_name_col].astype(str).str.strip() if category_name_col else ""
            ),
        }
    )
    # KOSIS national total can appear as "계" or region code "00".
    out["region_name"] = out["region_name"].replace({"계": "전국", "합계": "전국"})
    out.loc[out["region_code"].astype(str).str.strip() == "00", "region_name"] = "전국"
    out["region_name"] = out["region_name"].map(canonical_region)
    out["indicator_name"] = out["indicator_name"].replace("", pd.NA).fillna(out["indicator_code"])
    out["indicator_name"] = out["indicator_name"].replace("", "값")
    out = out[out["region_name"].isin(TARGET_REGIONS)].copy()
    out = out.dropna(subset=["period", "value"])
    out = out.sort_values(["region_name", "indicator_name", "category_name", "period"])
    out = out.drop_duplicates(
        subset=[
            "dataset_key",
            "region_name",
            "indicator_name",
            "category_name",
            "period",
        ],
        keep="last",
    )
    out["period_yyyymm"] = out["period"].dt.strftime("%Y%m")
    return out


def add_yoy(df: pd.DataFrame) -> pd.DataFrame:
    if df.empty:
        return df
    group_cols = ["dataset_key", "region_name", "indicator_name", "category_name"]
    out = df.sort_values(group_cols + ["period"]).copy()
    out["yoy_abs"] = out.groupby(group_cols, dropna=False)["value"].transform(lambda s: s - s.shift(12))
    prev = out.groupby(group_cols, dropna=False)["value"].shift(12)
    out["yoy_pct"] = np.where(prev == 0, np.nan, (out["value"] / prev - 1.0) * 100.0)
    return out


def series_filter(
    df: pd.DataFrame,
    dataset_key: str,
    region_name: str,
    indicator_name: Optional[str] = None,
    category_name: Optional[str] = None,
) -> pd.DataFrame:
    subset = df[(df["dataset_key"] == dataset_key) & (df["region_name"] == region_name)].copy()
    if indicator_name:
        subset = subset[subset["indicator_name"] == indicator_name]
    if category_name is not None and category_name != "":
        subset = subset[subset["category_name"] == category_name]
    subset = subset.sort_values("period")
    return subset


def _slice_recent_5y(series_df: pd.DataFrame) -> pd.DataFrame:
    if series_df.empty:
        return series_df
    latest_period = series_df["period"].max()
    cutoff = latest_period - pd.DateOffset(years=5)
    return series_df[series_df["period"] >= cutoff]


def _extreme(frame: pd.DataFrame, metric_col: str, mode: str) -> Tuple[float, pd.Timestamp]:
    valid = frame.dropna(subset=[metric_col])
    if valid.empty:
        return np.nan, pd.NaT
    idx = valid[metric_col].idxmax() if mode == "max" else valid[metric_col].idxmin()
    row = valid.loc[idx]
    return float(row[metric_col]), pd.Timestamp(row["period"])


def build_stats(series_df: pd.DataFrame) -> Dict[str, object]:
    stats: Dict[str, object] = {}
    if series_df.empty:
        return stats

    latest = series_df.iloc[-1]
    recent_5y = _slice_recent_5y(series_df)

    for metric_col, prefix in [("value", "level"), ("yoy_abs", "yoy_abs"), ("yoy_pct", "yoy_pct")]:
        max_all_v, max_all_p = _extreme(series_df, metric_col, "max")
        min_all_v, min_all_p = _extreme(series_df, metric_col, "min")
        max_5_v, max_5_p = _extreme(recent_5y, metric_col, "max")
        min_5_v, min_5_p = _extreme(recent_5y, metric_col, "min")

        stats[f"{prefix}_max_all_value"] = max_all_v
        stats[f"{prefix}_max_all_period"] = max_all_p
        stats[f"{prefix}_min_all_value"] = min_all_v
        stats[f"{prefix}_min_all_period"] = min_all_p
        stats[f"{prefix}_max_5y_value"] = max_5_v
        stats[f"{prefix}_max_5y_period"] = max_5_p
        stats[f"{prefix}_min_5y_value"] = min_5_v
        stats[f"{prefix}_min_5y_period"] = min_5_p

        latest_value = float(latest[metric_col]) if pd.notna(latest[metric_col]) else np.nan
        latest_period = pd.Timestamp(latest["period"])
        stats[f"{prefix}_latest_value"] = latest_value
        stats[f"{prefix}_latest_period"] = latest_period
        stats[f"{prefix}_is_new_max_all"] = pd.notna(max_all_p) and latest_period == max_all_p
        stats[f"{prefix}_is_new_min_all"] = pd.notna(min_all_p) and latest_period == min_all_p
        stats[f"{prefix}_is_new_max_5y"] = pd.notna(max_5_p) and latest_period == max_5_p
        stats[f"{prefix}_is_new_min_5y"] = pd.notna(min_5_p) and latest_period == min_5_p

    stats["latest_period"] = pd.Timestamp(latest["period"])
    stats["latest_value"] = float(latest["value"])
    stats["latest_unit"] = str(latest.get("unit", ""))
    return stats
