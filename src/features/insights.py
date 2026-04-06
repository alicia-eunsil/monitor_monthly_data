from typing import Any, Callable, Dict, List, Optional
from uuid import uuid4

import altair as alt
import numpy as np
import pandas as pd
import streamlit as st
import os

import src.config as app_config
from src.core.category_rules import ACTIVITY_INDICATOR_ORDER, norm_indicator_name
from src.core.formatters import escape_markdown_text, fmt_num, fmt_num_bold, fmt_period
from src.features.new_history import build_ai_insight_context
from src.services.insight_memory import build_prompt, compute_hash, load_memory, save_memory, select_memory_context
from src.services.openai_client import create_response_text

TARGET_REGIONS = app_config.TARGET_REGIONS
GYEONGGI_SIGUNGU = getattr(app_config, "GYEONGGI_SIGUNGU", [])


def _seeded_openai_key() -> str:
    try:
        secret_value = st.secrets.get("OPENAI_API_KEY", "") or st.secrets.get("openai_api_key", "")
    except Exception:
        secret_value = ""
    return str(secret_value or os.getenv("OPENAI_API_KEY", "") or os.getenv("openai_api_key", ""))


def _auto_summary_from_insight(text: str, max_lines: int = 3) -> str:
    if not text:
        return ""
    lines = [ln.strip() for ln in text.split("\n") if ln.strip()]
    cleaned: List[str] = []
    for ln in lines:
        if ln.startswith("#"):
            continue
        cleaned.append(ln)
    if not cleaned:
        cleaned = lines
    if not cleaned:
        return ""
    # Strip leading markdown markers if present.
    cleaned = [ln.lstrip("#").strip() for ln in cleaned if ln.lstrip("#").strip()]
    bullets = [ln for ln in cleaned if ln.startswith("-")]
    if bullets:
        return "\n".join(bullets[: max_lines])
    return " ".join(cleaned[: max_lines])[:240]


def _build_rule_based_insights(
    context: Dict[str, Any],
    region: str,
) -> List[str]:
    lines: List[str] = []
    scope_title = str(context.get("scope_title", "")).strip()
    selected_month = str(context.get("selected_month", "")).strip()
    stats = context.get("stats", {}) or {}
    total_events = int(stats.get("total_events", 0))
    max_events = int(stats.get("max_events", 0))
    min_events = int(stats.get("min_events", 0))
    prev_total = stats.get("prev_total_events")
    prev_max = stats.get("prev_max_events")
    prev_min = stats.get("prev_min_events")
    dominant = "최고 NEW" if max_events >= min_events else "최저 NEW"
    max_share = (max_events / total_events * 100.0) if total_events else 0.0
    min_share = (min_events / total_events * 100.0) if total_events else 0.0

    def _delta(cur: int, prev: object) -> str:
        if prev is None:
            return ""
        diff = int(cur) - int(prev)
        if diff > 0:
            return f" (+{diff:,})"
        if diff < 0:
            return f" (-{abs(diff):,})"
        return " (=)"

    if selected_month:
        delta_text = _delta(total_events, prev_total)
        lines.append(
            f"- {selected_month} 기준 {scope_title}({region}) NEW 이벤트는 **{total_events:,}건**{delta_text}입니다."
        )
    if total_events > 0:
        lines.append(
            f"- NEW 분포는 최고 **{max_events:,}건**({_delta(max_events, prev_max)}), "
            f"최저 **{min_events:,}건**({_delta(min_events, prev_min)})이며 "
            f"비중은 최고 {max_share:.1f}%, 최저 {min_share:.1f}%로 **{dominant} 우세**입니다."
        )

    focus_lines = [ln for ln in context.get("focus_lines", []) if str(ln).strip()]
    for ln in focus_lines[:2]:
        lines.append(f"- {str(ln).lstrip('-').strip()}")

    consecutive = [ln for ln in context.get("consecutive_lines", []) if str(ln).strip()]
    if consecutive:
        lines.append(f"- 연속 변화 신호: {consecutive[0].lstrip('-').strip()}")

    # Risk / next checks
    risk_lines: List[str] = []
    if min_events > max_events:
        risk_lines.append("최저 NEW 비중이 높아 리스크 신호가 상대적으로 우세합니다.")
    if total_events == 0:
        risk_lines.append("NEW 이벤트가 없어 뚜렷한 신호가 제한적입니다.")
    if consecutive:
        if "감소" in consecutive[0]:
            risk_lines.append("연속 감소 신호가 포함되어 있어 감소 흐름 지속 여부 점검이 필요합니다.")
        if "증가" in consecutive[0]:
            risk_lines.append("연속 증가 신호가 포함되어 상승 흐름 지속 여부 점검이 필요합니다.")
    if risk_lines:
        lines.append("- 의미/리스크: " + " ".join(risk_lines))

    lines.append(
        "- 다음 점검: 최고/최저 NEW 비중 변화, 연속 변화 신호의 지속 여부, "
        "최근 10년/5년 범위의 NEW 발생 지표가 반복되는지 확인하세요."
    )

    if not lines:
        lines.append("- 규칙 기반 인사이트를 만들 데이터가 부족합니다.")
    return lines


def pick_employment_indicator(indicators: List[str]) -> str:
    if not indicators:
        return ""
    for token in ["취업자수", "취업자", "취업"]:
        for name in indicators:
            if token in str(name):
                return str(name)
    return str(indicators[0])


def find_prev_period(periods: List[pd.Timestamp], latest: pd.Timestamp, lag: int) -> Optional[pd.Timestamp]:
    sorted_periods = sorted(pd.to_datetime(periods).dropna().tolist())
    if latest not in sorted_periods:
        return None
    idx = sorted_periods.index(latest)
    if idx < lag:
        return None
    return pd.Timestamp(sorted_periods[idx - lag])


def compute_contribution_table(
    df: pd.DataFrame,
    region: str,
    dataset_key: str,
    lag: int,
) -> tuple[pd.DataFrame, Dict[str, Any]]:
    meta: Dict[str, Any] = {
        "ok": False,
        "message": "",
        "dataset_title": "",
        "indicator": "",
        "latest_period": pd.NaT,
        "prev_period": pd.NaT,
        "total_delta": np.nan,
        "unit": "",
    }
    subset = df[(df["dataset_key"] == dataset_key) & (df["region_name"] == region)].copy()
    if subset.empty:
        meta["message"] = "해당 지역 데이터가 없습니다."
        return pd.DataFrame(), meta

    meta["dataset_title"] = str(subset["dataset_title"].iloc[0])
    indicators = sorted(subset["indicator_name"].dropna().unique().tolist())
    indicator = pick_employment_indicator(indicators)
    meta["indicator"] = indicator
    subset = subset[subset["indicator_name"] == indicator].copy()
    if subset.empty:
        meta["message"] = "취업자 지표 데이터가 없습니다."
        return pd.DataFrame(), meta

    categories = [c for c in subset["category_name"].dropna().unique().tolist() if str(c).strip()]
    if not categories:
        meta["message"] = "분해 가능한 분류 데이터가 없습니다."
        return pd.DataFrame(), meta

    periods = sorted(subset["period"].dropna().unique().tolist())
    if not periods:
        meta["message"] = "기간 데이터가 없습니다."
        return pd.DataFrame(), meta
    latest_period = pd.Timestamp(periods[-1])
    prev_period = find_prev_period(periods, latest_period, lag)
    if prev_period is None:
        meta["message"] = "비교 가능한 이전 시점 데이터가 부족합니다."
        return pd.DataFrame(), meta

    latest_df = subset[subset["period"] == latest_period][["category_name", "value", "unit"]].copy()
    prev_df = subset[subset["period"] == prev_period][["category_name", "value"]].copy()
    merged = latest_df.merge(prev_df, on="category_name", how="outer", suffixes=("_latest", "_prev"))
    merged["value_latest"] = pd.to_numeric(merged["value_latest"], errors="coerce")
    merged["value_prev"] = pd.to_numeric(merged["value_prev"], errors="coerce")
    merged["delta"] = merged["value_latest"] - merged["value_prev"]
    merged["category_name"] = merged["category_name"].astype(str).str.strip()
    merged = merged.dropna(subset=["delta"]).copy()
    if merged.empty:
        meta["message"] = "증감 계산 가능한 데이터가 없습니다."
        return pd.DataFrame(), meta

    total_row = merged[merged["category_name"].isin(["계", "합계", "전체"])]
    if not total_row.empty:
        total_delta = float(total_row["delta"].iloc[0])
    else:
        total_delta = float(merged["delta"].sum())
    merged["기여율(%)"] = np.where(total_delta == 0, np.nan, (merged["delta"] / total_delta) * 100.0)

    unit = ""
    unit_series = latest_df["unit"].dropna()
    if not unit_series.empty:
        unit = str(unit_series.iloc[0])

    view_df = merged.copy()
    view_df = view_df[~view_df["category_name"].isin(["계", "합계", "전체"])].copy()
    view_df = view_df.sort_values("delta", ascending=False)
    out_df = pd.DataFrame(
        {
            "분류": view_df["category_name"],
            "최신값": view_df["value_latest"].round(2),
            "비교값": view_df["value_prev"].round(2),
            "증감": view_df["delta"].round(2),
            "기여율(%)": view_df["기여율(%)"].round(2),
        }
    )

    meta.update(
        {
            "ok": True,
            "latest_period": latest_period,
            "prev_period": prev_period,
            "total_delta": total_delta,
            "unit": unit,
        }
    )
    return out_df, meta


def build_ai_contribution_commentary(table: pd.DataFrame, meta: Dict[str, Any], point_label: str, yoy_label: str) -> str:
    if table.empty or not meta.get("ok"):
        return "AI 해설을 생성할 데이터가 부족합니다."
    total_delta = float(meta.get("total_delta", np.nan))
    unit = str(meta.get("unit", ""))
    direction = "증가" if total_delta > 0 else "감소" if total_delta < 0 else "보합"
    latest_p = fmt_period(meta.get("latest_period"), "M" if point_label == "월" else "H")
    prev_p = fmt_period(meta.get("prev_period"), "M" if point_label == "월" else "H")
    top_pos = table[table["증감"] > 0].nlargest(3, "증감")
    top_neg = table[table["증감"] < 0].nsmallest(3, "증감")
    dominant = table.loc[table["증감"].abs().idxmax()] if not table.empty else None

    lines: List[str] = []

    def _fmt_pct_text(v: object) -> str:
        return "-" if v is None or pd.isna(v) else f"{float(v):,.1f}%"

    lines.append(
        f"- {latest_p} 기준 {meta['dataset_title']}({meta['indicator']})은 {prev_p} 대비 "
        f"총 {fmt_num_bold(total_delta, unit)} {direction}했습니다."
    )
    if not top_pos.empty:
        pos_text = ", ".join(
            [
                f"{escape_markdown_text(r['분류'])}({fmt_num(r['증감'], unit)}, {_fmt_pct_text(r['기여율(%)'])})"
                for _, r in top_pos.iterrows()
            ]
        )
        lines.append(f"- 증가요인 상위는 {pos_text} 입니다.")
    if not top_neg.empty:
        neg_text = ", ".join(
            [
                f"{escape_markdown_text(r['분류'])}({fmt_num(r['증감'], unit)}, {_fmt_pct_text(r['기여율(%)'])})"
                for _, r in top_neg.iterrows()
            ]
        )
        lines.append(f"- 감소요인 상위는 {neg_text} 입니다.")
    if dominant is not None:
        lines.append(
            f"- 가장 큰 변동은 **{dominant['분류']}**로, {yoy_label} 대비 "
            f"{fmt_num_bold(dominant['증감'], unit)} 변화했습니다."
        )
        lines.append(f"- 다음 점검 포인트: **{dominant['분류']}**의 변동이 다음 {point_label}에도 이어지는지 확인하세요.")
    lines.append(f"- 해석 기준은 `{yoy_label} 대비`이며, 기여율은 총증감 대비 비중입니다.")
    return "\n".join(lines)


def infer_lag_from_df(df: pd.DataFrame) -> int:
    if "prd_se" in df.columns and not df["prd_se"].dropna().empty:
        return 2 if str(df["prd_se"].dropna().iloc[0]).upper() == "H" else 12
    return 12


def build_activity_snapshot(df: pd.DataFrame, region: str, lag: int) -> tuple[pd.DataFrame, Dict[str, Any]]:
    meta: Dict[str, Any] = {
        "ok": False,
        "message": "",
        "prd_se": "M",
        "latest_period": pd.NaT,
        "prev_period": pd.NaT,
    }
    base = df[(df["dataset_key"] == "activity") & (df["region_name"] == region)].copy()
    if base.empty:
        meta["message"] = "경제활동인구현황 데이터가 없습니다."
        return pd.DataFrame(), meta
    base["period"] = pd.to_datetime(base["period"], errors="coerce")
    base = base.dropna(subset=["period"])
    if base.empty:
        meta["message"] = "기간 데이터가 없습니다."
        return pd.DataFrame(), meta
    if "prd_se" in base.columns and not base["prd_se"].dropna().empty:
        meta["prd_se"] = str(base["prd_se"].dropna().iloc[0]).upper()
    periods = sorted(base["period"].dropna().unique().tolist())
    latest_period = pd.Timestamp(periods[-1])
    prev_period = find_prev_period(periods, latest_period, lag)
    if prev_period is None:
        meta["message"] = "전년 비교 가능한 시점이 부족합니다."
        return pd.DataFrame(), meta

    base["norm_indicator"] = base["indicator_name"].apply(norm_indicator_name)
    rows: List[Dict[str, Any]] = []
    for ind in ACTIVITY_INDICATOR_ORDER:
        norm = norm_indicator_name(ind)
        ind_df = base[base["norm_indicator"] == norm].copy()
        if ind_df.empty:
            continue
        latest_rows = ind_df[ind_df["period"] == latest_period]
        prev_rows = ind_df[ind_df["period"] == prev_period]
        latest_val = pd.to_numeric(latest_rows["value"], errors="coerce").mean()
        prev_val = pd.to_numeric(prev_rows["value"], errors="coerce").mean()
        unit_candidates = pd.concat([latest_rows["unit"], prev_rows["unit"]], ignore_index=True)
        unit = str(unit_candidates.dropna().iloc[0]) if not unit_candidates.dropna().empty else ""
        rows.append(
            {
                "지표": ind,
                "norm_indicator": norm,
                "latest_value": latest_val,
                "prev_value": prev_val,
                "delta_value": latest_val - prev_val if pd.notna(latest_val) and pd.notna(prev_val) else np.nan,
                "unit": unit,
            }
        )
    out = pd.DataFrame(rows)
    if out.empty:
        meta["message"] = "표시 가능한 지표 데이터가 없습니다."
        return out, meta
    meta.update({"ok": True, "latest_period": latest_period, "prev_period": prev_period})
    return out, meta


def fmt_contrib_items(table: pd.DataFrame, unit: str, positive: bool, top_n: int = 3) -> str:
    if table.empty:
        return "-"
    if positive:
        view = table[table["증감"] > 0].nlargest(top_n, "증감")
    else:
        view = table[table["증감"] < 0].nsmallest(top_n, "증감")
    if view.empty:
        return "-"
    items = []
    for _, r in view.iterrows():
        pct_text = "-" if pd.isna(r.get("기여율(%)")) else f"{float(r['기여율(%)']):,.1f}%"
        items.append(f"{escape_markdown_text(r['분류'])}({fmt_num(r['증감'], unit)}, {pct_text})")
    return ", ".join(items)


def compute_gyeonggi_vs_national_contribution(
    df: pd.DataFrame,
    region_name: str = "경기도",
    base_region: str = "전국",
) -> tuple[pd.DataFrame, Dict[str, Any]]:
    meta: Dict[str, Any] = {
        "ok": False,
        "message": "",
        "region_name": str(region_name),
        "base_region": str(base_region),
        "base_region_derived": False,
        "indicator": "",
        "prd_se": "M",
        "latest_period": pd.NaT,
        "latest_nat_value": np.nan,
        "latest_gg_value": np.nan,
        "latest_share_pct": np.nan,
        "latest_contrib_pct": np.nan,
        "latest_nat_yoy_abs": np.nan,
        "latest_gg_yoy_abs": np.nan,
        "prev_year_period": pd.NaT,
        "prev_year_nat_value": np.nan,
        "prev_year_gg_value": np.nan,
        "prev_year_share_pct": np.nan,
        "share_yoy_change_pp": np.nan,
        "prev_year_contrib_pct": np.nan,
        "contrib_yoy_change_pp": np.nan,
        "recent_start_period": pd.NaT,
        "recent_start_share_pct": np.nan,
        "recent_end_share_pct": np.nan,
        "recent_change_pp": np.nan,
        "recent_max_period": pd.NaT,
        "recent_max_share_pct": np.nan,
        "recent_min_period": pd.NaT,
        "recent_min_share_pct": np.nan,
        "share_hist_percentile": np.nan,
        "contrib_abs_hist_percentile": np.nan,
        "share_change_pp": np.nan,
        "contrib_change_pp": np.nan,
        "same_direction_streak": 1,
        "unit": "",
    }

    base = df[df["dataset_key"] == "activity"].copy()
    if base.empty:
        meta["message"] = "경제활동인구현황 데이터가 없습니다."
        return pd.DataFrame(), meta

    indicators = sorted(base["indicator_name"].dropna().unique().tolist())
    indicator = pick_employment_indicator(indicators)
    meta["indicator"] = indicator
    base = base[base["indicator_name"] == indicator].copy()
    if base.empty:
        meta["message"] = "취업자 지표를 찾지 못했습니다."
        return pd.DataFrame(), meta

    region_names = base["region_name"].dropna().astype(str).str.strip().unique().tolist()
    national_candidate = str(base_region).strip() or (TARGET_REGIONS[0] if len(TARGET_REGIONS) >= 1 else "")
    target_region_candidate = str(region_name).strip()
    if target_region_candidate not in region_names:
        meta["message"] = f"{target_region_candidate} 데이터를 찾지 못했습니다."
        return pd.DataFrame(), meta

    if national_candidate in region_names:
        nat = (
            base[base["region_name"] == national_candidate]
            .groupby("period", as_index=False)
            .agg({"value": "sum", "yoy_abs": "sum"})
            .rename(columns={"value": "value_nat", "yoy_abs": "yoy_abs_nat"})
        )
    else:
        # Derive base totals from all regions in the provided scope.
        nat = (
            base.groupby("period", as_index=False)
            .agg({"value": "sum", "yoy_abs": "sum"})
            .rename(columns={"value": "value_nat", "yoy_abs": "yoy_abs_nat"})
        )
        meta["base_region_derived"] = True
    gg = (
        base[base["region_name"] == target_region_candidate]
        .groupby("period", as_index=False)
        .agg({"value": "sum", "yoy_abs": "sum"})
        .rename(columns={"value": "value_gg", "yoy_abs": "yoy_abs_gg"})
    )
    trend = nat.merge(gg, on="period", how="inner").sort_values("period")
    if trend.empty:
        meta["message"] = f"{national_candidate}/{target_region_candidate} 공통 시계열 구간이 없습니다."
        return pd.DataFrame(), meta

    prd_se = "M"
    if "prd_se" in base.columns and not base["prd_se"].dropna().empty:
        prd_se = str(base["prd_se"].dropna().iloc[0]).upper()
    lag = 12 if prd_se == "M" else 2

    trend["share_pct"] = np.where(trend["value_nat"] == 0, np.nan, trend["value_gg"] / trend["value_nat"] * 100.0)
    trend["contrib_pct"] = np.where(trend["yoy_abs_nat"] == 0, np.nan, trend["yoy_abs_gg"] / trend["yoy_abs_nat"] * 100.0)

    if len(trend) >= 2:
        trend["share_change_pp"] = trend["share_pct"].diff()
        trend["contrib_change_pp"] = trend["contrib_pct"].diff()
    else:
        trend["share_change_pp"] = np.nan
        trend["contrib_change_pp"] = np.nan

    latest_row = trend.iloc[-1]
    unit = ""
    unit_series = base["unit"].dropna()
    if not unit_series.empty:
        unit = str(unit_series.iloc[0])

    prev_year_row = None
    if len(trend) > lag:
        prev_year_row = trend.iloc[-1 - lag]

    recent_n = 12 if prd_se == "M" else 6
    recent = trend.tail(recent_n).copy()
    recent_start_row = recent.iloc[0] if not recent.empty else None
    recent_max_row = recent.loc[recent["share_pct"].idxmax()] if not recent["share_pct"].dropna().empty else None
    recent_min_row = recent.loc[recent["share_pct"].idxmin()] if not recent["share_pct"].dropna().empty else None

    share_series = trend["share_pct"].dropna()
    share_percentile = np.nan
    if not share_series.empty and pd.notna(latest_row["share_pct"]):
        share_percentile = float((share_series <= latest_row["share_pct"]).mean() * 100.0)

    contrib_abs_series = trend["contrib_pct"].abs().dropna()
    contrib_abs_percentile = np.nan
    latest_contrib_abs = abs(float(latest_row["contrib_pct"])) if pd.notna(latest_row["contrib_pct"]) else np.nan
    if not contrib_abs_series.empty and pd.notna(latest_contrib_abs):
        contrib_abs_percentile = float((contrib_abs_series <= latest_contrib_abs).mean() * 100.0)

    streak = 1
    if len(trend) >= 2:
        sign_series = np.sign(pd.to_numeric(trend["yoy_abs_gg"], errors="coerce").fillna(0))
        latest_sign = sign_series.iloc[-1]
        if latest_sign != 0:
            streak = 0
            for val in reversed(sign_series.tolist()):
                if val == latest_sign:
                    streak += 1
                else:
                    break

    meta.update(
        {
            "ok": True,
            "prd_se": prd_se,
            "latest_period": pd.Timestamp(latest_row["period"]),
            "latest_nat_value": float(latest_row["value_nat"]) if pd.notna(latest_row["value_nat"]) else np.nan,
            "latest_gg_value": float(latest_row["value_gg"]) if pd.notna(latest_row["value_gg"]) else np.nan,
            "latest_share_pct": float(latest_row["share_pct"]) if pd.notna(latest_row["share_pct"]) else np.nan,
            "latest_contrib_pct": float(latest_row["contrib_pct"]) if pd.notna(latest_row["contrib_pct"]) else np.nan,
            "latest_nat_yoy_abs": float(latest_row["yoy_abs_nat"]) if pd.notna(latest_row["yoy_abs_nat"]) else np.nan,
            "latest_gg_yoy_abs": float(latest_row["yoy_abs_gg"]) if pd.notna(latest_row["yoy_abs_gg"]) else np.nan,
            "prev_year_period": pd.Timestamp(prev_year_row["period"]) if prev_year_row is not None else pd.NaT,
            "prev_year_nat_value": float(prev_year_row["value_nat"]) if prev_year_row is not None and pd.notna(prev_year_row["value_nat"]) else np.nan,
            "prev_year_gg_value": float(prev_year_row["value_gg"]) if prev_year_row is not None and pd.notna(prev_year_row["value_gg"]) else np.nan,
            "prev_year_share_pct": float(prev_year_row["share_pct"]) if prev_year_row is not None and pd.notna(prev_year_row["share_pct"]) else np.nan,
            "share_yoy_change_pp": (
                float(latest_row["share_pct"] - prev_year_row["share_pct"])
                if prev_year_row is not None and pd.notna(latest_row["share_pct"]) and pd.notna(prev_year_row["share_pct"])
                else np.nan
            ),
            "prev_year_contrib_pct": float(prev_year_row["contrib_pct"]) if prev_year_row is not None and pd.notna(prev_year_row["contrib_pct"]) else np.nan,
            "contrib_yoy_change_pp": (
                float(latest_row["contrib_pct"] - prev_year_row["contrib_pct"])
                if prev_year_row is not None and pd.notna(latest_row["contrib_pct"]) and pd.notna(prev_year_row["contrib_pct"])
                else np.nan
            ),
            "recent_start_period": pd.Timestamp(recent_start_row["period"]) if recent_start_row is not None else pd.NaT,
            "recent_start_share_pct": float(recent_start_row["share_pct"]) if recent_start_row is not None and pd.notna(recent_start_row["share_pct"]) else np.nan,
            "recent_end_share_pct": float(latest_row["share_pct"]) if pd.notna(latest_row["share_pct"]) else np.nan,
            "recent_change_pp": (
                float(latest_row["share_pct"] - recent_start_row["share_pct"])
                if recent_start_row is not None and pd.notna(latest_row["share_pct"]) and pd.notna(recent_start_row["share_pct"])
                else np.nan
            ),
            "recent_max_period": pd.Timestamp(recent_max_row["period"]) if recent_max_row is not None else pd.NaT,
            "recent_max_share_pct": float(recent_max_row["share_pct"]) if recent_max_row is not None and pd.notna(recent_max_row["share_pct"]) else np.nan,
            "recent_min_period": pd.Timestamp(recent_min_row["period"]) if recent_min_row is not None else pd.NaT,
            "recent_min_share_pct": float(recent_min_row["share_pct"]) if recent_min_row is not None and pd.notna(recent_min_row["share_pct"]) else np.nan,
            "share_hist_percentile": share_percentile,
            "contrib_abs_hist_percentile": contrib_abs_percentile,
            "share_change_pp": float(latest_row["share_change_pp"]) if pd.notna(latest_row["share_change_pp"]) else np.nan,
            "contrib_change_pp": float(latest_row["contrib_change_pp"]) if pd.notna(latest_row["contrib_change_pp"]) else np.nan,
            "same_direction_streak": int(streak),
            "unit": unit,
        }
    )
    return trend, meta


def build_ai_gyeonggi_contribution_commentary(meta: Dict[str, Any], labels: Dict[str, str]) -> str:
    if not meta.get("ok"):
        return str(meta.get("message", "전국 대비 경기도 기여도를 계산할 수 없습니다."))

    base_region = str(meta.get("base_region", "전국"))
    region_name = str(meta.get("region_name", "경기도"))
    point = labels.get("point", "월")
    yoy = labels.get("yoy", "전년동월")
    prd_se = str(meta.get("prd_se", "M")).upper()
    latest = fmt_period(meta.get("latest_period"), prd_se)
    latest_nat_value = meta.get("latest_nat_value")
    latest_gg_value = meta.get("latest_gg_value")
    share = meta.get("latest_share_pct")
    contrib = meta.get("latest_contrib_pct")
    unit = str(meta.get("unit", ""))
    gg_delta = meta.get("latest_gg_yoy_abs")
    nat_delta = meta.get("latest_nat_yoy_abs")
    prev_period = meta.get("prev_year_period")
    prev_nat_value = meta.get("prev_year_nat_value")
    prev_gg_value = meta.get("prev_year_gg_value")
    prev_share = meta.get("prev_year_share_pct")
    share_yoy_change_pp = meta.get("share_yoy_change_pp")
    prev_contrib = meta.get("prev_year_contrib_pct")
    contrib_yoy_change_pp = meta.get("contrib_yoy_change_pp")
    recent_start_period = meta.get("recent_start_period")
    recent_start_share = meta.get("recent_start_share_pct")
    recent_end_share = meta.get("recent_end_share_pct")
    recent_change_pp = meta.get("recent_change_pp")
    recent_max_period = meta.get("recent_max_period")
    recent_max_share = meta.get("recent_max_share_pct")
    recent_min_period = meta.get("recent_min_period")
    recent_min_share = meta.get("recent_min_share_pct")

    contrib_text = "-" if pd.isna(contrib) else f"{float(contrib):,.1f}%"
    prev_period_text = fmt_period(prev_period, prd_se) if pd.notna(prev_period) else "-"
    share_yoy_text = "-" if pd.isna(share_yoy_change_pp) else f"{float(share_yoy_change_pp):+,.2f}%p"
    contrib_yoy_text = "-" if pd.isna(contrib_yoy_change_pp) else f"{float(contrib_yoy_change_pp):+,.1f}%p"
    recent_start_text = fmt_period(recent_start_period, prd_se) if pd.notna(recent_start_period) else "-"
    recent_max_period_text = fmt_period(recent_max_period, prd_se) if pd.notna(recent_max_period) else "-"
    recent_min_period_text = fmt_period(recent_min_period, prd_se) if pd.notna(recent_min_period) else "-"
    recent_change_text = "-" if pd.isna(recent_change_pp) else f"{float(recent_change_pp):+,.2f}%p"
    if pd.isna(nat_delta):
        nat_flow = "증감분"
    elif float(nat_delta) > 0:
        nat_flow = "증가분"
    elif float(nat_delta) < 0:
        nat_flow = "감소분"
    else:
        nat_flow = "변동분"

    lines = [
        (
            f"- **이번 {point} 기준** {region_name} 취업자는 {fmt_num_bold(latest_gg_value, unit)}, "
            f"{base_region} 취업자는 {fmt_num_bold(latest_nat_value, unit)}이며, "
            f"{region_name} 비중은 **{float(share):,.2f}%**입니다."
            if pd.notna(share)
            else f"- {latest} 기준 비중 계산값이 없습니다."
        ),
        (
            f"- **전년동월 비교**: {region_name} 취업자 {prev_period_text} {fmt_num_bold(prev_gg_value, unit)} → "
            f"{latest} {fmt_num_bold(latest_gg_value, unit)}, "
            f"{base_region} 취업자 {prev_period_text} {fmt_num_bold(prev_nat_value, unit)} → "
            f"{latest} {fmt_num_bold(latest_nat_value, unit)}. "
            f"비중은 {prev_period_text} **{float(prev_share):,.2f}%** → "
            f"{latest} 비중 **{float(share):,.2f}%** (**{share_yoy_text}**). "
            f"기여율은 {prev_period_text} **{'-' if pd.isna(prev_contrib) else f'{float(prev_contrib):,.1f}%'}** → "
            f"{latest} **{contrib_text}** "
            f"({region_name} 증감 {fmt_num(gg_delta, unit)} / {base_region} 증감 {fmt_num(nat_delta, unit)}, "
            f"{contrib_yoy_text})."
            if pd.notna(prev_share)
            else f"- 전년동월({yoy}) 기준 비교 데이터가 부족합니다."
        ),
        (
            f"- {latest}의 {yoy} 대비 증감은 {base_region} {fmt_num_bold(nat_delta, unit)}, "
            f"{region_name} {fmt_num_bold(gg_delta, unit)}이며, "
            f"{base_region} {nat_flow} 중 {region_name} 기여율은 **{contrib_text}**이며, "
            f"취업자수 기준으로 {region_name} {fmt_num_bold(gg_delta, unit)} / {base_region} {fmt_num_bold(nat_delta, unit)}입니다."
        ),
        (
            f"- **최근 12개월 비중 변화**: {recent_start_text} **{'-' if pd.isna(recent_start_share) else f'{float(recent_start_share):,.2f}%'}** → "
            f"{latest} **{'-' if pd.isna(recent_end_share) else f'{float(recent_end_share):,.2f}%'}** "
            f"(**{recent_change_text}**). "
            f"같은 기간 최고는 {recent_max_period_text} **{'-' if pd.isna(recent_max_share) else f'{float(recent_max_share):,.2f}%'}**, "
            f"최저는 {recent_min_period_text} **{'-' if pd.isna(recent_min_share) else f'{float(recent_min_share):,.2f}%'}**입니다."
        ),
        f"- 다음 점검 포인트: 다음 {point}에도 비중이 같은 방향으로 움직이는지 확인하세요.",
    ]
    return "\n".join(lines)


def render_ai_insights(
    df: pd.DataFrame,
    region_pool: List[str],
    labels: Dict[str, str],
    card_fn: Callable[[str, str, str], None],
    datasets: Optional[List[Any]] = None,
    events: Optional[pd.DataFrame] = None,
    report_scope: str = "경기도 전체",
    source_df: Optional[pd.DataFrame] = None,
    fixed_region: Optional[str] = None,
    selected_month: Optional[str] = None,
    show_ai: bool = True,
) -> None:
    if not show_ai:
        return
    st.subheader("AI INSIGHTS")
    region = ""
    base_region = "전국"
    if fixed_region:
        region = str(fixed_region)
        if region in GYEONGGI_SIGUNGU:
            base_region = "경기도"
    st.markdown(f"#### 영향요인분해({base_region} 내 {region or '지역'} 비중)")
    gy_trend, gy_meta = compute_gyeonggi_vs_national_contribution(
        df,
        region_name=str(region) if region else "경기도",
        base_region=base_region,
    )
    if not gy_meta.get("ok"):
        st.info(str(gy_meta.get("message", "전국 대비 경기도 기여도 계산이 불가능합니다.")))
    else:
        st.markdown("##### AI 해설")
        st.markdown(build_ai_gyeonggi_contribution_commentary(gy_meta, labels), unsafe_allow_html=True)
        c1, c2 = st.columns(2)
        with c1:
            share_sub = (
                f"{region} {fmt_num(gy_meta.get('latest_gg_value'), str(gy_meta.get('unit', '')))} / "
                f"{base_region} {fmt_num(gy_meta.get('latest_nat_value'), str(gy_meta.get('unit', '')))}"
            )
            card_fn(
                f"{base_region} 대비 {region} 비중",
                "-" if pd.isna(gy_meta.get("latest_share_pct")) else f"{float(gy_meta.get('latest_share_pct')):,.2f}%",
                share_sub,
            )
        with c2:
            contrib_sub = (
                f"{region} 증감 {fmt_num(gy_meta.get('latest_gg_yoy_abs'), str(gy_meta.get('unit', '')))} / "
                f"{base_region} 증감 {fmt_num(gy_meta.get('latest_nat_yoy_abs'), str(gy_meta.get('unit', '')))}"
            )
            card_fn(
                f"{base_region} 증감 기여율({labels.get('yoy', '전년동월')}대비)",
                "-" if pd.isna(gy_meta.get("latest_contrib_pct")) else f"{float(gy_meta.get('latest_contrib_pct')):,.1f}%",
                contrib_sub,
            )
        plot_df = gy_trend[["period", "share_pct", "contrib_pct"]].dropna(subset=["period"], how="any").copy()
        if not plot_df.empty:
            col_l, col_r = st.columns(2)
            with col_l:
                share_chart = (
                    alt.Chart(plot_df)
                    .mark_line(color="#2563eb", point=True)
                    .encode(
                        x=alt.X("period:T", title=labels.get("point", "월")),
                        y=alt.Y("share_pct:Q", title="전국 대비 경기도 비중(%)"),
                        tooltip=[
                            alt.Tooltip("yearmonth(period):T", title=labels.get("point", "월")),
                            alt.Tooltip("share_pct:Q", title="비중(%)", format=".2f"),
                        ],
                    )
                    .properties(height=280)
                )
                st.altair_chart(share_chart, use_container_width=True)
            with col_r:
                base = alt.Chart(plot_df).encode(
                    x=alt.X("period:T", title=labels.get("point", "월")),
                    tooltip=[
                        alt.Tooltip("yearmonth(period):T", title=labels.get("point", "월")),
                        alt.Tooltip("contrib_pct:Q", title="기여율(%)", format=".1f"),
                    ],
                )
                contrib_line = base.mark_line(color="#dc2626", point=True).encode(
                    y=alt.Y("contrib_pct:Q", title="전국 증감 기여율(%)")
                )
                zero = alt.Chart(pd.DataFrame({"zero": [0]})).mark_rule(
                    color="#9CA3AF",
                    strokeDash=[4, 4],
                ).encode(y="zero:Q")
                st.altair_chart(alt.layer(contrib_line, zero).properties(height=280), use_container_width=True)
    st.markdown("---")
    st.markdown(
        """
<div style="border:2px solid #0f172a; border-radius:14px; padding:16px 18px; background:#e0f2fe; margin-bottom:10px;">
  <div style="font-weight:900; font-size:1.25rem; color:#0f172a; letter-spacing:0.2px;">
    분석 지역
  </div>
  <div style="margin-top:6px; color:#1f2937; font-size:1.02rem; font-weight:700;">
    아래 모든 AI 분석(영향요인분해, 인사이트)에 공통 적용됩니다.
  </div>
</div>
""",
        unsafe_allow_html=True,
    )
    if fixed_region:
        st.caption(f"현재 선택된 시군 기준으로 분석합니다: **{region}**")
    else:
        gyeonggi_default = TARGET_REGIONS[9] if len(TARGET_REGIONS) >= 10 else (region_pool[0] if region_pool else "")
        region_default = gyeonggi_default if gyeonggi_default in region_pool else (region_pool[0] if region_pool else "")
        region = st.selectbox(
            "분석 지역 선택",
            region_pool,
            index=region_pool.index(region_default) if region_default in region_pool else 0,
            key="ai_region",
        )
    lag = 12
    if "prd_se" in df.columns and not df["prd_se"].dropna().empty:
        lag = 2 if str(df["prd_se"].dropna().iloc[0]).upper() == "H" else 12
    period_prd = "H" if lag == 2 else "M"
    st.markdown("#### 영향요인분해(지역별)")
    ds_options = {
        "연령별 취업자": "age",
        "종사상지위별 취업자": "status",
        "산업별 취업자수": "industry",
        "직종별 취업자수": "occupation",
    }
    ds_label = st.radio(
        "분해 축",
        list(ds_options.keys()),
        horizontal=True,
        key="ai_decomp_axis",
    )
    ds_key = ds_options[ds_label]
    contrib_df, contrib_meta = compute_contribution_table(df, region=region, dataset_key=ds_key, lag=lag)
    if not contrib_meta.get("ok"):
        st.info(str(contrib_meta.get("message", "분해 데이터를 계산할 수 없습니다.")))
    else:
        st.markdown("##### AI 해설")
        st.markdown(
            build_ai_contribution_commentary(contrib_df, contrib_meta, labels["point"], labels["yoy"]),
            unsafe_allow_html=True,
        )
        unit = str(contrib_meta.get("unit", ""))
        st.markdown(
            f"(기준시점: {fmt_period(contrib_meta['latest_period'], period_prd)} | "
            f"비교시점: {fmt_period(contrib_meta['prev_period'], period_prd)} | "
            f"총 증감: {fmt_num(contrib_meta['total_delta'], unit)})"
        )
        chart_df = contrib_df.copy()
        chart = (
            alt.Chart(chart_df)
            .mark_bar()
            .encode(
                x=alt.X("증감:Q", title="증감"),
                y=alt.Y("분류:N", sort="-x", title="분류"),
                color=alt.condition("datum.증감 >= 0", alt.value("#2563eb"), alt.value("#dc2626")),
                tooltip=[
                    alt.Tooltip("분류:N", title="분류"),
                    alt.Tooltip("증감:Q", title="증감", format=",.2f"),
                    alt.Tooltip("기여율(%):Q", title="기여율(%)", format=".2f"),
                ],
            )
            .properties(height=max(280, len(chart_df) * 22))
        )
        st.altair_chart(chart, use_container_width=True)
        st.dataframe(contrib_df, use_container_width=True, hide_index=True)
    st.markdown("---")

    if events is None or source_df is None or not datasets:
        return

    st.markdown("---")
    st.markdown("#### 규칙 기반 인사이트 + AI 보조")
    context = build_ai_insight_context(
        events=events,
        report_scope=report_scope,
        datasets=datasets,
        source_df=source_df,
        selected_region=region,
        selected_month=selected_month,
    )
    if not context.get("ok"):
        st.info(str(context.get("message", "인사이트 요약을 만들 수 없습니다.")))
        return

    context_lines = list(context.get("context_lines", []))
    focus_lines = list(context.get("focus_lines", []))
    consecutive_lines = list(context.get("consecutive_lines", []))
    context_title = str(context.get("context_title", ""))

    context_hash = compute_hash([context_title] + context_lines + focus_lines + consecutive_lines)
    memory_entries = load_memory(limit=400)
    selected_entries = select_memory_context(
        memory_entries,
        scope_title=str(context.get("scope_title", "")),
        region=str(region),
        limit=5,
        exact_hash=context_hash,
    )
    past_summaries: List[str] = []
    for entry in selected_entries:
        summary = str(entry.get("summary", "")).strip()
        if not summary:
            insight = str(entry.get("insight", "")).strip()
            summary = (insight[:140] + "...") if len(insight) > 140 else insight
        if summary:
            created = str(entry.get("created_at", ""))
            past_summaries.append(f"- ({created}) {summary}")

    st.markdown("##### 규칙 기반 핵심 요약")
    rule_lines = _build_rule_based_insights(context=context, region=str(region))
    st.markdown("\n".join(rule_lines))

    with st.expander("AI 보조 해석", expanded=True):
        st.markdown("##### OpenAI 설정")
        if st.session_state.get("ai_openai_model") in {None, "", "gpt-4.1"}:
            st.session_state["ai_openai_model"] = "gpt-5.2"
        model = st.text_input("모델", value=st.session_state.get("ai_openai_model", "gpt-5.2"), key="ai_openai_model")
        temperature = 0.3
        max_output_tokens = 800
        auto_save = st.toggle("생성 후 자동 저장", value=False, key="ai_memory_auto_save")

        st.markdown("##### 최신 데이터 요약")
        st.markdown("\n".join(context_lines + focus_lines + consecutive_lines))

        if past_summaries:
            st.markdown("##### 과거 인사이트 요약(참고)")
            st.markdown("\n".join(past_summaries))

        user_note = st.text_area("추가 메모", key="ai_memory_note", height=80)
        prompt = build_prompt(
            context_title=context_title,
            context_lines=context_lines,
            focus_lines=focus_lines,
            consecutive_lines=consecutive_lines,
            past_summaries=past_summaries,
            user_note=user_note,
        )
        st.text_area("LLM 프롬프트", value=prompt, height=260, key="ai_memory_prompt")

        st.markdown("##### AI 응답 저장")

        def _save_insight(insight: str, summary: str) -> bool:
            if not insight.strip():
                st.warning("AI 응답을 먼저 입력해 주세요.")
                return False
            summary_val = summary.strip()
            if not summary_val or summary_val.startswith("#"):
                summary_val = _auto_summary_from_insight(insight)
                st.session_state["ai_memory_summary"] = summary_val
            save_memory(
                {
                    "id": str(uuid4()),
                    "scope_title": str(context.get("scope_title", "")),
                    "region": str(region),
                    "selected_month": str(context.get("selected_month", "")),
                    "context_title": context_title,
                    "context_hash": context_hash,
                    "prompt": prompt,
                    "insight": insight.strip(),
                    "summary": summary_val,
                    "stats": context.get("stats", {}),
                }
            )
            st.success("인사이트를 저장했습니다.")
            return True

        if st.button("OpenAI로 생성", key="ai_memory_generate"):
            api_key = _seeded_openai_key()
            if not api_key:
                st.warning("OPENAI_API_KEY가 설정되지 않았습니다.")
            else:
                result = create_response_text(
                    api_key=api_key,
                    prompt=prompt,
                    model=str(model),
                    temperature=float(temperature),
                    max_output_tokens=int(max_output_tokens),
                )
                if not result.get("ok"):
                    st.error(str(result.get("error", "OpenAI 호출에 실패했습니다.")))
                else:
                    generated = str(result.get("text", "")).strip()
                    st.session_state["ai_memory_response"] = generated
                    current_summary = str(st.session_state.get("ai_memory_summary", "")).strip()
                    if not current_summary or current_summary.startswith("#"):
                        st.session_state["ai_memory_summary"] = _auto_summary_from_insight(generated)
                    if auto_save:
                        _save_insight(generated, st.session_state.get("ai_memory_summary", ""))

        insight_text = st.text_area("AI 응답", key="ai_memory_response", height=360)
        summary_text = st.text_area("요약(1~3줄)", key="ai_memory_summary", height=80)
        if st.button("인사이트 저장", key="ai_memory_save"):
            _save_insight(insight_text, summary_text)

        if memory_entries:
            st.markdown("##### 최근 저장된 인사이트")
            preview = pd.DataFrame(memory_entries[-15:]).copy()
            keep_cols = ["created_at", "scope_title", "region", "selected_month", "summary"]
            view_cols = [c for c in keep_cols if c in preview.columns]
            if view_cols:
                st.dataframe(preview[view_cols], use_container_width=True, hide_index=True)
