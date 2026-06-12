from typing import Any, Callable, Dict, List, Optional
from uuid import uuid4

import altair as alt
import numpy as np
import pandas as pd
import streamlit as st
import os

import src.config as app_config
from src.core.category_rules import ACTIVITY_INDICATOR_ORDER, _industry_code_token, norm_age_category, norm_indicator_name
from src.core.formatters import escape_markdown_text, fmt_num, fmt_num_bold, fmt_period
from src.features.new_history import build_ai_insight_context
from src.services.insight_memory import build_prompt, compute_hash, load_memory, save_memory, select_memory_context
from src.services.openai_client import DEFAULT_OPENAI_MODEL, create_response_text, normalize_model_name

TARGET_REGIONS = app_config.TARGET_REGIONS
GYEONGGI_SIGUNGU = getattr(app_config, "GYEONGGI_SIGUNGU", [])
ALLOWED_AGE_CONTRIB_CATEGORIES = {
    "15~19",
    "20~29",
    "30~39",
    "40~49",
    "50~59",
    "60세이상",
}
ALLOWED_INDUSTRY_CONTRIB_TOKENS = {"A", "BC", "D~U"}


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


def _build_extreme_summary_table(
    df: pd.DataFrame,
    value_col: str,
    period_col: str,
    period_prd: str,
) -> pd.DataFrame:
    if df.empty or value_col not in df.columns or period_col not in df.columns:
        return pd.DataFrame()
    work = df[[period_col, value_col]].copy()
    work[period_col] = pd.to_datetime(work[period_col], errors="coerce")
    work[value_col] = pd.to_numeric(work[value_col], errors="coerce")
    work = work.dropna(subset=[period_col, value_col]).sort_values(period_col)
    if work.empty:
        return pd.DataFrame()

    latest = work[period_col].max()
    windows = [
        ("전체기간", None),
        ("최근 10년", latest - pd.DateOffset(years=10)),
        ("최근 5년", latest - pd.DateOffset(years=5)),
    ]
    rows = []
    for label, cutoff in windows:
        scoped = work if cutoff is None else work[work[period_col] >= cutoff]
        if scoped.empty:
            rows.append({"구간": label, "최고": "-", "최저": "-"})
            continue
        max_idx = scoped[value_col].idxmax()
        min_idx = scoped[value_col].idxmin()
        max_row = scoped.loc[max_idx]
        min_row = scoped.loc[min_idx]
        max_is_new = pd.Timestamp(max_row[period_col]) == pd.Timestamp(latest)
        min_is_new = pd.Timestamp(min_row[period_col]) == pd.Timestamp(latest)
        max_period_text = fmt_period(max_row[period_col], period_prd)
        min_period_text = fmt_period(min_row[period_col], period_prd)
        if max_is_new:
            max_period_text = f"{max_period_text} NEW"
        if min_is_new:
            min_period_text = f"{min_period_text} NEW"
        rows.append(
            {
                "구간": label,
                "최고": f"{float(max_row[value_col]):,.2f}% ({max_period_text})",
                "최저": f"{float(min_row[value_col]):,.2f}% ({min_period_text})",
            }
        )
    return pd.DataFrame(rows)


def _style_new_in_extreme_table(df: pd.DataFrame):
    def _style_map(
        base: Any,
        func: Any,
        subset: List[str],
    ) -> Any:
        if hasattr(base, "map"):
            return base.map(func, subset=subset)
        return base.applymap(func, subset=subset)

    styler = df.style.set_properties(**{"text-align": "center"})
    target_cols = [col for col in ["최고", "최저"] if col in df.columns]
    if target_cols:
        styler = _style_map(
            styler,
            lambda v: "color:#f59e0b;font-weight:700;" if "NEW" in str(v).strip() else "",
            subset=target_cols,
        )
    return styler


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


def _filter_decomposition_categories(view_df: pd.DataFrame, dataset_key: str) -> pd.DataFrame:
    filtered = view_df.copy()
    if dataset_key == "age":
        filtered["_norm_age"] = filtered["category_name"].map(norm_age_category)
        filtered = filtered[filtered["_norm_age"].isin(ALLOWED_AGE_CONTRIB_CATEGORIES)].copy()
        filtered = filtered.drop(columns=["_norm_age"], errors="ignore")
    if dataset_key == "industry":
        filtered["_industry_token"] = filtered["category_name"].map(_industry_code_token)
        narrowed = filtered[filtered["_industry_token"].isin(ALLOWED_INDUSTRY_CONTRIB_TOKENS)].copy()
        if not narrowed.empty:
            filtered = narrowed
        filtered = filtered.drop(columns=["_industry_token"], errors="ignore")
    return filtered


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
    view_df = _filter_decomposition_categories(view_df, dataset_key)
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


def build_ai_comparison_commentary(
    table: pd.DataFrame,
    meta: Dict[str, Any],
    axis_label: str,
    labels: Dict[str, str],
) -> str:
    if table.empty or not meta.get("ok"):
        return str(meta.get("message", "비교 해설을 생성할 데이터가 부족합니다."))

    base_region = str(meta.get("base_region", "전국"))
    region_name = str(meta.get("region_name", "지역"))
    prd_se = str(meta.get("prd_se", "M")).upper()
    latest_text = fmt_period(meta.get("latest_period"), prd_se)
    unit = str(meta.get("unit", ""))
    contrib_col = f"{base_region} 증감 대비 지역 기여율(%)"
    base_delta_col = f"{base_region} 증감"
    region_delta_col = "지역 증감"

    work = table.copy()
    work[contrib_col] = pd.to_numeric(work[contrib_col], errors="coerce")
    work[base_delta_col] = pd.to_numeric(work[base_delta_col], errors="coerce")
    work[region_delta_col] = pd.to_numeric(work[region_delta_col], errors="coerce")
    work = work.dropna(subset=[contrib_col]).copy()
    if work.empty:
        return "상위지역 대비 해설을 생성할 비교 데이터가 부족합니다."

    top_pos = work[work[contrib_col] > 0].nlargest(3, contrib_col)
    top_neg = work[work[contrib_col] < 0].nsmallest(3, contrib_col)
    dominant = work.loc[work[contrib_col].abs().idxmax()] if not work.empty else None
    same_direction = work[np.sign(work[base_delta_col]) == np.sign(work[region_delta_col])].copy()
    opposite_direction = work[np.sign(work[base_delta_col]) != np.sign(work[region_delta_col])].copy()
    opposite_direction = opposite_direction[
        (work[base_delta_col] != 0) & (work[region_delta_col] != 0)
    ].copy()

    def _fmt_items(frame: pd.DataFrame) -> str:
        items: List[str] = []
        for _, row in frame.iterrows():
            items.append(
                f"{escape_markdown_text(str(row['분류']))}"
                f"({base_region} {fmt_num(row[base_delta_col], unit)} / "
                f"{region_name} {fmt_num(row[region_delta_col], unit)} / "
                f"기여율 {float(row[contrib_col]):,.1f}%)"
            )
        return ", ".join(items) if items else "-"

    direction_match_rate = float((np.sign(work[base_delta_col]) == np.sign(work[region_delta_col])).mean() * 100.0) if not work.empty else np.nan
    lines: List[str] = [
        f"- {latest_text} 기준 **{axis_label}**에서 각 분류별로 **{base_region} 증감 대비 {region_name} 증감 기여율**을 비교했습니다.",
    ]
    if pd.notna(direction_match_rate):
        lines.append(f"- 전체 분류 중 증감 방향이 같았던 비율은 **{direction_match_rate:,.1f}%**입니다.")
    if not top_pos.empty:
        lines.append(f"- **{base_region} 동일 분류 증가분 대비 {region_name} 기여가 컸던 항목**은 {_fmt_items(top_pos)} 입니다.")
    if not top_neg.empty:
        lines.append(f"- **{base_region} 동일 분류 감소분 대비 {region_name} 기여가 컸던 항목**은 {_fmt_items(top_neg)} 입니다.")
    if not opposite_direction.empty:
        top_opposite = opposite_direction.reindex(opposite_direction[contrib_col].abs().sort_values(ascending=False).index).head(3)
        lines.append(f"- **상위지역과 방향이 엇갈린 항목**은 {_fmt_items(top_opposite)} 입니다.")
    if dominant is not None and pd.notna(dominant[contrib_col]):
        lines.append(
            f"- 가장 두드러진 항목은 **{escape_markdown_text(str(dominant['분류']))}**이며, "
            f"{base_region} 증감이 {fmt_num_bold(dominant[base_delta_col], unit)}, "
            f"{region_name} 증감이 {fmt_num_bold(dominant[region_delta_col], unit)}로 "
            f"기여율은 **{float(dominant[contrib_col]):,.1f}%**입니다."
        )
    lines.append(
        f"- 해석 기준은 **같은 분류끼리 비교한 {base_region} 증감 대비 {region_name} 증감 비중**이며, "
        "지역 내부 기여율과는 다른 개념입니다."
    )
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


def _direction_label(base_delta: float, region_delta: float) -> str:
    if pd.isna(base_delta) or pd.isna(region_delta):
        return "비교 불가"
    if base_delta > 0 and region_delta > 0:
        return "동반 증가"
    if base_delta < 0 and region_delta < 0:
        return "동반 감소"
    if base_delta > 0 and region_delta < 0:
        return "전국 증가 · 지역 감소"
    if base_delta < 0 and region_delta > 0:
        return "전국 감소 · 지역 증가"
    if base_delta == 0 and region_delta == 0:
        return "동반 보합"
    if base_delta == 0:
        return "전국 보합 · 지역 변동"
    return "전국 변동 · 지역 보합"


def compute_industry_comparison_breakdown(
    df: pd.DataFrame,
    region_name: str,
    base_region: str = "전국",
) -> tuple[pd.DataFrame, Dict[str, Any]]:
    return compute_comparison_breakdown(df, region_name=region_name, dataset_key="industry", base_region=base_region)


def compute_comparison_breakdown(
    df: pd.DataFrame,
    region_name: str,
    dataset_key: str,
    base_region: str = "전국",
) -> tuple[pd.DataFrame, Dict[str, Any]]:
    meta: Dict[str, Any] = {
        "ok": False,
        "message": "",
        "latest_period": pd.NaT,
        "base_total_delta": np.nan,
        "region_total_delta": np.nan,
        "direction": "",
        "unit": "",
        "region_name": str(region_name),
        "base_region": str(base_region),
        "dataset_key": str(dataset_key),
        "prd_se": "M",
    }
    dataset_titles = {
        "industry": "산업별 취업자수",
        "age": "연령별 취업자",
        "occupation": "직종별 취업자수",
        "status": "종사상지위별 취업자",
    }
    axis_label = dataset_titles.get(dataset_key, dataset_key)
    meta["dataset_title"] = axis_label

    base = df[df["dataset_key"] == dataset_key].copy()
    if base.empty:
        meta["message"] = f"{axis_label} 데이터가 없습니다."
        return pd.DataFrame(), meta

    indicators = sorted(base["indicator_name"].dropna().unique().tolist())
    indicator = pick_employment_indicator(indicators)
    meta["indicator"] = indicator
    base = base[base["indicator_name"] == indicator].copy()
    if base.empty:
        meta["message"] = f"{axis_label} 지표를 찾지 못했습니다."
        return pd.DataFrame(), meta

    if "prd_se" in base.columns and not base["prd_se"].dropna().empty:
        meta["prd_se"] = str(base["prd_se"].dropna().iloc[0]).upper()

    drop_labels = {"시도별", "산업별", "산업명", "직업별", "직종별"}
    base["category_name"] = base["category_name"].astype(str).str.strip()
    base = base[(base["category_name"] != "") & (~base["category_name"].isin(drop_labels))].copy()
    if base.empty:
        meta["message"] = f"{axis_label} 분류 데이터가 없습니다."
        return pd.DataFrame(), meta

    regions = base["region_name"].dropna().astype(str).str.strip().unique().tolist()
    if region_name not in regions:
        meta["message"] = f"{region_name} {axis_label} 데이터를 찾지 못했습니다."
        return pd.DataFrame(), meta

    if base_region in regions:
        base_rows = base[base["region_name"] == base_region].copy()
    else:
        base_rows = base.copy()

    region_rows = base[base["region_name"] == region_name].copy()
    if region_rows.empty:
        meta["message"] = f"{region_name} {axis_label} 데이터를 찾지 못했습니다."
        return pd.DataFrame(), meta

    grp_cols = ["period", "category_name"]
    nat = base_rows.groupby(grp_cols, as_index=False).agg({"yoy_abs": "sum"})
    reg = region_rows.groupby(grp_cols, as_index=False).agg({"yoy_abs": "sum"})
    merged = nat.merge(reg, on=grp_cols, how="inner", suffixes=("_nat", "_reg"))
    if merged.empty:
        meta["message"] = f"{axis_label} 공통 비교 구간이 없습니다."
        return pd.DataFrame(), meta

    latest_period = pd.to_datetime(merged["period"], errors="coerce").max()
    view = merged[pd.to_datetime(merged["period"], errors="coerce") == latest_period].copy()
    if view.empty:
        meta["message"] = f"최신 시점 {axis_label} 데이터가 없습니다."
        return pd.DataFrame(), meta

    base_total_delta = float(view["yoy_abs_nat"].sum())
    region_total_delta = float(view["yoy_abs_reg"].sum())
    view[f"{base_region} 기여율(%)"] = np.where(
        base_total_delta == 0,
        np.nan,
        view["yoy_abs_nat"] / base_total_delta * 100.0,
    )
    view["지역 산업 기여율(%)"] = np.where(
        region_total_delta == 0,
        np.nan,
        view["yoy_abs_reg"] / region_total_delta * 100.0,
    )
    view[f"{base_region} 증감 대비 지역 기여율(%)"] = np.where(
        pd.to_numeric(view["yoy_abs_nat"], errors="coerce") == 0,
        np.nan,
        view["yoy_abs_reg"] / view["yoy_abs_nat"] * 100.0,
    )
    out = view.rename(
        columns={
            "category_name": "분류",
            "yoy_abs_nat": f"{base_region} 증감",
            "yoy_abs_reg": "지역 증감",
        }
    )[
        [
            "분류",
            f"{base_region} 증감",
            "지역 증감",
            f"{base_region} 기여율(%)",
            "지역 산업 기여율(%)",
            f"{base_region} 증감 대비 지역 기여율(%)",
        ]
    ].sort_values(f"{base_region} 증감 대비 지역 기여율(%)", ascending=False)

    unit = ""
    unit_series = base["unit"].dropna()
    if not unit_series.empty:
        unit = str(unit_series.iloc[0])

    meta.update(
        {
            "ok": True,
            "latest_period": pd.Timestamp(latest_period),
            "base_total_delta": base_total_delta,
            "region_total_delta": region_total_delta,
            "direction": _direction_label(base_total_delta, region_total_delta),
            "unit": unit,
        }
    )
    return out, meta


def compute_industry_comparison_trend(
    df: pd.DataFrame,
    region_name: str,
    base_region: str = "전국",
) -> tuple[pd.DataFrame, Dict[str, Any]]:
    meta: Dict[str, Any] = {"ok": False, "message": "", "prd_se": "M", "unit": ""}
    base = df[df["dataset_key"] == "industry"].copy()
    if base.empty:
        meta["message"] = "산업별 취업자수 데이터가 없습니다."
        return pd.DataFrame(), meta

    indicators = sorted(base["indicator_name"].dropna().unique().tolist())
    indicator = pick_employment_indicator(indicators)
    base = base[base["indicator_name"] == indicator].copy()
    if base.empty:
        meta["message"] = "산업별 취업자수 지표를 찾지 못했습니다."
        return pd.DataFrame(), meta

    prd_se = "M"
    if "prd_se" in base.columns and not base["prd_se"].dropna().empty:
        prd_se = str(base["prd_se"].dropna().iloc[0]).upper()

    drop_labels = {"시도별", "산업별", "산업명", "직업별", "직종별"}
    base["category_name"] = base["category_name"].astype(str).str.strip()
    base = base[(base["category_name"] != "") & (~base["category_name"].isin(drop_labels))].copy()
    if base.empty:
        meta["message"] = "산업 분류 데이터가 없습니다."
        return pd.DataFrame(), meta

    regions = base["region_name"].dropna().astype(str).str.strip().unique().tolist()
    if region_name not in regions:
        meta["message"] = f"{region_name} 산업별 데이터를 찾지 못했습니다."
        return pd.DataFrame(), meta

    if base_region in regions:
        base_rows = base[base["region_name"] == base_region].copy()
    else:
        base_rows = base.copy()
    region_rows = base[base["region_name"] == region_name].copy()

    grp_cols = ["period", "category_name"]
    nat = base_rows.groupby(grp_cols, as_index=False).agg({"yoy_abs": "sum"}).rename(columns={"yoy_abs": "yoy_abs_nat"})
    reg = region_rows.groupby(grp_cols, as_index=False).agg({"yoy_abs": "sum"}).rename(columns={"yoy_abs": "yoy_abs_reg"})
    merged = nat.merge(reg, on=grp_cols, how="inner")
    if merged.empty:
        meta["message"] = "산업별 추이 데이터가 없습니다."
        return pd.DataFrame(), meta

    merged["region_contrib_to_nat_pct"] = np.where(
        pd.to_numeric(merged["yoy_abs_nat"], errors="coerce") == 0,
        np.nan,
        merged["yoy_abs_reg"] / merged["yoy_abs_nat"] * 100.0,
    )
    merged["direction_match"] = np.where(
        np.sign(merged["yoy_abs_nat"]) == np.sign(merged["yoy_abs_reg"]),
        1,
        0,
    )

    unit = ""
    unit_series = base["unit"].dropna()
    if not unit_series.empty:
        unit = str(unit_series.iloc[0])
    meta.update({"ok": True, "prd_se": prd_se, "unit": unit})
    return merged, meta


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
    analysis_df = source_df if isinstance(source_df, pd.DataFrame) and not source_df.empty else df
    region = ""
    base_region = "전국"
    skip_base_comparison = False
    if fixed_region:
        region = str(fixed_region)
        if region == "전국":
            skip_base_comparison = True
        elif region in GYEONGGI_SIGUNGU:
            base_region = "경기도"
    else:
        gyeonggi_default = TARGET_REGIONS[9] if len(TARGET_REGIONS) >= 10 else (region_pool[0] if region_pool else "")
        region_default = gyeonggi_default if gyeonggi_default in region_pool else (region_pool[0] if region_pool else "")
        region = str(st.session_state.get("ai_region", region_default))
        if region not in region_pool and region_pool:
            region = region_default
    st.markdown(f"#### 영향요인분해({base_region} 내 {region or '지역'} 비중)")
    if skip_base_comparison:
        st.info("상단 시도 선택이 `전국`이어서 전국 대비 지역 비교 분석은 생략합니다.")
    else:
        gy_trend, gy_meta = compute_gyeonggi_vs_national_contribution(
            analysis_df,
            region_name=str(region) if region else "경기도",
            base_region=base_region,
        )
        if not gy_meta.get("ok"):
            st.info(str(gy_meta.get("message", "전국 대비 경기도 기여도 계산이 불가능합니다.")))
        else:
            analysis_region = str(region) if str(region).strip() else "경기도"
            st.markdown("##### 총량 방향 비교")
            latest_period_text = fmt_period(gy_meta.get("latest_period"), str(gy_meta.get("prd_se", "M")))
            unit = str(gy_meta.get("unit", ""))
            direction_text = _direction_label(
                float(gy_meta.get("latest_nat_yoy_abs")) if pd.notna(gy_meta.get("latest_nat_yoy_abs")) else np.nan,
                float(gy_meta.get("latest_gg_yoy_abs")) if pd.notna(gy_meta.get("latest_gg_yoy_abs")) else np.nan,
            )
            st.markdown(
                f"- **{latest_period_text} 기준** {labels.get('yoy', '전년동월')} 대비 증감 방향은 **{direction_text}**입니다. "
                f"({base_region} {fmt_num(gy_meta.get('latest_nat_yoy_abs'), unit)} / "
                f"{analysis_region} {fmt_num(gy_meta.get('latest_gg_yoy_abs'), unit)})"
            )

            ds_options = {
                "산업별 취업자수": "industry",
                "연령별 취업자": "age",
                "직종별 취업자수": "occupation",
                "종사상지위별 취업자": "status",
            }
            ds_label = st.radio(
                "분해 축",
                list(ds_options.keys()),
                horizontal=True,
                key=f"ai_compare_axis_{analysis_region}_{base_region}",
            )
            ds_key = ds_options[ds_label]

            st.markdown(f"##### {ds_label} 비교 및 기여도")
            comparison_df, comparison_meta = compute_comparison_breakdown(
                analysis_df,
                region_name=analysis_region,
                dataset_key=ds_key,
                base_region=base_region,
            )
            lag = infer_lag_from_df(analysis_df)
            internal_df, internal_meta = compute_contribution_table(
                analysis_df,
                region=analysis_region,
                dataset_key=ds_key,
                lag=lag,
            )
            if not comparison_meta.get("ok"):
                st.info(str(comparison_meta.get("message", f"{ds_label} 비교 데이터를 계산할 수 없습니다.")))
            else:
                st.markdown("###### AI 해설 - 상위지역 대비")
                st.markdown(
                    build_ai_comparison_commentary(comparison_df, comparison_meta, ds_label, labels),
                    unsafe_allow_html=True,
                )

                base_delta_col = f"{base_region} 증감"
                base_share_col = f"{base_region} 기여율(%)"
                contrib_col = f"{base_region} 증감 대비 지역 기여율(%)"
                chart_df = comparison_df.copy()
                st.markdown(f"###### 상위지역 대비 기여 ({base_region} 기준)")
                upper_df = chart_df[["분류", base_delta_col, "지역 증감", contrib_col]].copy()
                upper_fmt_df = upper_df.copy()
                for col in [base_delta_col, "지역 증감"]:
                    upper_fmt_df[col] = upper_fmt_df[col].apply(lambda v: fmt_num(v, str(comparison_meta.get("unit", ""))))
                upper_fmt_df[contrib_col] = upper_fmt_df[contrib_col].apply(
                    lambda v: "-" if pd.isna(v) else f"{float(v):,.1f}%"
                )
                st.dataframe(upper_fmt_df, use_container_width=True, hide_index=True)

                top_df = upper_df.nlargest(10, contrib_col)
                upper_bar = (
                    alt.Chart(top_df)
                    .mark_bar()
                    .encode(
                        x=alt.X(f"{contrib_col}:Q", title=contrib_col),
                        y=alt.Y("분류:N", sort="-x", title="분류"),
                        color=alt.condition(f"datum['{contrib_col}'] >= 0", alt.value("#2563eb"), alt.value("#dc2626")),
                        tooltip=[
                            alt.Tooltip("분류:N", title="분류"),
                            alt.Tooltip(f"{base_delta_col}:Q", title=base_delta_col, format=",.1f"),
                            alt.Tooltip("지역 증감:Q", title="지역 증감", format=",.1f"),
                            alt.Tooltip(f"{contrib_col}:Q", title="지역 기여율(%)", format=".1f"),
                        ],
                    )
                    .properties(height=320)
                )
                st.altair_chart(upper_bar, use_container_width=True)

                if internal_meta.get("ok"):
                    st.markdown("###### AI 해설 - 지역 내부")
                    st.markdown(
                        build_ai_contribution_commentary(internal_df, internal_meta, labels["point"], labels["yoy"]),
                        unsafe_allow_html=True,
                    )
                    st.markdown(f"###### 지역 내부 기여 ({analysis_region} 기준)")
                    internal_fmt_df = internal_df.copy()
                    for col in ["최신값", "비교값", "증감"]:
                        internal_fmt_df[col] = internal_fmt_df[col].apply(
                            lambda v: fmt_num(v, str(internal_meta.get("unit", "")))
                        )
                    internal_fmt_df["기여율(%)"] = internal_fmt_df["기여율(%)"].apply(
                        lambda v: "-" if pd.isna(v) else f"{float(v):,.1f}%"
                    )
                    st.dataframe(internal_fmt_df, use_container_width=True, hide_index=True)

                    internal_chart_df = internal_df.copy().nlargest(10, "기여율(%)")
                    internal_bar = (
                        alt.Chart(internal_chart_df)
                        .mark_bar()
                        .encode(
                            x=alt.X("기여율(%):Q", title=f"{analysis_region} 내부 기여율(%)"),
                            y=alt.Y("분류:N", sort="-x", title="분류"),
                            color=alt.condition("datum['기여율(%)'] >= 0", alt.value("#2563eb"), alt.value("#dc2626")),
                            tooltip=[
                                alt.Tooltip("분류:N", title="분류"),
                                alt.Tooltip("증감:Q", title="지역 증감", format=",.1f"),
                                alt.Tooltip("기여율(%):Q", title="지역 내부 기여율(%)", format=".1f"),
                            ],
                        )
                        .properties(height=320)
                    )
                    st.altair_chart(internal_bar, use_container_width=True)
                else:
                    st.markdown("###### AI 해설 - 지역 내부")
                    st.info(str(internal_meta.get("message", "지역 내부 해설 데이터를 계산할 수 없습니다.")))

                if ds_key == "industry":
                    st.markdown("##### 산업별 추이 진단")
                    trend_df, trend_meta = compute_industry_comparison_trend(
                        analysis_df,
                        region_name=analysis_region,
                        base_region=base_region,
                    )
                    if not trend_meta.get("ok"):
                        st.info(str(trend_meta.get("message", "산업별 추이를 계산할 수 없습니다.")))
                    else:
                        period_col = pd.to_datetime(trend_df["period"], errors="coerce")
                        trend_df = trend_df.assign(period=period_col).dropna(subset=["period"]).sort_values("period").copy()
                        period_options = trend_df["period"].drop_duplicates().tolist()
                        trend_prd = str(trend_meta.get("prd_se", "M"))
                        labels_opts = [fmt_period(p, trend_prd) for p in period_options]
                        if not labels_opts:
                            st.info("선택 가능한 산업별 추이 기간이 없습니다.")
                        else:
                            default_window = (labels_opts[max(0, len(labels_opts) - (6 if trend_prd == "H" else 12))], labels_opts[-1])
                            selected_window = st.select_slider(
                                "기간",
                                options=labels_opts,
                                value=default_window,
                                key=f"ai_industry_trend_period_{analysis_region}",
                            )
                            s_idx = labels_opts.index(selected_window[0])
                            e_idx = labels_opts.index(selected_window[1])
                            if s_idx > e_idx:
                                s_idx, e_idx = e_idx, s_idx
                            selected_periods = set(period_options[s_idx : e_idx + 1])
                            trend_view = trend_df[trend_df["period"].isin(selected_periods)].copy()

                            top_cats = (
                                trend_view.groupby("category_name", as_index=False)["region_contrib_to_nat_pct"]
                                .mean()
                                .reindex(columns=["category_name", "region_contrib_to_nat_pct"])
                                .sort_values("region_contrib_to_nat_pct", ascending=False)
                                .head(5)["category_name"]
                                .tolist()
                            )
                            all_cats = (
                                trend_view.groupby("category_name", as_index=False)["region_contrib_to_nat_pct"]
                                .mean()
                                .reindex(columns=["category_name", "region_contrib_to_nat_pct"])
                                .sort_values("region_contrib_to_nat_pct", ascending=False)["category_name"]
                                .tolist()
                            )
                            selected_cats = st.multiselect(
                                "표시할 산업 선택",
                                options=all_cats,
                                default=top_cats,
                                key=f"ai_industry_trend_categories_{analysis_region}",
                            )
                            plot_view = trend_view[trend_view["category_name"].isin(selected_cats)].copy()
                            dir_match_rate = float(plot_view["direction_match"].mean() * 100.0) if not plot_view.empty else np.nan
                            agg_by_cat = (
                                plot_view.groupby("category_name", as_index=False)
                                .agg(
                                    avg_contrib=("region_contrib_to_nat_pct", "mean"),
                                    latest_contrib=("region_contrib_to_nat_pct", "last"),
                                    pos_count=("region_contrib_to_nat_pct", lambda x: int((pd.to_numeric(x, errors="coerce") > 0).sum())),
                                    neg_count=("region_contrib_to_nat_pct", lambda x: int((pd.to_numeric(x, errors="coerce") < 0).sum())),
                                )
                            )
                            if not agg_by_cat.empty:
                                agg_by_cat["latest_vs_avg_pp"] = agg_by_cat["latest_contrib"] - agg_by_cat["avg_contrib"]
                                top_pos = agg_by_cat.sort_values("pos_count", ascending=False).head(1)
                                top_dev = agg_by_cat.reindex(agg_by_cat["latest_vs_avg_pp"].abs().sort_values(ascending=False).index).head(1)
                            else:
                                top_pos = pd.DataFrame()
                                top_dev = pd.DataFrame()

                            k1, k2, k3 = st.columns(3)
                            with k1:
                                card_fn("방향 일치율", "-" if pd.isna(dir_match_rate) else f"{dir_match_rate:,.1f}%", "선택 산업 기준")
                            with k2:
                                pos_text = "-" if top_pos.empty else f"{str(top_pos.iloc[0]['category_name'])} ({int(top_pos.iloc[0]['pos_count'])}회)"
                                card_fn("연속 +기여 우세", pos_text, "선택 산업·기간 기준")
                            with k3:
                                dev_text = "-"
                                dev_sub = "선택 산업 중 최대"
                                if not top_dev.empty and pd.notna(top_dev.iloc[0]["latest_vs_avg_pp"]):
                                    dev_text = f"{str(top_dev.iloc[0]['category_name'])} ({float(top_dev.iloc[0]['latest_vs_avg_pp']):+,.1f}%p)"
                                    latest_contrib = top_dev.iloc[0]["latest_contrib"]
                                    avg_contrib = top_dev.iloc[0]["avg_contrib"]
                                    latest_text = "-" if pd.isna(latest_contrib) else f"{float(latest_contrib):,.1f}%"
                                    avg_text = "-" if pd.isna(avg_contrib) else f"{float(avg_contrib):,.1f}%"
                                    dev_sub = f"최신 {latest_text} / 평균 {avg_text}"
                                card_fn("최신기여율 - 평균기여율", dev_text, dev_sub)
                            if not plot_view.empty:
                                trend_chart = (
                                    alt.Chart(plot_view)
                                    .mark_line(point=True)
                                    .encode(
                                        x=alt.X("period:T", title=labels.get("point", "월")),
                                        y=alt.Y("region_contrib_to_nat_pct:Q", title=contrib_col),
                                        color=alt.Color("category_name:N", title="산업"),
                                        tooltip=[
                                            alt.Tooltip("yearmonth(period):T", title=labels.get("point", "월")),
                                            alt.Tooltip("category_name:N", title="산업"),
                                            alt.Tooltip("region_contrib_to_nat_pct:Q", title="기여율(%)", format=".1f"),
                                        ],
                                    )
                                    .properties(height=280)
                                )
                                st.altair_chart(trend_chart, use_container_width=True)
                            else:
                                st.info("표시할 산업을 1개 이상 선택해 주세요.")

            st.markdown("##### 전국대비 추이(참고)")
            plot_df = gy_trend[["period", "share_pct", "contrib_pct"]].dropna(subset=["period"], how="any").copy()
            if not plot_df.empty:
                plot_df["period"] = pd.to_datetime(plot_df["period"], errors="coerce")
                plot_df = plot_df.dropna(subset=["period"]).sort_values("period").copy()
                period_options = plot_df["period"].drop_duplicates().tolist()
                chart_prd = "M"
                if "prd_se" in df.columns and not df["prd_se"].dropna().empty:
                    chart_prd = "H" if str(analysis_df["prd_se"].dropna().iloc[0]).upper() == "H" else "M"
                period_labels = [fmt_period(p, chart_prd) for p in period_options] if period_options else []
                default_window = (period_labels[0], period_labels[-1]) if period_labels else None

                col_l, col_r = st.columns(2)
                with col_l:
                    share_plot_df = plot_df.copy()
                    if period_labels and default_window:
                        selected_share_window = st.select_slider(
                            "전국대비 비중 기간",
                            options=period_labels,
                            value=default_window,
                            key=f"ai_share_period_{(region or 'all').strip()}",
                        )
                        share_start_idx = period_labels.index(selected_share_window[0])
                        share_end_idx = period_labels.index(selected_share_window[1])
                        if share_start_idx > share_end_idx:
                            share_start_idx, share_end_idx = share_end_idx, share_start_idx
                        share_periods = set(period_options[share_start_idx : share_end_idx + 1])
                        share_plot_df = share_plot_df[share_plot_df["period"].isin(share_periods)].copy()
                    share_vals = pd.to_numeric(share_plot_df["share_pct"], errors="coerce").dropna()
                    share_domain = None
                    if not share_vals.empty:
                        share_min = float(share_vals.min())
                        share_max = float(share_vals.max())
                        share_span = max(share_max - share_min, 0.5)
                        share_pad = share_span * 0.15
                        share_domain = [share_min - share_pad, share_max + share_pad]
                    share_chart = (
                    alt.Chart(share_plot_df)
                    .mark_line(color="#2563eb", point=True)
                    .encode(
                        x=alt.X("period:T", title=labels.get("point", "월")),
                        y=alt.Y("share_pct:Q", title=f"{base_region} 대비 {analysis_region} 비중(%)", scale=alt.Scale(domain=share_domain)),
                        tooltip=[
                            alt.Tooltip("yearmonth(period):T", title=labels.get("point", "월")),
                            alt.Tooltip("share_pct:Q", title="비중(%)", format=".2f"),
                        ],
                    )
                        .properties(height=280)
                    )
                    st.altair_chart(share_chart, use_container_width=True)
                    share_table = _build_extreme_summary_table(
                        share_plot_df,
                        value_col="share_pct",
                        period_col="period",
                        period_prd=chart_prd,
                    )
                    if not share_table.empty:
                        st.caption("전국대비 비중 최고·최저")
                        st.dataframe(_style_new_in_extreme_table(share_table), use_container_width=True, hide_index=True)
                with col_r:
                    contrib_plot_df = plot_df.copy()
                    if period_labels and default_window:
                        selected_contrib_window = st.select_slider(
                            "전국대비 증감 기여율 기간",
                            options=period_labels,
                            value=default_window,
                            key=f"ai_contrib_period_{(region or 'all').strip()}",
                        )
                        contrib_start_idx = period_labels.index(selected_contrib_window[0])
                        contrib_end_idx = period_labels.index(selected_contrib_window[1])
                        if contrib_start_idx > contrib_end_idx:
                            contrib_start_idx, contrib_end_idx = contrib_end_idx, contrib_start_idx
                        contrib_periods = set(period_options[contrib_start_idx : contrib_end_idx + 1])
                        contrib_plot_df = contrib_plot_df[contrib_plot_df["period"].isin(contrib_periods)].copy()
                    contrib_vals = pd.to_numeric(contrib_plot_df["contrib_pct"], errors="coerce").dropna()
                    contrib_domain = None
                    if not contrib_vals.empty:
                        contrib_min = float(contrib_vals.min())
                        contrib_max = float(contrib_vals.max())
                        contrib_span = max(contrib_max - contrib_min, 1.0)
                        contrib_pad = contrib_span * 0.15
                        contrib_domain = [contrib_min - contrib_pad, contrib_max + contrib_pad]
                    base = alt.Chart(contrib_plot_df).encode(
                        x=alt.X("period:T", title=labels.get("point", "월")),
                        tooltip=[
                            alt.Tooltip("yearmonth(period):T", title=labels.get("point", "월")),
                            alt.Tooltip("contrib_pct:Q", title="기여율(%)", format=".1f"),
                        ],
                    )
                    contrib_line = base.mark_line(color="#dc2626", point=True).encode(
                        y=alt.Y("contrib_pct:Q", title=f"{base_region} 증감 기여율(%)", scale=alt.Scale(domain=contrib_domain))
                    )
                    zero = alt.Chart(pd.DataFrame({"zero": [0]})).mark_rule(
                        color="#9CA3AF",
                        strokeDash=[4, 4],
                    ).encode(y="zero:Q")
                    st.altair_chart(alt.layer(contrib_line, zero).properties(height=280), use_container_width=True)
                    contrib_table = _build_extreme_summary_table(
                        contrib_plot_df,
                        value_col="contrib_pct",
                        period_col="period",
                        period_prd=chart_prd,
                    )
                    if not contrib_table.empty:
                        st.caption("증감 기여율 최고·최저")
                        st.dataframe(_style_new_in_extreme_table(contrib_table), use_container_width=True, hide_index=True)
    st.markdown("---")

    if not show_ai or events is None or source_df is None or not datasets:
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
    fact_lines = list(context.get("fact_lines", []))
    context_title = str(context.get("context_title", ""))

    context_hash = compute_hash([context_title] + context_lines + focus_lines + consecutive_lines + fact_lines)
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
        if st.session_state.get("ai_openai_model") in {None, "", "gpt-4.1", "gpt-5.2", "gpt-5.4-mini"}:
            st.session_state["ai_openai_model"] = DEFAULT_OPENAI_MODEL
        model = st.text_input(
            "모델",
            value=normalize_model_name(st.session_state.get("ai_openai_model", DEFAULT_OPENAI_MODEL)),
            key="ai_openai_model",
        )
        temperature = 0.3
        max_output_tokens = 800
        auto_save = st.toggle("생성 후 자동 저장", value=False, key="ai_memory_auto_save")

        st.markdown("##### 최신 데이터 요약")
        st.markdown("\n".join(context_lines + focus_lines + consecutive_lines + fact_lines))

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
            fact_lines=fact_lines,
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
