from __future__ import annotations

import os
from typing import Any, Dict, List

import altair as alt
import numpy as np
import pandas as pd
import streamlit as st

import src.config as app_config
from src.core.category_rules import (
    order_activity_indicators as _order_activity_indicators,
    order_age_categories as _order_age_categories,
    order_occupation_categories as _order_occupation_categories,
    order_province_industry_categories as _order_province_industry_categories,
    order_sigungu_age_categories as _order_sigungu_age_categories,
    order_sigungu_industry_categories as _order_sigungu_industry_categories,
    order_sigungu_occupation_categories as _order_sigungu_occupation_categories,
    order_sigungu_status_categories as _order_sigungu_status_categories,
    order_status_categories as _order_status_categories,
)
from src.core.formatters import (
    auto_y_domain as _auto_y_domain,
    fmt_num as _fmt_num,
    fmt_period as _fmt_period,
    new_badge as _new,
    remark_new as _remark_new,
    time_labels,
)
from src.features.insights import render_ai_insights as _render_ai_insights
from src.features.new_history import (
    collect_new_events as _collect_new_events,
    render_new_history_tab as _render_new_history_tab,
    render_new_monthly_report as _render_new_monthly_report,
)
from src.features.auto_discovery import render_auto_discovery_tab as _render_auto_discovery_tab
from src.features.report import render_report_template as _render_report_template
from src.services.loader import load_all_data_with_progress
from src.transform import build_stats, series_filter

# Keep runtime resilient even if a deployment temporarily mixes app/config versions.
GYEONGGI_SIGUNGU = getattr(app_config, "GYEONGGI_SIGUNGU", [])
TARGET_REGIONS = app_config.TARGET_REGIONS
DatasetConfig = getattr(app_config, "DatasetConfig", Any)
datasets_for_scope = getattr(
    app_config,
    "datasets_for_scope",
    lambda _scope: getattr(app_config, "DATASETS", []),
)

st.set_page_config(
    page_title="Data Monitoring",
    page_icon="🍦",
    layout="wide",
)

st.markdown(
    """
<style>
.stApp {
  font-family: "Malgun Gothic", "Apple SD Gothic Neo", "Noto Sans KR", sans-serif;
}
.stApp p, .stApp li, .stApp label, .stApp div[data-baseweb="radio"] span {
  font-size: 1.03rem;
}
.stApp h1 {
  font-size: 2.1rem;
}
.stApp h2 {
  font-size: 1.6rem;
}
.stApp h3 {
  font-size: 1.35rem;
}
.stApp div[data-baseweb="tab"] button {
  font-size: 1.0rem;
}
.stApp div[data-baseweb="radio"] > div {
  flex-wrap: nowrap;
  overflow-x: auto;
}
.stApp div[data-baseweb="radio"] label {
  white-space: nowrap;
}
.metric-card {
  border: 1px solid #e5e7eb;
  border-radius: 12px;
  padding: 14px 16px;
  background: #ffffff;
}
.metric-title {
  font-size: 1.0rem;
  color: #4b5563;
}
.metric-value {
  font-size: 1.5rem;
  font-weight: 700;
  margin-top: 4px;
}
.value-max {
  color: #b91c1c;
}
.value-min {
  color: #1d4ed8;
}
.metric-sub {
  margin-top: 3px;
  color: #6b7280;
  font-size: 0.95rem;
}
.new-badge {
  display: inline-block;
  margin-left: 6px;
  color: #f59e0b;
  font-size: 0.85rem;
  font-weight: 800;
}
</style>
""",
    unsafe_allow_html=True,
)

DATA_MODEL_VERSION = "2026-03-17-industry-order-v4"


def _seeded_api_key() -> str:
    try:
        secret_value = st.secrets.get("api_key", "") or st.secrets.get("API_KEY", "")
    except Exception:  # noqa: BLE001
        secret_value = ""
    return str(secret_value or os.getenv("api_key", "") or os.getenv("API_KEY", ""))


def _seeded_access_code() -> str:
    try:
        secret_value = st.secrets.get("access_code", "") or st.secrets.get("ACCESS_CODE", "")
    except Exception:  # noqa: BLE001
        secret_value = ""
    return str(secret_value or os.getenv("access_code", "") or os.getenv("ACCESS_CODE", ""))


def _require_access_gate() -> None:
    expected = _seeded_access_code().strip()
    if not expected:
        st.error("Access code is not set. Configure ACCESS_CODE (or access_code) in Streamlit secrets.")
        st.stop()

    if st.session_state.get("_access_granted_code", "") == expected:
        return

    st.title("Access Code")
    with st.form("access_code_form", clear_on_submit=True):
        entered = st.text_input("Access code", type="password", key="_access_code_input")
        submitted = st.form_submit_button("Sign in")
    if submitted:
        if str(entered).strip() == expected:
            st.session_state["_access_granted_code"] = expected
            st.rerun()
        st.error("Invalid access code.")
    st.stop()


def _time_labels(datasets: List[DatasetConfig]) -> Dict[str, str]:
    return time_labels([str(cfg.prd_se) for cfg in datasets])


def _card(title: str, value: str, sub: str, is_new: bool = False, value_class: str = "") -> None:
    value_cls = f"metric-value {value_class}".strip()
    st.markdown(
        f"""
<div class="metric-card">
  <div class="metric-title">{title}{_new(is_new)}</div>
  <div class="{value_cls}">{value}</div>
  <div class="metric-sub">{sub}</div>
</div>
""",
        unsafe_allow_html=True,
    )


def _style_extreme_table(df: pd.DataFrame) -> pd.io.formats.style.Styler:
    styler = df.style.set_properties(**{"text-align": "center"}).set_table_styles(
        [
            {"selector": "table", "props": [("width", "100%")]},
            {"selector": "th", "props": [("text-align", "center !important")]},
            {"selector": "td", "props": [("text-align", "center !important")]},
        ],
        overwrite=False,
    )
    if "최고" in df.columns:
        styler = styler.applymap(lambda _: "color:#b91c1c;font-weight:700;", subset=["최고"])
    if "최저" in df.columns:
        styler = styler.applymap(lambda _: "color:#1d4ed8;font-weight:700;", subset=["최저"])
    if "비고" in df.columns:
        styler = styler.applymap(
            lambda v: "color:#f59e0b;font-weight:700;" if "NEW" in str(v).strip() else "",
            subset=["비고"],
        )
    return styler


def _render_extreme_table(df: pd.DataFrame) -> None:
    html = _style_extreme_table(df).hide(axis="index").to_html(
        table_attributes='style="width:100%; table-layout:fixed; margin:0;"'
    )
    st.markdown(
        f"<div style='width:100%; overflow-x:auto;'>{html}</div>",
        unsafe_allow_html=True,
    )


def _extreme_rows(stats: Dict[str, object], prefix: str, unit: str, prd_se: str) -> pd.DataFrame:
    yoy_label = "전년동기" if str(prd_se).upper() == "H" else "전년동월"
    labels = {
        "level": "원자료",
        "yoy_abs": f"{yoy_label}대비 증감(절대)",
        "yoy_pct": f"{yoy_label}대비 증감률",
    }
    label = labels[prefix]
    if prefix == "level":
        display_unit = unit
    elif prefix == "yoy_abs":
        display_unit = "%p" if "%" in unit else unit
    else:
        display_unit = "%"
    rows = [
        {
            "지표": label,
            "구간": "전체기간",
            "최고": _fmt_num(stats.get(f"{prefix}_max_all_value"), display_unit),
            "최고 시점": _fmt_period(stats.get(f"{prefix}_max_all_period"), prd_se),
            "최저": _fmt_num(stats.get(f"{prefix}_min_all_value"), display_unit),
            "최저 시점": _fmt_period(stats.get(f"{prefix}_min_all_period"), prd_se),
            "비고": _remark_new(
                bool(stats.get(f"{prefix}_is_new_max_all")),
                bool(stats.get(f"{prefix}_is_new_min_all")),
            ),
        },
        {
            "지표": label,
            "구간": "최근 5년",
            "최고": _fmt_num(stats.get(f"{prefix}_max_5y_value"), display_unit),
            "최고 시점": _fmt_period(stats.get(f"{prefix}_max_5y_period"), prd_se),
            "최저": _fmt_num(stats.get(f"{prefix}_min_5y_value"), display_unit),
            "최저 시점": _fmt_period(stats.get(f"{prefix}_min_5y_period"), prd_se),
            "비고": _remark_new(
                bool(stats.get(f"{prefix}_is_new_max_5y")),
                bool(stats.get(f"{prefix}_is_new_min_5y")),
            ),
        },
    ]
    cols = ["지표", "구간", "최고", "최고 시점", "최저", "최저 시점", "비고"]
    return pd.DataFrame(rows)[cols]


def _render_dataset(
    df: pd.DataFrame,
    dataset_key: str,
    region_pool: List[str],
    default_region: str,
    datasets: List[DatasetConfig],
    is_gyeonggi31_mode: bool,
) -> None:
    cfg = next(x for x in datasets if x.key == dataset_key)
    subset = df[df["dataset_key"] == dataset_key].copy()
    st.subheader(cfg.title)
    if subset.empty:
        st.warning("해당 데이터가 없습니다.")
        return

    region_options = [r for r in region_pool if r in subset["region_name"].unique()]
    if not region_options:
        region_options = sorted(subset["region_name"].dropna().unique().tolist())
    default_region_index = region_options.index(default_region) if default_region in region_options else 0

    if dataset_key == "activity":
        # Keep activity indicators on one line by giving wider horizontal space.
        col1, col2 = st.columns([0.6, 5.4])
        col3 = st.container()
    elif dataset_key in {"industry", "occupation", "age", "status"}:
        # Narrow region control and widen classification area.
        col1, col2 = st.columns([0.6, 3.4])
        col3 = st.container()
    else:
        col1, col2, col3 = st.columns([0.6, 2.2, 2.2])
    with col1:
        region = st.selectbox(
            "지역",
            region_options,
            index=default_region_index,
            key=f"region_{dataset_key}",
        )

    indicators = sorted(subset["indicator_name"].dropna().unique().tolist())
    indicator = indicators[0] if indicators else ""
    category_container = col2
    if dataset_key == "activity":
        indicators = _order_activity_indicators(indicators)
        with col2:
            indicator = st.radio(
                "지표",
                indicators,
                key=f"indicator_{dataset_key}",
                horizontal=True,
            )
    elif dataset_key == "industry" and indicators:
        # Industry datasets can include multiple item IDs; auto-pick the one
        # with the broadest category coverage for the selected region.
        drop_labels = {"시도별", "산업별", "산업명", "직업별", "직종별"}
        region_slice = subset[subset["region_name"] == region]
        if region_slice.empty:
            region_slice = subset

        def _category_count(ind_name: str) -> int:
            pool = region_slice[region_slice["indicator_name"] == ind_name]
            cats = [
                c
                for c in pool["category_name"].dropna().unique().tolist()
                if str(c).strip() != "" and str(c).strip() not in drop_labels
            ]
            return len(cats)

        indicator = sorted(indicators, key=lambda x: (_category_count(x), x), reverse=True)[0]
    else:
        category_container = col2

    category = ""
    if cfg.has_category:
        category_pool = subset[subset["region_name"] == region]
        if indicator:
            category_pool = category_pool[category_pool["indicator_name"] == indicator]
        categories = sorted(
            c for c in category_pool["category_name"].dropna().unique().tolist() if str(c).strip() != ""
        )
        if not categories:
            categories = sorted(
                c for c in subset["category_name"].dropna().unique().tolist() if str(c).strip() != ""
            )
        if dataset_key in {"industry", "occupation"}:
            drop_labels = {"시도별", "산업별", "산업명", "직업별", "직종별"}
            cleaned = [c for c in categories if str(c).strip() not in drop_labels]
            if cleaned:
                categories = cleaned
        if is_gyeonggi31_mode:
            if dataset_key == "industry":
                categories = _order_sigungu_industry_categories(categories)
            if dataset_key == "age":
                categories = _order_sigungu_age_categories(categories)
            if dataset_key == "status":
                categories = _order_sigungu_status_categories(categories)
            if dataset_key == "occupation":
                categories = _order_sigungu_occupation_categories(categories)
        else:
            if dataset_key == "industry":
                categories = _order_province_industry_categories(categories)
            if dataset_key == "age":
                categories = _order_age_categories(categories)
            if dataset_key == "status":
                categories = _order_status_categories(categories)
            if dataset_key == "occupation":
                categories = _order_occupation_categories(categories)
        with category_container:
            if dataset_key in {"industry", "occupation", "age", "status"}:
                category = st.radio(
                    cfg.category_label,
                    categories,
                    key=f"category_radio_{dataset_key}",
                    horizontal=True,
                )
            else:
                category = st.selectbox(
                    cfg.category_label,
                    categories,
                    key=f"category_select_{dataset_key}",
                )

    series_df = series_filter(
        df=subset,
        dataset_key=dataset_key,
        region_name=region,
        indicator_name=indicator,
        category_name=category,
    )
    if series_df.empty:
        st.info("선택한 조건의 시계열 데이터가 없습니다.")
        return

    stats = build_stats(series_df)
    prd_se = str(series_df["prd_se"].iloc[-1]).upper() if "prd_se" in series_df.columns else "M"
    labels = _time_labels(datasets)
    latest_period = _fmt_period(stats.get("latest_period"), prd_se)
    unit = str(series_df["unit"].dropna().iloc[-1]) if not series_df["unit"].dropna().empty else ""

    st.caption(f"최신 기준{labels['point']}: {latest_period}")
    cols = st.columns(5)
    with cols[0]:
        _card(
            "최신",
            _fmt_num(stats.get("level_latest_value"), unit),
            latest_period,
            False,
        )
    with cols[1]:
        _card(
            "전체기간 최고",
            _fmt_num(stats.get("level_max_all_value"), unit),
            _fmt_period(stats.get("level_max_all_period"), prd_se),
            bool(stats.get("level_is_new_max_all")),
            "value-max",
        )
    with cols[2]:
        _card(
            "전체기간 최저",
            _fmt_num(stats.get("level_min_all_value"), unit),
            _fmt_period(stats.get("level_min_all_period"), prd_se),
            bool(stats.get("level_is_new_min_all")),
            "value-min",
        )
    with cols[3]:
        _card(
            "최근 5년 중 최고",
            _fmt_num(stats.get("level_max_5y_value"), unit),
            _fmt_period(stats.get("level_max_5y_period"), prd_se),
            bool(stats.get("level_is_new_max_5y")),
            "value-max",
        )
    with cols[4]:
        _card(
            "최근 5년 중 최저",
            _fmt_num(stats.get("level_min_5y_value"), unit),
            _fmt_period(stats.get("level_min_5y_period"), prd_se),
            bool(stats.get("level_is_new_min_5y")),
            "value-min",
        )

    st.markdown(f"#### {labels['trend']} 추이")
    level_df = series_df[["period", "value"]].dropna(subset=["value"]).copy()
    if level_df.empty:
        st.info(f"{labels['trend']} 추이 데이터가 없습니다.")
    else:
        level_title = "원자료" if not unit else f"원자료 ({unit})"
        level_domain = _auto_y_domain(level_df["value"])
        level_chart = (
            alt.Chart(level_df)
            .mark_line(color="#4C78A8")
            .encode(
                x=alt.X("period:T", title=labels["point"]),
                y=alt.Y("value:Q", title=level_title, scale=alt.Scale(domain=level_domain)),
                tooltip=[
                    alt.Tooltip("yearmonth(period):T", title=labels["point"]),
                    alt.Tooltip("value:Q", title=level_title, format=",.2f"),
                ],
            )
            .properties(height=320)
        )
        st.altair_chart(level_chart, use_container_width=True)

    st.markdown(f"#### {labels['yoy']}대비 증감(막대) / 증감률(선)")
    yoy_df = series_df[["period", "yoy_abs", "yoy_pct"]].dropna(
        subset=["yoy_abs", "yoy_pct"],
        how="all",
    )
    if yoy_df.empty:
        st.info("YoY 데이터가 없습니다.")
    else:
        base = alt.Chart(yoy_df).encode(
            x=alt.X("period:T", title=labels["point"]),
            tooltip=[
                alt.Tooltip("yearmonth(period):T", title=labels["point"]),
                alt.Tooltip("yoy_abs:Q", title=f"{labels['yoy']}대비 증감", format=",.2f"),
                alt.Tooltip("yoy_pct:Q", title=f"{labels['yoy']}대비 증감률(%)", format=".2f"),
            ],
        )
        bars = base.mark_bar(color="#4C78A8", opacity=0.55).encode(
            y=alt.Y("yoy_abs:Q", title=f"{labels['yoy']}대비 증감")
        )
        line = base.mark_line(color="#E45756", point=True).encode(
            y=alt.Y("yoy_pct:Q", title=f"{labels['yoy']}대비 증감률(%)")
        )
        zero = alt.Chart(pd.DataFrame({"zero": [0]})).mark_rule(
            color="#9CA3AF",
            strokeDash=[4, 4],
        ).encode(y="zero:Q")
        combo = alt.layer(bars, line, zero).resolve_scale(y="independent").properties(height=320)
        st.altair_chart(combo, use_container_width=True)

    st.markdown("#### 리포트 요약")
    summary_df = pd.concat(
        [
            _extreme_rows(stats, "level", unit, prd_se),
            _extreme_rows(stats, "yoy_abs", unit, prd_se),
            _extreme_rows(stats, "yoy_pct", unit, prd_se),
        ],
        ignore_index=True,
    )
    _render_extreme_table(summary_df)


_require_access_gate()

st.title("경제활동인구 모니터링")

with st.sidebar:
    st.subheader("데이터 제어")
    if st.button("데이터 새로고침"):
        st.session_state.pop("_loaded_api_key", None)
        st.session_state.pop("_loaded_data_version", None)
        st.session_state.pop("_loaded_scope_data", None)
        st.session_state.pop("_loaded_errors", None)
        st.session_state.pop("_loaded_debug_logs", None)
        st.rerun()
    sidebar_status = st.empty()
    sidebar_progress_box = st.empty()

api_key = _seeded_api_key()

if not api_key:
    st.warning("API key is not set.")
    st.stop()

loading_notice = st.empty()
loading_notice.info("데이터 불러오는 중... (약 10분 소요예정)")
loading_progress = st.empty()
try:
    if (
        st.session_state.get("_loaded_api_key") == api_key
        and st.session_state.get("_loaded_data_version") == DATA_MODEL_VERSION
        and "_loaded_scope_data" in st.session_state
    ):
        scope_data = st.session_state["_loaded_scope_data"]
        load_errors = st.session_state.get("_loaded_errors", [])
        debug_logs = st.session_state.get("_loaded_debug_logs", [])
    else:
        scope_data, load_errors, debug_logs = load_all_data_with_progress(
            api_key=api_key,
            status_box=sidebar_status,
            progress_box=sidebar_progress_box,
            main_status_box=loading_notice,
            main_progress_box=loading_progress,
        )
        st.session_state["_loaded_api_key"] = api_key
        st.session_state["_loaded_data_version"] = DATA_MODEL_VERSION
        st.session_state["_loaded_scope_data"] = scope_data
        st.session_state["_loaded_errors"] = load_errors
        st.session_state["_loaded_debug_logs"] = debug_logs
finally:
    loading_notice.empty()
    loading_progress.empty()

if load_errors:
    st.error("일부 데이터셋 조회 중 오류가 발생했습니다.")
    for err in load_errors:
        st.write(f"- {err}")

if debug_logs:
    with st.sidebar:
        with st.expander("진단 로그 보기", expanded=bool(load_errors)):
            st.code("\n".join(debug_logs[-300:]))

is_sido_mode = st.toggle(
    "시도",
    value=True,
    key="scope_toggle",
)
scope_label = "전국 17개 시도" if is_sido_mode else "경기 31개 시군"
region_scope = "province" if is_sido_mode else "gyeonggi31"
is_gyeonggi31_mode = region_scope == "gyeonggi31"
active_datasets = datasets_for_scope(region_scope)
st.caption(f"조회범위: {scope_label}")

data = scope_data.get(region_scope, pd.DataFrame())

if data.empty:
    st.warning("조회된 데이터가 없습니다. API 파라미터를 확인하세요.")
    st.stop()

region_pool = TARGET_REGIONS if region_scope == "province" else GYEONGGI_SIGUNGU
default_region = "경기도" if region_scope == "province" else (GYEONGGI_SIGUNGU[0] if GYEONGGI_SIGUNGU else "")
visible_data = data[data["region_name"].isin(region_pool)].copy()
if visible_data.empty:
    st.warning("선택한 범위에서 조회된 데이터가 없습니다.")
    st.stop()

if region_scope == "gyeonggi31":
    event_source = data[data["region_name"].isin(GYEONGGI_SIGUNGU + ["경기도"])].copy()
else:
    event_source = visible_data
events = _collect_new_events(event_source)
labels = _time_labels(active_datasets)

tab1, tab2, tab3, tab4, tab5, tab6, tab7, tab8, tab9 = st.tabs(
    [
        "① 경제활동인구현황",
        "② 연령별 취업자",
        "③ 종사상지위별 취업자",
        "④ 산업별 취업자수",
        "⑤ 직종별 취업자수",
        "⑥ NEW HISTORY",
        "⑦ 요약",
        "⑧ 리포트",
        "⑨ 자동 질문 발견",
    ]
)
with tab1:
    _render_dataset(visible_data, "activity", region_pool, default_region, active_datasets, is_gyeonggi31_mode)
with tab2:
    _render_dataset(visible_data, "age", region_pool, default_region, active_datasets, is_gyeonggi31_mode)
with tab3:
    _render_dataset(visible_data, "status", region_pool, default_region, active_datasets, is_gyeonggi31_mode)
with tab4:
    _render_dataset(visible_data, "industry", region_pool, default_region, active_datasets, is_gyeonggi31_mode)
with tab5:
    _render_dataset(visible_data, "occupation", region_pool, default_region, active_datasets, is_gyeonggi31_mode)
with tab6:
    _render_new_history_tab(events)
with tab7:
    st.subheader("요약")
    report_scope = st.radio(
        "리포트 범위",
        ["경기도 전체", "31개 시군"],
        index=0,
        horizontal=True,
        key="report_scope",
    )
    _render_new_monthly_report(
        events,
        report_scope=report_scope,
        datasets=active_datasets,
        source_df=data,
    )
    if region_scope == "province":
        st.markdown("---")
        _render_ai_insights(visible_data, region_pool, labels, card_fn=_card)
with tab8:
    st.subheader("리포트")
    _render_report_template(
        df=visible_data,
        province_df=scope_data.get("province", pd.DataFrame()),
        region_pool=region_pool,
        datasets=active_datasets,
        is_gyeonggi31_mode=is_gyeonggi31_mode,
        labels=labels,
    )
with tab9:
    _render_auto_discovery_tab(visible_data, labels=labels)

st.markdown(
    "<hr style='margin-top:2rem; margin-bottom:0.5rem;'>"
    "<p style='text-align:center; color:#6b7280; font-size:0.9rem;'>- created by alicia -</p>",
    unsafe_allow_html=True,
)




