from __future__ import annotations

import os
from typing import Any, Dict, List

import altair as alt
import pandas as pd
import streamlit as st

from src.config import DATASETS, TARGET_REGIONS, default_end_period
from src.kosis_client import KosisClient
from src.transform import add_yoy, build_stats, normalize_records, series_filter

st.set_page_config(
    page_title="KOSIS 월별 고용 모니터링",
    page_icon=":bar_chart:",
    layout="wide",
)

st.markdown(
    """
<style>
.metric-card {
  border: 1px solid #e5e7eb;
  border-radius: 12px;
  padding: 14px 16px;
  background: #ffffff;
}
.metric-title {
  font-size: 0.88rem;
  color: #4b5563;
}
.metric-value {
  font-size: 1.3rem;
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
  font-size: 0.82rem;
}
.new-badge {
  display: inline-block;
  margin-left: 6px;
  color: #b91c1c;
  font-size: 0.74rem;
  font-weight: 800;
}
</style>
""",
    unsafe_allow_html=True,
)

ACTIVITY_INDICATOR_ORDER = [
    "15세이상인구",
    "경제활동인구",
    "경제활동참가율",
    "비경제활동인구",
    "고용률",
    "취업자",
    "15~64세 고용률",
    "실업률",
    "실업자",
]


def _norm_indicator_name(text: str) -> str:
    s = str(text).strip()
    for token in [" ", "~", "-", "–", "ㅡ"]:
        s = s.replace(token, "")
    return s


def _order_activity_indicators(indicators: List[str]) -> List[str]:
    order_map = { _norm_indicator_name(name): idx for idx, name in enumerate(ACTIVITY_INDICATOR_ORDER) }
    return sorted(
        indicators,
        key=lambda x: (order_map.get(_norm_indicator_name(x), 999), x),
    )


def _seeded_api_key() -> str:
    try:
        secret_value = st.secrets.get("api_key", "") or st.secrets.get("API_KEY", "")
    except Exception:  # noqa: BLE001
        secret_value = ""
    return str(secret_value or os.getenv("api_key", "") or os.getenv("API_KEY", ""))


def _fmt_period(value: object) -> str:
    ts = pd.Timestamp(value)
    if pd.isna(ts):
        return "-"
    return ts.strftime("%Y-%m")


def _fmt_num(value: object, unit: str = "", digits: int = 1) -> str:
    if value is None or pd.isna(value):
        return "-"
    if unit and "%" in unit:
        digits = 2
    return f"{float(value):,.{digits}f}{unit}"


def _new(flag: bool) -> str:
    return "<span class='new-badge'>NEW</span>" if flag else ""


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
            lambda v: "color:#f59e0b;font-weight:700;" if str(v).strip() == "NEW" else "",
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


def _auto_y_domain(values: pd.Series, pad_ratio: float = 0.08) -> List[float] | None:
    valid = values.dropna()
    if valid.empty:
        return None
    vmin = float(valid.min())
    vmax = float(valid.max())
    if vmin == vmax:
        base = abs(vmin) if vmin != 0 else 1.0
        pad = base * pad_ratio
        return [vmin - pad, vmax + pad]
    span = vmax - vmin
    pad = span * pad_ratio
    return [vmin - pad, vmax + pad]


@st.cache_data(ttl=60 * 30, show_spinner=False)
def fetch_records_cached(
    api_key: str,
    dataset_key: str,
    end_period: str,
    config_signature: str,
) -> Dict[str, Any]:
    cfg = next((x for x in DATASETS if x.key == dataset_key), None)
    if cfg is None:
        raise RuntimeError(f"Unknown dataset key: {dataset_key}")
    client = KosisClient(api_key=api_key)
    records, debug_logs = client.fetch_with_debug(cfg, end_prd_de=end_period)
    return {"records": records, "debug_logs": debug_logs}


def load_data_with_progress(api_key: str) -> tuple[pd.DataFrame, List[str], List[str]]:
    end_period = default_end_period()
    total_steps = len(DATASETS) * 2 + 1
    step = 0
    frames: List[pd.DataFrame] = []
    errors: List[str] = []
    debug_logs: List[str] = []

    progress = st.progress(0)
    status = st.empty()

    for cfg in DATASETS:
        status.info(f"데이터 불러오는 중: {cfg.title}")
        try:
            config_signature = "|".join(
                [cfg.tbl_id, cfg.itm_id, cfg.obj_l1, cfg.obj_l2, cfg.output_fields, cfg.start_prd_de]
            )
            result = fetch_records_cached(
                api_key=api_key,
                dataset_key=cfg.key,
                end_period=end_period,
                config_signature=config_signature,
            )
            records = result.get("records", [])
            for line in result.get("debug_logs", []):
                debug_logs.append(f"[{cfg.key}] {line}")
            if records:
                sample = records[0]
                debug_logs.append(
                    f"[{cfg.key}] sample_keys={list(sample.keys())[:12]}"
                )
                debug_logs.append(
                    f"[{cfg.key}] sample_PRD_DE={sample.get('PRD_DE')} sample_DT={sample.get('DT')}"
                )
        except Exception as exc:  # noqa: BLE001
            records = []
            errors.append(f"{cfg.title}: {exc}")
            debug_logs.append(f"[{cfg.key}] ERROR: {exc}")
        step += 1
        progress.progress(min(100, int(step * 100 / total_steps)))

        status.info(f"파싱 중: {cfg.title}")
        parsed = normalize_records(cfg, records)
        debug_logs.append(f"[{cfg.key}] parsed_rows={len(parsed)} raw_rows={len(records)}")
        if not parsed.empty:
            frames.append(parsed)
        step += 1
        progress.progress(min(100, int(step * 100 / total_steps)))

    status.info("지표 계산 및 통합 중...")
    if not frames:
        progress.progress(100)
        status.error("데이터 로딩 실패")
        return pd.DataFrame(), errors, debug_logs

    combined = pd.concat(frames, ignore_index=True)
    combined = add_yoy(combined)
    debug_logs.append(f"[all] combined_rows={len(combined)}")
    step += 1
    progress.progress(100)
    status.success("로딩 완료")
    return combined, errors, debug_logs


def _extreme_rows(stats: Dict[str, object], prefix: str, unit: str) -> pd.DataFrame:
    labels = {
        "level": "원자료",
        "yoy_abs": "전년동월대비 증감(절대)",
        "yoy_pct": "전년동월대비 증감률",
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
            "최고 시점": _fmt_period(stats.get(f"{prefix}_max_all_period")),
            "최저": _fmt_num(stats.get(f"{prefix}_min_all_value"), display_unit),
            "최저 시점": _fmt_period(stats.get(f"{prefix}_min_all_period")),
            "비고": "NEW"
            if stats.get(f"{prefix}_is_new_max_all") or stats.get(f"{prefix}_is_new_min_all")
            else "",
        },
        {
            "지표": label,
            "구간": "최근 5년",
            "최고": _fmt_num(stats.get(f"{prefix}_max_5y_value"), display_unit),
            "최고 시점": _fmt_period(stats.get(f"{prefix}_max_5y_period")),
            "최저": _fmt_num(stats.get(f"{prefix}_min_5y_value"), display_unit),
            "최저 시점": _fmt_period(stats.get(f"{prefix}_min_5y_period")),
            "비고": "NEW"
            if stats.get(f"{prefix}_is_new_max_5y") or stats.get(f"{prefix}_is_new_min_5y")
            else "",
        },
    ]
    cols = ["지표", "구간", "최고", "최고 시점", "최저", "최저 시점", "비고"]
    return pd.DataFrame(rows)[cols]


def _render_dataset(df: pd.DataFrame, dataset_key: str) -> None:
    cfg = next(x for x in DATASETS if x.key == dataset_key)
    subset = df[df["dataset_key"] == dataset_key].copy()
    st.subheader(cfg.title)
    if subset.empty:
        st.warning("해당 데이터가 없습니다.")
        return

    region_options = [r for r in TARGET_REGIONS if r in subset["region_name"].unique()]
    if not region_options:
        region_options = sorted(subset["region_name"].dropna().unique().tolist())
    default_region_index = region_options.index("경기도") if "경기도" in region_options else 0

    col1, col2, col3 = st.columns([1, 1, 1])
    with col1:
        region = st.selectbox(
            "지역",
            region_options,
            index=default_region_index,
            key=f"region_{dataset_key}",
        )

    indicators = sorted(subset["indicator_name"].dropna().unique().tolist())
    indicator = indicators[0] if indicators else ""
    category_container = col3
    if dataset_key == "activity":
        indicators = _order_activity_indicators(indicators)
        with col2:
            indicator = st.radio(
                "지표",
                indicators,
                key=f"indicator_{dataset_key}",
                horizontal=True,
            )
    else:
        category_container = col2

    category = ""
    if cfg.has_category:
        categories = sorted(
            c for c in subset["category_name"].dropna().unique().tolist() if str(c).strip() != ""
        )
        with category_container:
            category = st.selectbox(cfg.category_label, categories, key=f"category_{dataset_key}")

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
    latest_period = _fmt_period(stats.get("latest_period"))
    unit = str(series_df["unit"].dropna().iloc[-1]) if not series_df["unit"].dropna().empty else ""

    st.caption(f"최신 기준월: {latest_period}")
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
            _fmt_period(stats.get("level_max_all_period")),
            bool(stats.get("level_is_new_max_all")),
            "value-max",
        )
    with cols[2]:
        _card(
            "전체기간 최저",
            _fmt_num(stats.get("level_min_all_value"), unit),
            _fmt_period(stats.get("level_min_all_period")),
            bool(stats.get("level_is_new_min_all")),
            "value-min",
        )
    with cols[3]:
        _card(
            "최근 5년 중 최고",
            _fmt_num(stats.get("level_max_5y_value"), unit),
            _fmt_period(stats.get("level_max_5y_period")),
            bool(stats.get("level_is_new_max_5y")),
            "value-max",
        )
    with cols[4]:
        _card(
            "최근 5년 중 최저",
            _fmt_num(stats.get("level_min_5y_value"), unit),
            _fmt_period(stats.get("level_min_5y_period")),
            bool(stats.get("level_is_new_min_5y")),
            "value-min",
        )

    st.markdown("#### 월별 추이")
    level_df = series_df[["period", "value"]].dropna(subset=["value"]).copy()
    if level_df.empty:
        st.info("월별 추이 데이터가 없습니다.")
    else:
        level_title = "원자료" if not unit else f"원자료 ({unit})"
        level_domain = _auto_y_domain(level_df["value"])
        level_chart = (
            alt.Chart(level_df)
            .mark_line(color="#4C78A8")
            .encode(
                x=alt.X("period:T", title="월"),
                y=alt.Y("value:Q", title=level_title, scale=alt.Scale(domain=level_domain)),
                tooltip=[
                    alt.Tooltip("yearmonth(period):T", title="월"),
                    alt.Tooltip("value:Q", title=level_title, format=",.2f"),
                ],
            )
            .properties(height=320)
        )
        st.altair_chart(level_chart, use_container_width=True)

    st.markdown("#### 전년동월대비 증감(막대) / 증감률(선)")
    yoy_df = series_df[["period", "yoy_abs", "yoy_pct"]].dropna(
        subset=["yoy_abs", "yoy_pct"],
        how="all",
    )
    if yoy_df.empty:
        st.info("YoY 데이터가 없습니다.")
    else:
        base = alt.Chart(yoy_df).encode(
            x=alt.X("period:T", title="월"),
            tooltip=[
                alt.Tooltip("yearmonth(period):T", title="월"),
                alt.Tooltip("yoy_abs:Q", title="전년동월대비 증감", format=",.2f"),
                alt.Tooltip("yoy_pct:Q", title="전년동월대비 증감률(%)", format=".2f"),
            ],
        )
        bars = base.mark_bar(color="#4C78A8", opacity=0.55).encode(
            y=alt.Y("yoy_abs:Q", title="전년동월대비 증감")
        )
        line = base.mark_line(color="#E45756", point=True).encode(
            y=alt.Y("yoy_pct:Q", title="전년동월대비 증감률(%)")
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
            _extreme_rows(stats, "level", unit),
            _extreme_rows(stats, "yoy_abs", unit),
            _extreme_rows(stats, "yoy_pct", unit),
        ],
        ignore_index=True,
    )
    _render_extreme_table(summary_df)

def _collect_new_events(df: pd.DataFrame) -> pd.DataFrame:
    rows: List[Dict[str, str]] = []
    key_cols = ["dataset_key", "dataset_title", "region_name", "indicator_name", "category_name"]
    for _, series in df.groupby(key_cols, dropna=False):
        series = series.sort_values("period")
        stats = build_stats(series)
        latest_month = _fmt_period(stats.get("latest_period"))
        base = {
            "데이터셋": str(series["dataset_title"].iloc[0]),
            "지역": str(series["region_name"].iloc[0]),
            "지표": str(series["indicator_name"].iloc[0]),
            "분류": str(series["category_name"].iloc[0]),
            "기준월": latest_month,
        }
        if stats.get("level_is_new_max_all"):
            rows.append({**base, "이벤트": "원자료 전체기간 최고 NEW"})
        if stats.get("level_is_new_min_all"):
            rows.append({**base, "이벤트": "원자료 전체기간 최저 NEW"})
        if stats.get("yoy_abs_is_new_max_all"):
            rows.append({**base, "이벤트": "YoY(절대) 전체기간 최고 NEW"})
        if stats.get("yoy_abs_is_new_min_all"):
            rows.append({**base, "이벤트": "YoY(절대) 전체기간 최저 NEW"})
        if stats.get("yoy_pct_is_new_max_all"):
            rows.append({**base, "이벤트": "YoY(증감률) 전체기간 최고 NEW"})
        if stats.get("yoy_pct_is_new_min_all"):
            rows.append({**base, "이벤트": "YoY(증감률) 전체기간 최저 NEW"})
    if not rows:
        return pd.DataFrame(columns=["데이터셋", "지역", "지표", "분류", "기준월", "이벤트"])
    return pd.DataFrame(rows)


st.title("경제활동인구 월별 모니터링")

if st.button("데이터 새로고침"):
    fetch_records_cached.clear()
    st.session_state.pop("_loaded_api_key", None)
    st.session_state.pop("_loaded_data", None)
    st.session_state.pop("_loaded_errors", None)
    st.session_state.pop("_loaded_debug_logs", None)
    st.rerun()

api_key = _seeded_api_key()

if not api_key:
    st.warning("API key is not set.")
    st.stop()

if st.session_state.get("_loaded_api_key") == api_key and "_loaded_data" in st.session_state:
    data = st.session_state["_loaded_data"]
    load_errors = st.session_state.get("_loaded_errors", [])
    debug_logs = st.session_state.get("_loaded_debug_logs", [])
else:
    data, load_errors, debug_logs = load_data_with_progress(api_key=api_key)
    st.session_state["_loaded_api_key"] = api_key
    st.session_state["_loaded_data"] = data
    st.session_state["_loaded_errors"] = load_errors
    st.session_state["_loaded_debug_logs"] = debug_logs

if load_errors:
    st.error("일부 데이터셋 조회 중 오류가 발생했습니다.")
    for err in load_errors:
        st.write(f"- {err}")

if debug_logs:
    with st.expander("진단 로그 보기", expanded=bool(load_errors)):
        st.code("\n".join(debug_logs[-300:]))

if data.empty:
    st.warning("조회된 데이터가 없습니다. API 파라미터를 확인하세요.")
    st.stop()

tab1, tab2, tab3, tab4 = st.tabs(["경제활동인구현황", "산업별 취업자수", "직종별 취업자수", "NEW 알림판"])
with tab1:
    _render_dataset(data, "activity")
with tab2:
    _render_dataset(data, "industry")
with tab3:
    _render_dataset(data, "occupation")
with tab4:
    st.subheader("최신값 갱신 NEW 이벤트")
    events = _collect_new_events(data)
    if events.empty:
        st.info("현재 기준월에서 새롭게 갱신된 최고/최저 이벤트가 없습니다.")
    else:
        st.dataframe(events, use_container_width=True, hide_index=True)
        st.markdown(
            "<p style='color:#b91c1c;font-weight:700'>NEW 이벤트는 최신 기준월에 극값(최고/최저)을 갱신한 경우만 표시합니다.</p>",
            unsafe_allow_html=True,
        )
