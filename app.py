from __future__ import annotations

import os
import re
import subprocess
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List
from urllib.parse import parse_qs, urlparse

import altair as alt
import numpy as np
import pandas as pd
import streamlit as st
import os

import src.config as app_config
from src.core.category_rules import (
    norm_indicator_name as _norm_indicator_name,
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
from src.features.insights import build_activity_snapshot as _build_activity_snapshot, render_ai_insights as _render_ai_insights
from src.features.new_history import (
    collect_new_events as _collect_new_events,
    get_report_period_options as _get_report_period_options,
    get_report_region_options as _get_report_region_options,
    render_consecutive_change_summary as _render_consecutive_change_summary,
    render_new_history_tab as _render_new_history_tab,
    render_new_monthly_report as _render_new_monthly_report,
)
from src.features.sigungu_typology import render_sigungu_typology_tab as _render_sigungu_typology_tab
from src.services.loader import load_all_data_with_progress, load_data_with_local_cache
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
.stApp [data-testid="stDialog"] [role="dialog"],
.stApp [data-testid="stDialog"] > div,
.stApp div[role="dialog"] {
  width: min(1200px, 96vw) !important;
  max-width: min(1200px, 96vw) !important;
}
.stApp [data-testid="stDialog"] iframe {
  width: 100% !important;
}
.stApp div[role="radiogroup"][aria-label="메뉴"] {
  display: flex;
  flex-wrap: wrap;
  gap: 0.4rem 0.45rem;
}
.stApp div[role="radiogroup"][aria-label="메뉴"] > label {
  border: 1px solid #cbd5e1;
  border-radius: 10px;
  background: #f8fafc;
  padding: 0.48rem 0.7rem;
  min-height: 2.2rem;
}
.stApp div[role="radiogroup"][aria-label="메뉴"] > label span {
  font-size: 0.98rem;
  font-weight: 700;
}
.stApp div[role="radiogroup"][aria-label="메뉴"] > label:has(input:checked) {
  border-color: #1e3a5f;
  background: #1e3a5f;
}
.stApp div[role="radiogroup"][aria-label="메뉴"] > label:has(input:checked) * {
  color: #ffffff !important;
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

DATA_MODEL_VERSION = "2026-06-05-activity-cache-guard-v1"
REQUIRED_SCOPE_COLUMNS = {"dataset_key", "region_name", "indicator_name", "category_name", "period", "value", "prd_se"}
SHOW_AI_FEATURES = str(os.getenv("SHOW_AI_FEATURES", "false")).strip().lower() in {"1", "true", "yes", "y"}


def _is_valid_scope_data(scope_data: object, required_scopes: List[str]) -> bool:
    if not isinstance(scope_data, dict):
        return False
    for scope_key in required_scopes:
        frame = scope_data.get(scope_key)
        if not isinstance(frame, pd.DataFrame):
            return False
        if frame.empty:
            return False
        if not REQUIRED_SCOPE_COLUMNS.issubset(set(frame.columns)):
            return False
        if "dataset_key" in frame.columns:
            expected = {
                str(getattr(cfg, "key", "")).strip()
                for cfg in datasets_for_scope(scope_key)
                if str(getattr(cfg, "key", "")).strip()
            }
            present = set(frame["dataset_key"].astype(str).str.strip().unique().tolist())
            if expected - present:
                return False
    return True


def _frame_signature(df: pd.DataFrame) -> str:
    if not isinstance(df, pd.DataFrame) or df.empty:
        return "empty"
    latest = pd.to_datetime(df["period"], errors="coerce").max() if "period" in df.columns else pd.NaT
    latest_text = "" if pd.isna(latest) else pd.Timestamp(latest).strftime("%Y-%m-%d")
    return f"{len(df)}|{latest_text}"


def _dataset_row_counts(df: pd.DataFrame) -> Dict[str, int]:
    if not isinstance(df, pd.DataFrame) or df.empty or "dataset_key" not in df.columns:
        return {}
    counts = df["dataset_key"].astype(str).str.strip().value_counts(dropna=False).to_dict()
    return {str(key): int(value) for key, value in counts.items()}


def _clear_derived_caches() -> None:
    st.session_state.pop("_dataset_subset_cache", None)
    st.session_state.pop("_series_stats_cache", None)
    st.session_state.pop("_events_cache", None)


def _get_cached_dataset_subset(df: pd.DataFrame, scope_tag: str, dataset_key: str) -> pd.DataFrame:
    cache = st.session_state.setdefault("_dataset_subset_cache", {})
    sig = _frame_signature(df)
    key = (scope_tag, dataset_key, sig)
    if key in cache:
        return cache[key]
    subset = df[df["dataset_key"].astype(str).str.strip() == str(dataset_key).strip()].copy()
    cache[key] = subset
    return subset


def _get_cached_series_and_stats(
    subset: pd.DataFrame,
    scope_tag: str,
    dataset_key: str,
    region_name: str,
    indicator_name: str,
    category_name: str,
) -> tuple[pd.DataFrame, Dict[str, object]]:
    cache = st.session_state.setdefault("_series_stats_cache", {})
    sig = _frame_signature(subset)
    key = (scope_tag, dataset_key, region_name, indicator_name, category_name, sig)
    if key in cache:
        return cache[key]
    series_df = series_filter(
        df=subset,
        dataset_key=dataset_key,
        region_name=region_name,
        indicator_name=indicator_name,
        category_name=category_name,
    )
    stats = build_stats(series_df) if not series_df.empty else {}
    cache[key] = (series_df, stats)
    return series_df, stats


def _get_cached_events(event_source: pd.DataFrame, scope_tag: str) -> pd.DataFrame:
    cache = st.session_state.setdefault("_events_cache", {})
    sig = _frame_signature(event_source)
    key = (scope_tag, sig)
    if key in cache:
        return cache[key]
    events = _collect_new_events(event_source)
    cache[key] = events
    return events


def _seeded_api_key() -> str:
    def _from_dotenv() -> str:
        dotenv_path = Path(".env")
        if not dotenv_path.exists():
            return ""
        try:
            for raw_line in dotenv_path.read_text(encoding="utf-8").splitlines():
                line = raw_line.strip()
                if not line or line.startswith("#") or "=" not in line:
                    continue
                key, value = line.split("=", 1)
                if key.strip() in {"api_key", "API_KEY", "KOSIS_API_KEY"}:
                    return value.strip().strip("'").strip('"')
        except Exception:
            return ""
        return ""

    try:
        secret_value = (
            st.secrets.get("api_key", "")
            or st.secrets.get("API_KEY", "")
            or st.secrets.get("KOSIS_API_KEY", "")
        )
    except Exception:  # noqa: BLE001
        secret_value = ""
    return str(
        secret_value
        or os.getenv("api_key", "")
        or os.getenv("API_KEY", "")
        or os.getenv("KOSIS_API_KEY", "")
        or _from_dotenv()
    )


def _seeded_access_code() -> str:
    try:
        secret_value = st.secrets.get("access_code", "") or st.secrets.get("ACCESS_CODE", "")
    except Exception:  # noqa: BLE001
        secret_value = ""
    return str(secret_value or os.getenv("access_code", "") or os.getenv("ACCESS_CODE", ""))


def _latest_git_commit_meta() -> Dict[str, str]:
    try:
        sha = subprocess.check_output(
            ["git", "rev-parse", "--short", "HEAD"],
            text=True,
            stderr=subprocess.DEVNULL,
        ).strip()
        committed_at = subprocess.check_output(
            ["git", "log", "-1", "--format=%cI"],
            text=True,
            stderr=subprocess.DEVNULL,
        ).strip()
        committed_at_display = committed_at
        try:
            dt = datetime.fromisoformat(committed_at)
            committed_at_display = dt.strftime("%Y-%m-%d %H:%M:%S (KST)")
        except Exception:
            committed_at_display = committed_at
        return {"sha": sha, "committed_at": committed_at_display}
    except Exception:
        return {"sha": "-", "committed_at": "-"}


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


def _extract_youtube_video_id(raw_url: str) -> str:
    url = str(raw_url or "").strip()
    if not url:
        return ""

    parsed = urlparse(url)
    host = parsed.netloc.lower()
    path = parsed.path.strip("/")

    if "youtube.com" in host:
        query = parse_qs(parsed.query)
        video_id = (query.get("v") or [""])[0].strip()
        if video_id:
            return video_id
        if path.startswith("live/"):
            live_id = path.split("/", 1)[1].strip()
            if live_id:
                return live_id
        if path.startswith("shorts/"):
            short_id = path.split("/", 1)[1].strip()
            if short_id:
                return short_id
        if path.startswith("embed/"):
            embed_id = path.split("/", 1)[1].strip()
            if embed_id:
                return embed_id
    elif "youtu.be" in host and path:
        short_id = path.split("/", 1)[0].strip()
        if short_id:
            return short_id

    direct_id = re.fullmatch(r"[A-Za-z0-9_-]{11}", url)
    if direct_id:
        return url
    return ""


def _render_youtube_player(video_id: str, audio_only_mode: bool, player_nonce: int = 0) -> None:
    embed_url = (
        f"https://www.youtube.com/embed/{video_id}"
        f"?autoplay=1&rel=0&playsinline=1&enablejsapi=1&nonce={int(player_nonce)}"
    )
    if audio_only_mode:
        hidden_player_html = (
            f'<iframe src="{embed_url}" '
            'allow="autoplay; encrypted-media; picture-in-picture" '
            'style="position:absolute; left:-99999px; width:1px; height:1px; border:0;" '
            'title="youtube-audio-only"></iframe>'
        )
        st.components.v1.html(hidden_player_html, height=1, scrolling=False)
    else:
        st.components.v1.iframe(embed_url, width=1100, height=560, scrolling=False)


def _render_youtube_display_content() -> None:
    st.caption("유튜브 URL 또는 영상 ID를 입력하면 재생됩니다.")
    if "_youtube_active_video_id" not in st.session_state:
        st.session_state["_youtube_active_video_id"] = ""
    if "_youtube_last_input" not in st.session_state:
        st.session_state["_youtube_last_input"] = ""
    if "_youtube_player_nonce" not in st.session_state:
        st.session_state["_youtube_player_nonce"] = 0

    input_url = st.text_input(
        "YouTube URL",
        key="youtube_popup_input_url",
        placeholder="예: https://www.youtube.com/watch?v=dQw4w9WgXcQ",
    )
    play_clicked = st.button("재생", key="youtube_play_btn", use_container_width=True)
    parsed_video_id = _extract_youtube_video_id(input_url)
    input_changed = input_url != str(st.session_state.get("_youtube_last_input", ""))
    st.session_state["_youtube_last_input"] = input_url

    # Enter 입력으로 rerun되는 경우와 재생 버튼 클릭 모두 즉시 반영
    if parsed_video_id and (play_clicked or input_changed):
        st.session_state["_youtube_active_video_id"] = parsed_video_id
        st.session_state["_youtube_player_nonce"] = int(st.session_state.get("_youtube_player_nonce", 0)) + 1
        st.rerun()
    elif play_clicked and not parsed_video_id:
        st.session_state["_youtube_active_video_id"] = ""
        st.warning("유효한 YouTube URL(또는 11자리 영상 ID)을 입력해 주세요.")

    audio_only_mode = st.toggle("오디오 전용(화면 숨김)", key="youtube_audio_only_mode", value=False)
    active_video_id = str(st.session_state.get("_youtube_active_video_id", "")).strip()
    player_nonce = int(st.session_state.get("_youtube_player_nonce", 0))
    if active_video_id:
        _render_youtube_player(video_id=active_video_id, audio_only_mode=audio_only_mode, player_nonce=player_nonce)
        if audio_only_mode:
            st.caption("오디오 전용 모드: 화면은 숨기고 소리만 재생 중입니다.")
        st.caption(f"영상 ID: {active_video_id}")

    if st.button("닫기", key="youtube_popup_close_btn"):
        st.session_state["_show_youtube_popup"] = False
        st.rerun()


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
    def _style_map(
        base: pd.io.formats.style.Styler,
        func: Any,
        subset: List[str],
    ) -> pd.io.formats.style.Styler:
        if hasattr(base, "map"):
            return base.map(func, subset=subset)
        return base.applymap(func, subset=subset)

    styler = df.style.set_properties(**{"text-align": "center"}).set_table_styles(
        [
            {"selector": "table", "props": [("width", "100%")]},
            {"selector": "th", "props": [("text-align", "center !important")]},
            {"selector": "td", "props": [("text-align", "center !important")]},
        ],
        overwrite=False,
    )
    if "최고" in df.columns:
        styler = _style_map(styler, lambda _: "color:#b91c1c;font-weight:700;", subset=["최고"])
    if "최저" in df.columns:
        styler = _style_map(styler, lambda _: "color:#1d4ed8;font-weight:700;", subset=["최저"])
    new_cols = [col for col in ["최고 시점", "최저 시점", "비고"] if col in df.columns]
    if new_cols:
        styler = _style_map(
            styler,
            lambda v: "color:#f59e0b;font-weight:700;" if "NEW" in str(v).strip() else "",
            subset=new_cols,
        )
    return styler


def _render_current_level_summary(df: pd.DataFrame, region: str, labels: Dict[str, str]) -> None:
    lag = 2 if ("prd_se" in df.columns and not df["prd_se"].dropna().empty and str(df["prd_se"].dropna().iloc[0]).upper() == "H") else 12
    snapshot_df, meta = _build_activity_snapshot(df, region=region, lag=lag)
    if not meta.get("ok") or snapshot_df.empty:
        return

    def _find_metric(tokens: List[str], exclude_tokens: List[str] | None = None) -> pd.Series | None:
        exclude_tokens = exclude_tokens or []
        for _, row in snapshot_df.iterrows():
            norm = _norm_indicator_name(row.get("지표", ""))
            if any(token in norm for token in tokens) and not any(ex in norm for ex in exclude_tokens):
                return row
        return None

    metric_rows = [
        ("취업자수", _find_metric(["취업자수", "취업자"], exclude_tokens=["고용률", "실업률"])),
        ("고용률", _find_metric(["고용률"], exclude_tokens=["15~64"])),
        ("실업률", _find_metric(["실업률"])),
        ("경제활동참가율", _find_metric(["경제활동참가율"])),
    ]

    available_metrics = [(title, row) for title, row in metric_rows if row is not None]
    if not available_metrics:
        return

    latest_period = _fmt_period(meta.get("latest_period"), str(meta.get("prd_se", "M")))
    st.markdown("#### 현재 수준 요약")
    cols = st.columns(len(available_metrics))
    for col, (title, row) in zip(cols, available_metrics):
        with col:
            unit = str(row.get("unit", ""))
            latest_value = row.get("latest_value")
            delta_value = row.get("delta_value")
            value_text = _fmt_num(latest_value, unit)
            if "%" in unit or "율" in title:
                delta_text = "-" if pd.isna(delta_value) else f"{float(delta_value):+,.1f}%p"
            else:
                delta_text = _fmt_num(delta_value, unit)
                if delta_text != "-":
                    delta_text = f"전년동월 대비 {delta_text}"
            sub = f"{latest_period} 기준"
            if delta_text != "-":
                sub = f"{sub} · {delta_text}"
            _card(title, value_text, sub)
    st.markdown("---")


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
            "구간": "최근 10년",
            "최고": _fmt_num(stats.get(f"{prefix}_max_10y_value"), display_unit),
            "최고 시점": _fmt_period(stats.get(f"{prefix}_max_10y_period"), prd_se),
            "최저": _fmt_num(stats.get(f"{prefix}_min_10y_value"), display_unit),
            "최저 시점": _fmt_period(stats.get(f"{prefix}_min_10y_period"), prd_se),
            "비고": _remark_new(
                bool(stats.get(f"{prefix}_is_new_max_10y")),
                bool(stats.get(f"{prefix}_is_new_min_10y")),
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
    scope_tag: str,
) -> None:
    cfg = next(x for x in datasets if x.key == dataset_key)
    subset = _get_cached_dataset_subset(df, scope_tag=scope_tag, dataset_key=dataset_key)
    st.subheader(cfg.title)
    if subset.empty:
        counts = _dataset_row_counts(df)
        st.warning("해당 데이터가 없습니다.")
        st.caption(f"진단: scope={scope_tag}, dataset_key={dataset_key}, available={counts}")
        return

    region_options = [r for r in region_pool if r in subset["region_name"].unique()]
    if not region_options:
        region_options = sorted(subset["region_name"].dropna().unique().tolist())
    default_region_index = region_options.index(default_region) if default_region in region_options else 0

    if dataset_key == "activity":
        # Keep activity indicators on one line by giving wider horizontal space.
        col1, col2 = st.columns([0.6, 5.4])
    elif dataset_key in {"industry", "occupation", "age", "status"}:
        # Narrow region control and widen classification area.
        col1, col2 = st.columns([0.6, 3.4])
    else:
        col1, col2, _ = st.columns([0.6, 2.2, 2.2])

    state_prefix = f"{scope_tag}_{dataset_key}"
    region_state_key = f"{state_prefix}_region"
    indicator_state_key = f"{state_prefix}_indicator"
    category_state_key = f"{state_prefix}_category"

    if st.session_state.get(region_state_key) not in region_options:
        st.session_state[region_state_key] = region_options[default_region_index]

    indicators = sorted(subset["indicator_name"].dropna().unique().tolist())
    if dataset_key == "activity":
        indicators = _order_activity_indicators(indicators)

    def _pick_auto_indicator(region_name: str) -> str:
        if not indicators:
            return ""
        drop_labels = {"시도별", "산업별", "산업명", "직업별", "직종별"}
        region_slice = subset[subset["region_name"] == region_name]
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

        def _priority(ind_name: str) -> tuple[int, int, str]:
            compact = str(ind_name).replace(" ", "")
            score = 0
            if "취업자(천명)" in compact:
                score += 300
            elif "취업자" in compact and "천명" in compact:
                score += 220
            elif "취업자" in compact:
                score += 120
            return (score, _category_count(ind_name), str(ind_name))

        return sorted(indicators, key=_priority, reverse=True)[0]

    if indicators and st.session_state.get(indicator_state_key) not in indicators:
        default_indicator = indicators[0]
        if dataset_key in {"industry", "occupation", "age", "status"}:
            default_indicator = _pick_auto_indicator(st.session_state[region_state_key])
        st.session_state[indicator_state_key] = default_indicator

    with col1:
        region_input = st.selectbox(
            "지역",
            region_options,
            index=region_options.index(st.session_state[region_state_key]),
            key=region_state_key,
        )

    indicator_input = st.session_state.get(indicator_state_key, indicators[0] if indicators else "")
    category_container = col2
    if dataset_key == "activity" and indicators:
        with col2:
            indicator_input = st.radio(
                "지표",
                indicators,
                index=indicators.index(indicator_input) if indicator_input in indicators else 0,
                key=indicator_state_key,
                horizontal=True,
            )
    elif dataset_key in {"industry", "occupation", "age", "status"} and indicators:
        indicator_input = _pick_auto_indicator(region_input)
        st.session_state[indicator_state_key] = indicator_input

    category_input = st.session_state.get(category_state_key, "")
    if cfg.has_category:
        category_pool = subset[subset["region_name"] == region_input]
        if indicator_input:
            category_pool = category_pool[category_pool["indicator_name"] == indicator_input]
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
        if categories:
            if category_input not in categories:
                category_input = categories[0]
                st.session_state[category_state_key] = category_input
            with category_container:
                if dataset_key in {"industry", "occupation", "age", "status"}:
                    category_input = st.radio(
                        cfg.category_label,
                        categories,
                        index=categories.index(category_input),
                        key=category_state_key,
                        horizontal=True,
                    )
                else:
                    category_input = st.selectbox(
                        cfg.category_label,
                        categories,
                        index=categories.index(category_input),
                        key=category_state_key,
                    )
        else:
            category_input = ""
            st.session_state[category_state_key] = ""

    region = str(st.session_state.get(region_state_key, region_input))
    indicator = str(st.session_state.get(indicator_state_key, indicator_input))
    category = str(st.session_state.get(category_state_key, category_input if cfg.has_category else ""))
    if dataset_key in {"industry", "occupation", "age", "status"} and indicator:
        st.caption(f"지표: {indicator}")

    series_df, stats = _get_cached_series_and_stats(
        subset=subset,
        scope_tag=scope_tag,
        dataset_key=dataset_key,
        region_name=region,
        indicator_name=indicator,
        category_name=category,
    )
    if series_df.empty:
        st.info("선택한 조건의 시계열 데이터가 없습니다.")
        return

    prd_se = str(series_df["prd_se"].iloc[-1]).upper() if "prd_se" in series_df.columns else "M"
    labels = _time_labels(datasets)
    latest_period = _fmt_period(stats.get("latest_period"), prd_se)
    unit = str(series_df["unit"].dropna().iloc[-1]) if not series_df["unit"].dropna().empty else ""

    st.caption(f"최신 기준{labels['point']}: {latest_period}")
    card_specs = [
        ("최신", stats.get("level_latest_value"), latest_period, False, ""),
        ("전체기간 최고", stats.get("level_max_all_value"), _fmt_period(stats.get("level_max_all_period"), prd_se), bool(stats.get("level_is_new_max_all")), "value-max"),
        ("전체기간 최저", stats.get("level_min_all_value"), _fmt_period(stats.get("level_min_all_period"), prd_se), bool(stats.get("level_is_new_min_all")), "value-min"),
        ("최근 10년 중 최고", stats.get("level_max_10y_value"), _fmt_period(stats.get("level_max_10y_period"), prd_se), bool(stats.get("level_is_new_max_10y")), "value-max"),
        ("최근 10년 중 최저", stats.get("level_min_10y_value"), _fmt_period(stats.get("level_min_10y_period"), prd_se), bool(stats.get("level_is_new_min_10y")), "value-min"),
        ("최근 5년 중 최고", stats.get("level_max_5y_value"), _fmt_period(stats.get("level_max_5y_period"), prd_se), bool(stats.get("level_is_new_max_5y")), "value-max"),
        ("최근 5년 중 최저", stats.get("level_min_5y_value"), _fmt_period(stats.get("level_min_5y_period"), prd_se), bool(stats.get("level_is_new_min_5y")), "value-min"),
    ]
    card_cols = st.columns(len(card_specs))
    for col, (title, raw_value, sub_text, is_new, value_class) in zip(card_cols, card_specs):
        with col:
            _card(
                title,
                _fmt_num(raw_value, unit),
                str(sub_text),
                bool(is_new),
                value_class,
            )

    chart_df = series_df
    window_options = ["전체", "최근 10년", "최근 5년"]
    st.markdown("<hr style='margin:18px 0 8px 0; border:0; border-top:1px solid #e5e7eb;'>", unsafe_allow_html=True)
    st.markdown("**데이터 기간**")
    window_sel = st.radio(
        "데이터 기간",
        window_options,
        horizontal=True,
        key=f"{state_prefix}_chart_window",
        label_visibility="collapsed",
    )
    st.markdown(f"#### {labels['trend']} 추이")
    if window_sel != "전체":
        years = 10 if "10" in window_sel else 5
        latest_ts = stats.get("latest_period")
        if pd.notna(latest_ts):
            cutoff = pd.Timestamp(latest_ts) - pd.DateOffset(years=years)
            chart_df = series_df[series_df["period"] >= cutoff].copy()
    level_df = chart_df[["period", "value"]].dropna(subset=["value"]).copy()
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
    yoy_df = chart_df[["period", "yoy_abs", "yoy_pct"]].dropna(
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

if "_show_youtube_popup" not in st.session_state:
    st.session_state["_show_youtube_popup"] = False

if hasattr(st, "dialog"):
    @st.dialog("경제활동인구 모니터링")
    def _open_youtube_dialog() -> None:
        _render_youtube_display_content()
else:
    def _open_youtube_dialog() -> None:
        pass

title_col, button_col = st.columns([0.94, 0.06])
with title_col:
    git_meta = _latest_git_commit_meta()
    st.markdown(
        (
            "<h1 style='margin:0;'>경제활동인구 모니터링 "
            f"<span style='font-size:0.50em; font-weight:500; color:#9ca3af;'>"
            f"(Commit: {git_meta.get('sha', '-')} | {git_meta.get('committed_at', '-')})"
            "</span></h1>"
        ),
        unsafe_allow_html=True,
    )
with button_col:
    if st.button("Y", key="open_youtube_dialog_btn", use_container_width=True, help="유튜브 디스플레이 열기"):
        st.session_state["_show_youtube_popup"] = True

if hasattr(st, "dialog") and st.session_state.get("_show_youtube_popup", False):
    _open_youtube_dialog()
    # Open request is one-shot; avoid reopening on unrelated reruns (tab switch, data load, etc.)
    st.session_state["_show_youtube_popup"] = False
elif not hasattr(st, "dialog") and st.session_state.get("_show_youtube_popup", False):
    with st.expander("경제활동인구 모니터링", expanded=True):
        _render_youtube_display_content()
    st.session_state["_show_youtube_popup"] = False

with st.sidebar:
    st.subheader("데이터 제어")
    if st.button("데이터 새로고침"):
        st.cache_data.clear()
        st.session_state["_force_data_refresh"] = True
        st.session_state.pop("_loaded_api_key", None)
        st.session_state.pop("_loaded_data_version", None)
        st.session_state.pop("_loaded_scope_data", None)
        st.session_state.pop("_loaded_errors", None)
        st.session_state.pop("_loaded_empty_data_warnings", None)
        st.session_state.pop("_loaded_debug_logs", None)
        st.session_state.pop("_schema_recovery_attempted", None)
        st.session_state.pop("_dataset_subset_cache", None)
        st.session_state.pop("_series_stats_cache", None)
        st.session_state.pop("_events_cache", None)
        st.rerun()
    sidebar_status = st.empty()
    sidebar_progress_box = st.empty()

is_sido_mode_default = bool(st.session_state.get("scope_toggle", True))
is_sido_mode = st.toggle(
    "시도",
    value=is_sido_mode_default,
    key="scope_toggle",
)
requested_scopes = ["province"] if is_sido_mode else ["province", "gyeonggi31"]

api_key = _seeded_api_key()

if not api_key:
    st.warning(
        "KOSIS API key가 설정되지 않았습니다. "
        "환경변수 `api_key`/`API_KEY`/`KOSIS_API_KEY`, `.env`, 또는 Streamlit secrets를 확인하세요."
    )
    st.stop()

loading_notice = st.empty()
loading_notice.info("데이터 불러오는 중... (기본: 시도, 시군은 필요 시 추가 로딩)")
loading_progress = st.empty()
try:
    force_data_refresh = bool(st.session_state.pop("_force_data_refresh", False))
    cached_scope_data = st.session_state.get("_loaded_scope_data")
    if (
        not force_data_refresh
        and
        st.session_state.get("_loaded_api_key") == api_key
        and st.session_state.get("_loaded_data_version") == DATA_MODEL_VERSION
        and isinstance(cached_scope_data, dict)
    ):
        scope_data = {
            "province": cached_scope_data.get("province", pd.DataFrame()),
            "gyeonggi31": cached_scope_data.get("gyeonggi31", pd.DataFrame()),
        }
        load_errors = st.session_state.get("_loaded_errors", [])
        empty_data_warnings = st.session_state.get("_loaded_empty_data_warnings", [])
        debug_logs = st.session_state.get("_loaded_debug_logs", [])
        missing_scopes = [scope for scope in requested_scopes if not _is_valid_scope_data(scope_data, [scope])]
        if missing_scopes:
            loading_notice.info(
                "데이터 추가 불러오는 중... "
                + ("경기 31개 시군" if "gyeonggi31" in missing_scopes else "전국 17개 시도")
            )
            new_scope_data, add_errors, add_debug_logs, add_warnings = load_data_with_local_cache(
                api_key=api_key,
                data_model_version=DATA_MODEL_VERSION,
                status_box=sidebar_status,
                progress_box=sidebar_progress_box,
                main_status_box=loading_notice,
                main_progress_box=loading_progress,
                scopes=missing_scopes,
                force_refresh=False,
                check_interval_hours=24,
            )
            for scope in missing_scopes:
                scope_data[scope] = new_scope_data.get(scope, pd.DataFrame())
            load_errors = [*load_errors, *add_errors]
            empty_data_warnings = [*empty_data_warnings, *add_warnings]
            debug_logs = [*debug_logs, *add_debug_logs]
            _clear_derived_caches()
    else:
        scope_data, load_errors, debug_logs, empty_data_warnings = load_data_with_local_cache(
            api_key=api_key,
            data_model_version=DATA_MODEL_VERSION,
            status_box=sidebar_status,
            progress_box=sidebar_progress_box,
            main_status_box=loading_notice,
            main_progress_box=loading_progress,
            scopes=requested_scopes,
            force_refresh=force_data_refresh,
            check_interval_hours=24,
        )
        scope_data = {
            "province": scope_data.get("province", pd.DataFrame()),
            "gyeonggi31": scope_data.get("gyeonggi31", pd.DataFrame()),
        }
        st.session_state["_loaded_api_key"] = api_key
        st.session_state["_loaded_data_version"] = DATA_MODEL_VERSION
        st.session_state["_loaded_scope_data"] = scope_data
        st.session_state["_loaded_errors"] = load_errors
        st.session_state["_loaded_empty_data_warnings"] = empty_data_warnings
        st.session_state["_loaded_debug_logs"] = debug_logs
        _clear_derived_caches()
    if not _is_valid_scope_data(scope_data, requested_scopes):
        raise RuntimeError("필수 조회 범위 데이터 로딩에 실패했습니다.")
    st.session_state["_loaded_scope_data"] = scope_data
    st.session_state["_loaded_errors"] = load_errors
    st.session_state["_loaded_empty_data_warnings"] = empty_data_warnings
    st.session_state["_loaded_debug_logs"] = debug_logs
finally:
    loading_notice.empty()
    loading_progress.empty()

if load_errors:
    st.error("일부 데이터셋 조회 중 오류가 발생했습니다.")
    for err in load_errors:
        st.write(f"- {err}")
if empty_data_warnings:
    st.warning("일부 데이터셋에서 API 응답이 비어 있거나 파싱 결과가 0건입니다.")
    for msg in empty_data_warnings:
        st.write(f"- {msg}")

if debug_logs:
    with st.sidebar:
        with st.expander("진단 로그 보기", expanded=bool(load_errors)):
            st.code("\n".join(debug_logs[-300:]))

scope_label = "전국 17개 시도" if is_sido_mode else "경기 31개 시군"
region_scope = "province" if is_sido_mode else "gyeonggi31"
is_gyeonggi31_mode = region_scope == "gyeonggi31"
active_datasets = datasets_for_scope(region_scope)
st.caption(f"조회범위: {scope_label}")

data = scope_data.get(region_scope, pd.DataFrame())

if data.empty:
    st.session_state.pop("_schema_recovery_attempted", None)
    st.warning("조회된 데이터가 없습니다. API 파라미터를 확인하세요.")
    st.stop()
missing_columns = sorted(REQUIRED_SCOPE_COLUMNS - set(data.columns))
if missing_columns:
    if not st.session_state.get("_schema_recovery_attempted", False):
        st.session_state["_schema_recovery_attempted"] = True
        st.cache_data.clear()
        st.session_state.pop("_loaded_api_key", None)
        st.session_state.pop("_loaded_data_version", None)
        st.session_state.pop("_loaded_scope_data", None)
        st.session_state.pop("_loaded_errors", None)
        st.session_state.pop("_loaded_empty_data_warnings", None)
        st.session_state.pop("_loaded_debug_logs", None)
        st.session_state.pop("_dataset_subset_cache", None)
        st.session_state.pop("_series_stats_cache", None)
        st.session_state.pop("_events_cache", None)
        st.warning("데이터 스키마 불일치를 감지해 자동으로 데이터를 다시 불러옵니다.")
        st.rerun()
    present_columns = sorted(map(str, data.columns.tolist()))
    preview_columns = ", ".join(present_columns[:20]) if present_columns else "(없음)"
    st.error(
        "데이터 스키마가 올바르지 않습니다. "
        f"누락 컬럼: {', '.join(missing_columns)}. 현재 컬럼: {preview_columns}. "
        "사이드바의 진단 로그를 확인해 주세요."
    )
    st.stop()
st.session_state.pop("_schema_recovery_attempted", None)

region_pool = TARGET_REGIONS if region_scope == "province" else GYEONGGI_SIGUNGU
default_region = "경기도" if region_scope == "province" else (GYEONGGI_SIGUNGU[0] if GYEONGGI_SIGUNGU else "")
visible_data = data[data["region_name"].isin(region_pool)].copy()
if visible_data.empty:
    st.warning("선택한 범위에서 조회된 데이터가 없습니다.")
    st.stop()

labels = _time_labels(active_datasets)

page_options = [
    "① NEW RECORDS",
    "② 경제활동인구현황",
    "③ 연령별 취업자",
    "④ 종사상지위별 취업자",
    "⑤ 산업별 취업자수",
    "⑥ 직종별 취업자수",
    "⑦ 요약",
    "⑧ 시군 유형화·정책매칭",
]
active_page = st.radio("메뉴", page_options, horizontal=True, key="active_page", label_visibility="collapsed")

needs_events = active_page in {"① NEW RECORDS", "⑦ 요약"}
events = pd.DataFrame()
if needs_events:
    if region_scope == "gyeonggi31":
        event_source = data[data["region_name"].isin(GYEONGGI_SIGUNGU)].copy()
    else:
        event_source = visible_data
    events = _get_cached_events(event_source, scope_tag=region_scope)

if active_page == "① NEW RECORDS":
    _render_new_history_tab(events)
elif active_page == "② 경제활동인구현황":
    _render_dataset(
        visible_data,
        "activity",
        region_pool,
        default_region,
        active_datasets,
        is_gyeonggi31_mode,
        scope_tag=region_scope,
    )
elif active_page == "③ 연령별 취업자":
    _render_dataset(
        visible_data,
        "age",
        region_pool,
        default_region,
        active_datasets,
        is_gyeonggi31_mode,
        scope_tag=region_scope,
    )
elif active_page == "④ 종사상지위별 취업자":
    _render_dataset(
        visible_data,
        "status",
        region_pool,
        default_region,
        active_datasets,
        is_gyeonggi31_mode,
        scope_tag=region_scope,
    )
elif active_page == "⑤ 산업별 취업자수":
    _render_dataset(
        visible_data,
        "industry",
        region_pool,
        default_region,
        active_datasets,
        is_gyeonggi31_mode,
        scope_tag=region_scope,
    )
elif active_page == "⑥ 직종별 취업자수":
    _render_dataset(
        visible_data,
        "occupation",
        region_pool,
        default_region,
        active_datasets,
        is_gyeonggi31_mode,
        scope_tag=region_scope,
    )
elif active_page == "⑦ 요약":
    st.subheader("요약(간략)")
    if region_scope == "gyeonggi31":
        summary_scope = "31개 시군"
        sigungu_options = _get_report_region_options(events, summary_scope)
        if not sigungu_options:
            sigungu_options = sorted(visible_data["region_name"].dropna().astype(str).str.strip().unique().tolist())

        selected_sigungu = ""
        selected_report_month = None
        c1, c2 = st.columns(2)
        with c1:
            if sigungu_options:
                default_sigungu = str(st.session_state.get("summary_sigungu", sigungu_options[0]))
                if default_sigungu not in sigungu_options:
                    default_sigungu = sigungu_options[0]
                selected_sigungu = st.selectbox(
                    "시군 선택",
                    sigungu_options,
                    index=sigungu_options.index(default_sigungu),
                    key="summary_sigungu",
                )
            else:
                st.selectbox(
                    "시군 선택",
                    ["선택 가능한 시군 없음"],
                    index=0,
                    key="summary_sigungu_empty",
                    disabled=True,
                )

        report_month_options = _get_report_period_options(
            events,
            summary_scope,
            selected_region=selected_sigungu or None,
        )
        with c2:
            if report_month_options:
                default_month = str(st.session_state.get("report_month", report_month_options[0]))
                if default_month not in report_month_options:
                    default_month = report_month_options[0]
                selected_report_month = st.selectbox(
                    f"리포트 기준{labels['point']}",
                    report_month_options,
                    index=report_month_options.index(default_month),
                    key="report_month",
                )
            else:
                st.selectbox(
                    f"리포트 기준{labels['point']}",
                    ["선택 가능한 기간 없음"],
                    index=0,
                    key="report_month_empty",
                    disabled=True,
                )

        _render_current_level_summary(data, selected_sigungu or "", labels)
        _render_new_monthly_report(
            events,
            report_scope=summary_scope,
            datasets=active_datasets,
            source_df=data,
            compact=True,
            include_consecutive_summary=False,
            selected_region=selected_sigungu or None,
            selected_month=selected_report_month,
        )
        _render_consecutive_change_summary(
            events=events,
            report_scope=summary_scope,
            datasets=active_datasets,
            source_df=data,
            selected_region=selected_sigungu or None,
            selected_month=selected_report_month,
        )
        st.markdown("---")
        _render_ai_insights(
            visible_data,
            sigungu_options if sigungu_options else region_pool,
            labels,
            card_fn=_card,
            datasets=active_datasets,
            events=events,
            report_scope=summary_scope,
            source_df=data,
            fixed_region=selected_sigungu or None,
            selected_month=selected_report_month,
            show_ai=SHOW_AI_FEATURES,
        )
    else:
        summary_scope = "17개 시도"
        province_options = _get_report_region_options(events, summary_scope)
        fallback_provinces = [
            region
            for region in region_pool
            if region == "전국" or region in visible_data["region_name"].dropna().astype(str).str.strip().unique().tolist()
        ]
        if not province_options:
            province_options = fallback_provinces
        else:
            province_options = [region for region in province_options if region != "전국"]
            province_options = ["전국"] + province_options

        selected_province = "전국"
        selected_report_month = None
        c1, c2 = st.columns(2)
        with c1:
            if province_options:
                default_province = str(st.session_state.get("summary_province", "전국"))
                if default_province not in province_options:
                    default_province = province_options[0]
                selected_province = st.selectbox(
                    "시도 선택",
                    province_options,
                    index=province_options.index(default_province),
                    key="summary_province",
                )
            else:
                st.selectbox(
                    "시도 선택",
                    ["선택 가능한 시도 없음"],
                    index=0,
                    key="summary_province_empty",
                    disabled=True,
                )

        report_month_options = _get_report_period_options(
            events,
            summary_scope,
            selected_region=selected_province,
        )
        with c2:
            if report_month_options:
                default_month = str(st.session_state.get("report_month_province", report_month_options[0]))
                if default_month not in report_month_options:
                    default_month = report_month_options[0]
                selected_report_month = st.selectbox(
                    f"리포트 기준{labels['point']}",
                    report_month_options,
                    index=report_month_options.index(default_month),
                    key="report_month_province",
                )
            else:
                st.selectbox(
                    f"리포트 기준{labels['point']}",
                    ["선택 가능한 기간 없음"],
                    index=0,
                    key="report_month_province_empty",
                    disabled=True,
                )

        _render_current_level_summary(data, selected_province, labels)
        _render_new_monthly_report(
            events,
            report_scope=summary_scope,
            datasets=active_datasets,
            source_df=data,
            compact=True,
            include_consecutive_summary=False,
            selected_region=selected_province,
            selected_month=selected_report_month,
        )
        _render_consecutive_change_summary(
            events=events,
            report_scope=summary_scope,
            datasets=active_datasets,
            source_df=data,
            selected_region=selected_province,
            selected_month=selected_report_month,
        )
        st.markdown("---")
        _render_ai_insights(
            visible_data,
            province_options if province_options else region_pool,
            labels,
            card_fn=_card,
            datasets=active_datasets,
            events=events,
            report_scope=summary_scope,
            source_df=data,
            fixed_region=selected_province,
            selected_month=selected_report_month,
            show_ai=SHOW_AI_FEATURES,
        )
elif active_page == "⑧ 시군 유형화·정책매칭":
    _render_sigungu_typology_tab(visible_data, is_gyeonggi31_mode=is_gyeonggi31_mode, datasets=active_datasets)

st.markdown(
    "<hr style='margin-top:2rem; margin-bottom:0.5rem;'>"
    "<p style='text-align:center; color:#6b7280; font-size:0.9rem;'>- created by alicia -</p>",
    unsafe_allow_html=True,
)




