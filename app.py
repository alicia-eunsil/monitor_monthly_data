from __future__ import annotations

import os
import re
from typing import Any, Dict, List, Optional

import altair as alt
import numpy as np
import pandas as pd
import streamlit as st

import src.config as app_config
from src.kosis_client import KosisClient
from src.transform import add_yoy, build_stats, normalize_records, series_filter

# Keep runtime resilient even if a deployment temporarily mixes app/config versions.
GYEONGGI_SIGUNGU = getattr(app_config, "GYEONGGI_SIGUNGU", [])
TARGET_REGIONS = app_config.TARGET_REGIONS
DatasetConfig = getattr(app_config, "DatasetConfig", Any)
datasets_for_scope = getattr(
    app_config,
    "datasets_for_scope",
    lambda _scope: getattr(app_config, "DATASETS", []),
)
default_end_period_by_prd_se = getattr(
    app_config,
    "default_end_period_by_prd_se",
    lambda _prd_se: app_config.default_end_period(),
)

st.set_page_config(
    page_title="Data Monitoring",
    page_icon="🍦",
    layout="wide",
)

st.markdown(
    """
<style>
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

OCCUPATION_CATEGORY_ORDER = [
    "관리자전문가",
    "관리자",
    "전문가및관련종사자",
    "사무종사자",
    "서비스판매종사자",
    "서비스종사자",
    "판매종사자",
    "농림어업숙련종사자",
    "기능기계조작조립단순노무종사자",
    "기능원및관련기능종사자",
    "장치기계조작및조립종사자",
    "단순노무종사자",
    "기타",
]

SIGUNGU_INDUSTRY_ORDER = [
    "계",
    "농업",
    "광제조업",
    "전기운수통신",
    "건설업",
    "도소매",
    "사업개인공공서비스",
]

SIGUNGU_OCCUPATION_ORDER = [
    "계",
    "관리자",
    "사무종사자",
    "서비스판매종사자",
    "농립어업종사자",
    "기능기계조작",
    "단순노무종사자",
]

SIGUNGU_STATUS_ORDER = [
    "계",
    "임금근로자",
    "-상용근로자",
    "-임시일용근로자",
    "비임금근로자",
]

SIGUNGU_AGE_ORDER = [
    "계",
    "15~29세",
    "30~49세",
    "50~64세",
    "65세이상",
    "15~64세",
    "55세이상",
]

STATUS_CATEGORY_ORDER = [
    "비임금근로자",
    "*자영업자",
    "-고용원이 있는 자영업자",
    "-고용원이 없는 자영업자",
    "-무급가족종사자",
    "임금근로자",
    "-상용근로자",
    "-임시근로자",
    "-일용근로자",
    "계",
]

AGE_CATEGORY_ORDER = [
    "15~24",
    "15~29",
    "15~64",
    "15~19",
    "20~29",
    "30~39",
    "40~49",
    "50~59",
    "60세 이상",
    "계",
]

DATA_MODEL_VERSION = "2026-03-13-halfyear-gyeonggi31-v1"


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


def _norm_occupation_category(text: str) -> str:
    s = str(text).strip()
    s = re.sub(r"^\*+\s*", "", s)
    s = re.sub(r"^\d+\s*", "", s)
    s = re.sub(r"\([^)]*\)", "", s)
    s = re.sub(r"[^0-9A-Za-z가-힣]", "", s)
    return s


def _order_occupation_categories(categories: List[str]) -> List[str]:
    order_map = {
        _norm_occupation_category(name): idx for idx, name in enumerate(OCCUPATION_CATEGORY_ORDER)
    }
    return sorted(
        categories,
        key=lambda x: (order_map.get(_norm_occupation_category(x), 999), x),
    )


def _norm_sigungu_industry(text: str) -> str:
    s = str(text).strip()
    s = re.sub(r"\s+", "", s)
    s = re.sub(r"[·ㆍ,/()\-]", "", s)
    return s


def _order_sigungu_industry_categories(categories: List[str]) -> List[str]:
    def _rank(cat: str) -> int:
        n = _norm_sigungu_industry(cat)
        if n in {"계", "합계", "전체"}:
            return 0
        if "농업" in n or "농림어업" in n:
            return 1
        if ("광" in n and "제조" in n) or "광업제조업" in n:
            return 2
        if "전기" in n and "운수" in n and "통신" in n:
            return 3
        if "건설" in n:
            return 4
        if "도소매" in n:
            return 5
        if "사업" in n and "개인" in n and ("공공" in n or "서비스" in n):
            return 6
        return 999

    return sorted(categories, key=lambda x: (_rank(x), x))


def _norm_age_category(text: str) -> str:
    s = str(text).strip()
    s = re.sub(r"\s+", "", s)
    s = s.replace("-", "~")
    s = s.replace("세이상", "이상")
    s = s.replace("세", "")
    return s


def _order_age_categories(categories: List[str]) -> List[str]:
    order_map = {_norm_age_category(name): idx for idx, name in enumerate(AGE_CATEGORY_ORDER)}
    return sorted(
        categories,
        key=lambda x: (order_map.get(_norm_age_category(x), 999), x),
    )


def _order_sigungu_age_categories(categories: List[str]) -> List[str]:
    def _rank(cat: str) -> int:
        n = _norm_age_category(cat)
        if n in {"계", "합계", "전체"}:
            return 0
        if "15~29" in n:
            return 1
        if "30~49" in n:
            return 2
        if "50~64" in n:
            return 3
        if "65이상" in n:
            return 4
        if "15~64" in n:
            return 5
        if "55이상" in n:
            return 6
        return 999

    return sorted(categories, key=lambda x: (_rank(x), x))


def _norm_status_category(text: str) -> str:
    s = str(text).strip()
    s = re.sub(r"^\*+\s*", "", s)
    s = re.sub(r"^-\s*", "", s)
    s = re.sub(r"\s+", "", s)
    return s


def _order_status_categories(categories: List[str]) -> List[str]:
    exact_order_map = {str(name).strip(): idx for idx, name in enumerate(STATUS_CATEGORY_ORDER)}
    norm_order_map = {_norm_status_category(name): idx for idx, name in enumerate(STATUS_CATEGORY_ORDER)}
    return sorted(
        categories,
        key=lambda x: (
            exact_order_map.get(str(x).strip(), norm_order_map.get(_norm_status_category(x), 999)),
            x,
        ),
    )


def _order_sigungu_status_categories(categories: List[str]) -> List[str]:
    def _rank(cat: str) -> int:
        n = _norm_status_category(cat)
        if n in {"계", "합계", "전체"}:
            return 0
        if n == "임금근로자":
            return 1
        if "상용근로자" in n:
            return 2
        if "임시일용근로자" in n or ("임시" in n and "일용" in n):
            return 3
        if "비임금근로자" in n:
            return 4
        return 999

    return sorted(categories, key=lambda x: (_rank(x), x))


def _order_sigungu_occupation_categories(categories: List[str]) -> List[str]:
    def _rank(cat: str) -> int:
        n = _norm_occupation_category(cat)
        if n in {"계", "합계", "전체"}:
            return 0
        if "관리자" in n:
            return 1
        if "사무종사자" in n:
            return 2
        if ("서비스" in n and "판매" in n) or "서비스판매종사자" in n:
            return 3
        if "농림어업종사자" in n or "농립어업종사자" in n or "농림어업" in n or "농립어업" in n:
            return 4
        if "기능" in n and "기계" in n and ("조작" in n or "조립" in n):
            return 5
        if "단순노무" in n:
            return 6
        return 999

    return sorted(categories, key=lambda x: (_rank(x), x))


def _seeded_api_key() -> str:
    try:
        secret_value = st.secrets.get("api_key", "") or st.secrets.get("API_KEY", "")
    except Exception:  # noqa: BLE001
        secret_value = ""
    return str(secret_value or os.getenv("api_key", "") or os.getenv("API_KEY", ""))


def _fmt_period(value: object, prd_se: str = "M") -> str:
    ts = pd.Timestamp(value)
    if pd.isna(ts):
        return "-"
    if str(prd_se).upper() == "H":
        month = int(ts.month)
        if month <= 6:
            return f"{ts.year}-상반기"
        return f"{ts.year}-하반기"
    return ts.strftime("%Y-%m")


def _time_labels(datasets: List[DatasetConfig]) -> Dict[str, str]:
    is_halfyear = bool(datasets) and all(str(cfg.prd_se).upper() == "H" for cfg in datasets)
    if is_halfyear:
        return {"point": "반기", "trend": "반기별", "yoy": "전년동기"}
    return {"point": "월", "trend": "월별", "yoy": "전년동월"}


def _fmt_num(value: object, unit: str = "", digits: int = 1) -> str:
    if value is None or pd.isna(value):
        return "-"
    if unit and "%" in unit:
        digits = 2
    return f"{float(value):,.{digits}f}{unit}"


def _new(flag: bool) -> str:
    return "<span class='new-badge'>NEW</span>" if flag else ""


def _remark_new(is_new_max: bool, is_new_min: bool) -> str:
    if is_new_max and is_new_min:
        return "최고·최저 NEW"
    if is_new_max:
        return "최고 NEW"
    if is_new_min:
        return "최저 NEW"
    return ""


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


def fetch_records_live(
    api_key: str,
    dataset_key: str,
    end_period: str,
    config_signature: str,
    datasets: List[DatasetConfig],
) -> Dict[str, Any]:
    cfg = next((x for x in datasets if x.key == dataset_key), None)
    if cfg is None:
        raise RuntimeError(f"Unknown dataset key: {dataset_key}")
    client = KosisClient(api_key=api_key)
    records, debug_logs = client.fetch_with_debug(cfg, end_prd_de=end_period)
    return {"records": records, "debug_logs": debug_logs}


def load_all_data_with_progress(
    api_key: str,
    status_box: Any,
    progress_box: Any,
    main_status_box: Optional[Any] = None,
    main_progress_box: Optional[Any] = None,
) -> tuple[Dict[str, pd.DataFrame], List[str], List[str]]:
    scope_defs = [
        ("province", "전국·17개 시도", datasets_for_scope("province")),
        ("gyeonggi31", "경기 31개 시군", datasets_for_scope("gyeonggi31")),
    ]
    total_steps = sum(len(ds) * 2 for _, _, ds in scope_defs) + len(scope_defs)
    step = 0
    frames_by_scope: Dict[str, List[pd.DataFrame]] = {k: [] for k, _, _ in scope_defs}
    errors: List[str] = []
    debug_logs: List[str] = []

    progress = progress_box.progress(0)
    main_progress = main_progress_box.progress(0) if main_progress_box is not None else None
    status = status_box
    main_status = main_status_box
    last_progress = -1

    def _set_progress(value: int) -> None:
        nonlocal last_progress
        safe_value = max(int(value), last_progress)
        last_progress = safe_value
        progress.progress(safe_value)
        if main_progress is not None:
            main_progress.progress(safe_value)

    def _set_info(message: str) -> None:
        status.info(message)
        if main_status is not None:
            main_status.info(message)

    def _set_error(message: str) -> None:
        status.error(message)

    def _set_success(message: str) -> None:
        status.success(message)

    for scope_key, scope_title, datasets in scope_defs:
        _set_info(f"{scope_title} 데이터셋 준비 중... ({step}/{total_steps})")
        step += 1
        _set_progress(min(100, int(step * 100 / total_steps)))
        for cfg in datasets:
            _set_info(f"[{scope_title}] 데이터 불러오는 중: {cfg.title} ({step}/{total_steps})")
            try:
                end_period = default_end_period_by_prd_se(cfg.prd_se)
                config_signature = "|".join(
                    [
                        cfg.tbl_id,
                        cfg.itm_id,
                        cfg.obj_l1,
                        cfg.obj_l2,
                        cfg.output_fields,
                        cfg.start_prd_de,
                        cfg.prd_se,
                        end_period,
                        scope_key,
                    ]
                )
                result = fetch_records_live(
                    api_key=api_key,
                    dataset_key=cfg.key,
                    end_period=end_period,
                    config_signature=config_signature,
                    datasets=datasets,
                )
                records = result.get("records", [])
                for line in result.get("debug_logs", []):
                    debug_logs.append(f"[{scope_key}:{cfg.key}] {line}")
                if records:
                    sample = records[0]
                    debug_logs.append(
                        f"[{scope_key}:{cfg.key}] sample_keys={list(sample.keys())[:12]}"
                    )
                    debug_logs.append(
                        f"[{scope_key}:{cfg.key}] sample_PRD_DE={sample.get('PRD_DE')} sample_DT={sample.get('DT')}"
                    )
            except Exception as exc:  # noqa: BLE001
                records = []
                errors.append(f"{scope_title} - {cfg.title}: {exc}")
                debug_logs.append(f"[{scope_key}:{cfg.key}] ERROR: {exc}")
            step += 1
            _set_progress(min(100, int(step * 100 / total_steps)))

            _set_info(f"[{scope_title}] 파싱 중: {cfg.title} ({step}/{total_steps})")
            parsed = normalize_records(cfg, records, region_scope=scope_key)
            debug_logs.append(
                f"[{scope_key}:{cfg.key}] parsed_rows={len(parsed)} raw_rows={len(records)}"
            )
            if not parsed.empty:
                frames_by_scope[scope_key].append(parsed)
            step += 1
            _set_progress(min(100, int(step * 100 / total_steps)))

    _set_info("지표 계산 및 통합 중...")
    data_by_scope: Dict[str, pd.DataFrame] = {}
    for scope_key, scope_title, _ in scope_defs:
        frames = frames_by_scope.get(scope_key, [])
        if frames:
            combined = pd.concat(frames, ignore_index=True)
            combined = add_yoy(combined)
            data_by_scope[scope_key] = combined
            debug_logs.append(f"[{scope_key}:all] combined_rows={len(combined)}")
        else:
            data_by_scope[scope_key] = pd.DataFrame()

    if all(df.empty for df in data_by_scope.values()):
        _set_progress(100)
        _set_error("데이터 로딩 실패")
        return data_by_scope, errors, debug_logs

    _set_progress(100)
    _set_success("로딩 완료")
    return data_by_scope, errors, debug_logs


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

    if dataset_key in {"industry", "occupation", "age", "status"}:
        col1, col2 = st.columns([1, 2])
        col3 = st.container()
    else:
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
            drop_labels = {"시도별", "산업별", "직업별", "직종별"}
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

def _collect_new_events(df: pd.DataFrame) -> pd.DataFrame:
    rows: List[Dict[str, str]] = []
    key_cols = ["dataset_key", "dataset_title", "region_name", "indicator_name", "category_name", "prd_se"]
    for _, series in df.groupby(key_cols, dropna=False):
        series = series.sort_values("period")
        prd_se = str(series["prd_se"].iloc[0]).upper() if "prd_se" in series.columns else "M"
        recent_window = 10 if prd_se == "H" else 60
        meta = {
            "데이터셋": str(series["dataset_title"].iloc[0]),
            "지역": str(series["region_name"].iloc[0]),
            "지표": str(series["indicator_name"].iloc[0]),
            "분류": str(series["category_name"].iloc[0]),
        }

        for metric_col, metric_label in [
            ("value", "원자료"),
            ("yoy_abs", "YoY(절대)"),
            ("yoy_pct", "YoY(증감률)"),
        ]:
            metric_df = series[["period", metric_col]].dropna(subset=[metric_col]).copy()
            if metric_df.empty:
                continue
            metric_df = metric_df.sort_values("period").reset_index(drop=True)

            prev_max = metric_df[metric_col].cummax().shift(1)
            prev_min = metric_df[metric_col].cummin().shift(1)
            prev_5y_max = metric_df[metric_col].shift(1).rolling(window=recent_window, min_periods=1).max()
            prev_5y_min = metric_df[metric_col].shift(1).rolling(window=recent_window, min_periods=1).min()

            for scope_label, scope_series in [("전체기간", prev_max), ("최근5년", prev_5y_max)]:
                is_new_max = metric_df[metric_col] > scope_series
                for _, row in metric_df[is_new_max.fillna(False)].iterrows():
                    rows.append(
                        {
                            **meta,
                            "기준월": _fmt_period(row["period"], prd_se),
                            "기준월_ts": pd.Timestamp(row["period"]),
                            "구분": metric_label,
                            "범위": scope_label,
                            "유형": "최고",
                            "이벤트": f"{metric_label} {scope_label} 최고 NEW",
                        }
                    )

            for scope_label, scope_series in [("전체기간", prev_min), ("최근5년", prev_5y_min)]:
                is_new_min = metric_df[metric_col] < scope_series
                for _, row in metric_df[is_new_min.fillna(False)].iterrows():
                    rows.append(
                        {
                            **meta,
                            "기준월": _fmt_period(row["period"], prd_se),
                            "기준월_ts": pd.Timestamp(row["period"]),
                            "구분": metric_label,
                            "범위": scope_label,
                            "유형": "최저",
                            "이벤트": f"{metric_label} {scope_label} 최저 NEW",
                        }
                    )
    if not rows:
        return pd.DataFrame(columns=["데이터셋", "지역", "지표", "분류", "기준월", "기준월_ts", "구분", "범위", "유형", "이벤트"])
    out = pd.DataFrame(rows).sort_values(
        ["기준월_ts", "데이터셋", "지역", "지표", "분류", "구분", "범위", "유형"],
        ascending=[False, True, True, True, True, True, True, True],
    )
    return out[["데이터셋", "지역", "지표", "분류", "기준월", "기준월_ts", "구분", "범위", "유형", "이벤트"]]


def _render_new_event_charts(events_view: pd.DataFrame, datasets: List[DatasetConfig]) -> None:
    if events_view.empty:
        st.info("그래프로 표시할 NEW 이벤트가 없습니다.")
        return

    view = events_view.copy()
    if "기준월_ts" in view.columns:
        view["기준월_dt"] = pd.to_datetime(view["기준월_ts"], errors="coerce")
    else:
        view["기준월_dt"] = pd.to_datetime(view["기준월"], errors="coerce")
    view = view.dropna(subset=["기준월_dt"])
    if view.empty:
        st.info("그래프로 표시할 NEW 이벤트가 없습니다.")
        return

    st.markdown("##### NEW 이벤트 분석 그래프")
    dataset_tab_order = [cfg.title for cfg in datasets]
    dataset_order_map = {name: idx for idx, name in enumerate(dataset_tab_order)}
    labels = _time_labels(datasets)

    def _render_six_charts(base_df: pd.DataFrame, section_title: str) -> None:
        st.markdown(f"###### {section_title}")
        if base_df.empty:
            st.info(f"{section_title} 데이터가 없습니다.")
            return

        row1_col1, row1_col2 = st.columns([1, 1])
        with row1_col1:
            month_summary = (
                base_df.groupby(["기준월", "기준월_dt"], as_index=False)
                .size()
                .rename(columns={"size": "NEW 건수"})
                .sort_values("기준월_dt")
            )
            month_chart = (
                alt.Chart(month_summary)
                .mark_line(point=True, color="#2563eb")
                .encode(
                    x=alt.X("기준월_dt:T", title=f"기준{labels['point']}"),
                    y=alt.Y("NEW 건수:Q", title="NEW 건수"),
                    tooltip=[
                        alt.Tooltip("기준월:N", title=f"기준{labels['point']}"),
                        alt.Tooltip("NEW 건수:Q", title="NEW 건수"),
                    ],
                )
                .properties(height=300, title=f"{labels['trend']} NEW 건수 추이")
            )
            st.altair_chart(month_chart, use_container_width=True)

        with row1_col2:
            type_summary = (
                base_df.groupby("유형", as_index=False)
                .size()
                .rename(columns={"size": "NEW 건수"})
                .sort_values("NEW 건수", ascending=False)
            )
            type_ratio_chart = (
                alt.Chart(type_summary)
                .mark_arc(innerRadius=65)
                .encode(
                    theta=alt.Theta("NEW 건수:Q"),
                    color=alt.Color("유형:N", title="유형", scale=alt.Scale(range=["#E45756", "#1D4ED8"])),
                    tooltip=[
                        alt.Tooltip("유형:N", title="유형"),
                        alt.Tooltip("NEW 건수:Q", title="NEW 건수"),
                    ],
                )
                .properties(height=300, title="유형별 NEW 비중")
            )
            st.altair_chart(type_ratio_chart, use_container_width=True)

        row2_col1, row2_col2 = st.columns([1, 1])
        with row2_col1:
            indicator_summary = (
                base_df.groupby("지표", as_index=False)
                .size()
                .rename(columns={"size": "NEW 건수"})
                .sort_values("NEW 건수", ascending=False)
            )
            indicator_chart = (
                alt.Chart(indicator_summary)
                .mark_bar(color="#2E8B57")
                .encode(
                    x=alt.X("NEW 건수:Q", title="NEW 건수"),
                    y=alt.Y("지표:N", sort="-x", title="지표"),
                    tooltip=[
                        alt.Tooltip("지표:N", title="지표"),
                        alt.Tooltip("NEW 건수:Q", title="NEW 건수"),
                    ],
                )
                .properties(height=max(260, len(indicator_summary) * 22), title="지표별 NEW 건수")
            )
            st.altair_chart(indicator_chart, use_container_width=True)

        with row2_col2:
            dataset_summary = (
                base_df.groupby("데이터셋", as_index=False)
                .size()
                .rename(columns={"size": "NEW 건수"})
            )
            dataset_summary["정렬순서"] = dataset_summary["데이터셋"].map(dataset_order_map).fillna(999)
            dataset_summary = dataset_summary.sort_values(["정렬순서", "데이터셋"]).drop(columns=["정렬순서"])
            dataset_chart = (
                alt.Chart(dataset_summary)
                .mark_bar(color="#7c3aed")
                .encode(
                    x=alt.X("NEW 건수:Q", title="NEW 건수"),
                    y=alt.Y("데이터셋:N", sort=dataset_tab_order, title="데이터셋"),
                    tooltip=[
                        alt.Tooltip("데이터셋:N", title="데이터셋"),
                        alt.Tooltip("NEW 건수:Q", title="NEW 건수"),
                    ],
                )
                .properties(height=max(260, len(dataset_summary) * 24), title="데이터셋별 NEW 건수")
            )
            st.altair_chart(dataset_chart, use_container_width=True)

        row3_col1, row3_col2 = st.columns([1, 1])
        with row3_col1:
            category_df = base_df.copy()
            category_df["분류"] = category_df["분류"].astype(str).str.strip().replace({"": "전체"})
            category_summary = (
                category_df.groupby("분류", as_index=False)
                .size()
                .rename(columns={"size": "NEW 건수"})
                .sort_values("NEW 건수", ascending=False)
                .head(15)
            )
            category_chart = (
                alt.Chart(category_summary)
                .mark_bar(color="#9467BD")
                .encode(
                    x=alt.X("NEW 건수:Q", title="NEW 건수"),
                    y=alt.Y("분류:N", sort="-x", title="분류"),
                    tooltip=[
                        alt.Tooltip("분류:N", title="분류"),
                        alt.Tooltip("NEW 건수:Q", title="NEW 건수"),
                    ],
                )
                .properties(height=max(260, len(category_summary) * 18), title="분류별 NEW 건수")
            )
            st.altair_chart(category_chart, use_container_width=True)

        with row3_col2:
            type_count_summary = (
                base_df.groupby("유형", as_index=False)
                .size()
                .rename(columns={"size": "NEW 건수"})
                .sort_values("NEW 건수", ascending=False)
            )
            type_count_chart = (
                alt.Chart(type_count_summary)
                .mark_bar(color="#f97316")
                .encode(
                    x=alt.X("유형:N", title="유형"),
                    y=alt.Y("NEW 건수:Q", title="NEW 건수"),
                    tooltip=[
                        alt.Tooltip("유형:N", title="유형"),
                        alt.Tooltip("NEW 건수:Q", title="NEW 건수"),
                    ],
                )
                .properties(height=300, title="유형별 NEW 건수")
            )
            st.altair_chart(type_count_chart, use_container_width=True)

    nation_df = view[view["지역"] == "전국"].copy()
    if nation_df.empty:
        nation_df = view.copy()
        _render_six_charts(nation_df, "전체 요약 (기본)")
    else:
        _render_six_charts(nation_df, "전국 요약 (기본)")

    region_options = sorted([r for r in view["지역"].dropna().unique().tolist() if r and r != "전국"])
    if not region_options:
        return

    st.markdown("---")
    st.markdown("###### 지역 선택 요약")
    default_region = "경기도" if "경기도" in region_options else region_options[0]
    selected_region = st.selectbox(
        "지역 선택",
        region_options,
        index=region_options.index(default_region),
        key="history_chart_region",
    )
    region_df = view[view["지역"] == selected_region].copy()
    _render_six_charts(region_df, f"{selected_region} 요약")


def _render_new_monthly_report(
    events: pd.DataFrame,
    report_scope: str,
    datasets: List[DatasetConfig],
) -> None:
    if events.empty:
        st.info("리포트로 표시할 NEW 이벤트가 없습니다.")
        return

    if report_scope == "경기도 전체":
        view = events[events["지역"] == "경기도"].copy()
        scope_title = "경기도 전체"
    else:
        view = events[events["지역"].isin(GYEONGGI_SIGUNGU)].copy()
        scope_title = "경기 31개 시군"
    if view.empty:
        st.info(f"{scope_title} 기준 NEW 이벤트가 없습니다.")
        return

    if "기준월_ts" in view.columns:
        view["기준월_dt"] = pd.to_datetime(view["기준월_ts"], errors="coerce")
    else:
        view["기준월_dt"] = pd.to_datetime(view["기준월"], errors="coerce")
    view = view.dropna(subset=["기준월_dt"])
    if view.empty:
        st.info(f"{scope_title} 기준 NEW 이벤트가 없습니다.")
        return

    month_table = (
        view[["기준월", "기준월_dt"]]
        .drop_duplicates()
        .sort_values("기준월_dt", ascending=False)
        .reset_index(drop=True)
    )
    month_list = month_table["기준월"].tolist()
    labels = _time_labels(datasets)
    selected_month = st.selectbox(f"리포트 기준{labels['point']}", month_list, key="report_month")
    month_df = view[view["기준월"] == selected_month].copy()
    if month_df.empty:
        st.info(f"선택한 기준{labels['point']}의 NEW 이벤트가 없습니다.")
        return
    selected_idx_list = month_table.index[month_table["기준월"] == selected_month].tolist()
    prev_month = (
        month_table.loc[selected_idx_list[0] + 1, "기준월"]
        if selected_idx_list and (selected_idx_list[0] + 1) < len(month_table)
        else None
    )
    prev_month_df = view[view["기준월"] == prev_month].copy() if prev_month else pd.DataFrame(columns=view.columns)

    def _fmt_delta(cur: int, prev: int | None) -> str:
        if prev is None:
            return ""
        diff = int(cur) - int(prev)
        if diff > 0:
            return f" (▲{diff:,})"
        if diff < 0:
            return f" (▼{abs(diff):,})"
        return " (→0)"

    st.markdown(f"#### {selected_month} NEW 리포트 ({scope_title})")
    total_count = len(month_df)
    max_count = int((month_df["유형"] == "최고").sum())
    min_count = int((month_df["유형"] == "최저").sum())
    prev_total_count = len(prev_month_df) if prev_month else None
    prev_max_count = int((prev_month_df["유형"] == "최고").sum()) if prev_month else None
    prev_min_count = int((prev_month_df["유형"] == "최저").sum()) if prev_month else None
    st.markdown(
        "\n".join(
            [
                f"##### {labels['point']} 요약",
                f"- 총 NEW 이벤트: **{total_count:,}건**{_fmt_delta(total_count, prev_total_count)}",
                f"- 최고 NEW: **{max_count:,}건**{_fmt_delta(max_count, prev_max_count)}",
                f"- 최저 NEW: **{min_count:,}건**{_fmt_delta(min_count, prev_min_count)}",
            ]
        )
    )

    ds_summary = (
        month_df.groupby("데이터셋", as_index=False)
        .size()
        .rename(columns={"size": "NEW 건수"})
    )
    dataset_tab_order = [cfg.title for cfg in datasets]
    ds_summary["정렬순서"] = ds_summary["데이터셋"].map(
        {name: idx for idx, name in enumerate(dataset_tab_order)}
    )
    ds_summary = ds_summary.sort_values(
        by=["정렬순서", "데이터셋"],
        ascending=[True, True],
        na_position="last",
    ).drop(columns=["정렬순서"])
    prev_ds_map: Dict[str, int] = {}
    if prev_month:
        prev_ds_map = (
            prev_month_df.groupby("데이터셋")
            .size()
            .astype(int)
            .to_dict()
        )
    ds_lines = ["##### 데이터셋별 건수"]
    ds_lines.extend(
        [
            f"- {row['데이터셋']}: **{int(row['NEW 건수']):,}건**"
            f"{_fmt_delta(int(row['NEW 건수']), prev_ds_map.get(str(row['데이터셋']))) if prev_month else ''}"
            for _, row in ds_summary.iterrows()
        ]
    )
    st.markdown("\n".join(ds_lines))

    type_summary = (
        month_df.groupby(["구분", "범위", "유형"], as_index=False)
        .size()
        .rename(columns={"size": "NEW 건수"})
    )
    metric_order = {"원자료": 0, "YoY(절대)": 1, "YoY(증감률)": 2}
    scope_order = {"전체기간": 0, "최근5년": 1}
    event_type_order = {"최고": 0, "최저": 1}
    type_summary["정렬_구분"] = type_summary["구분"].map(metric_order).fillna(999)
    type_summary["정렬_범위"] = type_summary["범위"].map(scope_order).fillna(999)
    type_summary["정렬_유형"] = type_summary["유형"].map(event_type_order).fillna(999)
    type_summary = type_summary.sort_values(
        ["정렬_구분", "정렬_범위", "정렬_유형", "구분", "범위", "유형"]
    ).drop(columns=["정렬_구분", "정렬_범위", "정렬_유형"])
    prev_type_map: Dict[tuple, int] = {}
    if prev_month:
        prev_type_map = (
            prev_month_df.groupby(["구분", "범위", "유형"])
            .size()
            .astype(int)
            .to_dict()
        )
    type_lines = ["##### 구분/범위/유형별 건수"]
    type_lines.extend(
        [
            f"- {row['구분']} / {row['범위']} / {row['유형']}: **{int(row['NEW 건수']):,}건**"
            f"{_fmt_delta(int(row['NEW 건수']), prev_type_map.get((row['구분'], row['범위'], row['유형']))) if prev_month else ''}"
            for _, row in type_summary.iterrows()
        ]
    )
    st.markdown("\n".join(type_lines))

    st.markdown("##### 상세 이벤트")
    detail_df = month_df[
        ["기준월", "데이터셋", "지역", "지표", "분류", "구분", "범위", "유형", "이벤트"]
    ].sort_values(["데이터셋", "지역", "지표", "분류", "구분", "범위", "유형"])
    max_detail_rows = 300
    detail_lines = []
    for _, row in detail_df.head(max_detail_rows).iterrows():
        category_text = str(row["분류"]).strip() if str(row["분류"]).strip() else "전체"
        detail_lines.append(
            f"- **[{row['기준월']}]** {row['데이터셋']} | {row['지역']} | {row['지표']} | {category_text} | {row['구분']} {row['범위']} {row['유형']}"
        )
    if len(detail_df) > max_detail_rows:
        detail_lines.append(f"- ... 총 {len(detail_df):,}건 중 상위 {max_detail_rows:,}건만 표시")
    st.markdown("\n".join(detail_lines))


def _pick_employment_indicator(indicators: List[str]) -> str:
    if not indicators:
        return ""
    for token in ["취업자수", "취업자", "취업"]:
        for name in indicators:
            if token in str(name):
                return str(name)
    return str(indicators[0])


def _find_prev_period(periods: List[pd.Timestamp], latest: pd.Timestamp, lag: int) -> Optional[pd.Timestamp]:
    sorted_periods = sorted(pd.to_datetime(periods).dropna().tolist())
    if latest not in sorted_periods:
        return None
    idx = sorted_periods.index(latest)
    if idx < lag:
        return None
    return pd.Timestamp(sorted_periods[idx - lag])


def _compute_contribution_table(
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
    indicator = _pick_employment_indicator(indicators)
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
    prev_period = _find_prev_period(periods, latest_period, lag)
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


def _build_ai_contribution_commentary(table: pd.DataFrame, meta: Dict[str, Any], point_label: str, yoy_label: str) -> str:
    if table.empty or not meta.get("ok"):
        return "AI 해설을 생성할 데이터가 부족합니다."
    total_delta = float(meta.get("total_delta", np.nan))
    unit = str(meta.get("unit", ""))
    direction = "증가" if total_delta > 0 else "감소" if total_delta < 0 else "보합"
    latest_p = _fmt_period(meta.get("latest_period"), "M" if point_label == "월" else "H")
    prev_p = _fmt_period(meta.get("prev_period"), "M" if point_label == "월" else "H")
    top_pos = table[table["증감"] > 0].nlargest(2, "증감")
    top_neg = table[table["증감"] < 0].nsmallest(2, "증감")

    lines: List[str] = []
    lines.append(
        f"- {latest_p} 기준 {meta['dataset_title']}({meta['indicator']})은 {prev_p} 대비 "
        f"총 **{_fmt_num(total_delta, unit)}** {direction}했습니다."
    )
    if not top_pos.empty:
        pos_text = ", ".join(
            [
                f"{r['분류']}({_fmt_num(r['증감'], unit)})"
                for _, r in top_pos.iterrows()
            ]
        )
        lines.append(f"- 증가 기여 상위는 {pos_text} 입니다.")
    if not top_neg.empty:
        neg_text = ", ".join(
            [
                f"{r['분류']}({_fmt_num(r['증감'], unit)})"
                for _, r in top_neg.iterrows()
            ]
        )
        lines.append(f"- 감소(상쇄) 요인은 {neg_text} 입니다.")
    lines.append(f"- 해석 기준은 `{yoy_label} 대비`이며, 기여율은 총증감 대비 비중입니다.")
    return "\n".join(lines)


def _robust_zscore(series: pd.Series) -> pd.Series:
    x = pd.to_numeric(series, errors="coerce")
    med = x.median(skipna=True)
    mad = (x - med).abs().median(skipna=True)
    if pd.isna(mad) or mad == 0:
        return pd.Series(np.nan, index=x.index)
    return 0.6745 * (x - med) / mad


def _compute_anomaly_table(df: pd.DataFrame, region: str, lag: int, lookback_periods: int = 36) -> pd.DataFrame:
    base = df[df["region_name"] == region].copy()
    if base.empty:
        return pd.DataFrame()
    rows: List[Dict[str, Any]] = []
    group_cols = ["dataset_key", "dataset_title", "region_name", "indicator_name", "category_name"]
    for _, g in base.groupby(group_cols, dropna=False):
        g = g.sort_values("period").copy()
        if len(g) < max(8, lag + 3):
            continue
        rz_value = _robust_zscore(g["value"]).abs()
        rz_yoy_abs = _robust_zscore(g["yoy_abs"]).abs()
        rz_yoy_pct = _robust_zscore(g["yoy_pct"]).abs()
        score_raw = pd.concat([rz_value, rz_yoy_abs, rz_yoy_pct], axis=1).max(axis=1, skipna=True)
        score = (score_raw / 5.0 * 100.0).clip(upper=100)
        for i, row in g.iterrows():
            s = score.loc[i]
            if pd.isna(s):
                continue
            component_scores = {
                "원자료": rz_value.loc[i],
                "증감": rz_yoy_abs.loc[i],
                "증감률": rz_yoy_pct.loc[i],
            }
            reason_key = max(component_scores, key=lambda k: -1 if pd.isna(component_scores[k]) else component_scores[k])
            reason_text_map = {
                "원자료": "원자료 값이 평소보다 크게 벗어남",
                "증감": "전년대비 증감폭이 평소보다 크게 벗어남",
                "증감률": "전년대비 증감률이 평소보다 크게 벗어남",
            }
            rows.append(
                {
                    "period": pd.Timestamp(row["period"]),
                    "기준시점": _fmt_period(row["period"], str(row.get("prd_se", "M"))),
                    "데이터셋": str(row["dataset_title"]),
                    "지역": str(row["region_name"]),
                    "지표": str(row["indicator_name"]),
                    "분류": str(row["category_name"]) if str(row["category_name"]).strip() else "전체",
                    "이상점수": float(s),
                    "이유": reason_text_map.get(reason_key, "평소 패턴 대비 변동이 큼"),
                }
            )
    if not rows:
        return pd.DataFrame()
    out = pd.DataFrame(rows)
    latest_p = out["period"].max()
    unique_p = sorted(out["period"].dropna().unique().tolist())
    if latest_p in unique_p:
        idx = unique_p.index(latest_p)
        start_idx = max(0, idx - lookback_periods + 1)
        cutoff = pd.Timestamp(unique_p[start_idx])
        out = out[out["period"] >= cutoff].copy()
    out = out.sort_values(["이상점수", "period"], ascending=[False, False])
    out = out.drop(columns=["period"], errors="ignore")
    return out


def _build_ai_anomaly_commentary(top_df: pd.DataFrame, point_label: str) -> str:
    if top_df.empty:
        return "이상 이벤트가 탐지되지 않았습니다."
    top = top_df.iloc[0]
    high_cnt = int((top_df["이상점수"] >= 75).sum())
    lines = [
        f"- 최우선 이상 이벤트는 **{top['기준시점']} / {top['지역']} / {top['지표']} / {top['분류']}** 입니다.",
        f"- 이상점수는 **{top['이상점수']:.1f}점**이며, 주된 이유는 `{top['이유']}`입니다.",
        f"- 최근 구간에서 경고 수준(75점 이상) 이벤트는 **{high_cnt}건**입니다.",
        f"- 점수는 Robust Z-score 기반이며, 평소 분포 대비 이례성을 의미합니다.",
    ]
    return "\n".join(lines)


def _compute_gyeonggi_vs_national_contribution(df: pd.DataFrame) -> tuple[pd.DataFrame, Dict[str, Any]]:
    meta: Dict[str, Any] = {
        "ok": False,
        "message": "",
        "indicator": "",
        "latest_period": pd.NaT,
        "latest_share_pct": np.nan,
        "latest_contrib_pct": np.nan,
        "latest_nat_yoy_abs": np.nan,
        "latest_gg_yoy_abs": np.nan,
        "unit": "",
    }

    base = df[df["dataset_key"] == "activity"].copy()
    if base.empty:
        meta["message"] = "경제활동인구현황 데이터가 없습니다."
        return pd.DataFrame(), meta

    indicators = sorted(base["indicator_name"].dropna().unique().tolist())
    indicator = _pick_employment_indicator(indicators)
    meta["indicator"] = indicator
    base = base[base["indicator_name"] == indicator].copy()
    if base.empty:
        meta["message"] = "취업자 지표를 찾지 못했습니다."
        return pd.DataFrame(), meta

    region_names = base["region_name"].dropna().astype(str).str.strip().unique().tolist()
    national_candidate = TARGET_REGIONS[0] if len(TARGET_REGIONS) >= 1 else ""
    gyeonggi_candidate = TARGET_REGIONS[9] if len(TARGET_REGIONS) >= 10 else ""
    if national_candidate not in region_names or gyeonggi_candidate not in region_names:
        meta["message"] = "전국 또는 경기도 데이터를 찾지 못했습니다."
        return pd.DataFrame(), meta

    nat = (
        base[base["region_name"] == national_candidate]
        .groupby("period", as_index=False)
        .agg({"value": "sum", "yoy_abs": "sum"})
        .rename(columns={"value": "value_nat", "yoy_abs": "yoy_abs_nat"})
    )
    gg = (
        base[base["region_name"] == gyeonggi_candidate]
        .groupby("period", as_index=False)
        .agg({"value": "sum", "yoy_abs": "sum"})
        .rename(columns={"value": "value_gg", "yoy_abs": "yoy_abs_gg"})
    )
    trend = nat.merge(gg, on="period", how="inner").sort_values("period")
    if trend.empty:
        meta["message"] = "전국/경기도 공통 시계열 구간이 없습니다."
        return pd.DataFrame(), meta

    trend["share_pct"] = np.where(
        trend["value_nat"] == 0,
        np.nan,
        trend["value_gg"] / trend["value_nat"] * 100.0,
    )
    trend["contrib_pct"] = np.where(
        trend["yoy_abs_nat"] == 0,
        np.nan,
        trend["yoy_abs_gg"] / trend["yoy_abs_nat"] * 100.0,
    )

    latest_row = trend.iloc[-1]
    unit = ""
    unit_series = base["unit"].dropna()
    if not unit_series.empty:
        unit = str(unit_series.iloc[0])

    meta.update(
        {
            "ok": True,
            "latest_period": pd.Timestamp(latest_row["period"]),
            "latest_share_pct": float(latest_row["share_pct"]) if pd.notna(latest_row["share_pct"]) else np.nan,
            "latest_contrib_pct": float(latest_row["contrib_pct"]) if pd.notna(latest_row["contrib_pct"]) else np.nan,
            "latest_nat_yoy_abs": float(latest_row["yoy_abs_nat"]) if pd.notna(latest_row["yoy_abs_nat"]) else np.nan,
            "latest_gg_yoy_abs": float(latest_row["yoy_abs_gg"]) if pd.notna(latest_row["yoy_abs_gg"]) else np.nan,
            "unit": unit,
        }
    )
    return trend, meta


def _build_ai_gyeonggi_contribution_commentary(meta: Dict[str, Any], labels: Dict[str, str]) -> str:
    if not meta.get("ok"):
        return str(meta.get("message", "전국 대비 경기도 기여도를 계산할 수 없습니다."))

    point = labels.get("point", "월")
    yoy = labels.get("yoy", "전년동월")
    prd_se = "M" if point == "월" else "H"
    latest = _fmt_period(meta.get("latest_period"), prd_se)
    share = meta.get("latest_share_pct")
    contrib = meta.get("latest_contrib_pct")
    unit = str(meta.get("unit", ""))
    gg_delta = meta.get("latest_gg_yoy_abs")
    nat_delta = meta.get("latest_nat_yoy_abs")

    contrib_text = "-" if pd.isna(contrib) else f"{float(contrib):,.1f}%"
    lines = [
        f"- {latest} 기준, 전국 취업자 중 경기도 비중은 **{float(share):,.2f}%**입니다." if pd.notna(share) else f"- {latest} 기준 비중 계산값이 없습니다.",
        f"- 같은 시점 {yoy} 대비 증감 기준으로, 경기도의 전국 기여율은 **{contrib_text}**입니다.",
        f"- 경기도 증감은 **{_fmt_num(gg_delta, unit)}**, 전국 증감은 **{_fmt_num(nat_delta, unit)}**입니다.",
        "- 참고: 전국 증감이 0에 가까우면 기여율(%)이 크게 흔들릴 수 있습니다.",
    ]
    return "\n".join(lines)


def _render_ai_insights(df: pd.DataFrame, region_pool: List[str], labels: Dict[str, str]) -> None:
    st.subheader("AI INSIGHTS")
    st.caption("?????? + AI ?? + AI ????(Robust Z-score)")
    st.markdown("#### ?? ?? ??? ??")
    gy_trend, gy_meta = _compute_gyeonggi_vs_national_contribution(df)
    if not gy_meta.get("ok"):
        st.info(str(gy_meta.get("message", "?? ?? ??? ??? ??? ??????.")))
    else:
        st.markdown("##### AI ??")
        st.markdown(_build_ai_gyeonggi_contribution_commentary(gy_meta, labels))
        period_prd = "H" if ("prd_se" in df.columns and not df["prd_se"].dropna().empty and str(df["prd_se"].dropna().iloc[0]).upper() == "H") else "M"
        latest_period_text = _fmt_period(gy_meta.get("latest_period"), period_prd)
        c1, c2, c3 = st.columns(3)
        with c1:
            _card(
                "?? ?? ??? ??",
                "-" if pd.isna(gy_meta.get("latest_share_pct")) else f"{float(gy_meta.get('latest_share_pct')):,.2f}%",
                latest_period_text,
            )
        with c2:
            _card(
                f"?? ?? ???({labels.get('yoy', '????')}??)",
                "-" if pd.isna(gy_meta.get("latest_contrib_pct")) else f"{float(gy_meta.get('latest_contrib_pct')):,.1f}%",
                latest_period_text,
            )
        with c3:
            _card(
                "??? ??",
                _fmt_num(gy_meta.get("latest_gg_yoy_abs"), str(gy_meta.get("unit", ""))),
                f"{labels.get('yoy', '????')}??",
            )
        plot_df = gy_trend[["period", "share_pct", "contrib_pct"]].dropna(subset=["period"], how="any").copy()
        if not plot_df.empty:
            col_l, col_r = st.columns(2)
            with col_l:
                share_chart = (
                    alt.Chart(plot_df)
                    .mark_line(color="#2563eb", point=True)
                    .encode(
                        x=alt.X("period:T", title=labels.get("point", "?")),
                        y=alt.Y("share_pct:Q", title="?? ?? ??? ??(%)"),
                        tooltip=[
                            alt.Tooltip("yearmonth(period):T", title=labels.get("point", "?")),
                            alt.Tooltip("share_pct:Q", title="??(%)", format=".2f"),
                        ],
                    )
                    .properties(height=280)
                )
                st.altair_chart(share_chart, use_container_width=True)
            with col_r:
                base = alt.Chart(plot_df).encode(
                    x=alt.X("period:T", title=labels.get("point", "?")),
                    tooltip=[
                        alt.Tooltip("yearmonth(period):T", title=labels.get("point", "?")),
                        alt.Tooltip("contrib_pct:Q", title="???(%)", format=".1f"),
                    ],
                )
                contrib_line = base.mark_line(color="#dc2626", point=True).encode(
                    y=alt.Y("contrib_pct:Q", title="?? ?? ???(%)")
                )
                zero = alt.Chart(pd.DataFrame({"zero": [0]})).mark_rule(
                    color="#9CA3AF",
                    strokeDash=[4, 4],
                ).encode(y="zero:Q")
                st.altair_chart(alt.layer(contrib_line, zero).properties(height=280), use_container_width=True)
    st.markdown("---")
    gyeonggi_default = TARGET_REGIONS[9] if len(TARGET_REGIONS) >= 10 else (region_pool[0] if region_pool else "")
    region_default = gyeonggi_default if gyeonggi_default in region_pool else (region_pool[0] if region_pool else "")
    region = st.selectbox(
        "?? ??",
        region_pool,
        index=region_pool.index(region_default) if region_default in region_pool else 0,
        key="ai_region",
    )
    lag = 12
    if "prd_se" in df.columns and not df["prd_se"].dropna().empty:
        lag = 2 if str(df["prd_se"].dropna().iloc[0]).upper() == "H" else 12
    period_prd = "H" if lag == 2 else "M"
    st.markdown("#### ??????")
    ds_options = {
        "??? ???": "age",
        "?????? ???": "status",
        "??? ????": "industry",
        "??? ????": "occupation",
    }
    ds_label = st.radio(
        "?? ?",
        list(ds_options.keys()),
        horizontal=True,
        key="ai_decomp_axis",
    )
    ds_key = ds_options[ds_label]
    contrib_df, contrib_meta = _compute_contribution_table(df, region=region, dataset_key=ds_key, lag=lag)
    if not contrib_meta.get("ok"):
        st.info(str(contrib_meta.get("message", "?? ???? ??? ? ????.")))
    else:
        st.markdown("##### AI ??")
        st.markdown(_build_ai_contribution_commentary(contrib_df, contrib_meta, labels["point"], labels["yoy"]))
        unit = str(contrib_meta.get("unit", ""))
        st.markdown(
            f"- ????: **{_fmt_period(contrib_meta['latest_period'], period_prd)}**  \n"
            f"- ????: **{_fmt_period(contrib_meta['prev_period'], period_prd)}**  \n"
            f"- ? ??: **{_fmt_num(contrib_meta['total_delta'], unit)}**"
        )
        chart_df = contrib_df.copy()
        chart = (
            alt.Chart(chart_df)
            .mark_bar()
            .encode(
                x=alt.X("???:Q", title="??"),
                y=alt.Y("???:N", sort="-x", title="??"),
                color=alt.condition("datum.??? >= 0", alt.value("#2563eb"), alt.value("#dc2626")),
                tooltip=[
                    alt.Tooltip("???:N", title="??"),
                    alt.Tooltip("???:Q", title="??", format=",.2f"),
                    alt.Tooltip("?????%):Q", title="???(%)", format=".2f"),
                ],
            )
            .properties(height=max(280, len(chart_df) * 22))
        )
        st.altair_chart(chart, use_container_width=True)
        st.dataframe(contrib_df, use_container_width=True, hide_index=True)
    st.markdown("#### AI ???? (Robust Z-score)")
    lookback = st.slider(
        f"???? ?? ?? ? ({labels.get('point', '?')})",
        min_value=12,
        max_value=60,
        value=36,
        step=6,
        key="ai_anomaly_lookback",
    )
    anomaly_df = _compute_anomaly_table(df, region=region, lag=lag, lookback_periods=lookback)
    if anomaly_df.empty:
        st.info("???? ??? ????.")
    else:
        top_df = anomaly_df.head(10).copy()
        st.markdown("##### AI ??")
        st.markdown(_build_ai_anomaly_commentary(top_df, labels["point"]))
        st.dataframe(top_df, use_container_width=True, hide_index=True)

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
loading_notice.info("데이터 불러오는 중...")
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

is_gyeonggi31_mode = st.toggle(
    "경기 31개 시군 모드",
    value=False,
    key="scope_toggle",
)
scope_label = "경기 31개 시군" if is_gyeonggi31_mode else "전국·17개 시도"
region_scope = "gyeonggi31" if is_gyeonggi31_mode else "province"
active_datasets = datasets_for_scope(region_scope)
st.caption(f"현재 조회 범위: {scope_label}")

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

tab1, tab2, tab3, tab4, tab5, tab6, tab7 = st.tabs(
    [
        "경제활동인구현황",
        "연령별 취업자",
        "종사상지위별 취업자",
        "산업별 취업자수",
        "직종별 취업자수",
        "NEW HISTORY",
        "REPORT",
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
    st.subheader("NEW HISTORY")
    if events.empty:
        st.info("집계된 NEW 이벤트 이력이 없습니다.")
    else:
        f1, f2 = st.columns([1, 1])
        with f1:
            metric_options = ["전체"] + sorted(events["구분"].dropna().unique().tolist())
            metric_sel = st.selectbox("구분", metric_options, key="history_metric_filter")
        with f2:
            type_options = ["전체", "최고", "최저"]
            type_sel = st.selectbox("유형", type_options, key="history_type_filter")

        view = events.copy()
        if metric_sel != "전체":
            view = view[view["구분"] == metric_sel]
        if type_sel != "전체":
            view = view[view["유형"] == type_sel]

        st.markdown("##### 상세 이벤트")
        detail_df = view[
            ["기준월", "데이터셋", "지역", "지표", "분류", "구분", "범위", "유형", "이벤트"]
        ].sort_values(
            ["기준월", "데이터셋", "지역", "지표", "분류", "구분", "범위", "유형"],
            ascending=[False, True, True, True, True, True, True, True],
        )
        max_history_rows = 500
        detail_lines = []
        for _, row in detail_df.head(max_history_rows).iterrows():
            category_text = str(row["분류"]).strip() if str(row["분류"]).strip() else "전체"
            detail_lines.append(
                f"- **[{row['기준월']}]** {row['데이터셋']} | {row['지역']} | {row['지표']} | {category_text} | {row['구분']} {row['범위']} {row['유형']}"
            )
        if len(detail_df) > max_history_rows:
            detail_lines.append(f"- ... 총 {len(detail_df):,}건 중 상위 {max_history_rows:,}건만 표시")
        st.markdown("\n".join(detail_lines) if detail_lines else "- 표시할 이벤트가 없습니다.")
        _render_new_event_charts(view, active_datasets)
        labels = _time_labels(active_datasets)
        st.markdown(
            f"<p style='color:#b91c1c;font-weight:700'>NEW 이벤트는 해당 {labels['point']} 시점 기준으로 전체기간/최근5년 최고·최저를 새로 갱신한 이력을 표시합니다.</p>",
            unsafe_allow_html=True,
        )
with tab7:
    st.subheader("REPORT")
    report_scope = st.radio(
        "리포트 범위",
        ["경기도 전체", "31개 시군"],
        index=0,
        horizontal=True,
        key="report_scope",
    )
    _render_new_monthly_report(events, report_scope=report_scope, datasets=active_datasets)

if region_scope == "province":
    st.markdown("---")
    _render_ai_insights(visible_data, region_pool, _time_labels(active_datasets))

st.markdown(
    "<hr style='margin-top:2rem; margin-bottom:0.5rem;'>"
    "<p style='text-align:center; color:#6b7280; font-size:0.9rem;'>- created by alicia -</p>",
    unsafe_allow_html=True,
)


