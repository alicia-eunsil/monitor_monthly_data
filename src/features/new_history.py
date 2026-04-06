from typing import Any, Dict, List, Optional

from collections import deque

import pandas as pd
import streamlit as st

import src.config as app_config
from src.core.category_rules import ACTIVITY_INDICATOR_ORDER, norm_indicator_name
from src.core.formatters import escape_markdown_text, fmt_num, fmt_period, time_labels
from src.features.new_event_summary import (
    build_dataset_count_lines,
    build_new_count_summary_lines,
    build_new_focus_line,
)
from src.features.streak_utils import current_streak_length as _current_streak_length

GYEONGGI_SIGUNGU = getattr(app_config, "GYEONGGI_SIGUNGU", [])


def _build_consecutive_change_lines(
    source_df: pd.DataFrame,
    region: str,
    asof_period: pd.Timestamp,
    labels: Dict[str, str],
) -> List[str]:
    if not region or source_df.empty or pd.isna(asof_period):
        return []

    point_label = labels.get("point", "월")
    yoy_label = labels.get("yoy", "전년동월")
    dataset_defs = [
        ("activity", "경제활동인구현황(취업자)"),
        ("industry", "산업별 취업자수"),
        ("occupation", "직종별 취업자수"),
        ("status", "종사상지위별 취업자"),
        ("age", "연령별 취업자"),
    ]

    def _duration_unit(prd_se: str) -> str:
        return "반기" if str(prd_se).upper() == "H" else "개월"

    def _format_item(cand: Dict[str, object], increase: bool, include_label: bool) -> str:
        direction = "증가" if increase else "감소"
        unit_text = _duration_unit(str(cand.get("prd_se", "M")))
        start_txt = fmt_period(cand["start"], str(cand.get("prd_se", "M")))
        end_txt = fmt_period(cand["end"], str(cand.get("prd_se", "M")))
        # Escape markdown-sensitive characters (notably "~") to avoid strikethrough rendering.
        range_txt = f"{escape_markdown_text(start_txt)}\\~{escape_markdown_text(end_txt)}"
        label_txt = escape_markdown_text(cand["label"])
        if include_label:
            return f"{label_txt} {range_txt} {cand['len']}{unit_text} 연속 {direction}"
        return f"{range_txt}, {cand['len']}{unit_text} 연속 {direction}"

    lines: List[str] = []
    min_show_all_len = 3
    for ds_key, ds_title in dataset_defs:
        ds = source_df[
            (source_df["dataset_key"].astype(str) == ds_key)
            & (source_df["region_name"].astype(str) == str(region))
            & (pd.to_datetime(source_df["period"], errors="coerce") <= pd.Timestamp(asof_period))
        ].copy()
        if ds.empty or "yoy_abs" not in ds.columns:
            continue
        ds["period"] = pd.to_datetime(ds["period"], errors="coerce")
        ds = ds.dropna(subset=["period"])
        if ds.empty:
            continue

        if ds_key == "activity":
            target_norm = norm_indicator_name("취업자")
            ds["norm_indicator"] = ds["indicator_name"].apply(norm_indicator_name)
            ds = ds[ds["norm_indicator"] == target_norm].copy()
            if ds.empty:
                continue

        up_items: List[Dict[str, object]] = []
        down_items: List[Dict[str, object]] = []
        group_cols = ["indicator_name", "category_name"]
        for _, g in ds.groupby(group_cols, dropna=False):
            g = g.sort_values("period")
            if g.empty:
                continue
            latest_period = pd.Timestamp(g["period"].iloc[-1])
            if latest_period != pd.Timestamp(asof_period):
                continue
            yoy_values = pd.to_numeric(g["yoy_abs"], errors="coerce")
            if yoy_values.empty or pd.isna(yoy_values.iloc[-1]):
                continue
            latest_yoy = float(yoy_values.iloc[-1])
            cat = str(g["category_name"].iloc[0]).strip()
            ind = str(g["indicator_name"].iloc[0]).strip()
            label = cat if cat else (ind if ind else "전체")
            prd_se = str(g["prd_se"].dropna().iloc[0]).upper() if "prd_se" in g.columns and not g["prd_se"].dropna().empty else "M"

            up_len = _current_streak_length(yoy_values, positive=True)
            if up_len > 0:
                start_p = pd.Timestamp(g["period"].iloc[len(g) - up_len])
                up_items.append(
                    {
                    "label": label,
                    "len": int(up_len),
                    "start": start_p,
                    "end": latest_period,
                    "latest_yoy": latest_yoy,
                    "prd_se": prd_se,
                    }
                )

            down_len = _current_streak_length(yoy_values, positive=False)
            if down_len > 0:
                start_p = pd.Timestamp(g["period"].iloc[len(g) - down_len])
                down_items.append(
                    {
                    "label": label,
                    "len": int(down_len),
                    "start": start_p,
                    "end": latest_period,
                    "latest_yoy": latest_yoy,
                    "prd_se": prd_se,
                    }
                )

        if not up_items and not down_items:
            continue

        up_items = sorted(up_items, key=lambda x: (-int(x["len"]), -abs(float(x["latest_yoy"])), str(x["label"])))
        down_items = sorted(down_items, key=lambda x: (-int(x["len"]), -abs(float(x["latest_yoy"])), str(x["label"])))

        up_show = [x for x in up_items if int(x["len"]) >= min_show_all_len]
        down_show = [x for x in down_items if int(x["len"]) >= min_show_all_len]
        if not up_show and up_items:
            up_show = [up_items[0]]
        if not down_show and down_items:
            down_show = [down_items[0]]

        include_label = ds_key != "activity"
        segs: List[str] = []
        if up_show:
            segs.extend([_format_item(x, increase=True, include_label=include_label) for x in up_show])
        if down_show:
            segs.extend([_format_item(x, increase=False, include_label=include_label) for x in down_show])
        if segs:
            lines.append(f"- {ds_title}: {', '.join(segs)} ({yoy_label}대비 증감 기준)")

    return lines


def _pick_streak_region(source_df: pd.DataFrame, report_scope: str) -> str:
    region_candidates = source_df["region_name"].dropna().astype(str).str.strip().unique().tolist()
    if "31" in str(report_scope):
        sigungu_candidates = sorted([r for r in GYEONGGI_SIGUNGU if r in region_candidates])
        return sigungu_candidates[0] if sigungu_candidates else ""
    return "경기도" if "경기도" in region_candidates else (region_candidates[0] if region_candidates else "")


def _build_dataset_new_event_lines(
    month_df: pd.DataFrame,
    datasets: List[Any],
    top_n_datasets: Optional[int] = None,
    per_dataset_events: int = 3,
) -> List[str]:
    if month_df.empty:
        return ["##### 데이터셋별 NEW 이벤트", "- 표시할 데이터가 없습니다."]

    ds_order = [str(getattr(cfg, "title", "")).strip() for cfg in datasets]
    metric_order = {"원자료": 0, "YoY(절대)": 1, "YoY(증감률)": 2}
    scope_order = {"전체기간": 0, "최근10년": 1, "최근5년": 2}
    event_type_order = {"최고": 0, "최저": 1}

    summary = month_df.groupby("데이터셋", as_index=False).size().rename(columns={"size": "건수"})
    summary["정렬_데이터셋"] = summary["데이터셋"].map({name: idx for idx, name in enumerate(ds_order)}).fillna(999)
    summary = summary.sort_values(["정렬_데이터셋", "데이터셋"]).drop(columns=["정렬_데이터셋"])
    ds_names = summary["데이터셋"].astype(str).tolist()
    if top_n_datasets is not None:
        ds_names = ds_names[: int(top_n_datasets)]

    lines: List[str] = ["##### 데이터셋별 NEW 이벤트"]
    for ds_name in ds_names:
        ds_view = month_df[month_df["데이터셋"].astype(str) == str(ds_name)].copy()
        if ds_view.empty:
            lines.append(f"- {ds_name}: 없음")
            continue
        ds_view["정렬_구분"] = ds_view["구분"].map(metric_order).fillna(999)
        ds_view["정렬_범위"] = ds_view["범위"].map(scope_order).fillna(999)
        ds_view["정렬_유형"] = ds_view["유형"].map(event_type_order).fillna(999)
        ds_view = ds_view.sort_values(["정렬_구분", "정렬_범위", "정렬_유형", "지표", "분류"]).drop(
            columns=["정렬_구분", "정렬_범위", "정렬_유형"]
        )

        tokens: List[str] = []
        for _, row in ds_view.head(int(per_dataset_events)).iterrows():
            category_text = str(row.get("분류", "")).strip() or "전체"
            tokens.append(
                f"{row['지표']}/{category_text}"
                f"({row['구분']} {row['범위']} {row['유형']})"
            )
        remain = int(len(ds_view) - len(tokens))
        if remain > 0:
            tokens.append(f"외 {remain:,}건")
        lines.append(f"- {ds_name}: " + ", ".join(tokens))

    if top_n_datasets is not None and len(summary) > len(ds_names):
        lines.append(f"- ... 데이터셋 {len(summary):,}개 중 상위 {len(ds_names):,}개만 표시")
    return lines


def collect_new_events(df: pd.DataFrame) -> pd.DataFrame:
    rows: List[Dict[str, str]] = []
    key_cols = ["dataset_key", "dataset_title", "region_name", "indicator_name", "category_name", "prd_se"]

    def _prev_window_extreme(metric_df: pd.DataFrame, metric_col: str, years: int, mode: str) -> pd.Series:
        # Exact date-based window (same boundary rule as build_stats/_slice_recent_years):
        # previous points with period >= (current_period - years) and < current_period.
        if metric_df.empty:
            return pd.Series(dtype="float64")
        periods = pd.to_datetime(metric_df["period"]).tolist()
        values = metric_df[metric_col].astype(float).tolist()
        n = len(values)
        out = [float("nan")] * n

        if mode == "max":
            mono = deque()  # decreasing by value
            def pop_cond(last_idx: int, new_val: float) -> bool:
                return values[last_idx] <= new_val
        else:
            mono = deque()  # increasing by value
            def pop_cond(last_idx: int, new_val: float) -> bool:
                return values[last_idx] >= new_val

        left = 0
        for i in range(n):
            cutoff = pd.Timestamp(periods[i]) - pd.DateOffset(years=int(years))
            while left < i and pd.Timestamp(periods[left]) < cutoff:
                if mono and mono[0] == left:
                    mono.popleft()
                left += 1

            if mono:
                out[i] = float(values[mono[0]])

            cur_val = float(values[i])
            while mono and pop_cond(mono[-1], cur_val):
                mono.pop()
            mono.append(i)

        return pd.Series(out, index=metric_df.index, dtype="float64")

    for _, series in df.groupby(key_cols, dropna=False):
        series = series.sort_values("period")
        prd_se = str(series["prd_se"].iloc[0]).upper() if "prd_se" in series.columns else "M"
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
            prev_10y_max = _prev_window_extreme(metric_df, metric_col, years=10, mode="max")
            prev_10y_min = _prev_window_extreme(metric_df, metric_col, years=10, mode="min")
            prev_5y_max = _prev_window_extreme(metric_df, metric_col, years=5, mode="max")
            prev_5y_min = _prev_window_extreme(metric_df, metric_col, years=5, mode="min")

            for scope_label, scope_series in [("전체기간", prev_max), ("최근10년", prev_10y_max), ("최근5년", prev_5y_max)]:
                is_new_max = metric_df[metric_col] > scope_series
                for _, row in metric_df[is_new_max.fillna(False)].iterrows():
                    rows.append(
                        {
                            **meta,
                            "기준월": fmt_period(row["period"], prd_se),
                            "기준월_ts": pd.Timestamp(row["period"]),
                            "구분": metric_label,
                            "범위": scope_label,
                            "유형": "최고",
                            "이벤트": f"{metric_label} {scope_label} 최고 NEW",
                        }
                    )

            for scope_label, scope_series in [("전체기간", prev_min), ("최근10년", prev_10y_min), ("최근5년", prev_5y_min)]:
                is_new_min = metric_df[metric_col] < scope_series
                for _, row in metric_df[is_new_min.fillna(False)].iterrows():
                    rows.append(
                        {
                            **meta,
                            "기준월": fmt_period(row["period"], prd_se),
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


def _find_prev_period(periods: List[pd.Timestamp], latest: pd.Timestamp, lag: int) -> Optional[pd.Timestamp]:
    sorted_periods = sorted(pd.to_datetime(periods).dropna().tolist())
    if latest not in sorted_periods:
        return None
    idx = sorted_periods.index(latest)
    if idx < lag:
        return None
    return pd.Timestamp(sorted_periods[idx - lag])


def _build_report_view(
    events: pd.DataFrame,
    report_scope: str,
    selected_region: Optional[str] = None,
) -> tuple[pd.DataFrame, str]:
    if events.empty:
        return events.copy(), str(report_scope or "")

    region_col = "지역" if "지역" in events.columns else ""
    if not region_col:
        return events.copy(), str(report_scope or "")

    if "31" in str(report_scope):
        view = events[events[region_col].isin(GYEONGGI_SIGUNGU)].copy()
        scope_title = "경기 31개 시군"
    else:
        region_text = events[region_col].astype(str)
        gyeonggi_mask = region_text.str.contains("경기", na=False)
        view = events[gyeonggi_mask].copy()
        scope_title = "경기도 전체"

    region_name = str(selected_region or "").strip()
    if region_name:
        scoped = view[view[region_col].astype(str).str.strip() == region_name].copy()
        if not scoped.empty:
            view = scoped
            scope_title = region_name
    return view, scope_title


def get_report_region_options(events: pd.DataFrame, report_scope: str) -> List[str]:
    if events.empty:
        return []
    view, _ = _build_report_view(events=events, report_scope=report_scope)
    if view.empty:
        return []
    return sorted(view["지역"].dropna().astype(str).str.strip().unique().tolist())


def get_report_period_options(
    events: pd.DataFrame,
    report_scope: str,
    selected_region: Optional[str] = None,
) -> List[str]:
    if events.empty:
        return []
    view, _ = _build_report_view(
        events=events,
        report_scope=report_scope,
        selected_region=selected_region,
    )
    if view.empty:
        return []
    if "기준월_ts" in view.columns:
        view["기준월_dt"] = pd.to_datetime(view["기준월_ts"], errors="coerce")
    else:
        view["기준월_dt"] = pd.to_datetime(view["기준월"], errors="coerce")
    view = view.dropna(subset=["기준월_dt"])
    if view.empty:
        return []
    month_table = (
        view[["기준월", "기준월_dt"]]
        .drop_duplicates()
        .sort_values("기준월_dt", ascending=False)
        .reset_index(drop=True)
    )
    return month_table["기준월"].astype(str).tolist()


def build_ai_insight_context(
    events: pd.DataFrame,
    report_scope: str,
    datasets: List[Any],
    source_df: pd.DataFrame,
    selected_region: Optional[str] = None,
    selected_month: Optional[str] = None,
) -> Dict[str, Any]:
    labels = time_labels([str(getattr(cfg, "prd_se", "M")) for cfg in datasets])
    if events.empty:
        return {
            "ok": False,
            "message": "NEW 이벤트가 없어 요약을 만들 수 없습니다.",
            "labels": labels,
            "scope_title": str(report_scope),
            "selected_month": "",
            "context_title": "",
            "context_lines": [],
            "focus_lines": [],
            "consecutive_lines": [],
            "stats": {},
        }

    view, scope_title = _build_report_view(
        events=events,
        report_scope=report_scope,
        selected_region=selected_region,
    )
    if view.empty:
        return {
            "ok": False,
            "message": f"{scope_title} 기준 NEW 이벤트가 없습니다.",
            "labels": labels,
            "scope_title": scope_title,
            "selected_month": "",
            "context_title": "",
            "context_lines": [],
            "focus_lines": [],
            "consecutive_lines": [],
            "stats": {},
        }

    if "기준월_ts" in view.columns:
        view["기준월_dt"] = pd.to_datetime(view["기준월_ts"], errors="coerce")
    else:
        view["기준월_dt"] = pd.to_datetime(view["기준월"], errors="coerce")
    view = view.dropna(subset=["기준월_dt"])
    if view.empty:
        return {
            "ok": False,
            "message": f"{scope_title} 기준 NEW 이벤트가 없습니다.",
            "labels": labels,
            "scope_title": scope_title,
            "selected_month": "",
            "context_title": "",
            "context_lines": [],
            "focus_lines": [],
            "consecutive_lines": [],
            "stats": {},
        }

    month_table = (
        view[["기준월", "기준월_dt"]]
        .drop_duplicates()
        .sort_values("기준월_dt", ascending=False)
        .reset_index(drop=True)
    )
    if month_table.empty:
        return {
            "ok": False,
            "message": f"{scope_title} 기준 NEW 이벤트가 없습니다.",
            "labels": labels,
            "scope_title": scope_title,
            "selected_month": "",
            "context_title": "",
            "context_lines": [],
            "focus_lines": [],
            "consecutive_lines": [],
            "stats": {},
        }

    month_list = month_table["기준월"].astype(str).tolist()
    if selected_month is None:
        selected_month = month_list[0]
    selected_month = str(selected_month)
    if selected_month not in month_list:
        selected_month = month_list[0]

    month_df = view[view["기준월"].astype(str) == selected_month].copy()
    selected_idx_list = month_table.index[month_table["기준월"].astype(str) == selected_month].tolist()
    prev_month = (
        month_table.loc[selected_idx_list[0] + 1, "기준월"]
        if selected_idx_list and (selected_idx_list[0] + 1) < len(month_table)
        else None
    )
    prev_month_df = view[view["기준월"] == prev_month].copy() if prev_month else pd.DataFrame(columns=view.columns)

    context_lines: List[str] = []
    context_lines.extend(build_new_count_summary_lines(month_df, prev_month_df if prev_month else None, labels))
    context_lines.extend(build_dataset_count_lines(month_df, prev_month_df if prev_month else None, datasets, top_n=3))

    point_label = str(labels.get("point", "월"))
    focus_lines = [
        f"- {build_new_focus_line(month_df, '최고', f'이번 {point_label} NEW 핵심')}",
        f"- {build_new_focus_line(month_df, '최저', f'이번 {point_label} NEW 리스크')}",
    ]

    selected_rows = month_table[month_table["기준월"].astype(str) == selected_month]
    selected_month_dt = (
        pd.to_datetime(selected_rows["기준월_dt"], errors="coerce").dropna().max() if not selected_rows.empty else pd.NaT
    )
    consecutive_lines: List[str] = []
    if pd.notna(selected_month_dt):
        consecutive_lines = _build_consecutive_change_lines(
            source_df=source_df,
            region=str(selected_region or "").strip() or _pick_streak_region(source_df, report_scope),
            asof_period=pd.Timestamp(selected_month_dt),
            labels=labels,
        )

    stats = {
        "total_events": int(len(month_df)),
        "max_events": int((month_df["유형"] == "최고").sum()) if "유형" in month_df.columns else 0,
        "min_events": int((month_df["유형"] == "최저").sum()) if "유형" in month_df.columns else 0,
    }

    context_title = f"{selected_month} NEW 리포트 ({scope_title})"
    return {
        "ok": True,
        "message": "",
        "labels": labels,
        "scope_title": scope_title,
        "selected_month": selected_month,
        "context_title": context_title,
        "context_lines": context_lines,
        "focus_lines": focus_lines,
        "consecutive_lines": consecutive_lines,
        "stats": stats,
    }


def render_new_monthly_report(
    events: pd.DataFrame,
    report_scope: str,
    datasets: List[Any],
    source_df: pd.DataFrame,
    compact: bool = False,
    include_consecutive_summary: bool = True,
    selected_region: Optional[str] = None,
    selected_month: Optional[str] = None,
) -> None:
    if events.empty:
        st.info("리포트로 표시할 NEW 이벤트가 없습니다.")
        return

    view, scope_title = _build_report_view(
        events=events,
        report_scope=report_scope,
        selected_region=selected_region,
    )
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
    if month_table.empty:
        st.info(f"{scope_title} 기준 NEW 이벤트가 없습니다.")
        return
    month_list = month_table["기준월"].tolist()
    labels = time_labels([str(getattr(cfg, "prd_se", "M")) for cfg in datasets])
    if selected_month is None:
        selected_month = st.selectbox(f"리포트 기준{labels['point']}", month_list, key="report_month")
    selected_month = str(selected_month)
    if selected_month not in month_table["기준월"].astype(str).tolist():
        selected_month = str(month_table.iloc[0]["기준월"])
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
            return f" (+{diff:,})"
        if diff < 0:
            return f" (-{abs(diff):,})"
        return " (=)"

    st.markdown(f"#### {selected_month} NEW 리포트 ({scope_title})")

    activity_lines: List[str] = []
    activity_df = source_df[source_df["dataset_key"] == "activity"].copy()
    activity_df["period"] = pd.to_datetime(activity_df["period"], errors="coerce")
    activity_df = activity_df.dropna(subset=["period"])
    if (not compact) and not activity_df.empty:
        region_candidates = activity_df["region_name"].dropna().astype(str).str.strip().unique().tolist()
        summary_region = str(selected_region or "").strip()
        if summary_region and summary_region not in region_candidates:
            summary_region = ""
        if not summary_region:
            summary_region = "경기도" if "경기도" in region_candidates else ""
        if not summary_region and report_scope == "31개 시군":
            sigungu_candidates = sorted([r for r in GYEONGGI_SIGUNGU if r in region_candidates])
            summary_region = sigungu_candidates[0] if sigungu_candidates else ""
        if summary_region:
            region_df = activity_df[activity_df["region_name"] == summary_region].copy()
            if not region_df.empty:
                prd_se = (
                    str(region_df["prd_se"].dropna().iloc[0]).upper()
                    if "prd_se" in region_df.columns and not region_df["prd_se"].dropna().empty
                    else "M"
                )
                region_df["period_label"] = region_df["period"].apply(lambda x: fmt_period(x, prd_se))
                latest_rows = region_df[region_df["period_label"] == selected_month].copy()
                if not latest_rows.empty:
                    latest_period = pd.Timestamp(latest_rows["period"].max())
                    all_periods = sorted(region_df["period"].dropna().unique().tolist())
                    lag = 2 if prd_se == "H" else 12
                    prev_period = _find_prev_period(all_periods, latest_period, lag)
                    prev_rows = (
                        region_df[region_df["period"] == prev_period].copy()
                        if prev_period is not None
                        else pd.DataFrame(columns=region_df.columns)
                    )

                    latest_map = (
                        latest_rows.groupby("indicator_name", as_index=False)
                        .agg({"value": "mean", "unit": "first"})
                        .rename(columns={"value": "latest_value", "unit": "latest_unit"})
                    )
                    prev_map = (
                        prev_rows.groupby("indicator_name", as_index=False)
                        .agg({"value": "mean", "unit": "first"})
                        .rename(columns={"value": "prev_value", "unit": "prev_unit"})
                    )
                    merged = latest_map.merge(prev_map, on="indicator_name", how="left")
                    merged["unit"] = merged["latest_unit"].fillna(merged["prev_unit"]).fillna("")
                    merged["delta_value"] = pd.to_numeric(merged["latest_value"], errors="coerce") - pd.to_numeric(
                        merged["prev_value"], errors="coerce"
                    )
                    activity_order_map = {
                        norm_indicator_name(name): idx for idx, name in enumerate(ACTIVITY_INDICATOR_ORDER)
                    }
                    merged["norm_indicator"] = merged["indicator_name"].apply(norm_indicator_name)
                    merged = merged[merged["norm_indicator"].isin(activity_order_map.keys())].copy()
                    merged["sort_key"] = merged["norm_indicator"].map(activity_order_map).fillna(999)
                    merged = merged.sort_values(["sort_key", "indicator_name"]).drop(columns=["sort_key"])

                    def _fmt_delta_signed(v: object, u: str) -> str:
                        if v is None or pd.isna(v):
                            return "-"
                        vv = float(v)
                        if vv > 0:
                            return f"▲{fmt_num(vv, u)}"
                        if vv < 0:
                            return f"▼{fmt_num(abs(vv), u)}"
                        return f"→{fmt_num(0, u)}"

                    yoy_label = labels["yoy"]
                    current_label = "당월" if labels["point"] == "월" else "당반기"
                    activity_lines.append(f"##### 경제활동인구현황 9개 지표 요약 ({summary_region})")
                    for _, r in merged.iterrows():
                        activity_lines.append(
                            f"- {str(r['indicator_name'])}: "
                            f"{current_label} **{fmt_num(r['latest_value'], str(r['unit']))}**, "
                            f"{yoy_label} **{fmt_num(r['prev_value'], str(r['unit']))}**, "
                            f"증감 **{_fmt_delta_signed(r['delta_value'], str(r['unit']))}**"
                        )

    if activity_lines:
        st.markdown("\n".join(activity_lines))

    selected_rows = month_table[month_table.iloc[:, 0].astype(str) == str(selected_month)]
    selected_month_dt = (
        pd.to_datetime(selected_rows.iloc[:, 1], errors="coerce").dropna().max() if not selected_rows.empty else pd.NaT
    )
    if include_consecutive_summary:
        streak_lines = _build_consecutive_change_lines(
            source_df=source_df,
            region=str(selected_region or "").strip() or _pick_streak_region(source_df, report_scope),
            asof_period=selected_month_dt,
            labels=labels,
        )
        if streak_lines:
            st.markdown("##### 연속 증가/감소 요약")
            st.markdown("\n".join(streak_lines))

    st.markdown(
        "\n".join(
            build_new_count_summary_lines(
                month_df=month_df,
                prev_month_df=prev_month_df if prev_month else None,
                labels=labels,
            )
        )
    )

    st.markdown(
        "\n".join(
            build_dataset_count_lines(
                month_df=month_df,
                prev_month_df=prev_month_df if prev_month else None,
                datasets=datasets,
                top_n=3 if compact else None,
            )
        )
    )
    st.markdown(
        "\n".join(
            _build_dataset_new_event_lines(
                month_df=month_df,
                datasets=datasets,
                top_n_datasets=3 if compact else None,
                per_dataset_events=2 if compact else 3,
            )
        )
    )
    if compact:
        point_label = str(labels.get("point", "월"))
        st.markdown(f"- {build_new_focus_line(month_df, '최고', f'이번 {point_label} NEW 핵심')}")
        st.markdown(f"- {build_new_focus_line(month_df, '최저', f'이번 {point_label} NEW 리스크')}")
        st.caption("See tab 1 (NEW HISTORY) for event details and tab 8 (Report) for full document view.")
        return

    type_summary = month_df.groupby(["구분", "범위", "유형"], as_index=False).size().rename(columns={"size": "NEW 건수"})
    metric_order = {"원자료": 0, "YoY(절대)": 1, "YoY(증감률)": 2}
    scope_order = {"전체기간": 0, "최근10년": 1, "최근5년": 2}
    event_type_order = {"최고": 0, "최저": 1}
    type_summary["정렬_구분"] = type_summary["구분"].map(metric_order).fillna(999)
    type_summary["정렬_범위"] = type_summary["범위"].map(scope_order).fillna(999)
    type_summary["정렬_유형"] = type_summary["유형"].map(event_type_order).fillna(999)
    type_summary = type_summary.sort_values(["정렬_구분", "정렬_범위", "정렬_유형", "구분", "범위", "유형"]).drop(columns=["정렬_구분", "정렬_범위", "정렬_유형"])
    prev_type_map: Dict[tuple, int] = {}
    if prev_month:
        prev_type_map = prev_month_df.groupby(["구분", "범위", "유형"]).size().astype(int).to_dict()
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


def render_consecutive_change_summary(
    events: pd.DataFrame,
    report_scope: str,
    datasets: List[Any],
    source_df: pd.DataFrame,
    selected_region: Optional[str] = None,
    selected_month: Optional[str] = None,
) -> None:
    if events.empty:
        return
    view, _ = _build_report_view(
        events=events,
        report_scope=report_scope,
        selected_region=selected_region,
    )
    if view.empty:
        return

    if "기준월_ts" in view.columns:
        view["기준월_dt"] = pd.to_datetime(view["기준월_ts"], errors="coerce")
    else:
        view["기준월_dt"] = pd.to_datetime(view["기준월"], errors="coerce")
    view = view.dropna(subset=["기준월_dt"])
    if view.empty:
        return

    month_table = (
        view[["기준월", "기준월_dt"]]
        .drop_duplicates()
        .sort_values("기준월_dt", ascending=False)
        .reset_index(drop=True)
    )
    if month_table.empty:
        return

    if selected_month is None:
        selected_month = str(st.session_state.get("report_month", month_table.iloc[0]["기준월"]))
    else:
        selected_month = str(selected_month)
    if selected_month not in month_table["기준월"].astype(str).tolist():
        selected_month = str(month_table.iloc[0]["기준월"])
    selected_rows = month_table[month_table["기준월"].astype(str) == selected_month]
    asof_period = pd.to_datetime(selected_rows["기준월_dt"], errors="coerce").dropna().max()
    if pd.isna(asof_period):
        return

    labels = time_labels([str(getattr(cfg, "prd_se", "M")) for cfg in datasets])
    streak_lines = _build_consecutive_change_lines(
        source_df=source_df,
        region=str(selected_region or "").strip() or _pick_streak_region(source_df, report_scope),
        asof_period=pd.Timestamp(asof_period),
        labels=labels,
    )
    if streak_lines:
        st.markdown("##### 연속 증가/감소 요약")
        st.markdown("\n".join(streak_lines))


def _build_indicator_region_extreme_table(df: pd.DataFrame) -> pd.DataFrame:
    if df.empty:
        return pd.DataFrame()

    group_cols = ["기준월", "데이터셋", "지표", "분류", "구분", "범위"]
    work = df.copy()
    work["지역"] = work["지역"].astype(str).str.strip()
    work["유형"] = work["유형"].astype(str).str.strip()

    def _region_list(series: pd.Series) -> str:
        regions = sorted({str(x).strip() for x in series if str(x).strip()})
        return ", ".join(regions) if regions else "-"

    high = (
        work[work["유형"] == "최고"]
        .groupby(group_cols, dropna=False)["지역"]
        .agg(_region_list)
        .reset_index(name="최고 지역")
    )
    low = (
        work[work["유형"] == "최저"]
        .groupby(group_cols, dropna=False)["지역"]
        .agg(_region_list)
        .reset_index(name="최저 지역")
    )

    out = high.merge(low, on=group_cols, how="outer")
    out["분류"] = out["분류"].astype(str).str.strip().replace("", "전체")
    out["최고 지역"] = out["최고 지역"].fillna("-")
    out["최저 지역"] = out["최저 지역"].fillna("-")

    return out.sort_values(
        ["기준월", "데이터셋", "지표", "분류", "구분", "범위"],
        ascending=[False, True, True, True, True, True],
    )


def render_new_history_tab(events: pd.DataFrame) -> None:
    st.subheader("NEW HISTORY")
    if events.empty:
        st.info("집계된 NEW 이벤트 이력이 없습니다.")
        return

    view = events.copy()
    f3, f4 = st.columns([1, 1])
    with f3:
        region_options = ["전체"] + sorted(view["지역"].dropna().unique().tolist())
        default_region_sel = "전체"
        region_sel = st.selectbox(
            "지역",
            region_options,
            index=region_options.index(default_region_sel),
            key="history_region_filter",
        )
    with f4:
        month_table = (
            view[["기준월", "기준월_ts"]]
            .drop_duplicates()
            .assign(기준월_dt=lambda d: pd.to_datetime(d["기준월_ts"], errors="coerce"))
            .dropna(subset=["기준월_dt"])
            .sort_values("기준월_dt", ascending=False)
        )
        month_options = month_table["기준월"].astype(str).tolist()
        date_options = ["전체"] + month_options if month_options else ["전체"]
        date_default_idx = 1 if month_options else 0
        date_sel = st.selectbox(
            "일자",
            date_options,
            index=date_default_idx,
            key="history_date_filter",
        )

    detail_view = view.copy()
    if region_sel != "전체":
        detail_view = detail_view[detail_view["지역"] == region_sel]
    if date_sel != "전체":
        detail_view = detail_view[detail_view["기준월"].astype(str) == str(date_sel)]

    detail_df = detail_view[
        ["기준월", "데이터셋", "지역", "지표", "분류", "구분", "범위", "유형", "이벤트"]
    ].sort_values(
        ["기준월", "데이터셋", "지역", "지표", "분류", "구분", "범위", "유형"],
        ascending=[False, True, True, True, True, True, True, True],
    )

    date_label = str(date_sel) if date_sel != "전체" else "전체"
    region_label = str(region_sel) if region_sel != "전체" else "전체"
    st.markdown(f"##### 상세 이벤트 (일자: {date_label} | 지역: {region_label})")
    detail_lines = []
    for _, row in detail_df.iterrows():
        category_text = str(row["분류"]).strip() if str(row["분류"]).strip() else "전체"
        detail_lines.append(
            f"- **[{row['기준월']}]** {row['데이터셋']} | {row['지역']} | {row['지표']} | {category_text} | {row['구분']} {row['범위']} {row['유형']}"
        )
    st.markdown("\n".join(detail_lines) if detail_lines else "- 표시할 이벤트가 없습니다.")

    by_indicator_source = view.copy()
    if date_sel != "전체":
        by_indicator_source = by_indicator_source[by_indicator_source["기준월"].astype(str) == str(date_sel)]
    by_indicator_df = _build_indicator_region_extreme_table(by_indicator_source)

    st.markdown(f"##### 지표별 최고/최저 지역 (일자: {date_label})")
    if region_sel != "전체":
        st.caption("지표별 비교를 위해 이 표에는 지역 필터를 적용하지 않았습니다.")
    if by_indicator_df.empty:
        st.info("지표별 최고/최저 지역 데이터가 없습니다.")
    else:
        st.dataframe(
            by_indicator_df,
            use_container_width=True,
            hide_index=True,
            column_config={
                "기준월": st.column_config.TextColumn("기준월", width="small"),
                "데이터셋": st.column_config.TextColumn("데이터셋", width="small"),
                "지표": st.column_config.TextColumn("지표", width="small"),
                "분류": st.column_config.TextColumn("분류", width="small"),
                "구분": st.column_config.TextColumn("구분", width="small"),
                "범위": st.column_config.TextColumn("범위", width="small"),
                "최고 지역": st.column_config.TextColumn("최고 지역", width="large"),
                "최저 지역": st.column_config.TextColumn("최저 지역", width="large"),
            },
        )
