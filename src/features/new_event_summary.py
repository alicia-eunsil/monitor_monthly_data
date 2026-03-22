from __future__ import annotations

from typing import Any, Dict, List, Optional

import pandas as pd


def _fmt_delta(cur: int, prev: Optional[int]) -> str:
    if prev is None:
        return ""
    diff = int(cur) - int(prev)
    if diff > 0:
        return f" (+{diff:,})"
    if diff < 0:
        return f" (-{abs(diff):,})"
    return " (=)"


def build_new_count_summary_lines(
    month_df: pd.DataFrame,
    prev_month_df: Optional[pd.DataFrame],
    labels: Dict[str, str],
) -> List[str]:
    prev_df = prev_month_df if prev_month_df is not None else pd.DataFrame(columns=month_df.columns)
    has_prev = prev_month_df is not None and not prev_df.empty

    total_count = len(month_df)
    max_count = int((month_df["유형"] == "최고").sum()) if "유형" in month_df.columns else 0
    min_count = int((month_df["유형"] == "최저").sum()) if "유형" in month_df.columns else 0
    prev_total = len(prev_df) if has_prev else None
    prev_max = int((prev_df["유형"] == "최고").sum()) if has_prev and "유형" in prev_df.columns else None
    prev_min = int((prev_df["유형"] == "최저").sum()) if has_prev and "유형" in prev_df.columns else None

    return [
        f"##### {labels.get('point', '월')} 요약",
        f"- 총 NEW 이벤트: **{total_count:,}건**{_fmt_delta(total_count, prev_total)}",
        f"- 최고 NEW: **{max_count:,}건**{_fmt_delta(max_count, prev_max)}",
        f"- 최저 NEW: **{min_count:,}건**{_fmt_delta(min_count, prev_min)}",
    ]


def build_dataset_count_lines(
    month_df: pd.DataFrame,
    prev_month_df: Optional[pd.DataFrame],
    datasets: List[Any],
    top_n: Optional[int] = None,
) -> List[str]:
    if month_df.empty:
        return ["##### 데이터셋별 건수", "- 표시할 데이터가 없습니다."]

    ds_summary = month_df.groupby("데이터셋", as_index=False).size().rename(columns={"size": "NEW 건수"})
    dataset_order = [getattr(cfg, "title", "") for cfg in datasets]
    ds_summary["정렬순서"] = ds_summary["데이터셋"].map({name: idx for idx, name in enumerate(dataset_order)})
    ds_summary = ds_summary.sort_values(by=["정렬순서", "데이터셋"], ascending=[True, True], na_position="last").drop(
        columns=["정렬순서"]
    )

    prev_map: Dict[str, int] = {}
    if prev_month_df is not None and not prev_month_df.empty:
        prev_map = prev_month_df.groupby("데이터셋").size().astype(int).to_dict()

    view = ds_summary.head(int(top_n)) if top_n is not None else ds_summary
    lines = ["##### 데이터셋별 건수"]
    lines.extend(
        [
            f"- {row['데이터셋']}: **{int(row['NEW 건수']):,}건**"
            f"{_fmt_delta(int(row['NEW 건수']), prev_map.get(str(row['데이터셋'])) if prev_map else None)}"
            for _, row in view.iterrows()
        ]
    )
    if top_n is not None and len(ds_summary) > len(view):
        lines.append(f"- ... 데이터셋 {len(ds_summary):,}개 중 상위 {len(view):,}개만 표시")
    return lines


def build_new_focus_line(
    events_df: pd.DataFrame,
    event_type: str,
    line_title: str,
    top_n: int = 2,
) -> str:
    if events_df is None or events_df.empty:
        return f"{line_title}: 없음"

    view = events_df[events_df["유형"].astype(str) == str(event_type)].copy()
    if view.empty:
        return f"{line_title}: 없음"

    ds_order = {
        "경제활동인구현황": 0,
        "산업별 취업자수": 1,
        "직종별 취업자수": 2,
        "종사상지위별 취업자": 3,
        "연령별 취업자": 4,
    }
    metric_order = {"원자료": 0, "YoY(절대)": 1, "YoY(증감률)": 2}
    scope_order = {"전체기간": 0, "최근5년": 1}
    view["_ds"] = view["데이터셋"].map(ds_order).fillna(99)
    view["_metric"] = view["구분"].map(metric_order).fillna(99)
    view["_scope"] = view["범위"].map(scope_order).fillna(99)
    view = view.sort_values(["_ds", "_metric", "_scope", "지표", "분류"])

    tokens: List[str] = []
    for _, row in view.head(int(top_n)).iterrows():
        cat = str(row.get("분류", "")).strip() or "전체"
        tokens.append(
            f"{row['데이터셋']} {row['지표']}/{cat} "
            f"({row['구분']} {row['범위']} {row['유형']})"
        )
    return f"{line_title}: " + (", ".join(tokens) if tokens else "없음")
