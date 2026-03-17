from typing import Any, Dict, List, Optional

import pandas as pd
import streamlit as st

import src.config as app_config
from src.core.category_rules import ACTIVITY_INDICATOR_ORDER, norm_indicator_name
from src.core.formatters import escape_markdown_text, fmt_num, fmt_period, time_labels

GYEONGGI_SIGUNGU = getattr(app_config, "GYEONGGI_SIGUNGU", [])


def _current_streak_length(values: pd.Series, positive: bool) -> int:
    if values.empty:
        return 0
    count = 0
    for v in reversed(values.tolist()):
        if pd.isna(v):
            break
        vv = float(v)
        if positive and vv > 0:
            count += 1
            continue
        if (not positive) and vv < 0:
            count += 1
            continue
        break
    return count


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


def collect_new_events(df: pd.DataFrame) -> pd.DataFrame:
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
                            "기준월": fmt_period(row["period"], prd_se),
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


def render_new_monthly_report(
    events: pd.DataFrame,
    report_scope: str,
    datasets: List[Any],
    source_df: pd.DataFrame,
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
    labels = time_labels([str(getattr(cfg, "prd_se", "M")) for cfg in datasets])
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

    activity_lines: List[str] = []
    activity_df = source_df[source_df["dataset_key"] == "activity"].copy()
    activity_df["period"] = pd.to_datetime(activity_df["period"], errors="coerce")
    activity_df = activity_df.dropna(subset=["period"])
    if not activity_df.empty:
        region_candidates = activity_df["region_name"].dropna().astype(str).str.strip().unique().tolist()
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
    region_candidates = source_df["region_name"].dropna().astype(str).str.strip().unique().tolist()
    if "31" in str(report_scope):
        sigungu_candidates = sorted([r for r in GYEONGGI_SIGUNGU if r in region_candidates])
        streak_region = sigungu_candidates[0] if sigungu_candidates else ""
    else:
        streak_region = "경기도" if "경기도" in region_candidates else (region_candidates[0] if region_candidates else "")
    streak_lines = _build_consecutive_change_lines(
        source_df=source_df,
        region=streak_region,
        asof_period=selected_month_dt,
        labels=labels,
    )
    if streak_lines:
        st.markdown("##### 연속 증가/감소 요약")
        st.markdown("\n".join(streak_lines))

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

    ds_summary = month_df.groupby("데이터셋", as_index=False).size().rename(columns={"size": "NEW 건수"})
    dataset_tab_order = [cfg.title for cfg in datasets]
    ds_summary["정렬순서"] = ds_summary["데이터셋"].map({name: idx for idx, name in enumerate(dataset_tab_order)})
    ds_summary = ds_summary.sort_values(by=["정렬순서", "데이터셋"], ascending=[True, True], na_position="last").drop(columns=["정렬순서"])
    prev_ds_map: Dict[str, int] = {}
    if prev_month:
        prev_ds_map = prev_month_df.groupby("데이터셋").size().astype(int).to_dict()
    ds_lines = ["##### 데이터셋별 건수"]
    ds_lines.extend(
        [
            f"- {row['데이터셋']}: **{int(row['NEW 건수']):,}건**"
            f"{_fmt_delta(int(row['NEW 건수']), prev_ds_map.get(str(row['데이터셋']))) if prev_month else ''}"
            for _, row in ds_summary.iterrows()
        ]
    )
    st.markdown("\n".join(ds_lines))

    type_summary = month_df.groupby(["구분", "범위", "유형"], as_index=False).size().rename(columns={"size": "NEW 건수"})
    metric_order = {"원자료": 0, "YoY(절대)": 1, "YoY(증감률)": 2}
    scope_order = {"전체기간": 0, "최근5년": 1}
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
