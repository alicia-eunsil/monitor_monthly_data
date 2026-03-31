from __future__ import annotations

from typing import Any, Dict, List

import numpy as np
import pandas as pd
import streamlit as st

import src.config as app_config
from src.core.formatters import fmt_period, time_labels
from src.features.new_history import collect_new_events
from src.features.streak_utils import current_streak_length

GYEONGGI_SIGUNGU = getattr(app_config, "GYEONGGI_SIGUNGU", [])


TYPE_ORDER = [
    "상승확산형",
    "하방압력형",
    "변동성경계형",
    "전환관찰형",
    "안정관리형",
]

TYPE_RULES = {
    "상승확산형": "연속 증가(3기 이상) 건수가 연속 감소 대비 뚜렷하게 많고, 최고 NEW가 우세한 시군",
    "하방압력형": "연속 감소(3기 이상) 건수가 연속 증가 대비 뚜렷하게 많고, 최저 NEW가 우세한 시군",
    "변동성경계형": "최신시점 NEW 발생이 많고, 증가/감소 신호가 동시에 나타나는 변동성 높은 시군",
    "전환관찰형": "증가·감소 신호가 혼재되어 방향 전환 가능성을 모니터링해야 하는 시군",
    "안정관리형": "최신시점 NEW와 연속신호가 상대적으로 적어 안정적 흐름을 보이는 시군",
}

POLICY_MATCH = {
    "상승확산형": {
        "정책방향": "성장세 확산 및 인력 미스매치 완화",
        "우선대상": "증가 기여 상위 산업·직종",
        "실행과제": "채용연계 강화 / 숙련전환 훈련 / 기업 맞춤 인력공급",
        "모니터링지표": "연속증가 기간, 원자료 최고 NEW 지속 여부",
    },
    "하방압력형": {
        "정책방향": "감소 충격 완화 및 재취업 전환 지원",
        "우선대상": "연속감소 장기화 분류",
        "실행과제": "집중 전직지원 / 고용유지 컨설팅 / 취약계층 맞춤지원",
        "모니터링지표": "연속감소 기간, 원자료 최저 NEW 재발 여부",
    },
    "변동성경계형": {
        "정책방향": "단기 변동 대응력 강화",
        "우선대상": "동시에 최고·최저 NEW가 발생한 분류",
        "실행과제": "월/반기 조기경보 / 현장점검 / 탄력적 사업배분",
        "모니터링지표": "최신 NEW 건수, 최고·최저 NEW 동시 발생 비율",
    },
    "전환관찰형": {
        "정책방향": "전환 국면 선제 관리",
        "우선대상": "증가·감소 혼재 분류",
        "실행과제": "핵심 분류 추적 / 지역 맞춤 패키지 / 분기별 정책 재조정",
        "모니터링지표": "증가/감소 균형도, 전환 직전 분류의 추세 지속성",
    },
    "안정관리형": {
        "정책방향": "현 수준 유지와 예방적 관리",
        "우선대상": "잠재 리스크 분류(변화 미미 구간)",
        "실행과제": "정기 모니터링 / 소규모 예방사업 / 데이터 품질 점검",
        "모니터링지표": "NEW 발생 빈도, 3기 이상 연속 신호 건수",
    },
}


def _infer_latest(df: pd.DataFrame) -> tuple[pd.Timestamp, str]:
    if df.empty:
        return pd.NaT, "M"
    latest = pd.to_datetime(df["period"], errors="coerce").dropna().max()
    prd_se = str(df["prd_se"].dropna().iloc[0]).upper() if "prd_se" in df.columns and not df["prd_se"].dropna().empty else "M"
    return pd.Timestamp(latest), prd_se


def _build_region_streak_features(df: pd.DataFrame, latest_period: pd.Timestamp) -> pd.DataFrame:
    base = df.copy()
    base["period"] = pd.to_datetime(base["period"], errors="coerce")
    base = base.dropna(subset=["period"])
    base = base[base["period"] <= latest_period].copy()
    if base.empty:
        return pd.DataFrame()

    rows: List[Dict[str, Any]] = []
    for region in sorted(base["region_name"].dropna().astype(str).unique().tolist()):
        r = base[base["region_name"].astype(str) == str(region)].copy()
        up3 = 0
        down3 = 0
        max_up = 0
        max_down = 0
        for _, g in r.groupby(["dataset_key", "indicator_name", "category_name"], dropna=False):
            g = g.sort_values("period")
            if g.empty or pd.Timestamp(g["period"].iloc[-1]) != latest_period:
                continue
            yoy = pd.to_numeric(g["yoy_abs"], errors="coerce")
            if yoy.empty or pd.isna(yoy.iloc[-1]):
                continue
            up_len = int(current_streak_length(yoy, positive=True))
            down_len = int(current_streak_length(yoy, positive=False))
            if up_len >= 3:
                up3 += 1
            if down_len >= 3:
                down3 += 1
            max_up = max(max_up, up_len)
            max_down = max(max_down, down_len)
        rows.append(
            {
                "시군": region,
                "연속증가(3기+)": int(up3),
                "연속감소(3기+)": int(down3),
                "최장증가": int(max_up),
                "최장감소": int(max_down),
            }
        )
    return pd.DataFrame(rows)


def _classify_regions(feature_df: pd.DataFrame) -> pd.DataFrame:
    if feature_df.empty:
        return feature_df
    view = feature_df.copy()
    new_q75 = float(view["최신 NEW"].quantile(0.75))
    new_q25 = float(view["최신 NEW"].quantile(0.25))

    def _label(row: pd.Series) -> str:
        up = int(row["연속증가(3기+)"])
        down = int(row["연속감소(3기+)"])
        hi = int(row["최고 NEW"])
        lo = int(row["최저 NEW"])
        new_total = int(row["최신 NEW"])

        if down >= up + 2 and lo >= hi:
            return "하방압력형"
        if up >= down + 2 and hi >= lo:
            return "상승확산형"
        if new_total >= new_q75 and up > 0 and down > 0:
            return "변동성경계형"
        if new_total <= new_q25 and (up + down) <= 2:
            return "안정관리형"
        return "전환관찰형"

    view["유형"] = view.apply(_label, axis=1)
    view["유형정렬"] = view["유형"].map({name: i for i, name in enumerate(TYPE_ORDER)}).fillna(999)
    return view.sort_values(["유형정렬", "시군"]).drop(columns=["유형정렬"])


def render_sigungu_typology_tab(df: pd.DataFrame, is_gyeonggi31_mode: bool, datasets: List[Any]) -> None:
    st.subheader("시군 유형화·정책 매칭")
    if not is_gyeonggi31_mode:
        st.info("경기 31개 시군 모드에서만 제공합니다.")
        return
    if df.empty:
        st.info("유형화할 데이터가 없습니다.")
        return

    work = df[df["region_name"].isin(GYEONGGI_SIGUNGU)].copy()
    if work.empty:
        st.info("시군 데이터가 없습니다.")
        return

    latest_period, prd_se = _infer_latest(work)
    if pd.isna(latest_period):
        st.info("기준시점을 찾을 수 없습니다.")
        return

    labels = time_labels([str(getattr(cfg, "prd_se", "M")) for cfg in datasets])
    latest_label = fmt_period(latest_period, prd_se)
    st.caption(f"기준{labels['point']}: {latest_label} (최신시점 기준)")

    events = collect_new_events(work)
    latest_events = events[events["기준월"].astype(str) == str(latest_label)].copy()
    if latest_events.empty:
        st.info("최신시점 NEW 이벤트가 없어 유형화를 생성할 수 없습니다.")
        return

    evt = (
        latest_events.groupby(["지역", "유형"], as_index=False)
        .size()
        .rename(columns={"size": "건수"})
        .pivot(index="지역", columns="유형", values="건수")
        .fillna(0)
    )
    for col in ["최고", "최저"]:
        if col not in evt.columns:
            evt[col] = 0
    evt["최신 NEW"] = evt["최고"] + evt["최저"]
    evt = evt.rename(columns={"최고": "최고 NEW", "최저": "최저 NEW"}).reset_index().rename(columns={"지역": "시군"})

    streak = _build_region_streak_features(work, latest_period)
    feature = evt.merge(streak, on="시군", how="left").fillna(0)
    for c in ["연속증가(3기+)", "연속감소(3기+)", "최장증가", "최장감소"]:
        feature[c] = feature[c].astype(int)

    typed = _classify_regions(feature)

    st.markdown("##### 유형 정의(규칙 기반)")
    rule_df = pd.DataFrame(
        [{"유형": k, "정의": TYPE_RULES[k]} for k in TYPE_ORDER if k in TYPE_RULES]
    )
    st.dataframe(rule_df, use_container_width=True, hide_index=True)

    st.markdown("##### 유형별 시군 현황")
    summary = (
        typed.groupby("유형", as_index=False)
        .agg(
            시군수=("시군", "count"),
            평균_NEW=("최신 NEW", "mean"),
            평균_연속증가=("연속증가(3기+)", "mean"),
            평균_연속감소=("연속감소(3기+)", "mean"),
        )
    )
    summary["평균_NEW"] = summary["평균_NEW"].round(1)
    summary["평균_연속증가"] = summary["평균_연속증가"].round(1)
    summary["평균_연속감소"] = summary["평균_연속감소"].round(1)
    summary["유형정렬"] = summary["유형"].map({name: i for i, name in enumerate(TYPE_ORDER)}).fillna(999)
    summary = summary.sort_values(["유형정렬", "유형"]).drop(columns=["유형정렬"])
    st.dataframe(summary, use_container_width=True, hide_index=True)

    st.markdown("##### 시군별 분류 결과")
    st.dataframe(
        typed[
            ["시군", "유형", "최신 NEW", "최고 NEW", "최저 NEW", "연속증가(3기+)", "연속감소(3기+)", "최장증가", "최장감소"]
        ],
        use_container_width=True,
        hide_index=True,
    )

    st.markdown("##### 유형별 정책 매칭표")
    policy_df = pd.DataFrame(
        [
            {
                "유형": t,
                "정책방향": POLICIES["정책방향"],
                "우선대상": POLICIES["우선대상"],
                "실행과제(권장)": POLICIES["실행과제"],
                "모니터링지표": POLICIES["모니터링지표"],
            }
            for t, POLICIES in POLICY_MATCH.items()
        ]
    )
    policy_df["유형정렬"] = policy_df["유형"].map({name: i for i, name in enumerate(TYPE_ORDER)}).fillna(999)
    policy_df = policy_df.sort_values(["유형정렬", "유형"]).drop(columns=["유형정렬"])
    st.dataframe(policy_df, use_container_width=True, hide_index=True)
