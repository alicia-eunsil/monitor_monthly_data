from __future__ import annotations

import re
from typing import Any, Dict, List, Tuple

import numpy as np
import pandas as pd
import streamlit as st

import src.config as app_config
from src.core.formatters import fmt_num, fmt_period, time_labels
from src.features.new_history import collect_new_events
from src.features.streak_utils import current_streak_length

GYEONGGI_SIGUNGU = getattr(app_config, "GYEONGGI_SIGUNGU", [])


TYPE_ORDER = [
    "전반확장형",
    "중장년견인형",
    "편중성장형",
    "혼조전환형",
    "구조조정압력형",
]

TYPE_RULES = {
    "전반확장형": "취업자 증가 + 연령/산업/직종에서 증가 분류 비중이 모두 높은 시군",
    "중장년견인형": "취업자 증가 + 연령구조에서 중장년 증가 기여가 청년보다 큰 시군",
    "편중성장형": "취업자 증가 + 산업/직종 중 일부 분류에 증가가 집중된 시군",
    "혼조전환형": "증가·감소 신호가 혼재되어 전환 국면 관찰이 필요한 시군",
    "구조조정압력형": "취업자 감소 또는 감소 신호 우세로 하방 압력이 큰 시군",
}

POLICY_MATCH = {
    "전반확장형": {
        "정책방향": "확산 유지 + 미스매치 완화",
        "핵심대상": "증가 폭이 큰 산업·직종",
        "권장과제": "채용연계 강화, 훈련-일자리 매칭, 구인난 직무 우선 대응",
        "점검지표": "확산도 유지 여부, 증가 상위 분류의 지속성",
    },
    "중장년견인형": {
        "정책방향": "세대 균형 회복",
        "핵심대상": "청년층 감소 분류, 중장년 집중 분류",
        "권장과제": "청년 채용 인센티브, 전환훈련, 세대혼합 채용모델",
        "점검지표": "청년/중장년 증감 격차, 청년 감소 연속기간",
    },
    "편중성장형": {
        "정책방향": "집중 리스크 분산",
        "핵심대상": "증가 집중 산업·직종, 취약 보완 분류",
        "권장과제": "편중 업종 외 보완투자, 업종 다변화 프로그램",
        "점검지표": "집중도(Top1 비중), 확산도 변화",
    },
    "혼조전환형": {
        "정책방향": "전환기 조기대응",
        "핵심대상": "증가·감소가 엇갈리는 분류",
        "권장과제": "분기 단위 신속 점검, 선택·집중형 사업 배분",
        "점검지표": "연속증가/감소 동시 발생, NEW 이벤트 방향성",
    },
    "구조조정압력형": {
        "정책방향": "하방 충격 완화",
        "핵심대상": "감소 장기화 분류, 취약계층·취약직무",
        "권장과제": "전직·재취업 패키지, 고용유지 컨설팅, 현장 밀착지원",
        "점검지표": "감소 연속기간, 최저 NEW 반복 여부",
    },
}


def _to_num(series: pd.Series) -> pd.Series:
    return pd.to_numeric(series, errors="coerce")


def _clean_text(value: object) -> str:
    return str(value or "").strip()


def _compact_text(value: object) -> str:
    return re.sub(r"\s+", "", _clean_text(value))


def _is_total_category(name: object) -> bool:
    n = _compact_text(name).lower()
    if not n:
        return True
    if n in {"계", "합계", "총계", "전체", "total"}:
        return True
    return False


def _extract_activity_employment_rows(df: pd.DataFrame) -> pd.DataFrame:
    if df.empty:
        return df
    work = df.copy()
    work["indicator_name"] = work["indicator_name"].astype(str)
    mask = work["indicator_name"].str.replace(" ", "", regex=False).str.contains("취업자", na=False)
    picked = work[mask].copy()
    return picked if not picked.empty else work


def _infer_latest(df: pd.DataFrame) -> Tuple[pd.Timestamp, str]:
    if df.empty:
        return pd.NaT, "M"
    latest = pd.to_datetime(df["period"], errors="coerce").dropna().max()
    prd_se = str(df["prd_se"].dropna().iloc[0]).upper() if "prd_se" in df.columns and not df["prd_se"].dropna().empty else "M"
    return pd.Timestamp(latest), prd_se


def _latest_slice(df: pd.DataFrame, region: str, dataset_key: str, latest_period: pd.Timestamp) -> pd.DataFrame:
    out = df[
        (df["region_name"].astype(str) == str(region))
        & (df["dataset_key"].astype(str) == str(dataset_key))
        & (pd.to_datetime(df["period"], errors="coerce") == pd.Timestamp(latest_period))
    ].copy()
    if out.empty:
        return out
    out["yoy_abs"] = _to_num(out["yoy_abs"])
    out["value"] = _to_num(out["value"])
    out = out.dropna(subset=["yoy_abs"])
    if dataset_key == "activity":
        out = _extract_activity_employment_rows(out)
    if dataset_key in {"age", "status", "industry", "occupation"}:
        out = out[~out["category_name"].map(_is_total_category)].copy()
    return out


def _breadth_and_concentration(view: pd.DataFrame) -> Tuple[float, float]:
    if view.empty:
        return np.nan, np.nan
    s = _to_num(view["yoy_abs"]).dropna()
    if s.empty:
        return np.nan, np.nan
    breadth = float((s > 0).sum()) / float(len(s))
    denom = float(s.abs().sum())
    concentration = (float(s.abs().max()) / denom) if denom > 0 else np.nan
    return breadth, concentration


def _top_pos_neg_label(view: pd.DataFrame) -> Tuple[str, str]:
    if view.empty:
        return "-", "-"
    s = view.copy()
    s["yoy_abs"] = _to_num(s["yoy_abs"])
    s = s.dropna(subset=["yoy_abs"])
    if s.empty:
        return "-", "-"
    pos = s[s["yoy_abs"] > 0].sort_values("yoy_abs", ascending=False)
    neg = s[s["yoy_abs"] < 0].sort_values("yoy_abs", ascending=True)

    def _fmt(row: pd.Series) -> str:
        cat = _clean_text(row.get("category_name", "")) or "전체"
        return f"{cat}({fmt_num(float(row['yoy_abs']))})"

    pos_label = _fmt(pos.iloc[0]) if not pos.empty else "-"
    neg_label = _fmt(neg.iloc[0]) if not neg.empty else "-"
    return pos_label, neg_label


def _is_youth_label(cat: str) -> bool:
    c = _compact_text(cat)
    return bool(re.search(r"(15[-~]?19|15[-~]?24|15[-~]?29|20[-~]?29|청년)", c))


def _is_senior_label(cat: str) -> bool:
    c = _compact_text(cat)
    return bool(re.search(r"(50[-~]?64|55세이상|60세이상|65세이상)", c))


def _age_balance(age_view: pd.DataFrame) -> Tuple[float, float]:
    if age_view.empty:
        return np.nan, np.nan
    s = age_view.copy()
    s["yoy_abs"] = _to_num(s["yoy_abs"])
    s = s.dropna(subset=["yoy_abs"])
    if s.empty:
        return np.nan, np.nan
    youth = float(s[s["category_name"].astype(str).map(_is_youth_label)]["yoy_abs"].sum())
    senior = float(s[s["category_name"].astype(str).map(_is_senior_label)]["yoy_abs"].sum())
    return youth, senior


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
            yoy = _to_num(g["yoy_abs"])
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


def _choose_type(row: pd.Series) -> str:
    total_yoy = float(row.get("취업자증감", np.nan))
    age_b = float(row.get("연령확산도", np.nan))
    ind_b = float(row.get("산업확산도", np.nan))
    occ_b = float(row.get("직종확산도", np.nan))
    ind_c = float(row.get("산업집중도", np.nan))
    occ_c = float(row.get("직종집중도", np.nan))
    youth = float(row.get("청년증감", np.nan))
    senior = float(row.get("중장년증감", np.nan))
    up3 = int(row.get("연속증가(3기+)", 0))
    down3 = int(row.get("연속감소(3기+)", 0))

    is_total_up = pd.notna(total_yoy) and total_yoy > 0
    strong_breadth = all(pd.notna(v) and v >= 0.6 for v in [age_b, ind_b, occ_b])
    biased_growth = any(pd.notna(v) and v >= 0.48 for v in [ind_c, occ_c])
    senior_driven = (
        is_total_up
        and pd.notna(senior)
        and senior > 0
        and pd.notna(youth)
        and youth <= 0
        and (senior - youth) >= max(5.0, 0.25 * abs(total_yoy))
    )
    pressure = (
        (pd.notna(total_yoy) and total_yoy < 0)
        or ((pd.notna(ind_b) and ind_b < 0.45) and (pd.notna(occ_b) and occ_b < 0.45))
        or (down3 >= up3 + 2)
    )

    if is_total_up and strong_breadth:
        return "전반확장형"
    if senior_driven:
        return "중장년견인형"
    if is_total_up and biased_growth:
        return "편중성장형"
    if pressure:
        return "구조조정압력형"
    return "혼조전환형"


def _reason_text(row: pd.Series) -> str:
    parts: List[str] = []
    if pd.notna(row.get("취업자증감")):
        sign = "증가" if float(row["취업자증감"]) > 0 else ("감소" if float(row["취업자증감"]) < 0 else "보합")
        parts.append(f"취업자 {sign}")
    if pd.notna(row.get("연령확산도")):
        parts.append(f"연령확산 {float(row['연령확산도']) * 100:.0f}%")
    if pd.notna(row.get("산업확산도")):
        parts.append(f"산업확산 {float(row['산업확산도']) * 100:.0f}%")
    if pd.notna(row.get("직종확산도")):
        parts.append(f"직종확산 {float(row['직종확산도']) * 100:.0f}%")
    if pd.notna(row.get("산업집중도")):
        parts.append(f"산업집중 {float(row['산업집중도']) * 100:.0f}%")
    if pd.notna(row.get("직종집중도")):
        parts.append(f"직종집중 {float(row['직종집중도']) * 100:.0f}%")
    return ", ".join(parts) if parts else "-"


def _build_features(work: pd.DataFrame, latest_period: pd.Timestamp) -> pd.DataFrame:
    regions = [r for r in GYEONGGI_SIGUNGU if r in work["region_name"].astype(str).unique().tolist()]
    rows: List[Dict[str, Any]] = []
    for region in regions:
        activity = _latest_slice(work, region, "activity", latest_period)
        age = _latest_slice(work, region, "age", latest_period)
        industry = _latest_slice(work, region, "industry", latest_period)
        occupation = _latest_slice(work, region, "occupation", latest_period)

        total_yoy = np.nan
        if not activity.empty:
            pick = activity.copy()
            pick = pick.sort_values("yoy_abs", ascending=False)
            total_yoy = float(_to_num(pick["yoy_abs"]).dropna().iloc[0]) if not _to_num(pick["yoy_abs"]).dropna().empty else np.nan

        age_b, _ = _breadth_and_concentration(age)
        ind_b, ind_c = _breadth_and_concentration(industry)
        occ_b, occ_c = _breadth_and_concentration(occupation)
        youth_yoy, senior_yoy = _age_balance(age)
        ind_pos, ind_neg = _top_pos_neg_label(industry)
        occ_pos, occ_neg = _top_pos_neg_label(occupation)

        rows.append(
            {
                "시군": region,
                "취업자증감": total_yoy,
                "연령확산도": age_b,
                "산업확산도": ind_b,
                "직종확산도": occ_b,
                "산업집중도": ind_c,
                "직종집중도": occ_c,
                "청년증감": youth_yoy,
                "중장년증감": senior_yoy,
                "산업 증가1위": ind_pos,
                "산업 감소1위": ind_neg,
                "직종 증가1위": occ_pos,
                "직종 감소1위": occ_neg,
            }
        )

    return pd.DataFrame(rows)


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
    st.caption(f"기준{labels['point']}: {fmt_period(latest_period, prd_se)} (최신시점 기준)")

    feature = _build_features(work, latest_period)
    if feature.empty:
        st.info("시군별 특징량을 만들 수 없습니다.")
        return

    events = collect_new_events(work)
    if not events.empty and "기준월_ts" in events.columns:
        evt = events.copy()
        evt["기준월_ts"] = pd.to_datetime(evt["기준월_ts"], errors="coerce")
        latest_evt = evt[evt["기준월_ts"] == pd.Timestamp(latest_period)].copy()
        latest_evt_count = (
            latest_evt.groupby("지역", as_index=False)
            .size()
            .rename(columns={"size": "최신NEW"})
            .rename(columns={"지역": "시군"})
        )
        feature = feature.merge(latest_evt_count, on="시군", how="left")
    if "최신NEW" not in feature.columns:
        feature["최신NEW"] = 0
    feature["최신NEW"] = feature["최신NEW"].fillna(0).astype(int)

    streak_df = _build_region_streak_features(work, latest_period)
    feature = feature.merge(streak_df, on="시군", how="left")
    for c in ["연속증가(3기+)", "연속감소(3기+)", "최장증가", "최장감소"]:
        feature[c] = feature[c].fillna(0).astype(int)

    feature["유형"] = feature.apply(_choose_type, axis=1)
    feature["판정근거"] = feature.apply(_reason_text, axis=1)
    feature["유형정렬"] = feature["유형"].map({name: i for i, name in enumerate(TYPE_ORDER)}).fillna(999)
    feature = feature.sort_values(["유형정렬", "시군"]).drop(columns=["유형정렬"])

    st.markdown("##### 유형 정의(규칙 기반)")
    rule_df = pd.DataFrame([{"유형": k, "정의": TYPE_RULES[k]} for k in TYPE_ORDER])
    st.dataframe(rule_df, use_container_width=True, hide_index=True)

    st.markdown("##### 유형별 시군 수")
    summary = feature.groupby("유형", as_index=False).agg(
        시군수=("시군", "count"),
        평균_취업자증감=("취업자증감", "mean"),
        평균_연령확산도=("연령확산도", "mean"),
        평균_산업확산도=("산업확산도", "mean"),
        평균_직종확산도=("직종확산도", "mean"),
    )
    summary["평균_취업자증감"] = summary["평균_취업자증감"].round(1)
    for col in ["평균_연령확산도", "평균_산업확산도", "평균_직종확산도"]:
        summary[col] = (summary[col] * 100).round(1)
    summary["유형정렬"] = summary["유형"].map({name: i for i, name in enumerate(TYPE_ORDER)}).fillna(999)
    summary = summary.sort_values(["유형정렬", "유형"]).drop(columns=["유형정렬"])
    st.dataframe(summary, use_container_width=True, hide_index=True)

    st.markdown("##### 시군별 분류 결과")
    view = feature.copy()
    for col in ["연령확산도", "산업확산도", "직종확산도", "산업집중도", "직종집중도"]:
        view[col] = (view[col] * 100).round(1)
    view["취업자증감"] = view["취업자증감"].map(lambda v: fmt_num(v) if pd.notna(v) else "-")
    view["청년증감"] = view["청년증감"].map(lambda v: fmt_num(v) if pd.notna(v) else "-")
    view["중장년증감"] = view["중장년증감"].map(lambda v: fmt_num(v) if pd.notna(v) else "-")
    st.dataframe(
        view[
            [
                "시군",
                "유형",
                "취업자증감",
                "연령확산도",
                "산업확산도",
                "직종확산도",
                "산업집중도",
                "직종집중도",
                "청년증감",
                "중장년증감",
                "연속증가(3기+)",
                "연속감소(3기+)",
                "최신NEW",
                "산업 증가1위",
                "산업 감소1위",
                "직종 증가1위",
                "직종 감소1위",
                "판정근거",
            ]
        ],
        use_container_width=True,
        hide_index=True,
    )

    st.markdown("##### 유형별 정책 매칭표")
    policy_df = pd.DataFrame(
        [
            {
                "유형": t,
                "정책방향": policy["정책방향"],
                "핵심대상": policy["핵심대상"],
                "권장과제": policy["권장과제"],
                "점검지표": policy["점검지표"],
            }
            for t, policy in POLICY_MATCH.items()
        ]
    )
    policy_df["유형정렬"] = policy_df["유형"].map({name: i for i, name in enumerate(TYPE_ORDER)}).fillna(999)
    policy_df = policy_df.sort_values(["유형정렬", "유형"]).drop(columns=["유형정렬"])
    st.dataframe(policy_df, use_container_width=True, hide_index=True)
