from __future__ import annotations

import re
from typing import Any, Dict, List, Tuple

import numpy as np
import pandas as pd
import streamlit as st

import src.config as app_config
from src.core.formatters import fmt_num, fmt_period, time_labels
from src.features.new_history import collect_new_events

GYEONGGI_SIGUNGU = getattr(app_config, "GYEONGGI_SIGUNGU", [])


TYPE_ORDER = [
    "전반 호조",
    "호조-감속",
    "산업·직종 재편",
    "구조 취약",
    "하방 압력",
]

TYPE_RULES = {
    "전반 호조": "취업자·고용률 등 핵심 지표가 개선되고 최고 NEW가 우세한 시군",
    "호조-감속": "총량은 양호하나 전년동기 대비 증감률 둔화/최저 신호가 함께 나타나는 시군",
    "산업·직종 재편": "제조·기능계열 약화와 서비스·사무/판매계열 강화가 동시에 나타나는 시군",
    "구조 취약": "청년·핵심 직종 약세와 최저 NEW 누적이 나타나는 시군",
    "하방 압력": "취업자 감소, 고용률 약화, 실업률 상승 등 하방 신호가 우세한 시군",
}

POLICY_MATCH = {
    "전반 호조": {
        "정책방향": "성장세 유지와 확산",
        "핵심대상": "증가 기여 상위 산업·직종",
        "권장과제": "채용연계 강화, 인력 미스매치 완화, 성장 업종 맞춤훈련",
        "점검지표": "최고 NEW 지속 여부, 확산도(증가 분류 비중)",
    },
    "호조-감속": {
        "정책방향": "둔화 구간 선제 대응",
        "핵심대상": "증감률 둔화 분류(고용률, 취업자 증감률)",
        "권장과제": "증감률 하락 분류 집중점검, 단기 보강사업 투입",
        "점검지표": "YoY 최저 NEW 재발 여부, 연속감소 전환 여부",
    },
    "산업·직종 재편": {
        "정책방향": "전환기 구조 대응",
        "핵심대상": "제조·기능 약세 / 서비스·사무·판매 강세 분류",
        "권장과제": "전직·재배치 훈련, 업종 전환형 일자리 매칭, 취약업종 완충",
        "점검지표": "제조 vs 서비스 증감 격차, 기능 vs 서비스판매 격차",
    },
    "구조 취약": {
        "정책방향": "취약구간 회복",
        "핵심대상": "청년층, 기능·기계조작·조립, 단순노무 등 약세 분류",
        "권장과제": "청년 맞춤 채용지원, 취약직종 전환훈련, 생활권 단위 취업지원",
        "점검지표": "최저 NEW 누적, 청년 증감, 취약직종 연속감소",
    },
    "하방 압력": {
        "정책방향": "하방 리스크 완화",
        "핵심대상": "취업자 감소·실업률 상승 동반 시군",
        "권장과제": "집중 고용안정 패키지, 재취업 연계, 현장점검 강화",
        "점검지표": "취업자/고용률/실업률 3대 신호 동시 악화 여부",
    },
}


def _to_num(series: pd.Series) -> pd.Series:
    return pd.to_numeric(series, errors="coerce")


def _compact_text(value: object) -> str:
    return re.sub(r"\s+", "", str(value or "").strip())


def _is_total_category(name: object) -> bool:
    n = _compact_text(name).lower()
    return n in {"", "계", "합계", "총계", "전체", "total"}


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
    if dataset_key in {"age", "status", "industry", "occupation"}:
        out = out[~out["category_name"].map(_is_total_category)].copy()
    return out


def _pick_activity_signal(activity: pd.DataFrame) -> Dict[str, float]:
    result = {
        "취업자증감": np.nan,
        "고용률증감": np.nan,
        "실업률증감": np.nan,
        "15세이상인구증감": np.nan,
    }
    if activity.empty:
        return result
    v = activity.copy()
    v["name_compact"] = v["indicator_name"].astype(str).map(_compact_text)

    def _get_val(include: List[str], exclude: List[str] | None = None) -> float:
        c = v["name_compact"].astype(str)
        mask = pd.Series(True, index=v.index)
        for token in include:
            mask &= c.str.contains(token, na=False)
        if exclude:
            for token in exclude:
                mask &= ~c.str.contains(token, na=False)
        picked = v[mask].copy()
        if picked.empty:
            return np.nan
        picked = picked.dropna(subset=["yoy_abs"])
        if picked.empty:
            return np.nan
        return float(picked.iloc[0]["yoy_abs"])

    result["취업자증감"] = _get_val(["취업자"])
    result["고용률증감"] = _get_val(["고용률"], exclude=["15~64", "1564"])
    result["실업률증감"] = _get_val(["실업률"])
    result["15세이상인구증감"] = _get_val(["15세이상인구"])
    return result


def _sum_yoy_by_keywords(view: pd.DataFrame, keywords: List[str]) -> float:
    if view.empty:
        return np.nan
    c = view["category_name"].astype(str).map(_compact_text)
    mask = pd.Series(False, index=view.index)
    for k in keywords:
        mask |= c.str.contains(_compact_text(k), na=False)
    picked = view[mask].copy()
    if picked.empty:
        return np.nan
    s = _to_num(picked["yoy_abs"]).dropna()
    return float(s.sum()) if not s.empty else np.nan


def _age_youth_senior_signal(age: pd.DataFrame) -> Tuple[float, float]:
    youth = _sum_yoy_by_keywords(age, ["15~19", "15~24", "15~29", "20~29", "청년"])
    senior = _sum_yoy_by_keywords(age, ["50~64", "55세이상", "60세이상", "65세이상"])
    return youth, senior


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
        return f"{str(row['category_name']).strip()}({fmt_num(float(row['yoy_abs']))})"

    return (_fmt(pos.iloc[0]) if not pos.empty else "-", _fmt(neg.iloc[0]) if not neg.empty else "-")


def _latest_event_enriched(work: pd.DataFrame, latest_period: pd.Timestamp, prd_se: str) -> pd.DataFrame:
    events = collect_new_events(work)
    if events.empty:
        return pd.DataFrame()
    latest_label = fmt_period(latest_period, prd_se)
    evt = events[events["기준월"].astype(str) == str(latest_label)].copy()
    if evt.empty:
        return evt

    title_map = (
        work[["dataset_key", "dataset_title"]]
        .drop_duplicates()
        .assign(dataset_title=lambda d: d["dataset_title"].astype(str))
        .set_index("dataset_title")["dataset_key"]
        .to_dict()
    )
    evt["dataset_key"] = evt["데이터셋"].astype(str).map(title_map).fillna("")
    return evt


def _event_count(evt_region: pd.DataFrame, dataset_key: str, metric: str, event_type: str) -> int:
    if evt_region.empty:
        return 0
    v = evt_region[
        (evt_region["dataset_key"].astype(str) == str(dataset_key))
        & (evt_region["구분"].astype(str) == str(metric))
        & (evt_region["유형"].astype(str) == str(event_type))
    ]
    return int(len(v))


def _classify(feature: pd.Series) -> str:
    emp = float(feature.get("취업자증감", np.nan))
    emp_rate = float(feature.get("고용률증감", np.nan))
    unemp_rate = float(feature.get("실업률증감", np.nan))
    pop15 = float(feature.get("15세이상인구증감", np.nan))
    youth = float(feature.get("청년증감", np.nan))
    senior = float(feature.get("중장년증감", np.nan))
    manuf = float(feature.get("제조계열증감", np.nan))
    service = float(feature.get("서비스계열증감", np.nan))
    func_occ = float(feature.get("기능계열증감", np.nan))
    svc_occ = float(feature.get("서비스·사무계열증감", np.nan))
    high_n = int(feature.get("최고NEW", 0))
    low_n = int(feature.get("최저NEW", 0))
    activity_yoy_low = int(feature.get("활동지표_YoY최저NEW", 0))

    down_pressure = 0
    down_pressure += 1 if pd.notna(emp) and emp < 0 else 0
    down_pressure += 1 if pd.notna(emp_rate) and emp_rate < 0 else 0
    down_pressure += 1 if pd.notna(unemp_rate) and unemp_rate > 0 else 0
    down_pressure += 1 if low_n >= high_n + 2 else 0
    down_pressure += 1 if activity_yoy_low >= 3 else 0

    vulnerability = 0
    vulnerability += 1 if pd.notna(youth) and youth < 0 else 0
    vulnerability += 1 if pd.notna(func_occ) and func_occ < 0 else 0
    vulnerability += 1 if low_n >= 4 else 0
    vulnerability += 1 if pd.notna(pop15) and pop15 < 0 else 0

    restructure = 0
    restructure += 1 if pd.notna(manuf) and pd.notna(service) and manuf < 0 < service else 0
    restructure += 1 if pd.notna(func_occ) and pd.notna(svc_occ) and func_occ < 0 < svc_occ else 0
    restructure += 1 if high_n >= 1 and low_n >= 1 else 0

    good = 0
    good += 1 if pd.notna(emp) and emp > 0 else 0
    good += 1 if high_n >= 4 else 0
    good += 1 if pd.notna(emp_rate) and emp_rate >= 0 else 0
    good += 1 if pd.notna(unemp_rate) and unemp_rate <= 0 else 0

    slowdown = 0
    slowdown += 1 if pd.notna(emp) and emp > 0 else 0
    slowdown += 1 if activity_yoy_low >= 2 else 0
    slowdown += 1 if pd.notna(emp_rate) and emp_rate < 0 else 0
    slowdown += 1 if pd.notna(pop15) and pop15 < 0 else 0

    if down_pressure >= 3:
        return "하방 압력"
    if vulnerability >= 3:
        return "구조 취약"
    if restructure >= 2 and pd.notna(emp) and emp > 0:
        return "산업·직종 재편"
    if good >= 3 and slowdown >= 2:
        return "호조-감속"
    if good >= 3:
        return "전반 호조"
    if slowdown >= 2:
        return "호조-감속"
    if restructure >= 2:
        return "산업·직종 재편"
    if vulnerability >= 2:
        return "구조 취약"
    return "하방 압력" if (pd.notna(emp) and emp < 0) else "호조-감속"


def _reason_text(feature: pd.Series, label: str) -> str:
    parts: List[str] = []
    if pd.notna(feature.get("취업자증감")):
        parts.append("취업자 증가" if float(feature["취업자증감"]) > 0 else "취업자 감소")
    if pd.notna(feature.get("고용률증감")) and pd.notna(feature.get("실업률증감")):
        parts.append(
            f"고용률 {'상승' if float(feature['고용률증감']) >= 0 else '하락'} / 실업률 {'하락' if float(feature['실업률증감']) <= 0 else '상승'}"
        )
    if pd.notna(feature.get("제조계열증감")) and pd.notna(feature.get("서비스계열증감")):
        if float(feature["제조계열증감"]) < 0 < float(feature["서비스계열증감"]):
            parts.append("제조 약세·서비스 강세")
    if pd.notna(feature.get("기능계열증감")) and pd.notna(feature.get("서비스·사무계열증감")):
        if float(feature["기능계열증감"]) < 0 < float(feature["서비스·사무계열증감"]):
            parts.append("기능직 약세·서비스/사무 강세")
    if pd.notna(feature.get("청년증감")) and pd.notna(feature.get("중장년증감")):
        if float(feature["청년증감"]) < 0 < float(feature["중장년증감"]):
            parts.append("청년 약세·중장년 강세")
    if int(feature.get("최저NEW", 0)) > int(feature.get("최고NEW", 0)):
        parts.append("최저 NEW 우세")

    if not parts:
        return f"{label} 신호"
    return ", ".join(parts[:3])


def _build_feature_table(work: pd.DataFrame, latest_period: pd.Timestamp, prd_se: str) -> pd.DataFrame:
    regions = [r for r in GYEONGGI_SIGUNGU if r in work["region_name"].astype(str).unique().tolist()]
    latest_events = _latest_event_enriched(work, latest_period, prd_se)

    rows: List[Dict[str, Any]] = []
    for region in regions:
        activity = _latest_slice(work, region, "activity", latest_period)
        age = _latest_slice(work, region, "age", latest_period)
        industry = _latest_slice(work, region, "industry", latest_period)
        occupation = _latest_slice(work, region, "occupation", latest_period)

        activity_sig = _pick_activity_signal(activity)
        youth, senior = _age_youth_senior_signal(age)

        manufact = _sum_yoy_by_keywords(industry, ["광공업", "제조업", "광·제조", "광제조", "제조"])
        service = _sum_yoy_by_keywords(industry, ["도소매", "숙박", "사업·개인·공공서비스", "전기·운수·통신·금융", "사회간접자본"])
        occ_func = _sum_yoy_by_keywords(occupation, ["기능·기계조작·조립", "기능기계조작조립", "기능원", "장치기계조작", "단순노무"])
        occ_service = _sum_yoy_by_keywords(occupation, ["서비스·판매", "서비스판매", "서비스 종사자", "판매 종사자", "사무 종사자"])

        ind_pos, ind_neg = _top_pos_neg_label(industry)
        occ_pos, occ_neg = _top_pos_neg_label(occupation)

        evt_region = latest_events[latest_events["지역"].astype(str) == str(region)].copy() if not latest_events.empty else pd.DataFrame()
        high_n = int((evt_region["유형"].astype(str) == "최고").sum()) if not evt_region.empty else 0
        low_n = int((evt_region["유형"].astype(str) == "최저").sum()) if not evt_region.empty else 0
        activity_yoy_low = (
            _event_count(evt_region, "activity", "YoY(절대)", "최저")
            + _event_count(evt_region, "activity", "YoY(증감률)", "최저")
        )

        row: Dict[str, Any] = {
            "시군": region,
            "취업자증감": activity_sig["취업자증감"],
            "고용률증감": activity_sig["고용률증감"],
            "실업률증감": activity_sig["실업률증감"],
            "15세이상인구증감": activity_sig["15세이상인구증감"],
            "청년증감": youth,
            "중장년증감": senior,
            "제조계열증감": manufact,
            "서비스계열증감": service,
            "기능계열증감": occ_func,
            "서비스·사무계열증감": occ_service,
            "최고NEW": high_n,
            "최저NEW": low_n,
            "활동지표_YoY최저NEW": int(activity_yoy_low),
            "산업 증가요인": ind_pos,
            "산업 감소요인": ind_neg,
            "직종 증가요인": occ_pos,
            "직종 감소요인": occ_neg,
        }
        row["유형"] = _classify(pd.Series(row))
        row["핵심판단"] = _reason_text(pd.Series(row), str(row["유형"]))
        rows.append(row)

    out = pd.DataFrame(rows)
    if out.empty:
        return out
    out["유형정렬"] = out["유형"].map({name: i for i, name in enumerate(TYPE_ORDER)}).fillna(999)
    out = out.sort_values(["유형정렬", "시군"]).drop(columns=["유형정렬"])
    return out


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

    feature = _build_feature_table(work, latest_period, prd_se)
    if feature.empty:
        st.info("유형화할 결과가 없습니다.")
        return

    st.markdown("##### 유형 정의(규칙 기반)")
    rule_df = pd.DataFrame([{"유형": k, "정의": TYPE_RULES[k]} for k in TYPE_ORDER])
    st.dataframe(rule_df, use_container_width=True, hide_index=True)

    st.markdown("##### 시군별 유형 결과")
    view = feature.copy()
    for c in [
        "취업자증감",
        "고용률증감",
        "실업률증감",
        "15세이상인구증감",
        "청년증감",
        "중장년증감",
    ]:
        view[c] = view[c].map(lambda v: fmt_num(v) if pd.notna(v) else "-")

    st.dataframe(
        view[
            [
                "시군",
                "유형",
                "핵심판단",
                "취업자증감",
                "고용률증감",
                "실업률증감",
                "청년증감",
                "중장년증감",
                "산업 증가요인",
                "산업 감소요인",
                "직종 증가요인",
                "직종 감소요인",
                "최고NEW",
                "최저NEW",
            ]
        ],
        use_container_width=True,
        hide_index=True,
    )

    with st.expander("상세 신호 보기", expanded=False):
        detailed = feature.copy()
        for c in [
            "취업자증감",
            "고용률증감",
            "실업률증감",
            "15세이상인구증감",
            "청년증감",
            "중장년증감",
            "제조계열증감",
            "서비스계열증감",
            "기능계열증감",
            "서비스·사무계열증감",
        ]:
            detailed[c] = detailed[c].map(lambda v: fmt_num(v) if pd.notna(v) else "-")
        st.dataframe(detailed, use_container_width=True, hide_index=True)

    st.markdown("##### 유형별 정책 매칭표")
    policy_df = pd.DataFrame(
        [
            {
                "유형": t,
                "정책방향": p["정책방향"],
                "핵심대상": p["핵심대상"],
                "권장과제": p["권장과제"],
                "점검지표": p["점검지표"],
            }
            for t, p in POLICY_MATCH.items()
        ]
    )
    policy_df["유형정렬"] = policy_df["유형"].map({name: i for i, name in enumerate(TYPE_ORDER)}).fillna(999)
    policy_df = policy_df.sort_values(["유형정렬", "유형"]).drop(columns=["유형정렬"])
    st.dataframe(policy_df, use_container_width=True, hide_index=True)
