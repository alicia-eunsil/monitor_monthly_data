from __future__ import annotations

import re
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import streamlit as st

from src.core.formatters import fmt_num, fmt_period


def _to_numeric(series: pd.Series) -> pd.Series:
    return pd.to_numeric(series, errors="coerce")


def _default_config() -> Dict[str, Any]:
    return {
        "top_n_regions": 5,
        "risk_w_yoy": 0.55,
        "risk_w_slope": 0.25,
        "risk_w_streak": 0.20,
        "sector_quantile": 0.15,
        "sector_min_drop_pct": -8.0,
    }


def _infer_lag(prd_se: str) -> int:
    return 2 if str(prd_se).upper() == "H" else 12


def _pick_primary_indicator(df: pd.DataFrame) -> str:
    indicators = sorted(df["indicator_name"].dropna().astype(str).unique().tolist())
    if not indicators:
        return ""
    preferred = ["취업자", "취업", "고용률", "실업률", "경제활동"]
    for token in preferred:
        for name in indicators:
            if token in name:
                return name
    # Fallback: pick indicator with broadest coverage.
    counts = (
        df.groupby("indicator_name", dropna=False)["value"]
        .count()
        .sort_values(ascending=False)
        .to_dict()
    )
    return str(max(indicators, key=lambda x: counts.get(x, 0)))


def _pct_worse(series: pd.Series, lower_is_worse: bool) -> pd.Series:
    s = _to_numeric(series)
    if s.dropna().empty:
        return pd.Series(np.nan, index=s.index)
    if lower_is_worse:
        return s.rank(pct=True, ascending=False) * 100.0
    return s.rank(pct=True, ascending=True) * 100.0


def _latest_slope(values: pd.Series, window: int = 4) -> float:
    valid = _to_numeric(values).dropna()
    if len(valid) < window:
        return np.nan
    tail = valid.tail(window)
    return float((tail.iloc[-1] - tail.iloc[0]) / (window - 1))


def _negative_streak(values: pd.Series, max_len: int = 3) -> int:
    valid = _to_numeric(values).dropna()
    if valid.empty:
        return 0
    streak = 0
    for v in reversed(valid.tolist()):
        if v < 0:
            streak += 1
            if streak >= max_len:
                return max_len
        else:
            break
    return streak


def _activity_panel(df: pd.DataFrame) -> tuple[pd.DataFrame, Dict[str, Any]]:
    info: Dict[str, Any] = {"ok": False, "message": "", "prd_se": "M", "indicator": "", "latest_period": pd.NaT}
    base = df[df["dataset_key"] == "activity"].copy()
    if base.empty:
        info["message"] = "경제활동인구현황 데이터가 없습니다."
        return pd.DataFrame(), info

    indicator = _pick_primary_indicator(base)
    if indicator:
        base = base[base["indicator_name"] == indicator].copy()
    if base.empty:
        info["message"] = "활동지표에서 분석 가능한 시계열이 없습니다."
        return pd.DataFrame(), info

    prd_se = str(base["prd_se"].dropna().iloc[0]).upper() if "prd_se" in base.columns and not base["prd_se"].dropna().empty else "M"
    lag = _infer_lag(prd_se)
    base = (
        base.groupby(["region_name", "period", "prd_se"], as_index=False, dropna=False)
        .agg({"value": "sum"})
        .sort_values(["region_name", "period"])
    )
    base["value"] = _to_numeric(base["value"])
    base["yoy_abs"] = base.groupby("region_name", dropna=False)["value"].diff(lag)
    prev = base.groupby("region_name", dropna=False)["value"].shift(lag)
    base["yoy_pct"] = np.where(prev == 0, np.nan, (base["value"] / prev - 1.0) * 100.0)

    if base["period"].dropna().empty:
        info["message"] = "시점 정보가 없어 질문 생성이 불가합니다."
        return pd.DataFrame(), info

    info.update(
        {
            "ok": True,
            "prd_se": prd_se,
            "lag": lag,
            "indicator": indicator,
            "latest_period": pd.Timestamp(base["period"].max()),
        }
    )
    return base, info


def _format_top_region_list(df: pd.DataFrame, col: str, unit: str = "") -> str:
    if df.empty:
        return "-"
    items: List[str] = []
    for _, row in df.iterrows():
        region = str(row.get("region_name", ""))
        score_text = fmt_num(row.get(col), unit)
        items.append(f"{region}({score_text})")
    return ", ".join(items)


def _question_fall_risk_top5(df: pd.DataFrame, cfg: Dict[str, Any]) -> Optional[Dict[str, Any]]:
    panel, info = _activity_panel(df)
    if not info.get("ok"):
        return None

    latest_period = pd.Timestamp(info["latest_period"])
    latest = panel[panel["period"] == latest_period].copy()
    if latest.empty:
        return None

    slopes = (
        panel.groupby("region_name", dropna=False)["value"]
        .apply(_latest_slope)
        .rename("slope_recent")
        .reset_index()
    )
    streaks = (
        panel.groupby("region_name", dropna=False)["yoy_abs"]
        .apply(_negative_streak)
        .rename("neg_streak")
        .reset_index()
    )

    latest = latest.merge(slopes, on="region_name", how="left").merge(streaks, on="region_name", how="left")
    latest["score_yoy"] = _pct_worse(latest["yoy_pct"], lower_is_worse=True)
    latest["score_slope"] = _pct_worse(latest["slope_recent"], lower_is_worse=True)
    latest["score_streak"] = (_to_numeric(latest["neg_streak"]).fillna(0).clip(lower=0, upper=3) / 3.0) * 100.0
    w_yoy = float(cfg.get("risk_w_yoy", 0.55))
    w_slope = float(cfg.get("risk_w_slope", 0.25))
    w_streak = float(cfg.get("risk_w_streak", 0.20))
    w_sum = w_yoy + w_slope + w_streak
    if w_sum <= 0:
        w_yoy, w_slope, w_streak, w_sum = 0.55, 0.25, 0.20, 1.0
    latest["risk_score"] = (
        latest["score_yoy"].fillna(0) * (w_yoy / w_sum)
        + latest["score_slope"].fillna(0) * (w_slope / w_sum)
        + latest["score_streak"].fillna(0) * (w_streak / w_sum)
    )

    top_n = int(cfg.get("top_n_regions", 5))
    top = latest.sort_values("risk_score", ascending=False).head(top_n).copy()
    if top.empty:
        return None

    yoy_text = _format_top_region_list(top[["region_name", "yoy_pct"]], "yoy_pct", "%")
    risk_text = _format_top_region_list(top[["region_name", "risk_score"]], "risk_score", "")
    prd_se = str(info.get("prd_se", "M"))
    point = fmt_period(latest_period, prd_se)
    score = float(_to_numeric(top["risk_score"]).mean())
    confidence = float(min(100.0, latest["yoy_pct"].notna().mean() * 100.0))

    return {
        "question_type": f"하락위험 Top{top_n}",
        "question": f"다음 달(또는 다음 반기) 하락 위험이 큰 지역 Top{top_n}는 어디인가?",
        "answer": f"위험점수 상위는 {risk_text} 입니다. 최신 YoY 기준으로는 {yoy_text} 순으로 약세입니다.",
        "evidence": (
            f"기준시점 {point}, 지표 {info.get('indicator','-')}, "
            f"위험점수=YoY({w_yoy/w_sum:.0%})+최근기울기({w_slope/w_sum:.0%})+연속악화({w_streak/w_sum:.0%})"
        ),
        "priority_score": round(score, 1),
        "confidence": round(confidence, 1),
        "action_hint": f"상위 {top_n}개 지역의 세부 카테고리(연령/산업/직종) 점검",
        "latest_period": point,
    }


def _question_gap_vs_average(df: pd.DataFrame, cfg: Dict[str, Any]) -> Optional[Dict[str, Any]]:
    panel, info = _activity_panel(df)
    if not info.get("ok"):
        return None

    latest_period = pd.Timestamp(info["latest_period"])
    latest = panel[panel["period"] == latest_period].copy()
    if latest.empty or latest["yoy_pct"].dropna().empty:
        return None

    latest["yoy_pct"] = _to_numeric(latest["yoy_pct"])
    scope_avg = float(latest["yoy_pct"].mean(skipna=True))
    latest["gap_pp"] = latest["yoy_pct"] - scope_avg
    top_n = int(cfg.get("top_n_regions", 5))
    worst = latest.nsmallest(top_n, "gap_pp")
    if worst.empty:
        return None

    gap_text = _format_top_region_list(worst[["region_name", "gap_pp"]], "gap_pp", "%p")
    score = float(np.clip(_to_numeric(worst["gap_pp"]).abs().mean() * 4.0 + 35.0, 0, 100))
    confidence = float(min(100.0, latest["yoy_pct"].notna().mean() * 100.0))
    point = fmt_period(latest_period, str(info.get("prd_se", "M")))

    return {
        "question_type": "평균 대비 악화",
        "question": "평균 대비 악화 속도가 더 빠른 지역은 어디인가?",
        "answer": f"평균 YoY({fmt_num(scope_avg, '%')}) 대비 하방 격차가 큰 지역은 {gap_text} 입니다.",
        "evidence": f"기준시점 {point}, 지표 {info.get('indicator','-')}, gap=지역 YoY - 평균 YoY",
        "priority_score": round(score, 1),
        "confidence": round(confidence, 1),
        "action_hint": "평균 대비 격차가 큰 지역부터 원인분석 우선",
        "latest_period": point,
    }


def _is_youth_category(name: object) -> bool:
    text = str(name or "").replace(" ", "")
    if not text:
        return False
    if "청년" in text or "20대" in text:
        return True
    if re.search(r"15\D*29", text):
        return True
    if re.search(r"15\D*34", text):
        return True
    return False


def _question_youth_two_period_decline(df: pd.DataFrame, cfg: Dict[str, Any]) -> Optional[Dict[str, Any]]:
    base = df[df["dataset_key"] == "age"].copy()
    if base.empty:
        return None

    base = base[base["category_name"].map(_is_youth_category)].copy()
    if base.empty:
        return None

    indicator = _pick_primary_indicator(base)
    if indicator:
        base = base[base["indicator_name"] == indicator].copy()
    if base.empty:
        return None

    prd_se = str(base["prd_se"].dropna().iloc[0]).upper() if "prd_se" in base.columns and not base["prd_se"].dropna().empty else "M"
    lag = _infer_lag(prd_se)

    agg = (
        base.groupby(["region_name", "period"], as_index=False, dropna=False)
        .agg({"value": "sum"})
        .sort_values(["region_name", "period"])
    )
    agg["value"] = _to_numeric(agg["value"])
    agg["yoy_abs"] = agg.groupby("region_name", dropna=False)["value"].diff(lag)

    latest_period = agg["period"].max()
    if pd.isna(latest_period):
        return None
    latest_period = pd.Timestamp(latest_period)
    latest = agg[agg["period"] == latest_period][["region_name", "yoy_abs"]].rename(columns={"yoy_abs": "yoy_abs_latest"})
    prev_rows = (
        agg[agg["period"] < latest_period]
        .sort_values("period")
        .groupby("region_name", as_index=False, dropna=False)
        .tail(1)[["region_name", "yoy_abs"]]
        .rename(columns={"yoy_abs": "yoy_abs_prev"})
    )
    merged = latest.merge(prev_rows, on="region_name", how="left")
    merged["yoy_abs_latest"] = _to_numeric(merged["yoy_abs_latest"])
    merged["yoy_abs_prev"] = _to_numeric(merged["yoy_abs_prev"])

    focus = merged[(merged["yoy_abs_latest"] < 0) & (merged["yoy_abs_prev"] < 0)].copy()
    point = fmt_period(latest_period, prd_se)
    if focus.empty:
        return {
            "question_type": "청년층 연속 악화",
            "question": "청년층 고용이 2기 연속 악화된 지역은 어디인가?",
            "answer": "현재 기준으로 2기 연속 악화 지역은 확인되지 않았습니다.",
            "evidence": f"기준시점 {point}, 청년 카테고리 합산(15~29/20대/청년 패턴) YoY 절대증감 확인",
            "priority_score": 35.0,
            "confidence": 70.0,
            "action_hint": "동일 조건으로 다음 시점 연속성 재확인",
            "latest_period": point,
        }

    focus["severity"] = focus["yoy_abs_latest"].abs() + focus["yoy_abs_prev"].abs()
    top_n = int(cfg.get("top_n_regions", 5))
    top = focus.sort_values("severity", ascending=False).head(top_n)
    top_text = _format_top_region_list(top[["region_name", "severity"]], "severity")
    score = float(np.clip(_to_numeric(top["severity"]).mean() / 15.0 + 45.0, 0, 100))
    confidence = float(min(100.0, merged["yoy_abs_latest"].notna().mean() * 100.0))

    return {
        "question_type": "청년층 연속 악화",
        "question": "청년층 고용이 2기 연속 악화된 지역은 어디인가?",
        "answer": f"연속 악화가 확인된 지역 상위는 {top_text} 입니다.",
        "evidence": f"기준시점 {point}, 최신/직전 YoY 절대증감이 모두 음수인 지역 선별",
        "priority_score": round(score, 1),
        "confidence": round(confidence, 1),
        "action_hint": "청년층 대상 산업/직종 세부탭과 교차 확인",
        "latest_period": point,
    }


def _question_sector_shock(df: pd.DataFrame, cfg: Dict[str, Any]) -> Optional[Dict[str, Any]]:
    base = df[df["dataset_key"].isin(["industry", "occupation"])].copy()
    if base.empty:
        return None

    latest_period = base["period"].max()
    if pd.isna(latest_period):
        return None
    latest_period = pd.Timestamp(latest_period)

    base["yoy_pct"] = _to_numeric(base["yoy_pct"])
    latest = base[base["period"] == latest_period].dropna(subset=["yoy_pct"]).copy()
    if latest.empty:
        return None

    q = float(latest["yoy_pct"].quantile(float(cfg.get("sector_quantile", 0.15))))
    threshold = min(float(cfg.get("sector_min_drop_pct", -8.0)), q)
    shock = latest[latest["yoy_pct"] <= threshold].copy()
    if shock.empty:
        shock = latest.nsmallest(int(cfg.get("top_n_regions", 5)), "yoy_pct").copy()
    shock = shock.sort_values("yoy_pct").head(max(6, int(cfg.get("top_n_regions", 5))))

    lines = []
    for _, row in shock.iterrows():
        region = str(row.get("region_name", ""))
        dkey = str(row.get("dataset_key", ""))
        category = str(row.get("category_name", "")).strip() or "전체"
        lines.append(f"{region}/{dkey}/{category}({fmt_num(row.get('yoy_pct'), '%')})")
    answer_text = ", ".join(lines) if lines else "-"

    prd_se = str(base["prd_se"].dropna().iloc[0]).upper() if "prd_se" in base.columns and not base["prd_se"].dropna().empty else "M"
    point = fmt_period(latest_period, prd_se)
    score = float(np.clip(_to_numeric(shock["yoy_pct"]).abs().mean() * 3.5, 0, 100))
    confidence = float(min(100.0, latest["yoy_pct"].notna().mean() * 100.0))

    return {
        "question_type": "산업/직종 급락",
        "question": "산업/직종별 급락이 발생한 지역은 어디인가?",
        "answer": f"급락 신호 상위는 {answer_text} 입니다.",
        "evidence": (
            f"기준시점 {point}, 산업/직종 최신 YoY 하위 {int(float(cfg.get('sector_quantile', 0.15))*100)}% "
            f"(최소 {fmt_num(float(cfg.get('sector_min_drop_pct', -8.0)), '%')} 이하) 중심 추출"
        ),
        "priority_score": round(score, 1),
        "confidence": round(confidence, 1),
        "action_hint": "급락 카테고리의 직전 6개 시점 추세 점검",
        "latest_period": point,
    }


def discover_question_candidates(df: pd.DataFrame, cfg: Optional[Dict[str, Any]] = None) -> pd.DataFrame:
    required = {"dataset_key", "region_name", "period", "value", "indicator_name", "category_name", "prd_se"}
    if df is None or df.empty:
        return pd.DataFrame()
    if not required.issubset(set(df.columns)):
        return pd.DataFrame()
    config = _default_config()
    if cfg:
        config.update(cfg)

    builders = [
        _question_fall_risk_top5,
        _question_gap_vs_average,
        _question_youth_two_period_decline,
        _question_sector_shock,
    ]
    rows: List[Dict[str, Any]] = []
    for fn in builders:
        row = fn(df, config)
        if row:
            rows.append(row)
    if not rows:
        return pd.DataFrame()
    out = pd.DataFrame(rows)
    out["priority_score"] = _to_numeric(out["priority_score"]).fillna(0)
    out["confidence"] = _to_numeric(out["confidence"]).fillna(0)
    out = out.sort_values(["priority_score", "confidence"], ascending=[False, False]).reset_index(drop=True)
    out.insert(0, "rank", np.arange(1, len(out) + 1))
    return out


def render_auto_discovery_tab(df: pd.DataFrame, labels: Dict[str, str]) -> None:
    st.subheader("자동 질문 발견")
    st.caption(
        "데이터를 자동 스캔해 질문과 답변 후보를 생성합니다. "
        "탐색 단계에서 무엇을 먼저 볼지 정하는 데 초점을 둔 화면입니다."
    )

    with st.expander("탐색 설정", expanded=False):
        top_n_regions = st.slider("지역 Top N", min_value=3, max_value=10, value=5, step=1)
        st.markdown("**하락위험 점수 가중치**")
        risk_w_yoy = st.slider("YoY 가중치", min_value=0.1, max_value=0.8, value=0.55, step=0.05)
        risk_w_slope = st.slider("최근기울기 가중치", min_value=0.1, max_value=0.6, value=0.25, step=0.05)
        risk_w_streak = st.slider("연속악화 가중치", min_value=0.1, max_value=0.6, value=0.20, step=0.05)
        st.markdown("**산업/직종 급락 조건**")
        sector_quantile = st.slider("하위 분위수(%)", min_value=5, max_value=30, value=15, step=5)
        sector_min_drop_pct = st.slider("최소 급락 임계값(%)", min_value=-20.0, max_value=-3.0, value=-8.0, step=1.0)

    cfg = {
        "top_n_regions": top_n_regions,
        "risk_w_yoy": risk_w_yoy,
        "risk_w_slope": risk_w_slope,
        "risk_w_streak": risk_w_streak,
        "sector_quantile": float(sector_quantile) / 100.0,
        "sector_min_drop_pct": sector_min_drop_pct,
    }
    candidates = discover_question_candidates(df, cfg=cfg)
    if candidates.empty:
        st.info("자동 생성 가능한 질문이 없습니다.")
        return

    top_n = st.slider("표시할 질문 수", min_value=1, max_value=10, value=min(6, len(candidates)), step=1)
    view = candidates.head(top_n).copy()

    c1, c2, c3 = st.columns(3)
    with c1:
        st.metric("생성 질문 수", len(candidates))
    with c2:
        st.metric("최고 우선순위", f"{view['priority_score'].max():.1f}")
    with c3:
        st.metric("분석 기준", labels.get("point", "시점"))

    table_cols = ["rank", "question_type", "priority_score", "confidence", "question"]
    st.dataframe(
        view[table_cols].rename(
            columns={
                "rank": "순위",
                "question_type": "질문 유형",
                "priority_score": "우선순위",
                "confidence": "신뢰도",
                "question": "질문",
            }
        ),
        use_container_width=True,
        hide_index=True,
    )

    st.markdown("#### 자동 생성 결과")
    for _, row in view.iterrows():
        title = f"[{int(row['rank'])}] {row['question']}"
        with st.expander(title, expanded=int(row["rank"]) == 1):
            st.markdown(f"**답변**: {row['answer']}")
            st.markdown(f"**근거**: {row['evidence']}")
            st.markdown(f"**권장 액션**: {row['action_hint']}")
            st.caption(f"우선순위 {row['priority_score']:.1f} / 신뢰도 {row['confidence']:.1f}")
