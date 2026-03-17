import re
from typing import Dict, List

import pandas as pd


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


def norm_indicator_name(text: str) -> str:
    s = str(text).strip()
    for token in [" ", "~", "-", "–", "ㅡ"]:
        s = s.replace(token, "")
    return s


def order_activity_indicators(indicators: List[str]) -> List[str]:
    order_map = {norm_indicator_name(name): idx for idx, name in enumerate(ACTIVITY_INDICATOR_ORDER)}
    return sorted(indicators, key=lambda x: (order_map.get(norm_indicator_name(x), 999), x))


def norm_occupation_category(text: str) -> str:
    s = str(text).strip()
    s = re.sub(r"^\*+\s*", "", s)
    s = re.sub(r"^\d+\s*", "", s)
    s = re.sub(r"\([^)]*\)", "", s)
    s = re.sub(r"[^0-9A-Za-z가-힣]", "", s)
    return s


def order_occupation_categories(categories: List[str]) -> List[str]:
    order_map = {norm_occupation_category(name): idx for idx, name in enumerate(OCCUPATION_CATEGORY_ORDER)}
    return sorted(categories, key=lambda x: (order_map.get(norm_occupation_category(x), 999), x))


def norm_sigungu_industry(text: str) -> str:
    s = str(text).strip()
    s = re.sub(r"\s+", "", s)
    s = re.sub(r"[·ㆍ,/()\-]", "", s)
    return s


def order_sigungu_industry_categories(categories: List[str]) -> List[str]:
    def _rank(cat: str) -> int:
        n = norm_sigungu_industry(cat)
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


def _industry_code_token(text: str) -> str:
    s = str(text or "").strip().upper()
    if not s:
        return ""

    raw_text = str(text or "").strip()
    norm_text = re.sub(r"[\s\u00A0·ㆍ,./()\-]", "", raw_text)
    if norm_text in {"\uACC4", "\uD569\uACC4", "\uC804\uCCB4"} or s == "TOTAL":
        return "TOTAL"

    def _normalize_token(raw: str) -> str:
        token_raw = re.sub(r"[^A-Z~,\-]", "", str(raw or "").upper())
        if not token_raw or not re.search(r"[A-Z]", token_raw):
            return ""
        normalized = token_raw.replace(",", "").replace("-", "~")
        letters_only = re.sub(r"[^A-Z]", "", normalized)
        if letters_only == "ELU" or normalized == "EL~U":
            return "EL~U"
        if letters_only == "DU" or normalized == "D~U":
            return "D~U"
        if letters_only in {"BC", "GI", "DHJK", "A", "C", "F"}:
            return letters_only
        return letters_only

    # Prefer explicit code in parenthesis: e.g. (BC), (G,I), (D,H,J,K), (EL~U), (D-U)
    m = re.search(r"\(([^()]*)\)", s)
    if m:
        token = _normalize_token(m.group(1))
        if token:
            return token

    # Fallback to leading alphabet class code: e.g. "A 농업...", "C 제조업...", "F 건설업..."
    # Allow non-alphabet prefixes such as full-width star/bullet.
    m = re.match(r"^[^A-Z0-9]*([A-Z])", s)
    if m:
        return m.group(1)
    return ""


def order_province_industry_categories(categories: List[str]) -> List[str]:
    # Requested fixed UI order (province mode):
    # ? -> A ??... -> * ???(BC) -> C ???... -> * ?????? ? ??????(D~U)
    # -> F ???... -> * ??????????(GI) -> * ??????????? ? ??(EL~U)
    # -> * ???????????(DHJK)
    ordered_tokens = ["TOTAL", "A", "BC", "C", "D~U", "F", "GI", "EL~U", "DHJK"]
    order_map = {tok: idx for idx, tok in enumerate(ordered_tokens)}
    return sorted(categories, key=lambda x: (order_map.get(_industry_code_token(x), 999), str(x)))


def norm_age_category(text: str) -> str:
    s = str(text).strip()
    s = re.sub(r"\s+", "", s)
    s = s.replace("-", "~")
    s = s.replace("세이상", "이상")
    s = s.replace("세", "")
    return s


def order_age_categories(categories: List[str]) -> List[str]:
    order_map = {norm_age_category(name): idx for idx, name in enumerate(AGE_CATEGORY_ORDER)}
    return sorted(categories, key=lambda x: (order_map.get(norm_age_category(x), 999), x))


def order_sigungu_age_categories(categories: List[str]) -> List[str]:
    def _rank(cat: str) -> int:
        n = norm_age_category(cat)
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


def norm_status_category(text: str) -> str:
    s = str(text).strip()
    s = re.sub(r"^\*+\s*", "", s)
    s = re.sub(r"^-\s*", "", s)
    s = re.sub(r"\s+", "", s)
    return s


def order_status_categories(categories: List[str]) -> List[str]:
    exact_order_map = {str(name).strip(): idx for idx, name in enumerate(STATUS_CATEGORY_ORDER)}
    norm_order_map = {norm_status_category(name): idx for idx, name in enumerate(STATUS_CATEGORY_ORDER)}
    return sorted(
        categories,
        key=lambda x: (exact_order_map.get(str(x).strip(), norm_order_map.get(norm_status_category(x), 999)), x),
    )


def order_sigungu_status_categories(categories: List[str]) -> List[str]:
    def _rank(cat: str) -> int:
        n = norm_status_category(cat)
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


def order_sigungu_occupation_categories(categories: List[str]) -> List[str]:
    def _rank(cat: str) -> int:
        n = norm_occupation_category(cat)
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


def is_valid_industry_category(name: object, code: object = "") -> bool:
    s_name = str(name).strip()
    s_code = str(code).strip()
    if not s_name:
        return False
    if s_name.startswith("*"):
        return False
    return bool(re.search(r"[A-Za-z]", s_name) or re.search(r"[A-Za-z]", s_code))


def apply_industry_category_filter(df: pd.DataFrame) -> tuple[pd.DataFrame, Dict[str, int]]:
    # Keep all industry categories as-is.
    if df.empty or "dataset_key" not in df.columns:
        return df, {"before_rows": 0, "after_rows": 0, "removed_rows": 0}

    out = df.copy()
    industry_mask = out["dataset_key"].astype(str) == "industry"
    before_rows = int(industry_mask.sum())
    if before_rows == 0:
        return out, {"before_rows": 0, "after_rows": 0, "removed_rows": 0}
    after_rows = before_rows
    return out, {
        "before_rows": before_rows,
        "after_rows": after_rows,
        "removed_rows": 0,
    }


def order_categories_like_ui(categories: List[str], dataset_key: str, is_gyeonggi31_mode: bool) -> List[str]:
    ordered = sorted(c for c in categories if str(c).strip() != "")
    if dataset_key in {"industry", "occupation"}:
        drop_labels = {"시도별", "산업별", "산업명", "직업별", "직종별"}
        cleaned = [c for c in ordered if str(c).strip() not in drop_labels]
        if cleaned:
            ordered = cleaned
    if is_gyeonggi31_mode:
        if dataset_key == "industry":
            return order_sigungu_industry_categories(ordered)
        if dataset_key == "age":
            return order_sigungu_age_categories(ordered)
        if dataset_key == "status":
            return order_sigungu_status_categories(ordered)
        if dataset_key == "occupation":
            return order_sigungu_occupation_categories(ordered)
    else:
        if dataset_key == "industry":
            return order_province_industry_categories(ordered)
        if dataset_key == "age":
            return order_age_categories(ordered)
        if dataset_key == "status":
            return order_status_categories(ordered)
        if dataset_key == "occupation":
            return order_occupation_categories(ordered)
    return ordered
