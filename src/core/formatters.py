from typing import Dict, List

import pandas as pd


def fmt_period(value: object, prd_se: str = "M") -> str:
    ts = pd.Timestamp(value)
    if pd.isna(ts):
        return "-"
    if str(prd_se).upper() == "H":
        month = int(ts.month)
        if month <= 6:
            return f"{ts.year}-상반기"
        return f"{ts.year}-하반기"
    return ts.strftime("%Y-%m")


def time_labels(prd_se_values: List[str]) -> Dict[str, str]:
    is_halfyear = bool(prd_se_values) and all(str(v).upper() == "H" for v in prd_se_values)
    if is_halfyear:
        return {"point": "반기", "trend": "반기별", "yoy": "전년동기"}
    return {"point": "월", "trend": "월별", "yoy": "전년동월"}


def fmt_num(value: object, unit: str = "", digits: int = 1) -> str:
    if value is None or pd.isna(value):
        return "-"
    if unit and "%" in unit:
        digits = 2
    return f"{float(value):,.{digits}f}{unit}"


def fmt_num_bold(value: object, unit: str = "", digits: int = 1) -> str:
    return f"<strong>{fmt_num(value, unit, digits)}</strong>"


def new_badge(flag: bool) -> str:
    return "<span class='new-badge'>NEW</span>" if flag else ""


def remark_new(is_new_max: bool, is_new_min: bool) -> str:
    if is_new_max and is_new_min:
        return "최고·최저 NEW"
    if is_new_max:
        return "최고 NEW"
    if is_new_min:
        return "최저 NEW"
    return ""


def auto_y_domain(values: pd.Series, pad_ratio: float = 0.08) -> List[float] | None:
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


def fmt_triangle_delta(v: object, unit: str, fmt_num_func=fmt_num) -> str:
    if v is None or pd.isna(v):
        return "-"
    vv = float(v)
    if vv > 0:
        return f"▲{fmt_num_func(vv, unit)}"
    if vv < 0:
        return f"▼{fmt_num_func(abs(vv), unit)}"
    return f"→{fmt_num_func(0, unit)}"


def escape_markdown_text(text: object) -> str:
    s = str(text)
    for ch in ["\\", "`", "*", "_", "{", "}", "[", "]", "(", ")", "#", "+", "-", "!", "|", "~"]:
        s = s.replace(ch, f"\\{ch}")
    return s
