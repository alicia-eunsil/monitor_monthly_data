from __future__ import annotations

import pandas as pd


def current_streak_length(values: pd.Series, positive: bool) -> int:
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
