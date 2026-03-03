from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.config import DATASETS, default_end_period
from src.kosis_client import KosisClient
from src.transform import add_yoy, normalize_records


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="KOSIS API 수집 스크립트")
    parser.add_argument(
        "--api-key",
        default=os.getenv("api_key", ""),
        help="KOSIS API Key",
    )
    parser.add_argument("--end-prd-de", default=default_end_period(), help="종료월 YYYYMM")
    parser.add_argument("--outdir", default="data", help="저장 폴더")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if not args.api_key:
        raise SystemExit("API Key가 없습니다. --api-key 또는 api_key 환경변수를 설정하세요.")

    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)

    client = KosisClient(api_key=args.api_key)
    frames: list[pd.DataFrame] = []
    for cfg in DATASETS:
        records = client.fetch(cfg, end_prd_de=args.end_prd_de)
        raw_path = outdir / f"raw_{cfg.key}.json"
        pd.DataFrame(records).to_json(raw_path, orient="records", force_ascii=False, indent=2)

        normalized = normalize_records(cfg, records)
        normalized_path = outdir / f"normalized_{cfg.key}.csv"
        normalized.to_csv(normalized_path, index=False, encoding="utf-8-sig")
        frames.append(normalized)
        print(f"[OK] {cfg.title}: {len(normalized):,} rows")

    if frames:
        combined = add_yoy(pd.concat(frames, ignore_index=True))
        combined.to_csv(outdir / "normalized_all_with_yoy.csv", index=False, encoding="utf-8-sig")
        print(f"[DONE] total rows: {len(combined):,}")


if __name__ == "__main__":
    main()
