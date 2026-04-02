from typing import Any, Dict, List, Optional

import pandas as pd

import src.config as app_config
from src.core.category_rules import apply_industry_category_filter
from src.kosis_client import KosisClient
from src.transform import add_yoy, normalize_records

DatasetConfig = getattr(app_config, "DatasetConfig", Any)
datasets_for_scope = getattr(
    app_config,
    "datasets_for_scope",
    lambda _scope: getattr(app_config, "DATASETS", []),
)
default_end_period_by_prd_se = getattr(
    app_config,
    "default_end_period_by_prd_se",
    lambda _prd_se: app_config.default_end_period(),
)


def fetch_records_live(
    api_key: str,
    dataset_key: str,
    end_period: str,
    config_signature: str,
    datasets: List[DatasetConfig],
) -> Dict[str, Any]:
    cfg = next((x for x in datasets if x.key == dataset_key), None)
    if cfg is None:
        raise RuntimeError(f"Unknown dataset key: {dataset_key}")
    client = KosisClient(api_key=api_key)
    records, debug_logs = client.fetch_with_debug(cfg, end_prd_de=end_period)
    return {"records": records, "debug_logs": debug_logs}


def load_all_data_with_progress(
    api_key: str,
    status_box: Any,
    progress_box: Any,
    main_status_box: Optional[Any] = None,
    main_progress_box: Optional[Any] = None,
) -> tuple[Dict[str, pd.DataFrame], List[str], List[str], List[str]]:
    scope_defs = [
        ("province", "전국·17개 시도", datasets_for_scope("province")),
        ("gyeonggi31", "경기 31개 시군", datasets_for_scope("gyeonggi31")),
    ]
    total_steps = sum(len(ds) * 2 for _, _, ds in scope_defs) + len(scope_defs)
    step = 0
    frames_by_scope: Dict[str, List[pd.DataFrame]] = {k: [] for k, _, _ in scope_defs}
    errors: List[str] = []
    debug_logs: List[str] = []
    empty_data_warnings: List[str] = []

    progress = progress_box.progress(0)
    main_progress = main_progress_box.progress(0) if main_progress_box is not None else None
    status = status_box
    main_status = main_status_box
    last_progress = -1

    def _set_progress(value: int) -> None:
        nonlocal last_progress
        safe_value = max(int(value), last_progress)
        last_progress = safe_value
        progress.progress(safe_value)
        if main_progress is not None:
            main_progress.progress(safe_value)

    def _set_info(message: str) -> None:
        status.info(message)
        if main_status is not None:
            main_status.info(message)

    def _set_error(message: str) -> None:
        status.error(message)

    def _set_success(message: str) -> None:
        status.success(message)

    for scope_key, scope_title, datasets in scope_defs:
        _set_info(f"{scope_title} 데이터셋 준비 중... ({step}/{total_steps})")
        step += 1
        _set_progress(min(100, int(step * 100 / total_steps)))
        for cfg in datasets:
            _set_info(f"[{scope_title}] 데이터 불러오는 중: {cfg.title} ({step}/{total_steps})")
            try:
                end_period = default_end_period_by_prd_se(cfg.prd_se)
                config_signature = "|".join(
                    [
                        cfg.tbl_id,
                        cfg.itm_id,
                        cfg.obj_l1,
                        cfg.obj_l2,
                        cfg.output_fields,
                        cfg.start_prd_de,
                        cfg.prd_se,
                        end_period,
                        scope_key,
                    ]
                )
                result = fetch_records_live(
                    api_key=api_key,
                    dataset_key=cfg.key,
                    end_period=end_period,
                    config_signature=config_signature,
                    datasets=datasets,
                )
                records = result.get("records", [])
                for line in result.get("debug_logs", []):
                    debug_logs.append(f"[{scope_key}:{cfg.key}] {line}")
                if records:
                    sample = records[0]
                    debug_logs.append(f"[{scope_key}:{cfg.key}] sample_keys={list(sample.keys())[:12]}")
                    debug_logs.append(
                        f"[{scope_key}:{cfg.key}] sample_PRD_DE={sample.get('PRD_DE')} sample_DT={sample.get('DT')}"
                    )
            except Exception as exc:  # noqa: BLE001
                records = []
                errors.append(f"{scope_title} - {cfg.title}: {exc}")
                debug_logs.append(f"[{scope_key}:{cfg.key}] ERROR: {exc}")
            step += 1
            _set_progress(min(100, int(step * 100 / total_steps)))

            _set_info(f"[{scope_title}] 파싱 중: {cfg.title} ({step}/{total_steps})")
            parsed = normalize_records(cfg, records, region_scope=scope_key)
            debug_logs.append(f"[{scope_key}:{cfg.key}] parsed_rows={len(parsed)} raw_rows={len(records)}")
            if len(records) == 0:
                empty_data_warnings.append(
                    f"{scope_title} - {cfg.title}: API 응답이 비어 있습니다 (end={end_period}, prd_se={cfg.prd_se})."
                )
            elif parsed.empty:
                empty_data_warnings.append(
                    f"{scope_title} - {cfg.title}: API 원본 {len(records)}건 수신했지만 파싱 후 0건입니다."
                )
            if not parsed.empty:
                frames_by_scope[scope_key].append(parsed)
            step += 1
            _set_progress(min(100, int(step * 100 / total_steps)))

    _set_info("지표 계산 및 통합 중...")
    data_by_scope: Dict[str, pd.DataFrame] = {}
    for scope_key, _, _ in scope_defs:
        frames = frames_by_scope.get(scope_key, [])
        if frames:
            combined = pd.concat(frames, ignore_index=True)
            combined = add_yoy(combined)
            combined, filter_stats = apply_industry_category_filter(combined)
            debug_logs.append(
                "[{}:industry_filter] before_rows={} after_rows={} removed_rows={}".format(
                    scope_key,
                    filter_stats["before_rows"],
                    filter_stats["after_rows"],
                    filter_stats["removed_rows"],
                )
            )
            data_by_scope[scope_key] = combined
            debug_logs.append(f"[{scope_key}:all] combined_rows={len(combined)}")
        else:
            data_by_scope[scope_key] = pd.DataFrame()

    if all(df.empty for df in data_by_scope.values()):
        _set_progress(100)
        _set_error("데이터 로딩 실패")
        return data_by_scope, errors, debug_logs, empty_data_warnings

    _set_progress(100)
    _set_success("로딩 완료")
    return data_by_scope, errors, debug_logs, empty_data_warnings
