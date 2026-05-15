from typing import Any, Dict, List, Optional

import pandas as pd

import src.config as app_config
from src.core.category_rules import apply_industry_category_filter
from src.kosis_client import KosisClient
from src.transform import add_yoy, normalize_records

REQUIRED_SCOPE_COLUMNS = {
    "dataset_key",
    "region_name",
    "indicator_name",
    "category_name",
    "period",
    "value",
    "prd_se",
}

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
    scopes: Optional[List[str]] = None,
) -> tuple[Dict[str, pd.DataFrame], List[str], List[str], List[str]]:
    all_scope_defs = [
        ("province", "?꾧뎅쨌17媛??쒕룄", datasets_for_scope("province")),
        ("gyeonggi31", "寃쎄린 31媛??쒓뎔", datasets_for_scope("gyeonggi31")),
    ]
    requested_scopes = set(scopes or ["province", "gyeonggi31"])
    scope_defs = [scope_def for scope_def in all_scope_defs if scope_def[0] in requested_scopes]
    if not scope_defs:
        return {}, [], [], []
    scope_title_map = {k: t for k, t, _ in scope_defs}
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
        _set_info(f"{scope_title} ?곗씠?곗뀑 以鍮?以?.. ({step}/{total_steps})")
        step += 1
        _set_progress(min(100, int(step * 100 / total_steps)))
        for cfg in datasets:
            _set_info(f"[{scope_title}] ?곗씠??遺덈윭?ㅻ뒗 以? {cfg.title} ({step}/{total_steps})")
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

            _set_info(f"[{scope_title}] ?뚯떛 以? {cfg.title} ({step}/{total_steps})")
            parsed = normalize_records(cfg, records, region_scope=scope_key)
            debug_logs.append(f"[{scope_key}:{cfg.key}] parsed_rows={len(parsed)} raw_rows={len(records)}")
            if len(records) == 0:
                empty_data_warnings.append(
                    f"{scope_title} - {cfg.title}: API ?묐떟??鍮꾩뼱 ?덉뒿?덈떎 (end={end_period}, prd_se={cfg.prd_se})."
                )
            elif parsed.empty:
                empty_data_warnings.append(
                    f"{scope_title} - {cfg.title}: API ?먮낯 {len(records)}嫄??섏떊?덉?留??뚯떛 ??0嫄댁엯?덈떎."
                )
            if not parsed.empty:
                frames_by_scope[scope_key].append(parsed)
            step += 1
            _set_progress(min(100, int(step * 100 / total_steps)))

    _set_info("吏??怨꾩궛 諛??듯빀 以?..")
    data_by_scope: Dict[str, pd.DataFrame] = {}
    for scope_key, _, _ in scope_defs:
        frames = frames_by_scope.get(scope_key, [])
        if frames:
            combined = pd.concat(frames, ignore_index=True)
            missing_required = sorted(REQUIRED_SCOPE_COLUMNS - set(combined.columns))
            debug_logs.append(f"[{scope_key}:all] combined_cols={list(combined.columns)}")
            if missing_required:
                errors.append(
                    f"{scope_title_map.get(scope_key, scope_key)} - ?듯빀 ?곗씠???ㅽ궎留??꾨씫: {', '.join(missing_required)}"
                )
                debug_logs.append(f"[{scope_key}:all] missing_required_cols={missing_required}")
                data_by_scope[scope_key] = pd.DataFrame()
                continue
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
        _set_error("?곗씠??濡쒕뵫 ?ㅽ뙣")
        return data_by_scope, errors, debug_logs, empty_data_warnings

    _set_progress(100)
    _set_success("濡쒕뵫 ?꾨즺")
    return data_by_scope, errors, debug_logs, empty_data_warnings

