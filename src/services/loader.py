from typing import Any, Dict, List, Optional
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime, timedelta, timezone
from pathlib import Path
import json
import os
import shutil

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

CACHE_ROOT = Path("data/cache")
MANIFEST_PATH = CACHE_ROOT / "manifest.json"


def _kosis_max_workers(dataset_count: int) -> int:
    raw = str(os.getenv("KOSIS_MAX_WORKERS", "1")).strip()
    try:
        value = int(raw)
    except ValueError:
        value = 1
    return max(1, min(value, dataset_count))


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
        ("province", "전국·17개 시도", datasets_for_scope("province")),
        ("gyeonggi31", "경기 31개 시군", datasets_for_scope("gyeonggi31")),
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

    class _NoopProgress:
        def progress(self, _value: int) -> None:
            return None

    class _NoopStatus:
        def info(self, _message: str) -> None:
            return None

        def error(self, _message: str) -> None:
            return None

        def success(self, _message: str) -> None:
            return None

    status = status_box if status_box is not None else _NoopStatus()
    main_status = main_status_box if main_status_box is not None else None
    progress = progress_box.progress(0) if progress_box is not None else _NoopProgress()
    main_progress = main_progress_box.progress(0) if main_progress_box is not None else None
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

        fetch_outputs: Dict[str, Dict[str, Any]] = {}
        max_workers = _kosis_max_workers(len(datasets))
        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            future_map = {}
            for cfg in datasets:
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
                future = executor.submit(
                    fetch_records_live,
                    api_key=api_key,
                    dataset_key=cfg.key,
                    end_period=end_period,
                    config_signature=config_signature,
                    datasets=datasets,
                )
                future_map[future] = (cfg, end_period)

            for future in as_completed(future_map):
                cfg, end_period = future_map[future]
                _set_info(f"[{scope_title}] 데이터 불러오는 중: {cfg.title} ({step}/{total_steps})")
                try:
                    result = future.result()
                    records = result.get("records", [])
                    fetch_outputs[cfg.key] = {"records": records, "end_period": end_period}
                    for line in result.get("debug_logs", []):
                        debug_logs.append(f"[{scope_key}:{cfg.key}] {line}")
                    if records:
                        sample = records[0]
                        debug_logs.append(f"[{scope_key}:{cfg.key}] sample_keys={list(sample.keys())[:12]}")
                        debug_logs.append(
                            f"[{scope_key}:{cfg.key}] sample_PRD_DE={sample.get('PRD_DE')} sample_DT={sample.get('DT')}"
                        )
                except Exception as exc:  # noqa: BLE001
                    fetch_outputs[cfg.key] = {"records": [], "end_period": end_period}
                    errors.append(f"{scope_title} - {cfg.title}: {exc}")
                    debug_logs.append(f"[{scope_key}:{cfg.key}] ERROR: {exc}")
                step += 1
                _set_progress(min(100, int(step * 100 / total_steps)))

        for cfg in datasets:
            fetch_meta = fetch_outputs.get(cfg.key, {})
            records = fetch_meta.get("records", [])
            end_period = fetch_meta.get("end_period", default_end_period_by_prd_se(cfg.prd_se))
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
            missing_required = sorted(REQUIRED_SCOPE_COLUMNS - set(combined.columns))
            debug_logs.append(f"[{scope_key}:all] combined_cols={list(combined.columns)}")
            if missing_required:
                errors.append(
                    f"{scope_title_map.get(scope_key, scope_key)} - 통합 데이터 스키마 누락: {', '.join(missing_required)}"
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
        _set_error("데이터 로딩 실패")
        return data_by_scope, errors, debug_logs, empty_data_warnings

    _set_progress(100)
    _set_success("로딩 완료")
    return data_by_scope, errors, debug_logs, empty_data_warnings


def _now_utc_iso() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def _parse_utc_iso(value: str) -> Optional[datetime]:
    text = str(value or "").strip()
    if not text:
        return None
    try:
        parsed = datetime.fromisoformat(text)
    except ValueError:
        return None
    if parsed.tzinfo is None:
        return parsed.replace(tzinfo=timezone.utc)
    return parsed.astimezone(timezone.utc)


def _scope_cache_path(scope_key: str) -> Path:
    return CACHE_ROOT / scope_key / "all.parquet"


def _scope_backup_cache_path(scope_key: str) -> Path:
    return CACHE_ROOT / scope_key / "all.backup.parquet"


def _default_manifest() -> Dict[str, Any]:
    return {
        "schema_version": "",
        "last_check_at_utc": "",
        "last_refresh_at_utc": "",
        "scopes": {},
    }


def _read_manifest() -> Dict[str, Any]:
    if not MANIFEST_PATH.exists():
        return _default_manifest()
    try:
        payload = json.loads(MANIFEST_PATH.read_text(encoding="utf-8"))
        if isinstance(payload, dict):
            out = _default_manifest()
            out.update(payload)
            if not isinstance(out.get("scopes"), dict):
                out["scopes"] = {}
            return out
    except Exception:
        pass
    return _default_manifest()


def _write_manifest_atomic(manifest: Dict[str, Any]) -> None:
    CACHE_ROOT.mkdir(parents=True, exist_ok=True)
    tmp_path = MANIFEST_PATH.with_suffix(".tmp")
    tmp_path.write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")
    tmp_path.replace(MANIFEST_PATH)


def _is_valid_scope_frame(frame: object) -> bool:
    if not isinstance(frame, pd.DataFrame):
        return False
    if frame.empty:
        return False
    return REQUIRED_SCOPE_COLUMNS.issubset(set(frame.columns))


def _dataset_row_counts(frame: object) -> Dict[str, int]:
    if not isinstance(frame, pd.DataFrame) or frame.empty or "dataset_key" not in frame.columns:
        return {}
    counts = frame["dataset_key"].astype(str).str.strip().value_counts(dropna=False).to_dict()
    return {str(key): int(value) for key, value in counts.items()}


def _read_valid_scope_cache(path: Path, scope_key: str) -> pd.DataFrame:
    if not path.exists():
        return pd.DataFrame()
    try:
        frame = pd.read_parquet(path)
        if not _is_valid_scope_frame(frame):
            return pd.DataFrame()
        has_all_datasets, _missing = _scope_has_expected_datasets(frame, scope_key)
        if not has_all_datasets:
            return pd.DataFrame()
        return frame
    except Exception:
        return pd.DataFrame()


def _read_scope_cache(scope_key: str) -> pd.DataFrame:
    path = _scope_cache_path(scope_key)
    frame = _read_valid_scope_cache(path, scope_key)
    if not frame.empty:
        return frame
    backup_path = _scope_backup_cache_path(scope_key)
    return _read_valid_scope_cache(backup_path, scope_key)


def _write_scope_cache_atomic(scope_key: str, frame: pd.DataFrame) -> None:
    path = _scope_cache_path(scope_key)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp_path = path.with_suffix(".tmp.parquet")
    frame.to_parquet(tmp_path, index=False)
    tmp_path.replace(path)
    backup_path = _scope_backup_cache_path(scope_key)
    shutil.copy2(path, backup_path)


def _latest_period_text(frame: pd.DataFrame) -> str:
    if not isinstance(frame, pd.DataFrame) or frame.empty or "period" not in frame.columns:
        return ""
    latest = pd.to_datetime(frame["period"], errors="coerce").max()
    if pd.isna(latest):
        return ""
    return pd.Timestamp(latest).strftime("%Y-%m-%d")


def _scope_has_expected_datasets(frame: pd.DataFrame, scope_key: str) -> tuple[bool, List[str]]:
    expected = [
        str(getattr(cfg, "key", "")).strip()
        for cfg in datasets_for_scope(scope_key)
        if getattr(cfg, "required_for_scope", True)
    ]
    expected = [key for key in expected if key]
    if not isinstance(frame, pd.DataFrame) or frame.empty or "dataset_key" not in frame.columns:
        return False, expected
    present = set(frame["dataset_key"].astype(str).str.strip().unique().tolist())
    missing = [key for key in expected if key not in present]
    return len(missing) == 0, missing


def _scope_debug_summary(scope_key: str, label: str, frame: object) -> str:
    if not isinstance(frame, pd.DataFrame):
        return f"[{scope_key}:{label}] no_frame"
    row_counts = _dataset_row_counts(frame)
    return f"[{scope_key}:{label}] rows={len(frame)} dataset_rows={row_counts}"


def _probe_scope_latest_period(
    api_key: str,
    scope_key: str,
) -> tuple[str, List[str]]:
    debug_logs: List[str] = []
    datasets = datasets_for_scope(scope_key)
    activity_cfg = next((x for x in datasets if str(getattr(x, "key", "")) == "activity"), None)
    if activity_cfg is None:
        return "", debug_logs
    end_period = default_end_period_by_prd_se(activity_cfg.prd_se)
    config_signature = "|".join(
        [
            activity_cfg.tbl_id,
            activity_cfg.itm_id,
            activity_cfg.obj_l1,
            activity_cfg.obj_l2,
            activity_cfg.output_fields,
            activity_cfg.start_prd_de,
            activity_cfg.prd_se,
            end_period,
            scope_key,
        ]
    )
    result = fetch_records_live(
        api_key=api_key,
        dataset_key="activity",
        end_period=end_period,
        config_signature=config_signature,
        datasets=datasets,
    )
    records = result.get("records", [])
    for line in result.get("debug_logs", []):
        debug_logs.append(f"[probe:{scope_key}:activity] {line}")
    parsed = normalize_records(activity_cfg, records, region_scope=scope_key)
    if parsed.empty or "period" not in parsed.columns:
        return "", debug_logs
    latest = pd.to_datetime(parsed["period"], errors="coerce").max()
    if pd.isna(latest):
        return "", debug_logs
    return pd.Timestamp(latest).strftime("%Y-%m-%d"), debug_logs


def load_data_with_local_cache(
    api_key: str,
    data_model_version: str,
    status_box: Any,
    progress_box: Any,
    main_status_box: Optional[Any] = None,
    main_progress_box: Optional[Any] = None,
    scopes: Optional[List[str]] = None,
    force_refresh: bool = False,
    check_interval_hours: int = 24,
) -> tuple[Dict[str, pd.DataFrame], List[str], List[str], List[str]]:
    requested_scopes = list(scopes or ["province", "gyeonggi31"])
    manifest = _read_manifest()
    errors: List[str] = []
    debug_logs: List[str] = []
    warnings: List[str] = []

    scope_data: Dict[str, pd.DataFrame] = {scope: _read_scope_cache(scope) for scope in requested_scopes}
    for scope in requested_scopes:
        debug_logs.append(_scope_debug_summary(scope, "cache_read", scope_data.get(scope)))
    missing_scopes = [scope for scope in requested_scopes if not _is_valid_scope_frame(scope_data.get(scope))]

    schema_mismatch = str(manifest.get("schema_version", "")) != str(data_model_version)
    last_check_at = _parse_utc_iso(str(manifest.get("last_check_at_utc", "")))
    now_utc = datetime.now(timezone.utc)
    check_due = last_check_at is None or (now_utc - last_check_at) >= timedelta(hours=max(1, int(check_interval_hours)))

    scopes_to_refresh: List[str] = []
    if force_refresh or schema_mismatch:
        scopes_to_refresh = requested_scopes
    elif missing_scopes:
        scopes_to_refresh = missing_scopes
    elif check_due:
        for scope in requested_scopes:
            try:
                remote_latest, probe_logs = _probe_scope_latest_period(api_key=api_key, scope_key=scope)
                debug_logs.extend(probe_logs)
                local_latest = str((manifest.get("scopes", {}) or {}).get(scope, {}).get("latest_period", ""))
                if remote_latest and (not local_latest or remote_latest > local_latest):
                    scopes_to_refresh.append(scope)
            except Exception as exc:  # noqa: BLE001
                errors.append(f"{scope} 업데이트 확인 실패: {exc}")
        manifest["last_check_at_utc"] = _now_utc_iso()

    if scopes_to_refresh:
        fetched, fetch_errors, fetch_logs, fetch_warnings = load_all_data_with_progress(
            api_key=api_key,
            status_box=status_box,
            progress_box=progress_box,
            main_status_box=main_status_box,
            main_progress_box=main_progress_box,
            scopes=scopes_to_refresh,
        )
        errors.extend(fetch_errors)
        debug_logs.extend(fetch_logs)
        warnings.extend(fetch_warnings)

        for scope in scopes_to_refresh:
            new_df = fetched.get(scope, pd.DataFrame())
            debug_logs.append(_scope_debug_summary(scope, "refresh_result", new_df))
            if _is_valid_scope_frame(new_df):
                dedup_cols = ["dataset_key", "region_name", "indicator_name", "category_name", "period"]
                dedup_df = new_df.drop_duplicates(subset=[c for c in dedup_cols if c in new_df.columns], keep="last").copy()
                has_all_datasets, missing_dataset_keys = _scope_has_expected_datasets(dedup_df, scope)
                if not has_all_datasets:
                    errors.append(
                        f"{scope} 저장 건너뜀: 일부 데이터셋 누락({', '.join(missing_dataset_keys)})으로 기존 캐시를 유지합니다."
                    )
                    debug_logs.append(f"[{scope}:cache] skipped_incomplete_scope missing={missing_dataset_keys}")
                    debug_logs.append(_scope_debug_summary(scope, "cache_kept", scope_data.get(scope)))
                    continue
                _write_scope_cache_atomic(scope, dedup_df)
                scope_data[scope] = dedup_df
                debug_logs.append(_scope_debug_summary(scope, "cache_saved", dedup_df))
                scope_meta = (manifest.get("scopes", {}) or {}).get(scope, {})
                scope_meta.update(
                    {
                        "latest_period": _latest_period_text(dedup_df),
                        "rows": int(len(dedup_df)),
                        "updated_at_utc": _now_utc_iso(),
                        "path": str(_scope_cache_path(scope)),
                    }
                )
                manifest.setdefault("scopes", {})[scope] = scope_meta
            else:
                errors.append(f"{scope} 저장 건너뜀: 유효한 데이터가 없어 기존 캐시를 유지합니다.")

        manifest["schema_version"] = str(data_model_version)
        manifest["last_refresh_at_utc"] = _now_utc_iso()
        manifest["last_check_at_utc"] = _now_utc_iso()
        _write_manifest_atomic(manifest)
    elif check_due:
        manifest["schema_version"] = str(data_model_version)
        _write_manifest_atomic(manifest)

    return scope_data, errors, debug_logs, warnings

