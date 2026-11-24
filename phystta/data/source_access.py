from __future__ import annotations

import json
import os
import shutil
from pathlib import Path
from typing import Any
from urllib.parse import urlparse

import yaml


def _repo_root(repo_root: str | Path | None = None) -> Path:
    if repo_root is not None:
        return Path(repo_root).expanduser().resolve()
    return Path(__file__).resolve().parents[2]


def source_access_registry_path(repo_root: str | Path | None = None) -> Path:
    return _repo_root(repo_root) / "registry" / "source_access_registry.json"


def load_source_access_registry(repo_root: str | Path | None = None) -> dict[str, Any]:
    payload = json.loads(source_access_registry_path(repo_root).read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise RuntimeError("source_access_registry.json must contain a JSON object")
    sources = payload.get("sources")
    if not isinstance(sources, dict):
        raise RuntimeError("source_access_registry.json must contain a 'sources' object")
    return payload


def normalize_source_name(name: str, *, repo_root: str | Path | None = None) -> str:
    registry = load_source_access_registry(repo_root)
    target = str(name).strip().lower()
    for canonical, spec in dict(registry.get("sources", {})).items():
        aliases = {str(canonical).lower(), *(str(item).lower() for item in spec.get("aliases", []))}
        if target in aliases:
            return canonical
    return target


def _ordered_unique_strings(values: list[str]) -> list[str]:
    seen: set[str] = set()
    ordered: list[str] = []
    for value in values:
        if value in seen:
            continue
        seen.add(value)
        ordered.append(value)
    return ordered


def infer_required_sources_from_config(
    config_path: str | Path | None,
    *,
    repo_root: str | Path | None = None,
) -> list[str]:
    if config_path is None:
        return []
    path = Path(config_path).expanduser().resolve()
    if not path.exists():
        return []
    payload = yaml.safe_load(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        return []
    data = payload.get("data")
    if not isinstance(data, dict):
        return []
    rows = data.get("sources")
    if not isinstance(rows, list):
        return []
    out: list[str] = []
    seen: set[str] = set()
    for row in rows:
        if not isinstance(row, dict):
            continue
        raw_name = row.get("name") or row.get("domain_label")
        if not raw_name:
            continue
        canonical = normalize_source_name(str(raw_name), repo_root=repo_root)
        if canonical in seen:
            continue
        seen.add(canonical)
        out.append(canonical)
    return out


def _resolve_override_root(spec: dict[str, Any]) -> Path | None:
    root_env = str(spec.get("root_env", "")).strip()
    if not root_env:
        return None
    candidate = os.environ.get(root_env)
    if not candidate:
        return None
    return Path(candidate).expanduser()


def _canonical_root(spec: dict[str, Any]) -> Path:
    return Path(str(spec["canonical_root"])).expanduser()


def _canonical_paths(spec: dict[str, Any], *, current_root: Path | None = None) -> dict[str, Path]:
    canonical_root = _canonical_root(spec)
    effective_root = (current_root.expanduser() if current_root is not None else _current_root(spec).expanduser())
    out = {"root": effective_root}
    for key, value in dict(spec.get("canonical_paths", {})).items():
        configured = Path(str(value)).expanduser()
        if configured.is_absolute():
            try:
                rel = configured.relative_to(canonical_root)
            except ValueError:
                resolved = configured
            else:
                resolved = effective_root / rel
        else:
            resolved = effective_root / configured
        out[str(key)] = resolved
    return out


def _current_root(spec: dict[str, Any]) -> Path:
    return (_resolve_override_root(spec) or _canonical_root(spec)).expanduser()


def _ensure_path(value: str | Path | None) -> Path | None:
    if value is None:
        return None
    return Path(value).expanduser()


def _tool_payload(tool: str) -> dict[str, Any]:
    resolved = shutil.which(tool)
    return {
        "tool": tool,
        "present": resolved is not None,
        "path": resolved,
    }


def _redact_url(value: str | None) -> str | None:
    if not value:
        return None
    if value.startswith("s3://"):
        return "s3://<redacted>"
    parsed = urlparse(value)
    if parsed.scheme and parsed.netloc:
        base = f"{parsed.scheme}://{parsed.netloc}{parsed.path}"
        if parsed.query:
            return f"{base}?<redacted>"
        return base
    return "<redacted>"


def _credential_presence(name: str) -> bool:
    value = os.environ.get(name)
    return bool(value and value.strip())


def _workspace_bundle_root(bundle_root: Path) -> Path:
    return Path("/workspace") / bundle_root.name


def _workspace_repo_root(bundle_root: Path) -> Path:
    return _workspace_bundle_root(bundle_root) / "repo" / "phystta_active_20260321"


def _resolve_kaggle_json_candidate_paths(repo_root: Path) -> list[Path]:
    out: list[Path] = []
    env_path = os.environ.get("KAGGLE_JSON_PATH")
    if env_path:
        out.append(Path(env_path).expanduser())
    config_dir = os.environ.get("KAGGLE_CONFIG_DIR")
    if config_dir:
        out.append(Path(config_dir).expanduser() / "kaggle.json")
    out.append(Path.home() / ".kaggle" / "kaggle.json")
    deduped: list[Path] = []
    seen: set[str] = set()
    for item in out:
        text = str(item)
        if text in seen:
            continue
        seen.add(text)
        deduped.append(item)
    return deduped


def _resolve_credentials(spec: dict[str, Any], *, repo_root: Path) -> dict[str, Any]:
    backend = str(spec.get("backend"))
    if backend == "aws_s3":
        access_present = _credential_presence("AWS_ACCESS_KEY_ID")
        secret_present = _credential_presence("AWS_SECRET_ACCESS_KEY")
        session_present = _credential_presence("AWS_SESSION_TOKEN")
        profile_present = _credential_presence("AWS_PROFILE")
        region_present = _credential_presence("AWS_DEFAULT_REGION")
        profile_name = os.environ.get("AWS_PROFILE")
        if profile_present:
            mode = "aws_profile"
            ready = True
        elif access_present and secret_present:
            mode = "env"
            ready = True
        else:
            mode = "missing"
            ready = False
        return {
            "mode": mode,
            "ready": ready,
            "required_env_vars": ["AWS_ACCESS_KEY_ID", "AWS_SECRET_ACCESS_KEY"],
            "optional_env_vars": ["AWS_SESSION_TOKEN", "AWS_DEFAULT_REGION", "AWS_PROFILE"],
            "present": {
                "AWS_ACCESS_KEY_ID": access_present,
                "AWS_SECRET_ACCESS_KEY": secret_present,
                "AWS_SESSION_TOKEN": session_present,
                "AWS_DEFAULT_REGION": region_present,
                "AWS_PROFILE": profile_present,
            },
            "missing": (
                []
                if ready
                else ["AWS_ACCESS_KEY_ID", "AWS_SECRET_ACCESS_KEY"]
            ),
            "profile_name": profile_name if profile_present else None,
        }
    if backend == "azure_blob_sas":
        env_name = str(spec.get("locator_env", ""))
        present = _credential_presence(env_name)
        return {
            "mode": "env" if present else "missing",
            "ready": present,
            "required_env_vars": [env_name] if env_name else [],
            "optional_env_vars": [],
            "present": {env_name: present} if env_name else {},
            "missing": [] if present or not env_name else [env_name],
        }
    if backend == "kaggle_dataset":
        username_present = _credential_presence("KAGGLE_USERNAME")
        key_present = _credential_presence("KAGGLE_KEY")
        candidates = _resolve_kaggle_json_candidate_paths(repo_root)
        resolved_json = next((path for path in candidates if path.exists()), None)
        locator_env = str(spec.get("locator_env", ""))
        locator_present = _credential_presence(locator_env)
        if username_present and key_present:
            mode = "env"
            ready = True
        elif resolved_json is not None:
            mode = "kaggle_json"
            ready = True
        else:
            mode = "missing"
            ready = False
        return {
            "mode": mode,
            "ready": ready,
            "required_env_vars": ["KAGGLE_USERNAME", "KAGGLE_KEY", locator_env] if locator_env else ["KAGGLE_USERNAME", "KAGGLE_KEY"],
            "optional_env_vars": ["KAGGLE_JSON_PATH", "KAGGLE_CONFIG_DIR"],
            "present": {
                "KAGGLE_USERNAME": username_present,
                "KAGGLE_KEY": key_present,
                locator_env: locator_present,
            }
            if locator_env
            else {
                "KAGGLE_USERNAME": username_present,
                "KAGGLE_KEY": key_present,
            },
            "missing": (
                []
                if ready
                else (
                    ["KAGGLE_JSON_PATH", "KAGGLE_USERNAME", "KAGGLE_KEY"]
                )
            ),
            "kaggle_json_candidates": [str(path) for path in candidates],
            "resolved_kaggle_json_path": str(resolved_json) if resolved_json is not None else None,
            "locator_env": locator_env or None,
            "locator_present": locator_present,
        }
    if backend in {"direct_url", "local_existing"}:
        env_name = str(spec.get("locator_env", "")).strip()
        present = _credential_presence(env_name) if env_name else True
        return {
            "mode": "none" if backend == "local_existing" else ("env" if present else "missing"),
            "ready": present,
            "required_env_vars": [env_name] if env_name else [],
            "optional_env_vars": [],
            "present": {env_name: present} if env_name else {},
            "missing": [] if present or not env_name else [env_name],
        }
    return {
        "mode": "missing",
        "ready": False,
        "required_env_vars": [],
        "optional_env_vars": [],
        "present": {},
        "missing": [],
    }


def _resolve_locator(spec: dict[str, Any], *, credentials: dict[str, Any]) -> dict[str, Any]:
    env_name = str(spec.get("locator_env", "")).strip()
    if not env_name:
        return {
            "env": None,
            "present": False,
            "redacted_preview": None,
        }
    raw = os.environ.get(env_name)
    return {
        "env": env_name,
        "present": bool(raw and raw.strip()),
        "redacted_preview": _redact_url(raw),
    }


def _current_source_status(
    *,
    source_name: str,
    spec: dict[str, Any],
    repo_root: Path,
) -> dict[str, Any]:
    backend = str(spec.get("backend"))
    current_root = _current_root(spec)
    canonical_paths = _canonical_paths(spec, current_root=current_root)
    tools = [_tool_payload(str(tool)) for tool in spec.get("required_tools", [])]
    credentials = _resolve_credentials(spec, repo_root=repo_root)
    locator = _resolve_locator(spec, credentials=credentials)
    missing_canonical = [name for name, path in canonical_paths.items() if not path.exists()]

    fetched_needs_prepare = False
    if backend == "aws_s3":
        labels_csv = current_root / "mimic-cxr-2.0.0-chexpert.csv"
        labels_gz = current_root / "mimic-cxr-2.0.0-chexpert.csv.gz"
        meta_csv = current_root / "mimic-cxr-2.0.0-metadata.csv"
        meta_gz = current_root / "mimic-cxr-2.0.0-metadata.csv.gz"
        fetched_needs_prepare = current_root.exists() and ((labels_gz.exists() and not labels_csv.exists()) or (meta_gz.exists() and not meta_csv.exists()))
    elif backend == "azure_blob_sas":
        nested_train = current_root / "CheXpert-v1.0" / "train.csv"
        nested_small = current_root / "CheXpert-v1.0-small" / "train.csv"
        fetched_needs_prepare = current_root.exists() and not canonical_paths["train_csv"].exists() and (nested_train.exists() or nested_small.exists())
    elif backend == "kaggle_dataset":
        images_dir = current_root / "images"
        fragment_dirs = list(current_root.glob("images_*/images"))
        fetched_needs_prepare = current_root.exists() and canonical_paths["metadata_csv"].exists() and not images_dir.exists() and bool(fragment_dirs)

    if all(path.exists() for path in canonical_paths.values()):
        status = "ready_canonical"
        blocker_codes: list[str] = []
    elif fetched_needs_prepare:
        status = "fetched_needs_prepare"
        blocker_codes = ["blocked_by_failed_normalization"]
    else:
        missing_tools = [row["tool"] for row in tools if not row["present"]]
        if backend == "local_existing":
            status = "blocked_by_missing_contract"
            blocker_codes = ["blocked_by_missing_source_contract"]
        elif missing_tools:
            status = "tool_missing"
            code_map = {
                "aws": "blocked_by_missing_aws_cli",
                "azcopy": "blocked_by_missing_azcopy",
                "kaggle": "blocked_by_missing_kaggle_cli",
            }
            blocker_codes = [code_map.get(tool, "blocked_by_missing_contract") for tool in missing_tools]
        elif not credentials.get("ready", False):
            status = "credentials_missing"
            if backend == "aws_s3":
                blocker_codes = ["blocked_by_missing_aws_credentials"]
            elif backend == "kaggle_dataset":
                if credentials.get("resolved_kaggle_json_path") is None and not (
                    credentials.get("present", {}).get("KAGGLE_USERNAME") and credentials.get("present", {}).get("KAGGLE_KEY")
                ):
                    blocker_codes = ["blocked_by_missing_kaggle_credentials", "blocked_by_missing_kaggle_json"]
                else:
                    blocker_codes = ["blocked_by_missing_kaggle_credentials"]
            else:
                blocker_codes = ["blocked_by_missing_contract"]
        elif backend in {"aws_s3", "azure_blob_sas", "kaggle_dataset", "direct_url"} and not locator.get("present", False):
            status = "blocked_by_missing_contract"
            if backend == "azure_blob_sas":
                blocker_codes = ["blocked_by_missing_azure_sas"]
            else:
                blocker_codes = ["blocked_by_missing_source_contract"]
        else:
            status = "fetch_required"
            blocker_codes = []
    return {
        "backend": backend,
        "current_root": str(current_root),
        "canonical_paths": {name: str(path) for name, path in canonical_paths.items()},
        "status": status,
        "blocker_codes": blocker_codes,
        "missing_canonical_paths": [str(canonical_paths[name]) for name in missing_canonical],
        "required_tools": tools,
        "credentials": credentials,
        "locator": locator,
    }


def _prepare_steps(
    spec: dict[str, Any],
    *,
    repo_root: Path,
    bundle_root: Path,
    campaign_id: str,
) -> list[dict[str, Any]]:
    source = str(spec.get("source"))
    root = _current_root(spec)
    scripts_root = _workspace_repo_root(bundle_root) / "scripts"
    if source == "mimic":
        return [
            {
                "step_id": "prepare_mimic_root_csvs",
                "kind": "normalize_root_contract",
                "required": True,
                "command_preview": (
                    f"python3 {scripts_root / 'source_access_runtime.py'} --bundle_root {_workspace_bundle_root(bundle_root)} "
                    f"--repo_root {_workspace_repo_root(bundle_root)} --campaign_id {campaign_id} "
                    f"--source mimic --action prepare --exec"
                ),
                "expected_outputs": [
                    str(root / "mimic-cxr-2.0.0-chexpert.csv"),
                    str(root / "mimic-cxr-2.0.0-metadata.csv"),
                ],
            },
            {
                "step_id": "prepare_mimic_smoke_metadata",
                "kind": "smoke_metadata",
                "required": False,
                "command_preview": (
                    f"python3 {scripts_root / 'prepare_mimic_smoke_metadata.py'} --root {root} --out {root / 'mimic-cxr-2.0.0-chexpert.smoke.csv'}"
                ),
                "expected_outputs": [str(root / "mimic-cxr-2.0.0-chexpert.smoke.csv")],
            },
        ]
    if source == "chexpert":
        return [
            {
                "step_id": "normalize_chexpert_root_contract",
                "kind": "canonical_root_layout",
                "required": True,
                "command_preview": (
                    f"python3 {scripts_root / 'source_access_runtime.py'} --bundle_root {_workspace_bundle_root(bundle_root)} "
                    f"--repo_root {_workspace_repo_root(bundle_root)} --campaign_id {campaign_id} "
                    f"--source chexpert --action prepare --exec"
                ),
                "expected_outputs": [str(root / "train.csv")],
            },
            {
                "step_id": "prepare_chexpert_smoke_metadata",
                "kind": "smoke_metadata",
                "required": False,
                "command_preview": (
                    f"python3 {scripts_root / 'prepare_chexpert_smoke_metadata.py'} --root {root} --smoke_out {root / 'train.smoke.csv'}"
                ),
                "expected_outputs": [str(root / "train.smoke.csv")],
            },
        ]
    if source == "nih_chestxray14":
        return [
            {
                "step_id": "normalize_nih_layout",
                "kind": "canonical_image_layout",
                "required": True,
                "command_preview": (
                    f"python3 {scripts_root / 'normalize_nih_layout.py'} --root {root} --images_dir {root / 'images'}"
                ),
                "expected_outputs": [str(root / "images")],
            },
            {
                "step_id": "prepare_nih_smoke_metadata",
                "kind": "smoke_metadata",
                "required": False,
                "command_preview": (
                    f"python3 {scripts_root / 'prepare_open_cxr_metadata.py'} nih --root {root} --out {root / 'Data_Entry_2017.normalized.csv'}"
                ),
                "expected_outputs": [str(root / "Data_Entry_2017.normalized.csv")],
            },
        ]
    return []


def _fetch_steps(spec: dict[str, Any], *, repo_root: Path) -> list[dict[str, Any]]:
    source = str(spec.get("source"))
    root = _current_root(spec)
    backend = str(spec.get("backend"))
    if backend == "aws_s3":
        return [
            {
                "step_id": "aws_s3_sync",
                "kind": "fetch",
                "required": True,
                "command_preview": f"aws s3 sync \"$PHYSTTA_MIMIC_S3_URI\" \"{root}\" --no-progress",
            }
        ]
    if backend == "azure_blob_sas":
        return [
            {
                "step_id": "azure_blob_copy",
                "kind": "fetch",
                "required": True,
                "command_preview": f"azcopy copy \"$CHEXPERT_SAS_URL\" \"{root}\" --recursive=true",
            }
        ]
    if backend == "kaggle_dataset":
        return [
            {
                "step_id": "kaggle_dataset_download",
                "kind": "fetch",
                "required": True,
                "command_preview": (
                    f"kaggle datasets download \"$PHYSTTA_NIH_KAGGLE_DATASET\" -p \"{root}\" --unzip"
                ),
            }
        ]
    if backend == "direct_url":
        return [
            {
                "step_id": "direct_url_fetch",
                "kind": "fetch",
                "required": True,
                "command_preview": f"curl -L \"$PHYSTTA_SOURCE_URL\" -o \"{root}\"",
            }
        ]
    return []


def _ranked_source_needs(autonomous_preview: dict[str, Any]) -> list[dict[str, Any]]:
    frontier = dict(autonomous_preview.get("frontier_preview", {}))
    task_rows = {str(row.get("family")): row for row in frontier.get("task_contracts", [])}
    candidates = [
        row
        for row in list(frontier.get("family_frontier", []))
        if str(row.get("status")) in {"available", "available_template_backed_override"}
    ]
    ranked = sorted(
        candidates,
        key=lambda row: (
            -int(row.get("selection_priority_score", 0)),
            int(row.get("campaign_priority", 999)),
            str(row.get("family", "")),
        ),
    )
    source_rank: dict[str, dict[str, Any]] = {}
    for family_rank, family_row in enumerate(ranked, start=1):
        family = str(family_row.get("family"))
        synthesis = dict(task_rows.get(family, {}))
        branch_ids = [str(item) for item in synthesis.get("candidate_branch_ids", []) if item]
        required_sources = [str(item) for item in synthesis.get("required_sources", []) if item]
        for index, source in enumerate(required_sources, start=1):
            if source in source_rank:
                continue
            source_rank[source] = {
                "source": source,
                "family": family,
                "family_rank": family_rank,
                "source_rank_within_family": index,
                "selection_priority_score": int(family_row.get("selection_priority_score", 0)),
                "required_by_branch_ids": branch_ids,
                "required_by_families": [family],
                "why_this_dataset_now": [
                    f"selected_family={family}",
                    f"family_rank={family_rank}",
                    *list(family_row.get("selection_why_now", [])),
                ],
            }
    return sorted(source_rank.values(), key=lambda row: (row["family_rank"], row["source_rank_within_family"], row["source"]))


def build_source_access_contracts(
    *,
    bundle_root: str | Path,
    repo_root: str | Path,
    campaign_id: str,
    autonomous_preview: dict[str, Any],
) -> dict[str, dict[str, Any]]:
    bundle_root = Path(bundle_root).expanduser().resolve()
    repo_root = Path(repo_root).expanduser().resolve()
    workspace_bundle_root = _workspace_bundle_root(bundle_root)
    workspace_repo_root = _workspace_repo_root(bundle_root)
    registry = load_source_access_registry(repo_root)
    sources = dict(registry.get("sources", {}))
    ranked_needs = _ranked_source_needs(autonomous_preview)
    frontier = dict(autonomous_preview.get("frontier_preview", {}))
    task_rows = {str(row.get("family")): row for row in frontier.get("task_contracts", [])}
    ranked_families = sorted(
        [
            row
            for row in list(frontier.get("family_frontier", []))
            if str(row.get("status")) in {"available", "available_template_backed_override"}
        ],
        key=lambda row: (
            -int(row.get("selection_priority_score", 0)),
            int(row.get("campaign_priority", 999)),
            str(row.get("family", "")),
        ),
    )
    needed_now = []
    if ranked_families:
        top_family = str(ranked_families[0].get("family"))
        needed_now = [
            normalize_source_name(str(item), repo_root=repo_root)
            for item in list(dict(task_rows.get(top_family, {})).get("required_sources", []))
            if item
        ]
        needed_now = _ordered_unique_strings(needed_now)
    all_ranked = [row["source"] for row in ranked_needs]

    readiness_sources: dict[str, Any] = {}
    fetch_rows: list[dict[str, Any]] = []
    prepare_rows: list[dict[str, Any]] = []
    next_actions: list[dict[str, Any]] = []

    for source_name, spec_raw in sources.items():
        spec = dict(spec_raw)
        spec["source"] = source_name
        state = _current_source_status(source_name=source_name, spec=spec, repo_root=repo_root)
        need_row = next((row for row in ranked_needs if row["source"] == source_name), None)
        required_now = source_name in needed_now
        required_later = source_name in all_ranked and not required_now
        if required_now:
            if state["status"] == "ready_canonical":
                stage_need_state = "source_ready_existing"
            elif state["status"] == "fetched_needs_prepare":
                stage_need_state = "source_prepare_required_for_next_stage"
            elif state["status"] == "fetch_required":
                stage_need_state = "source_fetch_required_for_next_stage"
            else:
                stage_need_state = "source_access_blocked_for_next_stage"
        elif required_later:
            stage_need_state = "source_not_needed_yet"
        else:
            stage_need_state = "source_not_needed_yet"
        fetch_steps = _fetch_steps(spec, repo_root=repo_root)
        prepare_steps = _prepare_steps(
            spec,
            repo_root=repo_root,
            bundle_root=bundle_root,
            campaign_id=campaign_id,
        )
        runtime_commands = {
            "fetch_dry_run": (
                f"python3 {workspace_repo_root / 'scripts' / 'source_access_runtime.py'} "
                f"--bundle_root {workspace_bundle_root} --repo_root {workspace_repo_root} "
                f"--campaign_id {campaign_id} --source {source_name} --action fetch"
            ),
            "fetch_exec": (
                f"python3 {workspace_repo_root / 'scripts' / 'source_access_runtime.py'} "
                f"--bundle_root {workspace_bundle_root} --repo_root {workspace_repo_root} "
                f"--campaign_id {campaign_id} --source {source_name} --action fetch --exec"
            ),
            "prepare_dry_run": (
                f"python3 {workspace_repo_root / 'scripts' / 'source_access_runtime.py'} "
                f"--bundle_root {workspace_bundle_root} --repo_root {workspace_repo_root} "
                f"--campaign_id {campaign_id} --source {source_name} --action prepare"
            ),
            "prepare_exec": (
                f"python3 {workspace_repo_root / 'scripts' / 'source_access_runtime.py'} "
                f"--bundle_root {workspace_bundle_root} --repo_root {workspace_repo_root} "
                f"--campaign_id {campaign_id} --source {source_name} --action prepare --exec"
            ),
            "fetch_prepare_dry_run": (
                f"python3 {workspace_repo_root / 'scripts' / 'source_access_runtime.py'} "
                f"--bundle_root {workspace_bundle_root} --repo_root {workspace_repo_root} "
                f"--campaign_id {campaign_id} --source {source_name} --action fetch_prepare"
            ),
            "fetch_prepare_exec": (
                f"python3 {workspace_repo_root / 'scripts' / 'source_access_runtime.py'} "
                f"--bundle_root {workspace_bundle_root} --repo_root {workspace_repo_root} "
                f"--campaign_id {campaign_id} --source {source_name} --action fetch_prepare --exec"
            ),
        }
        readiness_sources[source_name] = {
            "source": source_name,
            "display_name": spec.get("display_name"),
            "backend": state["backend"],
            "status": state["status"],
            "stage_need_state": stage_need_state,
            "required_for_current_stage": required_now,
            "required_for_current_campaign_frontier": source_name in all_ranked,
            "why_this_dataset_now": list((need_row or {}).get("why_this_dataset_now", [])),
            "required_by_families": list((need_row or {}).get("required_by_families", [])),
            "required_by_branch_ids": list((need_row or {}).get("required_by_branch_ids", [])),
            "current_stage_rank": need_row.get("family_rank") if need_row else None,
            "current_root": state["current_root"],
            "canonical_paths": state["canonical_paths"],
            "missing_canonical_paths": state["missing_canonical_paths"],
            "required_tools": state["required_tools"],
            "credential_resolution": state["credentials"],
            "locator_contract": state["locator"],
            "blocker_codes": list(state["blocker_codes"]),
            "fetch_steps": fetch_steps,
            "prepare_steps": prepare_steps,
            "runtime_commands": runtime_commands,
        }
        if source_name in all_ranked:
            fetch_rows.append(
                {
                    "source": source_name,
                    "status": state["status"],
                    "stage_need_state": stage_need_state,
                    "current_stage_rank": need_row.get("family_rank") if need_row else None,
                    "why_this_dataset_now": list((need_row or {}).get("why_this_dataset_now", [])),
                    "recommended_action": (
                        "use_ready_source"
                        if state["status"] == "ready_canonical"
                        else "prepare_fetched_source"
                        if state["status"] == "fetched_needs_prepare"
                        else "fetch_and_prepare_source"
                        if state["status"] == "fetch_required"
                        else "resolve_source_blocker"
                    ),
                    "fetch_steps": fetch_steps,
                    "prepare_steps": prepare_steps,
                    "blocker_codes": list(state["blocker_codes"]),
                    "runtime_commands": runtime_commands,
                }
            )
        if state["status"] != "ready_canonical" and source_name in all_ranked:
            next_actions.append(
                {
                    "source": source_name,
                    "action_class": (
                        "source_prepare_required"
                        if state["status"] == "fetched_needs_prepare"
                        else "source_fetch_required"
                        if state["status"] == "fetch_required"
                        else "source_access_blocked"
                    ),
                    "required_for_current_stage": required_now,
                    "why_this_dataset_now": list((need_row or {}).get("why_this_dataset_now", [])),
                    "blocker_codes": list(state["blocker_codes"]),
                    "runtime_commands": runtime_commands,
                }
            )
        prepare_rows.append(
            {
                "source": source_name,
                "status": state["status"],
                "canonical_paths": state["canonical_paths"],
                "prepare_steps": prepare_steps,
                "post_prepare_target_status": "ready_canonical",
            }
        )

    summary = {
        "ready_canonical_sources": sorted(
            name for name, row in readiness_sources.items() if row["status"] == "ready_canonical"
        ),
        "fetch_required_sources": sorted(
            name for name, row in readiness_sources.items() if row["status"] == "fetch_required"
        ),
        "credentials_missing_sources": sorted(
            name for name, row in readiness_sources.items() if row["status"] == "credentials_missing"
        ),
        "tool_missing_sources": sorted(
            name for name, row in readiness_sources.items() if row["status"] == "tool_missing"
        ),
        "fetched_needs_prepare_sources": sorted(
            name for name, row in readiness_sources.items() if row["status"] == "fetched_needs_prepare"
        ),
        "prepare_required_sources": sorted(
            name for name, row in readiness_sources.items() if row["status"] == "fetched_needs_prepare"
        ),
        "blocked_sources": sorted(
            name for name, row in readiness_sources.items() if row["status"] in {"blocked_by_missing_contract", "credentials_missing", "tool_missing"}
        ),
        "needed_now_sources": needed_now,
        "needed_later_sources": [name for name in all_ranked if name not in needed_now],
        "current_stage_top_source": needed_now[0] if needed_now else None,
    }
    return {
        "source_access_readiness": {
            "schema_version": "1.0",
            "campaign_id": campaign_id,
            "bundle_root": str(bundle_root),
            "repo_root": str(repo_root),
            "summary": summary,
            "sources": readiness_sources,
        },
        "source_fetch_plan": {
            "schema_version": "1.0",
            "campaign_id": campaign_id,
            "workspace_contract": {
                "workspace_bundle_root": str(workspace_bundle_root),
                "workspace_repo_root": str(workspace_repo_root),
            },
            "ordered_sources": fetch_rows,
            "next_actions": next_actions,
        },
        "source_prepare_manifest": {
            "schema_version": "1.0",
            "campaign_id": campaign_id,
            "sources": prepare_rows,
        },
    }


def source_stage_signal_from_required_sources(
    required_sources: list[str],
    *,
    repo_root: str | Path | None = None,
) -> dict[str, Any]:
    repo_root = _repo_root(repo_root)
    registry = load_source_access_registry(repo_root)
    normalized = [normalize_source_name(item, repo_root=repo_root) for item in required_sources]
    if not normalized:
        return {
            "selection_source_state": "no_source_requirement",
            "selection_next_action_reason": ["no_dataset_fetch_needed"],
            "selection_required_sources_ready": True,
            "selection_source_blocker_codes": [],
        }
    blockers: list[str] = []
    fetchable = False
    prepare_required = False
    ready_all = True
    for source_name in normalized:
        spec = dict(registry["sources"].get(source_name, {}))
        if not spec:
            blockers.append("blocked_by_missing_source_contract")
            ready_all = False
            continue
        spec["source"] = source_name
        state = _current_source_status(source_name=source_name, spec=spec, repo_root=repo_root)
        if state["status"] == "fetch_required":
            fetchable = True
            ready_all = False
        elif state["status"] == "fetched_needs_prepare":
            prepare_required = True
            ready_all = False
        elif state["status"] == "ready_canonical":
            continue
        else:
            blockers.extend(list(state["blocker_codes"]))
            ready_all = False
    blockers = sorted(dict.fromkeys(blockers))
    if ready_all:
        return {
            "selection_source_state": "all_required_sources_ready",
            "selection_next_action_reason": ["source_ready_existing"],
            "selection_required_sources_ready": True,
            "selection_source_blocker_codes": [],
        }
    if fetchable and not blockers:
        return {
            "selection_source_state": "source_fetch_required",
            "selection_next_action_reason": ["source_fetch_required_for_next_stage"],
            "selection_required_sources_ready": False,
            "selection_source_blocker_codes": [],
        }
    if prepare_required and not blockers:
        return {
            "selection_source_state": "source_prepare_required",
            "selection_next_action_reason": ["source_prepare_required_for_next_stage"],
            "selection_required_sources_ready": False,
            "selection_source_blocker_codes": [],
        }
    return {
        "selection_source_state": "source_blocked",
        "selection_next_action_reason": ["blocked_by_missing_source_access"],
        "selection_required_sources_ready": False,
        "selection_source_blocker_codes": blockers or ["blocked_by_missing_source_contract"],
    }
