from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional


ROOT = Path(__file__).resolve().parents[1]
LOCAL_CONFIG_PATH = ROOT / "scripts" / "stageS_extension_local_config.json"


@dataclass(frozen=True)
class StageSExtensionSpec:
    """
    Defines one Stage S-extension condition.

    train_method:
        Method name passed into scripts.train_continuous --method
        if the condition can be represented directly by the current parser.

    extra_cli_args:
        Any additional CLI arguments needed for this condition.

    ready:
        False means the semantic mapping has not yet been finalized.

    notes:
        Human-readable explanation of the intended condition.
    """
    name: str
    train_method: Optional[str]
    extra_cli_args: Dict[str, str] = field(default_factory=dict)
    ready: bool = False
    notes: str = ""


def _default_specs() -> List[StageSExtensionSpec]:
    """
    Default Stage S-extension conditions.

    SAFE DEFAULTS:
    - sac_cb1 and sac_cs1 are marked ready because they can be approximated
      from the current known parser methods.
    - lilac_base / lilac_cs1 / lilac_cs2 / lilac_cs3 remain unresolved until
      the user maps them explicitly in stageS_extension_local_config.json.
    """
    return [
        StageSExtensionSpec(
            name="lilac_base",
            train_method=None,
            extra_cli_args={},
            ready=False,
            notes=(
                "LILAC backbone without the new thesis constraint mechanisms. "
                "Must be mapped explicitly based on the local codebase."
            ),
        ),
        StageSExtensionSpec(
            name="lilac_cs1",
            train_method=None,
            extra_cli_args={},
            ready=False,
            notes=(
                "LILAC + stationary counterpart of Cb1. "
                "Must be mapped explicitly based on the local codebase."
            ),
        ),
        StageSExtensionSpec(
            name="lilac_cs2",
            train_method=None,
            extra_cli_args={},
            ready=False,
            notes=(
                "LILAC + stationary counterpart of Cb2. "
                "Must be mapped explicitly based on the local codebase."
            ),
        ),
        StageSExtensionSpec(
            name="lilac_cs3",
            train_method=None,
            extra_cli_args={},
            ready=False,
            notes=(
                "LILAC + stationary counterpart of Cb3. "
                "Must be mapped explicitly based on the local codebase."
            ),
        ),
        StageSExtensionSpec(
            name="sac_cb1",
            train_method="cb",
            extra_cli_args={},
            ready=True,
            notes=(
                "Representative SAC portability test for proactive constraint type 1. "
                "Uses current parser method 'cb'."
            ),
        ),
        StageSExtensionSpec(
            name="sac_cs1",
            train_method="fixed_full_A",
            extra_cli_args={},
            ready=True,
            notes=(
                "Representative SAC stationary counterpart for portability testing. "
                "Uses current parser method 'fixed_full_A'."
            ),
        ),
    ]


def _load_local_overrides() -> Dict[str, dict]:
    """
    Load user/project-specific overrides from JSON if present.

    Expected format:
    {
      "lilac_base": {
        "train_method": "cb",
        "extra_cli_args": {"some_flag": "some_value"},
        "ready": true,
        "notes": "..."
      },
      ...
    }
    """
    if not LOCAL_CONFIG_PATH.exists():
        return {}

    data = json.loads(LOCAL_CONFIG_PATH.read_text(encoding="utf-8"))
    if not isinstance(data, dict):
        raise ValueError(
            f"{LOCAL_CONFIG_PATH} must contain a JSON object mapping method names to configs."
        )
    return data


def _apply_overrides(
    specs: List[StageSExtensionSpec],
    overrides: Dict[str, dict],
) -> List[StageSExtensionSpec]:
    result: List[StageSExtensionSpec] = []

    for spec in specs:
        cfg = overrides.get(spec.name)
        if cfg is None:
            result.append(spec)
            continue

        if not isinstance(cfg, dict):
            raise ValueError(
                f"Override for '{spec.name}' must be a JSON object in {LOCAL_CONFIG_PATH}."
            )

        train_method = cfg.get("train_method", spec.train_method)
        extra_cli_args = cfg.get("extra_cli_args", spec.extra_cli_args)
        ready = cfg.get("ready", spec.ready)
        notes = cfg.get("notes", spec.notes)

        if extra_cli_args is None:
            extra_cli_args = {}
        if not isinstance(extra_cli_args, dict):
            raise ValueError(
                f"'extra_cli_args' for '{spec.name}' must be an object in {LOCAL_CONFIG_PATH}."
            )

        result.append(
            StageSExtensionSpec(
                name=spec.name,
                train_method=train_method,
                extra_cli_args={str(k): str(v) for k, v in extra_cli_args.items()},
                ready=bool(ready),
                notes=str(notes),
            )
        )

    return result


def get_stage_s_extension_specs() -> List[StageSExtensionSpec]:
    specs = _default_specs()
    overrides = _load_local_overrides()
    return _apply_overrides(specs, overrides)


def get_stage_s_extension_spec_map() -> Dict[str, StageSExtensionSpec]:
    specs = get_stage_s_extension_specs()
    return {spec.name: spec for spec in specs}