from __future__ import annotations

from dataclasses import dataclass, fields
from pathlib import Path
from typing import Any, Dict

import yaml


@dataclass(frozen=True)
class PersonaProfile:
    """Base behavioral priors for a customer segment."""

    visit_prob: float
    browse_prob: float
    search_prob: float
    add_to_cart_prob: float
    remove_from_cart_prob: float
    purchase_given_cart_prob: float
    purchase_given_visit_prob: float
    coupon_open_prob: float
    coupon_redeem_prob: float
    avg_order_mean: float
    avg_order_std: float
    churn_sensitivity: float
    price_sensitivity: float
    recovery_prob: float
    treatment_lift: float
    acquisition_weight: float


_PERSONA_FIELD_NAMES = frozenset(f.name for f in fields(PersonaProfile))


def _default_personas_yaml_path() -> Path:
    return Path(__file__).resolve().parents[2] / "config" / "simulator_config.yaml"


def _persona_from_mapping(name: str, raw: Dict[str, Any]) -> PersonaProfile:
    missing = _PERSONA_FIELD_NAMES - raw.keys()
    if missing:
        raise ValueError(f"Persona '{name}' missing fields: {sorted(missing)}")
    extra = set(raw.keys()) - _PERSONA_FIELD_NAMES
    if extra:
        raise ValueError(f"Persona '{name}' unknown fields: {sorted(extra)}")
    return PersonaProfile(**{f.name: raw[f.name] for f in fields(PersonaProfile)})


def load_personas_from_yaml(path: Path | None = None) -> Dict[str, PersonaProfile]:
    """Load personas from YAML. Expects top-level key `personas:`."""
    cfg_path = path or _default_personas_yaml_path()
    with cfg_path.open("r", encoding="utf-8") as f:
        doc = yaml.safe_load(f)
    if not isinstance(doc, dict):
        raise ValueError("simulator_config.yaml must be a mapping at the top level.")
    raw_personas = doc.get("personas")
    if not isinstance(raw_personas, dict) or not raw_personas:
        raise ValueError("simulator_config.yaml must define a non-empty `personas` mapping.")
    return {str(k): _persona_from_mapping(str(k), v) for k, v in raw_personas.items() if isinstance(v, dict)}


def _default_personas() -> Dict[str, PersonaProfile]:
    path = _default_personas_yaml_path()
    if not path.is_file():
        raise FileNotFoundError(
            f"Personas are loaded only from YAML. Create {path} with a `personas:` section."
        )
    try:
        return load_personas_from_yaml(path)
    except (OSError, ValueError, yaml.YAMLError) as e:
        raise RuntimeError(f"Failed to load personas from {path}: {e}") from e


DEFAULT_PERSONAS: Dict[str, PersonaProfile] = _default_personas()


def get_persona_names():
    return list(DEFAULT_PERSONAS.keys())
