"""
Experiment configuration presets for different ablation studies and model variants.

Defines named configurations that override the base config.yaml for specific
experiment runs (e.g., no_fairness, no_diversity, etc.).
"""
from dataclasses import dataclass, field
from typing import Dict, Any

@dataclass
class ExperimentConfig:
    """Base experiment configuration."""
    name: str = "default"
    loss_overrides: Dict[str, Any] = field(default_factory=dict)
    training_overrides: Dict[str, Any] = field(default_factory=dict)
    prompt_variant: str = "default"

# Pre-defined experiment presets
EXPERIMENT_PRESETS = {
    "full_model": ExperimentConfig(
        name="full_model",
        loss_overrides={},
        training_overrides={},
        prompt_variant="default",
    ),
    "ablation_no_fairness": ExperimentConfig(
        name="no_fairness",
        loss_overrides={"adv_weight": 0.0},
        prompt_variant="default",
    ),
    "ablation_no_diversity": ExperimentConfig(
        name="no_diversity",
        loss_overrides={"div_weight": 0.0},
    ),
    "ablation_no_debiasing": ExperimentConfig(
        name="no_debiasing",
        loss_overrides={"pop_weight": 0.0},
    ),
    "ablation_no_augmentation": ExperimentConfig(
        name="no_augmentation",
        loss_overrides={"aug_weight": 0.0},
    ),
    "prompt_variant_gender_neutral": ExperimentConfig(
        name="gender_neutral_prompt",
        prompt_variant="gender_neutral",
    ),
    "prompt_variant_accessibility": ExperimentConfig(
        name="accessibility_prompt",
        prompt_variant="accessibility",
    ),
}

def get_experiment_config(name: str) -> ExperimentConfig:
    """Retrieve a named experiment configuration."""
    if name not in EXPERIMENT_PRESETS:
        raise ValueError(f"Unknown experiment: {name}. Available: {list(EXPERIMENT_PRESETS.keys())}")
    return EXPERIMENT_PRESETS[name]