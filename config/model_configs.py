"""
Model configurations for all compared models.
"""
from dataclasses import dataclass, field
from typing import Optional, List, Dict
import yaml
import os

@dataclass
class BLAIRConfig:
    """BLAIR baseline model configuration."""
    model_name: str = "hyp1231/blair-roberta-base"
    embedding_dim: int = 768
    max_seq_length: int = 64
    pooler_type: str = "cls"
    temperature: float = 0.05
    do_mlm: bool = False
    mlm_weight: float = 0.1


@dataclass
class DeepSeekConfig:
    """DeepSeek API model configuration."""
    api_base_url: str = "https://api.deepseek.com"
    api_key: Optional[str] = None
    embedding_model: str = "deepseek-chat"
    chat_model: str = "deepseek-v4-pro"
    max_tokens: int = 4096
    embedding_dim: int = 4096

    def __post_init__(self):
        if self.api_key is None:
            self.api_key = os.environ.get("DEEPSEEK_API_KEY")


@dataclass
class ClaudeVoyageConfig:
    """Anthropic Claude + Voyage AI configuration."""
    anthropic_api_key: Optional[str] = None
    voyage_api_key: Optional[str] = None
    voyage_model: str = "voyage-3"
    embedding_dim: int = 1024
    claude_model: str = "claude-sonnet-4-20250514"
    max_tokens: int = 4096

    def __post_init__(self):
        if self.anthropic_api_key is None:
            self.anthropic_api_key = os.environ.get("ANTHROPIC_API_KEY")
        if self.voyage_api_key is None:
            self.voyage_api_key = os.environ.get("VOYAGE_API_KEY")


@dataclass
class CustomModelConfig:
    """Custom model with multi-objective loss and custom prompts."""
    backbone: str = "hyp1231/blair-roberta-base"
    embedding_dim: int = 768
    adapter_type: str = "moe"
    num_experts: int = 8
    lora_rank: int = 8
    lora_alpha: int = 16
    use_pca: bool = True
    pca_dim: int = 256


@dataclass
class LossConfig:
    """Multi-objective loss configuration."""
    bpr_weight: float = 1.0
    adv_weight: float = 0.1
    adv_hidden_dim: int = 128
    div_weight: float = 0.05
    div_topk: int = 10
    pop_weight: float = 0.1
    aug_weight: float = 0.2
    aug_temperature: float = 0.07
    protected_attribute: Optional[str] = None


@dataclass
class PromptConfig:
    """Custom prompt configuration."""
    prompt_a_temperature: float = 0.7
    prompt_b_temperature: float = 0.3
    augmentation_batch_size: int = 32
    max_augmented_chars: int = 500


def load_config(config_path: str = "config/config.yaml") -> Dict:
    """Load configuration from YAML file."""
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    return config