"""
Augmentation pipeline: generates inclusive item descriptions using LLMs
(Prompt A) and caches results.
"""
import os
import json
import logging
from typing import Dict, List, Optional, Callable
from tqdm import tqdm

from .prompt_templates import format_prompt

logger = logging.getLogger(__name__)


class ItemDescriptionAugmenter:
    """
    Augments item metadata with inclusive descriptions using an LLM.
    Supports multiple backends: DeepSeek, Claude, or a local model.
    """

    def __init__(
            self,
            llm_backend: Callable[[str, str], str],  # function(system_prompt, user_prompt) -> response
            prompt_variant: str = "default",
            cache_path: Optional[str] = None,
    ):
        self.llm_backend = llm_backend
        self.prompt_variant = prompt_variant
        self.cache: Dict[str, str] = {}
        self.cache_path = cache_path
        if cache_path and os.path.exists(cache_path):
            with open(cache_path, 'r') as f:
                self.cache = json.load(f)

    def augment(self, item_id: str, original_desc: str) -> str:
        """Generate an inclusive description for a single item, with caching."""
        if item_id in self.cache:
            return self.cache[item_id]

        system_prompt = "You are a helpful assistant that creates inclusive product descriptions."
        user_prompt = format_prompt("prompt_a", variant=self.prompt_variant,
                                    original_description=original_desc)
        try:
            augmented = self.llm_backend(system_prompt, user_prompt)
            self.cache[item_id] = augmented
            if self.cache_path:
                self._save_cache()
            return augmented
        except Exception as e:
            logger.warning(f"Augmentation failed for {item_id}: {e}")
            return original_desc  # fallback

    def augment_batch(self, item_texts: Dict[str, str], verbose: bool = True) -> Dict[str, str]:
        """Augment a batch of items, returning updated item_id -> augmented text."""
        results = {}
        items = list(item_texts.items())
        for item_id, text in tqdm(items, desc="Augmenting items", disable=not verbose):
            results[item_id] = self.augment(item_id, text)
        return results

    def _save_cache(self):
        """Save cache to JSON."""
        os.makedirs(os.path.dirname(self.cache_path), exist_ok=True)
        with open(self.cache_path, 'w') as f:
            json.dump(self.cache, f, indent=2)
        logger.info(f"Augmentation cache saved to {self.cache_path}")