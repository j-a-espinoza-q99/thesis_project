"""
LLM interaction helpers: system/user message formatting for different providers.
Provides a unified wrapper to call DeepSeek, Anthropic, or local models for
prompt-based augmentation and ranking.
"""
import os
from typing import List, Dict, Optional, Union

# DeepSeek
from openai import OpenAI

# Anthropic
import anthropic

import logging

logger = logging.getLogger(__name__)


class LLMBackend:
    """Factory for LLM backends based on provider name."""

    @staticmethod
    def create(provider: str, api_key: Optional[str] = None, model: Optional[str] = None):
        if provider == "deepseek":
            return DeepSeekBackend(api_key=api_key, model=model or "deepseek-v4-pro")
        elif provider == "claude":
            return ClaudeBackend(api_key=api_key, model=model or "claude-sonnet-4-20250514")
        else:
            raise ValueError(f"Unsupported LLM provider: {provider}")

class DeepSeekBackend:
    """DeepSeek API backend."""
    def __init__(self, api_key: Optional[str] = None, model: str = "deepseek-v4-pro"):
        self.api_key = api_key or os.environ.get("DEEPSEEK_API_KEY")
        self.model = model
        self.client = OpenAI(api_key=self.api_key, base_url="https://api.deepseek.com")

    def generate(self, messages: List[Dict[str, str]], max_tokens: int = 500, temperature: float = 0.7) -> str:
        resp = self.client.chat.completions.create(
            model=self.model,
            messages=messages,
            max_tokens=max_tokens,
            temperature=temperature,
        )
        return resp.choices[0].message.content

class ClaudeBackend:
    """Anthropic Claude backend."""
    def __init__(self, api_key: Optional[str] = None, model: str = "claude-sonnet-4-20250514"):
        self.api_key = api_key or os.environ.get("ANTHROPIC_API_KEY")
        self.model = model
        self.client = anthropic.Anthropic(api_key=self.api_key)

    def generate(self, system_prompt: str, user_message: str, max_tokens: int = 500, temperature: float = 0.7) -> str:
        resp = self.client.messages.create(
            model=self.model,
            max_tokens=max_tokens,
            temperature=temperature,
            system=system_prompt,
            messages=[{"role": "user", "content": user_message}],
        )
        return resp.content[0].text