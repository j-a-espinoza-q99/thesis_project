"""
DeepSeek API integration for recommendation embeddings and ranking.

Uses DeepSeek's OpenAI-compatible API:
- Embeddings: /v1/embeddings endpoint
- Chat: /v1/chat/completions for prompt-based ranking
"""
import os
import time
import numpy as np
import torch
from typing import List, Optional, Dict
from openai import OpenAI
import logging

logger = logging.getLogger(__name__)


class DeepSeekRecommender:
    """
    DeepSeek-based recommender using API embeddings and chat completion.
    """

    def __init__(
            self,
            api_key: Optional[str] = None,
            api_base_url: str = "https://api.deepseek.com",
            embedding_model: str = "deepseek-chat",
            chat_model: str = "deepseek-v4-pro",
            max_retries: int = 3,
            retry_delay: float = 1.0,
            device: str = "cuda",
    ):
        self.api_key = api_key or os.environ.get("DEEPSEEK_API_KEY")
        if not self.api_key:
            raise ValueError("DEEPSEEK_API_KEY not set")

        self.client = OpenAI(
            api_key=self.api_key,
            base_url=api_base_url,
        )
        self.embedding_model = embedding_model
        self.chat_model = chat_model
        self.max_retries = max_retries
        self.retry_delay = retry_delay
        self.device = device

    def _retry_request(self, func, *args, **kwargs):
        """Retry API requests with exponential backoff."""
        for attempt in range(self.max_retries):
            try:
                return func(*args, **kwargs)
            except Exception as e:
                if attempt == self.max_retries - 1:
                    raise
                logger.warning(f"API request failed (attempt {attempt + 1}): {e}")
                time.sleep(self.retry_delay * (2 ** attempt))

    def encode(
            self,
            texts: List[str],
            batch_size: int = 32,
    ) -> np.ndarray:
        """
        Encode texts using DeepSeek embedding API.

        Note: DeepSeek embeddings are accessed via the chat model's embedding layer.
        """
        all_embeddings = []

        for i in range(0, len(texts), batch_size):
            batch = texts[i:i + batch_size]

            def _encode_batch():
                response = self.client.embeddings.create(
                    model=self.embedding_model,
                    input=batch,
                )
                return [item.embedding for item in response.data]

            embeddings = self._retry_request(_encode_batch)
            all_embeddings.extend(embeddings)

        return np.array(all_embeddings, dtype=np.float32)

    def chat(
            self,
            messages: List[Dict[str, str]],
            max_tokens: int = 1024,
            temperature: float = 0.3,
    ) -> str:
        """Make a chat completion request."""
        def _chat():
            response = self.client.chat.completions.create(
                model=self.chat_model,
                messages=messages,
                max_tokens=max_tokens,
                temperature=temperature,
            )
            return response.choices[0].message.content

        return self._retry_request(_chat)

    def rank_items(
            self,
            query: str,
            candidate_items: List[str],
            user_history: Optional[List[str]] = None,
    ) -> List[int]:
        """
        Rank candidate items using DeepSeek chat model with fairness-aware prompting.
        """
        # Build prompt
        prompt = self._build_ranking_prompt(query, candidate_items, user_history)

        messages = [
            {"role": "system", "content": "You are a fair and unbiased recommendation system. "
                                          "You help users find relevant items while ensuring diversity and fairness."},
            {"role": "user", "content": prompt},
        ]

        response = self.chat(messages, max_tokens=500, temperature=0.3)

        # Parse ranked item indices
        ranked_indices = self._parse_ranking_response(response, len(candidate_items))

        return ranked_indices

    def _build_ranking_prompt(
            self,
            query: str,
            candidate_items: List[str],
            user_history: Optional[List[str]] = None,
    ) -> str:
        """Build ranking prompt with fairness instructions."""
        prompt_parts = []

        if user_history:
            prompt_parts.append(f"User's purchase history: {', '.join(user_history[:10])}")

        prompt_parts.append(f"User's current need: {query}")
        prompt_parts.append("\nCandidate items to rank:")

        for i, item in enumerate(candidate_items):
            # Truncate long item descriptions
            item_short = item[:200] + "..." if len(item) > 200 else item
            prompt_parts.append(f"[{i}] {item_short}")

        prompt_parts.append("\nPlease rank the top 10 most relevant items by their index numbers.")
        prompt_parts.append("Ensure your ranking:")
        prompt_parts.append("1. Prioritizes relevance to the user's need")
        prompt_parts.append("2. Includes diverse item types, not just the most popular ones")
        prompt_parts.append("3. Avoids reinforcing stereotypes based on user history")
        prompt_parts.append("\nReturn ONLY a comma-separated list of indices, e.g.: 3,7,1,15,4")

        return "\n".join(prompt_parts)

    def _parse_ranking_response(self, response: str, num_candidates: int) -> List[int]:
        """Parse ranking response into list of indices."""
        try:
            # Extract numbers from response
            import re
            numbers = re.findall(r'\d+', response)
            indices = [int(n) for n in numbers if 0 <= int(n) < num_candidates]

            # Remove duplicates while preserving order
            seen = set()
            unique_indices = []
            for idx in indices:
                if idx not in seen:
                    seen.add(idx)
                    unique_indices.append(idx)

            return unique_indices[:10] if unique_indices else list(range(min(10, num_candidates)))
        except Exception as e:
            logger.warning(f"Failed to parse ranking response: {e}")
            return list(range(min(10, num_candidates)))

    def generate_augmented_description(
            self,
            original_description: str,
            prompt_template: str,
            temperature: float = 0.7,
    ) -> str:
        """
        Generate augmented item description using Prompt A template.
        """
        messages = [
            {"role": "system", "content": "You are an expert at creating inclusive and "
                                          "fair product descriptions that appeal to diverse audiences."},
            {"role": "user", "content": prompt_template.format(
                original_description=original_description
            )},
        ]

        return self.chat(messages, max_tokens=500, temperature=temperature)