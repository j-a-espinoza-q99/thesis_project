"""
Anthropic Claude + Voyage AI integration for recommendation.

Uses Voyage AI for embedding generation and Claude for chat-based ranking,
following Anthropic's recommended approach.
"""
import os
import time
import logging
from typing import List, Optional, Dict

import numpy as np
import voyageai
from anthropic import Anthropic

logger = logging.getLogger(__name__)


class ClaudeVoyageRecommender:
    """
    Claude + Voyage AI recommender.
    Uses Voyage AI embeddings (Anthropic's recommended embedding provider)
    and Claude for ranking/fair candidate selection.
    """

    def __init__(
            self,
            anthropic_api_key: Optional[str] = None,
            voyage_api_key: Optional[str] = None,
            voyage_model: str = "voyage-3",
            claude_model: str = "claude-sonnet-4-20250514",
            max_retries: int = 3,
            retry_delay: float = 1.0,
    ):
        self.anthropic_api_key = anthropic_api_key or os.environ.get("ANTHROPIC_API_KEY")
        self.voyage_api_key = voyage_api_key or os.environ.get("VOYAGE_API_KEY")

        if not self.anthropic_api_key:
            raise ValueError("ANTHROPIC_API_KEY not set")
        if not self.voyage_api_key:
            raise ValueError("VOYAGE_API_KEY not set")

        # Initialize Voyage AI client
        self.voyage_client = voyageai.Client(api_key=self.voyage_api_key)
        self.voyage_model = voyage_model

        # Initialize Anthropic client
        self.anthropic_client = Anthropic(api_key=self.anthropic_api_key)
        self.claude_model = claude_model

        self.max_retries = max_retries
        self.retry_delay = retry_delay

    def _retry(self, func, *args, **kwargs):
        """Retry with exponential backoff."""
        for attempt in range(self.max_retries):
            try:
                return func(*args, **kwargs)
            except Exception as e:
                if attempt == self.max_retries - 1:
                    raise
                logger.warning(f"API call failed (attempt {attempt + 1}): {e}")
                time.sleep(self.retry_delay * (2 ** attempt))

    def encode(self, texts: List[str], input_type: str = "document") -> np.ndarray:
        """
        Generate Voyage AI embeddings for texts.

        Args:
            texts: List of text strings
            input_type: "query" or "document" (affects embedding behavior)
        """
        def _encode():
            result = self.voyage_client.embed(
                texts=texts,
                model=self.voyage_model,
                input_type=input_type,
            )
            return np.array(result.embeddings, dtype=np.float32)

        return self._retry(_encode)

    def encode_items_batch(
            self,
            item_texts: List[str],
            batch_size: int = 128,
    ) -> np.ndarray:
        """Encode item metadata texts in batches."""
        all_embeddings = []
        for i in range(0, len(item_texts), batch_size):
            batch = item_texts[i:i + batch_size]
            embeddings = self.encode(batch, input_type="document")
            all_embeddings.append(embeddings)
        return np.concatenate(all_embeddings, axis=0)

    def encode_query(self, query: str) -> np.ndarray:
        """Encode a single query."""
        return self.encode([query], input_type="query")[0]

    def rank_items_with_claude(
            self,
            query: str,
            candidate_items: List[str],
            user_history: Optional[List[str]] = None,
            use_fair_prompt: bool = True,
    ) -> List[int]:
        """
        Use Claude to rank candidate items with fairness-aware prompting.
        """
        # Build candidates string
        candidates_str = "\n".join([
            f"[{i}] {item[:200]}..." if len(item) > 200 else f"[{i}] {item}"
            for i, item in enumerate(candidate_items)
        ])

        history_str = ", ".join(user_history[:10]) if user_history else "None available"

        system_prompt = (
            "You are a fair, unbiased recommendation system. You help users find "
            "relevant items while ensuring diversity and avoiding stereotypes. "
            "Apply gentle debiasing: slightly reduce the weight of extremely popular "
            "items, ensure diverse representation, and avoid reinforcing demographic "
            "stereotypes based on user history."
        )

        user_prompt = f"""We want to make a fair recommendation.

User's purchase history: {history_str}

User's current need: {query}

Candidate items:
{candidates_str}

Rank the top 10 most relevant items by their index numbers.
Your ranking should:
1. Prioritize relevance to the user's need
2. Include diverse items, not just the most popular ones
3. Avoid reinforcing stereotypes based on the user's history
4. Balance popularity with novelty

Return ONLY a comma-separated list of indices, e.g.: 3,7,1,15,4"""

        def _rank():
            response = self.anthropic_client.messages.create(
                model=self.claude_model,
                max_tokens=200,
                temperature=0.3,
                system=system_prompt,
                messages=[{"role": "user", "content": user_prompt}],
            )
            return response.content[0].text

        result = self._retry(_rank)

        # Parse indices
        import re
        numbers = re.findall(r'\d+', result)
        indices = [int(n) for n in numbers if 0 <= int(n) < len(candidate_items)]

        # Remove duplicates
        seen = set()
        unique = []
        for idx in indices:
            if idx not in seen:
                seen.add(idx)
                unique.append(idx)

        return unique[:10] if unique else list(range(min(10, len(candidate_items))))

    def generate_inclusive_description(
            self,
            original_description: str,
    ) -> str:
        """
        Generate an inclusive item description using Claude (Prompt A).
        """
        system_prompt = (
            "You are an expert at creating inclusive product descriptions that "
            "appeal to diverse audiences without stereotypes."
        )

        user_prompt = f"""The original product description: '{original_description}'

Create an inclusive description that:
1. Highlights universally appealing features
2. Avoids gender, age, and cultural stereotypes
3. Uses inclusive, people-first language
4. Describes accessibility and universal design aspects

Return ONLY the new description."""

        def _generate():
            response = self.anthropic_client.messages.create(
                model=self.claude_model,
                max_tokens=500,
                temperature=0.7,
                system=system_prompt,
                messages=[{"role": "user", "content": user_prompt}],
            )
            return response.content[0].text

        return self._retry(_generate)