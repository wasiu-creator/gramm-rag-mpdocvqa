"""
src/generation/vlm_client.py
─────────────────────────────
API wrapper for VLM generation calls in GraMM-RAG.

Supports:
  - Together.ai  → Llama-3.3-70B-Instruct-Turbo  (serverless, ~$0.88/1M tokens)
  - OpenAI       → GPT-4o-mini               (cheap dev fallback)

LIMIT: max_questions_per_benchmark=100 in config/base.yaml controls total calls.
← EXTEND: set to None in base.yaml for full paper runs.

API keys read from environment variables:
  TOGETHER_API_KEY
  OPENAI_API_KEY
"""

import os
import logging
import time
from typing import Optional

logger = logging.getLogger(__name__)


class VLMClient:
    """
    Unified client for Together.ai and OpenAI generation APIs.
    Automatically retries on rate-limit errors (429) with exponential backoff.
    """

    def __init__(
        self,
        provider: str = "together",           # "together" | "openai"
        together_model: str = "meta-llama/Llama-3.3-70B-Instruct-Turbo",
        openai_model: str = "gpt-4o-mini",
        max_retries: int = 3,
        retry_delay: float = 2.0,
    ):
        self.provider = provider
        self.together_model = together_model
        self.openai_model = openai_model
        self.max_retries = max_retries
        self.retry_delay = retry_delay
        self._client = None
        self._init_client()

    def _init_client(self):
        if self.provider == "together":
            api_key = os.environ.get("TOGETHER_API_KEY", "")
            if not api_key:
                # Auto-fallback to extractive when no key is set
                logger.warning(
                    "TOGETHER_API_KEY not set — falling back to extractive local VLM. "
                    "# ← EXTEND: set TOGETHER_API_KEY for Llama-3.3-70B paper results."
                )
                self.provider = "extractive"
                return
            try:
                from together import Together
                self._client = Together(api_key=api_key)
                logger.info(f"Together.ai client ready: {self.together_model}")
            except ImportError:
                raise ImportError("Run: pip install together")

        elif self.provider == "openai":
            api_key = os.environ.get("OPENAI_API_KEY", "")
            if not api_key:
                logger.warning(
                    "OPENAI_API_KEY not set — falling back to extractive local VLM."
                )
                self.provider = "extractive"
                return
            try:
                from openai import OpenAI
                self._client = OpenAI(api_key=api_key)
                logger.info(f"OpenAI client ready: {self.openai_model}")
            except ImportError:
                raise ImportError("Run: pip install openai")

        elif self.provider == "extractive":
            logger.info("Extractive local VLM: no API calls, uses retrieved context.")

    def generate(
        self,
        prompt: str,
        temperature: float = 0.0,
        max_tokens: int = 512,
    ) -> str:
        """
        Generate a response for the given prompt.

        Retries up to max_retries times on API errors.
        Sleep 0.5s between calls to avoid rate limits.

        Args:
            prompt:      Full text prompt (from prompt_builder.build_prompt).
            temperature: Sampling temperature (0.0 = greedy).
            max_tokens:  Max output tokens.

        Returns:
            Generated answer string.
        """
        if self.provider == "extractive":
            return self._extractive_generate(prompt)

        for attempt in range(self.max_retries):
            try:
                if self.provider == "together":
                    return self._together_generate(prompt, temperature, max_tokens)
                else:
                    return self._openai_generate(prompt, temperature, max_tokens)

            except Exception as e:
                if attempt < self.max_retries - 1:
                    wait = self.retry_delay * (2 ** attempt)
                    logger.warning(f"API error (attempt {attempt+1}): {e}. Retrying in {wait}s")
                    time.sleep(wait)
                else:
                    logger.error(f"VLM generation failed after {self.max_retries} attempts: {e}")
                    return "[GENERATION_ERROR]"

        return "[GENERATION_ERROR]"

    def _together_generate(self, prompt: str, temperature: float, max_tokens: int) -> str:
        resp = self._client.chat.completions.create(
            model=self.together_model,
            messages=[{"role": "user", "content": prompt}],
            temperature=temperature,
            max_tokens=max_tokens,
        )
        time.sleep(0.5)   # rate-limit courtesy delay
        return resp.choices[0].message.content.strip()

    def _openai_generate(self, prompt: str, temperature: float, max_tokens: int) -> str:
        resp = self._client.chat.completions.create(
            model=self.openai_model,
            messages=[{"role": "user", "content": prompt}],
            temperature=temperature,
            max_tokens=max_tokens,
        )
        time.sleep(0.5)
        return resp.choices[0].message.content.strip()

    def _extractive_generate(self, prompt: str) -> str:
        """
        Local extractive fallback — no API key required.
        Parses the structured prompt (built by prompt_builder) and finds the
        most keyword-overlapping text snippet for the question.
        # ← EXTEND: replace with real VLM (set TOGETHER_API_KEY) for paper results.
        """
        import re
        # Extract question
        question = ""
        for line in prompt.split("\n"):
            if line.startswith("Question:"):
                question = line[len("Question:"):].strip()
                break

        # Extract text blocks between the === separators
        body_match = re.search(r"={3,}(.+?)={3,}", prompt, re.DOTALL)
        context = body_match.group(1).strip() if body_match else ""

        # Parse [TYPE | Page N] blocks — extract only the text content
        text_snippets = []
        for block in re.split(r"\n---\n", context):
            block = block.strip()
            # Remove the [TYPE | Page N] header line
            lines = block.split("\n")
            content_lines = [l for l in lines if not l.startswith("[") and l.strip()]
            if content_lines:
                text_snippets.append(" ".join(content_lines))

        if not text_snippets:
            return ""

        # Score each snippet by keyword overlap with question
        stopwords = {"the", "a", "an", "is", "are", "was", "were", "what",
                     "which", "how", "when", "who", "of", "in", "on", "at",
                     "to", "for", "with", "this", "that", "it", "its", "do"}
        q_words = {w for w in question.lower().split() if w not in stopwords and len(w) > 2}

        best_text, best_score = "", 0
        for snippet in text_snippets:
            score = sum(1 for w in q_words if w in snippet.lower())
            if score > best_score:
                best_score, best_text = score, snippet
            # Break ties with shorter snippets (more specific)
            elif score == best_score and 0 < len(snippet) < len(best_text):
                best_text = snippet

        return best_text[:200] if best_text else (text_snippets[0][:200] if text_snippets else "")
