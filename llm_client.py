from __future__ import annotations

from dataclasses import dataclass

from config import ConfigurationError, Settings


class LLMCallError(RuntimeError):
    """Raised when the OpenAI-compatible LLM call fails."""


@dataclass
class LLMClient:
    settings: Settings

    def complete(
        self,
        system_prompt: str,
        user_prompt: str,
        *,
        temperature: float = 0.2,
        max_tokens: int | None = None,
    ) -> str:
        self.settings.require_llm()
        try:
            from openai import OpenAI
        except ImportError as exc:
            raise ConfigurationError("Missing dependency: openai. Install requirements.txt first.") from exc

        client = OpenAI(base_url=self.settings.llm_base_url, api_key=self.settings.llm_api_key)
        kwargs = {
            "model": self.settings.llm_model,
            "messages": [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt},
            ],
            "temperature": temperature,
        }
        if max_tokens is not None:
            kwargs["max_tokens"] = max_tokens

        try:
            response = client.chat.completions.create(**kwargs)
        except Exception as exc:  # pragma: no cover - depends on remote LLM runtime.
            raise LLMCallError(f"LLM request failed: {exc}") from exc

        content = response.choices[0].message.content
        if not content:
            raise LLMCallError("LLM returned an empty response.")
        return content.strip()
