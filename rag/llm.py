"""Talk to a local LLM through Ollama's HTTP API (free, runs on your machine).

Install Ollama from https://ollama.com, then `ollama pull llama3.2`.
Settings (environment variables):
- OLLAMA_URL   default http://localhost:11434
- OLLAMA_MODEL default llama3.2
"""

import os
from typing import Protocol

import httpx


class LLMError(RuntimeError):
    """The LLM could not be reached or returned an error."""


class LLM(Protocol):
    name: str

    def chat(self, system: str, user: str) -> str: ...


class OllamaLLM:
    def __init__(
        self,
        model: str | None = None,
        base_url: str | None = None,
        temperature: float = 0.1,
        timeout: float = 120.0,
    ):
        self.model = model or os.getenv("OLLAMA_MODEL", "llama3.2")
        self.base_url = (base_url or os.getenv("OLLAMA_URL", "http://localhost:11434")).rstrip("/")
        self.temperature = temperature
        self.timeout = timeout
        self.name = f"ollama:{self.model}"

    def chat(self, system: str, user: str) -> str:
        payload = {
            "model": self.model,
            "messages": [
                {"role": "system", "content": system},
                {"role": "user", "content": user},
            ],
            "stream": False,
            "options": {"temperature": self.temperature},
        }
        try:
            r = httpx.post(f"{self.base_url}/api/chat", json=payload, timeout=self.timeout)
        except httpx.ConnectError as e:
            raise LLMError(
                f"Can't reach Ollama at {self.base_url}. Start the Ollama app "
                f"(or run `ollama serve`) and run `ollama pull {self.model}`."
            ) from e
        except httpx.TimeoutException as e:
            raise LLMError(f"Ollama took longer than {self.timeout:.0f}s to answer.") from e
        if r.status_code == 404:
            raise LLMError(f"Ollama doesn't have the model '{self.model}'. Run `ollama pull {self.model}`.")
        if r.status_code >= 400:
            raise LLMError(f"Ollama returned HTTP {r.status_code}: {r.text[:300]}")
        return r.json().get("message", {}).get("content", "").strip()


def get_llm() -> LLM:
    return OllamaLLM()
