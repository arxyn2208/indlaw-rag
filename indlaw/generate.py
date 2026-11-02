"""Grounded answer generation over retrieved passages, with citation checking.

A generator is any object with ``complete(prompt: str) -> str``. Two are provided:

* ``ChatEndpoint``: any OpenAI-compatible chat endpoint. Ollama running locally
  (``http://localhost:11434/v1``) needs no key. Hosted services with a free tier
  (Groq, Gemini, OpenRouter) use the same protocol and need your own key.
* ``LocalGGUF``: a quantised model file run in-process with llama-cpp-python.

The prompt tells the model to use only the numbered passages and to cite them
as [n]. ``check_citations`` then verifies the output instead of trusting it.
"""

from __future__ import annotations

import json
import os
import re
import time
import urllib.error
import urllib.request
from dataclasses import dataclass, field

from .answer import MIN_COVERAGE, coverage
from .index import Index
from .text import Chunk, tokens

SYSTEM = (
    "You answer questions about Indian court judgments using ONLY the numbered passages given. "
    "Cite every sentence with the passage numbers it relies on, like [1] or [2][3]. "
    "If the passages do not answer the question, reply exactly: INSUFFICIENT. "
    "Do not use outside knowledge. This is not legal advice."
)


def build_prompt(question: str, chunks: list[Chunk]) -> str:
    passages = "\n\n".join(f"[{i}] ({c.title}, {c.label}) {c.text}" for i, c in enumerate(chunks, start=1))
    return f"{SYSTEM}\n\nPassages:\n{passages}\n\nQuestion: {question}\nAnswer:"


class GenerationError(RuntimeError):
    """The model service failed after retries."""


class ChatEndpoint:
    def __init__(self, base_url: str | None = None, model: str | None = None, api_key: str | None = None,
                 timeout: int = 120, retries: int = 3, backoff: float = 2.0):
        self.base_url = (base_url or os.getenv("INDLAW_LLM_URL", "http://localhost:11434/v1")).rstrip("/")
        self.model = model or os.getenv("INDLAW_LLM_MODEL", "llama3.1")
        self.api_key = api_key or os.getenv("INDLAW_LLM_KEY", "")
        self.timeout = timeout
        self.retries = retries
        self.backoff = backoff

    def complete(self, prompt: str) -> str:
        body = json.dumps({"model": self.model, "temperature": 0,
                           "messages": [{"role": "user", "content": prompt}]}).encode()
        headers = {"Content-Type": "application/json"}
        if self.api_key:
            headers["Authorization"] = f"Bearer {self.api_key}"
        req = urllib.request.Request(f"{self.base_url}/chat/completions", body, headers)
        for attempt in range(self.retries + 1):
            try:
                with urllib.request.urlopen(req, timeout=self.timeout) as resp:
                    return json.load(resp)["choices"][0]["message"]["content"].strip()
            except urllib.error.HTTPError as e:
                if e.code in (429, 500, 502, 503, 504) and attempt < self.retries:
                    time.sleep(2 ** attempt * self.backoff)
                    continue
                raise GenerationError(f"The language model service returned HTTP {e.code}. Try again shortly.") from e
            except (urllib.error.URLError, TimeoutError) as e:
                if attempt < self.retries:
                    time.sleep(2 ** attempt * self.backoff)
                    continue
                raise GenerationError(f"Could not reach the language model service: {e}") from e
        raise GenerationError("The language model service did not respond.")


class LocalGGUF:
    def __init__(self, path: str, n_ctx: int = 4096):
        from llama_cpp import Llama  # optional dependency

        self.llm = Llama(model_path=path, n_ctx=n_ctx, verbose=False)

    def complete(self, prompt: str) -> str:
        out = self.llm.create_chat_completion(messages=[{"role": "user", "content": prompt}],
                                              temperature=0, max_tokens=400)
        return out["choices"][0]["message"]["content"].strip()


def generator_from_env():
    """INDLAW_GGUF=path uses a local model file; INDLAW_LLM_URL / _MODEL / _KEY use a chat endpoint."""
    gguf = os.getenv("INDLAW_GGUF")
    if gguf:
        return LocalGGUF(gguf)
    if os.getenv("INDLAW_LLM_URL") or os.getenv("INDLAW_LLM_MODEL"):
        return ChatEndpoint()
    return None


@dataclass
class RagAnswer:
    mode: str  # "rag", "abstained" or "extractive"
    text: str
    sources: list[Chunk] = field(default_factory=list)
    cited: list[int] = field(default_factory=list)
    uncited_sentences: list[str] = field(default_factory=list)
    bad_citations: list[int] = field(default_factory=list)


def check_citations(text: str, n_passages: int) -> tuple[list[int], list[str], list[int]]:
    """``(valid cited numbers, sentences with no citation, out-of-range numbers)``."""
    cited, bad, uncited = [], [], []
    for sentence in re.split(r"(?<=[.;?!])\s+(?=[A-Z\[])", text.strip()):
        numbers = [int(n) for n in re.findall(r"\[(\d+)\]", sentence)]
        if not numbers and len(tokens(sentence)) >= 4:
            uncited.append(sentence)
        for n in numbers:
            (cited if 1 <= n <= n_passages else bad).append(n)
    return sorted(set(cited)), uncited, sorted(set(bad))


def answer_rag(index: Index, question: str, generator, k: int = 4,
               min_coverage: float = MIN_COVERAGE) -> RagAnswer:
    hits = [c for c, _ in index.search(question, k=k)]
    terms = set(tokens(question))
    if not hits or max(coverage(terms, c) for c in hits) < min_coverage:
        return RagAnswer("abstained", "The loaded judgments do not appear to cover this question.", hits)
    reply = generator.complete(build_prompt(question, hits))
    if reply.strip().rstrip(".").upper() == "INSUFFICIENT":
        return RagAnswer("abstained", "The model found the retrieved passages insufficient.", hits)
    cited, uncited, bad = check_citations(reply, len(hits))
    return RagAnswer("rag", reply, hits, cited, uncited, bad)


def annotate(text: str, sources: list[Chunk]) -> list[tuple[str, list[Chunk], str]]:
    """Split an answer into sentences and attach the passages each one cites.

    Status is "cited" (every number is valid), "invalid" (a number does not match a
    retrieved passage) or "uncited" (no citation at all, so unverified).
    """
    rows = []
    for sentence in re.split(r"(?<=[.;?!])\s+(?=[A-Z\[])", text.strip()):
        if not sentence.strip():
            continue
        numbers = [int(n) for n in re.findall(r"\[(\d+)\]", sentence)]
        valid = [sources[n - 1] for n in numbers if 1 <= n <= len(sources)]
        if not numbers:
            status = "uncited"
        elif len(valid) < len(numbers):
            status = "invalid"
        else:
            status = "cited"
        rows.append((sentence, valid, status))
    return rows
