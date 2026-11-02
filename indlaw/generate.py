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
    "Reply with at most 3 bullet lines. Each line has: a short claim in your own words, the passage number "
    "in brackets, then an exact quotation of 5 to 30 words copied character for character from that passage, "
    "in double quotes. Example: - The court held that X. [2] \"words copied exactly from passage 2\". "
    "Only say the court held something if the passage says the court held it. If a passage reports an argument, "
    "a contention or a view that the court rejected, say that in the claim. "
    "Do not answer yes or no unless a passage states it. "
    "If no passage answers the question, reply exactly: INSUFFICIENT. "
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
class Statement:
    claim: str
    passage: int | None
    quote: str
    status: str  # verified_quote, uncited, invalid, no_quote, quote_not_found, off_point
    warning: str = ""


@dataclass
class RagAnswer:
    mode: str  # "rag" or "abstained"
    text: str
    sources: list[Chunk] = field(default_factory=list)
    statements: list[Statement] = field(default_factory=list)

    @property
    def shown(self) -> list[Statement]:
        return [st for st in self.statements if st.status == "verified_quote"]

    @property
    def removed(self) -> list[Statement]:
        return [st for st in self.statements if st.status != "verified_quote"]


REASONS = {
    "uncited": "cites no passage",
    "invalid": "cites a passage number that was not retrieved",
    "no_quote": "gives no quotation",
    "quote_not_found": "its quotation does not appear in the cited passage",
    "off_point": "its claim shares too few words with its quotation",
}
CUES = ("rejected", "rejecting", "contended", "contention", "argued", "argument", "submitted", "urged",
        "negatived", "not accepted", "dismissed")
MIN_OVERLAP = 0.4


def _norm(text: str) -> str:
    text = text.replace("\u2018", "'").replace("\u2019", "'").replace("\u201c", '"').replace("\u201d", '"')
    return re.sub(r"\s+", " ", text).strip().lower()


def _quote_in(quote: str, passage: str) -> bool:
    fragments = [f.strip(" .,;:-") for f in re.split(r"\.\.\.|\u2026", quote)]
    fragments = [f for f in fragments if f]
    hay = _norm(passage)
    return bool(fragments) and all(len(f.split()) >= 3 and _norm(f) in hay for f in fragments)


def parse_statements(text: str, sources: list[Chunk]) -> list[Statement]:
    """Turn a model reply into checked statements.

    A statement is "verified_quote" only when it cites a retrieved passage, gives a quotation that is
    found verbatim in that passage, and its claim shares enough words with the quotation. This proves the
    words exist in the source. It does not prove the claim follows from them.
    """
    out = []
    for line in text.splitlines():
        line = re.sub(r"^[\s\-*\u2022\d.)]+", "", line).strip()
        if len(tokens(line)) < 4:
            continue
        numbers = [int(n) for n in re.findall(r"\[(\d+)\]", line)]
        quotes = re.findall(r'["\u201c]([^"\u201c\u201d]{12,})["\u201d]', line)
        claim = re.sub(r'\s+', ' ', re.sub(r'\[\d+\]|["\u201c][^"\u201c\u201d]*["\u201d]', '', line)).strip(" :-")
        quote = quotes[0].strip() if quotes else ""
        n = numbers[0] if numbers else None
        if not numbers:
            status = "uncited"
        elif any(not 1 <= x <= len(sources) for x in numbers):
            status = "invalid"
        elif not quote:
            status = "no_quote"
        elif not all(_quote_in(quote, sources[x - 1].text) for x in numbers[:1]):
            status = "quote_not_found"
        else:
            claim_terms, quote_terms = set(tokens(claim)), set(tokens(quote))
            overlap = len(claim_terms & quote_terms) / len(claim_terms) if claim_terms else 0.0
            status = "verified_quote" if overlap >= MIN_OVERLAP else "off_point"
        warning = ""
        if status == "verified_quote" and any(c in _norm(sources[n - 1].text) for c in CUES):
            warning = ("This passage mentions an argument or a rejected view. Read the quotation in context: "
                       "it may not be the court's own holding.")
        out.append(Statement(claim, n, quote, status, warning))
    return out


def answer_rag(index: Index, question: str, generator, k: int = 4,
               min_coverage: float = MIN_COVERAGE) -> RagAnswer:
    hits = [c for c, _ in index.search(question, k=k)]
    terms = set(tokens(question))
    if not hits or max(coverage(terms, c) for c in hits) < min_coverage:
        return RagAnswer("abstained", "The loaded judgments do not appear to cover this question.", hits)
    reply = generator.complete(build_prompt(question, hits))
    if reply.strip().rstrip(".").upper() == "INSUFFICIENT":
        return RagAnswer("abstained", "The model found the retrieved passages insufficient.", hits)
    statements = parse_statements(reply, hits)
    if not any(st.status == "verified_quote" for st in statements):
        return RagAnswer("abstained", "The model gave no statement whose quotation could be found in a cited passage, "
                         "so no answer is shown.", hits, statements)
    return RagAnswer("rag", reply, hits, statements)
