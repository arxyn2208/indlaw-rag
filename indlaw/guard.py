"""Usage limits for a shared demo, so one visitor cannot use up a free model quota for everyone.

State lives in memory in one server process: a restart resets the counts. These limits cap this app's own
calls; they are not a guarantee about the provider's quota, which other uses of the same key can also spend.
"""

from __future__ import annotations

import threading
import time
from collections import deque

from .generate import GenerationError

MAX_QUESTION_CHARS = 300
MAX_CALLS_PER_QUESTION = 6  # one answer call plus one judge call per statement


class UsageGuard:
    def __init__(self, per_session: int = 8, per_hour: int = 30, per_day: int = 100, clock=time.time):
        self.per_session, self.per_hour, self.per_day = per_session, per_hour, per_day
        self.clock = clock
        self._stamps: deque[float] = deque()
        self._lock = threading.Lock()

    def check(self, question: str, session_used: int) -> str | None:
        """Return a reason the model may NOT be used, or None when it may. Does not count a use."""
        if len(question) > MAX_QUESTION_CHARS:
            return f"Questions are limited to {MAX_QUESTION_CHARS} characters."
        if session_used >= self.per_session:
            return f"This session has used its {self.per_session} model answers."
        now = self.clock()
        with self._lock:
            while self._stamps and now - self._stamps[0] > 86400:
                self._stamps.popleft()
            if len(self._stamps) >= self.per_day:
                return "The demo's daily model budget is used up."
            if sum(1 for t in self._stamps if now - t <= 3600) >= self.per_hour:
                return "The demo's hourly model budget is used up."
        return None

    def record(self) -> None:
        with self._lock:
            self._stamps.append(self.clock())


class CappedGenerator:
    """Wraps a generator so one question cannot make more than ``limit`` model calls."""

    def __init__(self, inner, limit: int = MAX_CALLS_PER_QUESTION):
        self.inner, self.limit, self.calls = inner, limit, 0

    def complete(self, prompt: str) -> str:
        if self.calls >= self.limit:
            raise GenerationError("This question reached its model call limit.")
        self.calls += 1
        return self.inner.complete(prompt)
