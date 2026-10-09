"""
belief.py — Symbolic belief representation and knowledge base.

Uses AGM (Alchourrón-Gärdenfors-Makinson) belief revision:
  Adding a new belief about subject:predicate replaces all stale beliefs
  for that slot, so the KB always holds the most recent fact per predicate.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Optional, Set


@dataclass(frozen=True)
class Belief:
    """
    A single first-order-logic ground atom representing one fact.

    Attributes:
        subject   : Entity the belief is about, e.g. "A2".
        predicate : Predicate slot, e.g. "position", "heading", "status".
        value     : Any JSON-serializable value, e.g. (120.5, 340.2).
        confidence: [0, 1] epistemic certainty.
        timestamp : Simulation tick when this belief was created.
        source    : "SENSOR" | "COMM" | "DEDUCTION"
    """

    subject: str
    predicate: str
    value: Any
    confidence: float
    timestamp: int
    source: str


class KnowledgeBase:
    """
    AGM-compliant belief base.

    - add(belief): revise by replacing any stale same-slot belief.
    - query(subject, predicate): return the current belief or None.
    - history(subject, predicate, n): return last n beliefs (for ToM).
    """

    def __init__(self):
        # Main belief set (single-slot monotonic revision)
        self._current: dict[str, Belief] = {}
        # Per-slot history for Theory-of-Mind trajectory analysis
        self._history: dict[str, list[Belief]] = {}
        self._max_history = 30

    # ------------------------------------------------------------------ #
    # Mutation                                                             #
    # ------------------------------------------------------------------ #
    def add(self, belief: Belief) -> None:
        """AGM revision: replace stale same-slot belief, keep history."""
        key = f"{belief.subject}:{belief.predicate}"
        self._current[key] = belief
        self._history.setdefault(key, []).append(belief)
        # prune old history
        if len(self._history[key]) > self._max_history:
            self._history[key].pop(0)

    def remove_all(self, subject: str, predicate: str) -> None:
        key = f"{subject}:{predicate}"
        self._current.pop(key, None)
        self._history.pop(key, None)

    # ------------------------------------------------------------------ #
    # Query                                                                #
    # ------------------------------------------------------------------ #
    def query(self, subject: str, predicate: str) -> Optional[Belief]:
        """Return the most recent belief for this slot, or None."""
        return self._current.get(f"{subject}:{predicate}")

    def history(self, subject: str, predicate: str, n: int = 5) -> list[Belief]:
        """Return last n beliefs for this slot (oldest first)."""
        return self._history.get(f"{subject}:{predicate}", [])[-n:]

    def get_all(self) -> Set[Belief]:
        return set(self._current.values())

    def all_subjects(self) -> set[str]:
        return {b.subject for b in self._current.values()}
