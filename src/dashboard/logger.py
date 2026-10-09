"""
logger.py — Structured JSON Lines event ledger.

All significant simulation events are written as one JSON object per line.
This log is the single source of truth for:
  - The Streamlit research dashboard (reads in real time via tail).
  - Post-hoc evaluation: precision/recall, ROC, replanning latency.
  - Explainability: every trust decision is traceable to a specific paradox
    with the full Bayesian math trace attached.

Event types (field "event")
----------------------------
  SIMULATION_START       — Simulation initialized.
  SIMULATION_END         — Finished with final summary stats.
  TRUST_UPDATE           — Beta trust score changed for a peer (with full trace).
  EPISTEMIC_PARADOX      — Z3 SMT or kinematic clash detected.
  DBN_STEP               — Full forward-filter math trace for one DBN update.
  REPLAN                 — D* Lite replanning event with cost delta.
  HAZARD_SNAPSHOT        — Periodic serialized hazard field snapshot.
  ATTACK_WINDOW          — Ground-truth attack active (for evaluation overlay).
  CRYPTO_FAULT           — AES-GCM tag verification failed.
"""
from __future__ import annotations

import json
import time
from pathlib import Path
from typing import Any, Dict, Optional


class StructuredEventLedger:
    """
    Append-only JSON Lines event ledger.

    Parameters
    ----------
    log_path : File path for the JSONL log (created fresh each run).
    echo     : If True, also print events to stdout (verbose mode).
    """

    def __init__(
        self,
        log_path: str = "logs/execution_ledger.jsonl",
        echo:     bool = False,
    ) -> None:
        path = Path(log_path)
        path.parent.mkdir(parents=True, exist_ok=True)
        self._file  = open(path, "w", encoding="utf-8", buffering=1)  # line-buffered
        self.echo   = echo
        self._count = 0

    # ------------------------------------------------------------------ #
    # Generic write                                                        #
    # ------------------------------------------------------------------ #
    def log(self, tick: int, event_type: str, payload: Dict[str, Any]) -> None:
        """Append one event record to the ledger."""
        record = {
            "timestamp": time.time(),
            "tick":      tick,
            "event":     event_type,
            **payload,
        }
        line = json.dumps(record, default=str)
        self._file.write(line + "\n")
        self._count += 1
        if self.echo:
            _print_colored(event_type, line)

    # ------------------------------------------------------------------ #
    # Typed convenience methods                                            #
    # ------------------------------------------------------------------ #
    def log_simulation_start(
        self,
        tick: int,
        n_agents: int,
        max_ticks: int,
        use_ecdh: bool,
        seed: int,
        cfg_summary: Optional[dict] = None,
    ) -> None:
        self.log(tick, "SIMULATION_START", {
            "n_agents":   n_agents,
            "max_ticks":  max_ticks,
            "use_ecdh":   use_ecdh,
            "seed":       seed,
            "cfg":        cfg_summary or {},
        })

    def log_simulation_end(self, tick: int, summary: dict) -> None:
        self.log(tick, "SIMULATION_END", summary)

    def log_trust_update(self, tick: int, evaluator: str, upd: dict) -> None:
        """
        Log a Beta trust update event with the full calculation trace.

        upd is the dict returned by BetaTrustEngine.update(), which now
        includes alpha, beta, grudge_tier, recovery_throttled, etc.
        """
        self.log(tick, "TRUST_UPDATE", {
            "evaluator": evaluator,
            "peer_id":   upd.get("peer_id"),
            **{k: v for k, v in upd.items() if k != "peer_id"},
        })

    def log_epistemic_paradox(
        self,
        tick:            int,
        evaluator:       str,
        peer_id:         str,
        delta:           float,
        reason:          str,
        explanation:     str,
        intent_posteriors: Dict[str, float],
        smt_reason:      str = "",
    ) -> None:
        """Log a full epistemic paradox detection event."""
        self.log(tick, "EPISTEMIC_PARADOX", {
            "evaluator":        evaluator,
            "peer_id":          peer_id,
            "delta":            round(delta, 3),
            "reason":           reason,
            "explanation":      explanation,
            "intent_posteriors": {k: round(v, 4) for k, v in intent_posteriors.items()},
            "smt_reason":       smt_reason,
            "best_intent":      max(intent_posteriors, key=intent_posteriors.get, default="none")
                                if intent_posteriors else "none",
            "best_intent_p":    round(max(intent_posteriors.values(), default=0.0), 4),
        })

    def log_dbn_step(
        self,
        tick:       int,
        evaluator:  str,
        peer_id:    str,
        dbn_result: dict,
    ) -> None:
        """
        Log a full DBN forward-filter step with the complete math trace.

        dbn_result is the dict returned by DynamicBayesianClassifier.classify().
        The Streamlit Bayesian Inspector tab reads these events to display
        step-by-step probability tables.
        """
        self.log(tick, "DBN_STEP", {
            "evaluator":  evaluator,
            "peer_id":    peer_id,
            "p_attack":   round(dbn_result.get("p_attack", 0), 5),
            "p_fault":    round(dbn_result.get("p_fault",  0), 5),
            "p_honest":   round(dbn_result.get("p_honest", 0), 5),
            "label":      dbn_result.get("label", "HONEST"),
            "math_trace": dbn_result.get("math_trace", {}),
        })

    def log_replan(
        self,
        tick:     int,
        agent_id: str,
        trigger:  str,
        stats:    dict,
    ) -> None:
        """
        Log a D* Lite replanning event with cost comparison.

        stats is the dict from TrustDStarLite.last_replan_stats.
        """
        self.log(tick, "REPLAN", {
            "agent_id": agent_id,
            "trigger":  trigger,
            **stats,
        })

    def log_hazard_snapshot(
        self,
        tick:       int,
        agent_id:   str,
        hazard:     Dict[str, float],
    ) -> None:
        """
        Log a periodic hazard field snapshot for dashboard visualization.
        Written every N ticks (controlled by caller) to limit log size.
        """
        self.log(tick, "HAZARD_SNAPSHOT", {
            "agent_id": agent_id,
            "n_cells":  len(hazard),
            "hazard":   hazard,    # {"x,y": cost, ...}
        })

    def log_attack_window(
        self,
        tick:        int,
        attacker_id: int,
        target_id:   int,
        mode:        str,
    ) -> None:
        self.log(tick, "ATTACK_WINDOW", {
            "attacker_id": attacker_id,
            "target_id":   target_id,
            "mode":        mode,
        })

    def log_crypto_fault(
        self,
        tick:      int,
        evaluator: str,
        peer_id:   str,
        error:     str,
    ) -> None:
        self.log(tick, "CRYPTO_FAULT", {
            "evaluator": evaluator,
            "peer_id":   peer_id,
            "error":     error,
        })

    # ------------------------------------------------------------------ #
    # Lifecycle                                                            #
    # ------------------------------------------------------------------ #
    def close(self) -> None:
        self._file.flush()
        self._file.close()

    @property
    def event_count(self) -> int:
        return self._count


# ------------------------------------------------------------------ #
# Terminal color helpers                                               #
# ------------------------------------------------------------------ #
_COLORS = {
    "EPISTEMIC_PARADOX": "\033[91m",   # red
    "ATTACK_WINDOW":     "\033[93m",   # yellow
    "TRUST_UPDATE":      "\033[94m",   # blue
    "REPLAN":            "\033[96m",   # cyan
    "SIMULATION_START":  "\033[92m",   # green
    "SIMULATION_END":    "\033[92m",
    "DBN_STEP":          "\033[95m",   # magenta
}
_RESET = "\033[0m"


def _print_colored(event_type: str, line: str) -> None:
    color = _COLORS.get(event_type, "")
    print(f"{color}[LOG] {event_type}{_RESET} {line[:200]}")
