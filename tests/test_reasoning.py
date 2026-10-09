"""
test_reasoning.py — Unit tests for BDI engine, beliefs, and BayesianToM.
"""
import pytest
from src.reasoning.belief import Belief, KnowledgeBase
from src.reasoning.theory_of_mind import BayesianToM
from src.reasoning.bdi_engine import BDIEngine


class TestKnowledgeBase:

    @pytest.fixture
    def kb(self):
        return KnowledgeBase()

    def test_add_and_query(self, kb):
        b = Belief("A2", "position", (100.0, 200.0), 0.95, 5, "COMM")
        kb.add(b)
        result = kb.query("A2", "position")
        assert result is not None
        assert result.value == (100.0, 200.0)

    def test_agm_revision_replaces_stale(self, kb):
        """New belief replaces old belief for same subject:predicate."""
        b1 = Belief("A2", "position", (100.0, 200.0), 0.95, 5, "COMM")
        b2 = Belief("A2", "position", (110.0, 205.0), 0.95, 6, "COMM")
        kb.add(b1)
        kb.add(b2)
        result = kb.query("A2", "position")
        assert result.value == (110.0, 205.0)

    def test_history_retained(self, kb):
        for i in range(5):
            b = Belief("A2", "position", (float(i * 10), 0.0), 0.95, i, "COMM")
            kb.add(b)
        hist = kb.history("A2", "position", n=3)
        assert len(hist) == 3
        assert hist[-1].value == (40.0, 0.0)

    def test_query_missing_returns_none(self, kb):
        assert kb.query("A99", "position") is None

    def test_all_subjects(self, kb):
        kb.add(Belief("A1", "pos", (0.0, 0.0), 1.0, 0, "SENSOR"))
        kb.add(Belief("A2", "pos", (0.0, 0.0), 1.0, 0, "SENSOR"))
        assert "A1" in kb.all_subjects()
        assert "A2" in kb.all_subjects()


class TestBayesianToM:

    @pytest.fixture
    def tom(self):
        return BayesianToM(
            max_velocity=3.0,
            pos_tolerance=18.0,
            obstacles=[(200.0, 200.0, 100.0, 100.0)],
            rationality=2.0,
        )

    @pytest.fixture
    def kb_with_history(self):
        kb = KnowledgeBase()
        kb.add(Belief("A2", "position", (100.0, 100.0), 0.95, 0, "COMM"))
        return kb

    def test_normal_move_no_contradiction(self, tom, kb_with_history):
        goals = {"forage": (500.0, 500.0), "return": (0.0, 0.0)}
        clash, delta, expl, post = tom.verify(
            peer_id="A2",
            claimed_pos=(102.0, 101.0),  # tiny move
            claimed_tick=1,
            kb=kb_with_history,
            goals=goals,
        )
        assert clash is False
        assert delta < 5.0

    def test_teleport_detected(self, tom, kb_with_history):
        goals = {"forage": (500.0, 500.0)}
        clash, delta, expl, post = tom.verify(
            peer_id="A2",
            claimed_pos=(400.0, 400.0),   # 424px jump in 1 tick
            claimed_tick=1,
            kb=kb_with_history,
            goals=goals,
        )
        assert clash is True

    def test_first_contact_no_clash(self, tom):
        """No history = no baseline = no contradiction (benefit of the doubt)."""
        kb   = KnowledgeBase()   # empty
        goals = {"forage": (500.0, 500.0)}
        clash, delta, expl, post = tom.verify(
            peer_id="A3",
            claimed_pos=(100.0, 100.0),
            claimed_tick=1,
            kb=kb,
            goals=goals,
        )
        assert clash is False

    def test_intent_posteriors_sum_to_one(self, tom, kb_with_history):
        goals = {"forage": (500.0, 500.0), "return": (0.0, 0.0), "explore": (300.0, 300.0)}
        _, _, _, post = tom.verify(
            peer_id="A2",
            claimed_pos=(103.0, 103.0),
            claimed_tick=1,
            kb=kb_with_history,
            goals=goals,
        )
        assert len(post) == 3
        assert abs(sum(post.values()) - 1.0) < 1e-6


class TestBDIEngine:

    @pytest.fixture
    def bdi(self):
        return BDIEngine(
            agent_id="A0",
            obstacles=[(200.0, 200.0, 100.0, 100.0)],
            max_velocity=3.0,
            pos_tolerance=18.0,
            goals={"forage": (680.0, 480.0), "return": (50.0, 50.0)},
        )

    def test_normal_message_no_contradiction(self, bdi):
        msgs = [{"sender_id": 1, "tick": 1, "x": 60.0, "y": 540.0,
                 "heading": 0.0, "status": "FORAGE"}]
        # First contact: no baseline
        events = bdi.update_beliefs(msgs)
        assert len(events) == 1
        assert events[0]["peer_id"] == "A1"

    def test_corrupted_message_flagged(self, bdi):
        msgs = [{"CORRUPTED": True, "sender_id": 2, "error": "tag mismatch"}]
        events = bdi.update_beliefs(msgs)
        assert events[0]["contradiction"] is True
        assert "CRYPTO" in events[0]["reason"]

    def test_second_message_teleport_detected(self, bdi):
        # Seed history
        bdi.update_beliefs([{"sender_id": 2, "tick": 0, "x": 100.0, "y": 100.0,
                              "heading": 0.0, "status": "EXPLORE"}])
        # Teleport
        events = bdi.update_beliefs([{"sender_id": 2, "tick": 1, "x": 600.0, "y": 600.0,
                                      "heading": 0.0, "status": "EXPLORE"}])
        assert events[0]["contradiction"] is True
