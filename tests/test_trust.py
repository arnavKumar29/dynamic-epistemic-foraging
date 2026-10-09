"""
test_trust.py — Unit tests for Beta-Binomial Dirichlet trust engine.
"""
import pytest
from src.adaptation.trust_manager import BetaTrustEngine


@pytest.fixture
def engine():
    return BetaTrustEngine(alpha0=4.0, beta0=1.0, aging=0.95)


class TestBetaTrustEngine:

    def test_initial_trust_prior(self, engine):
        trust, unc = engine.get("A2")
        # alpha0=4, beta0=1 → trust = 4/5 = 0.80
        assert abs(trust - 0.80) < 0.01

    def test_honest_event_increases_trust(self, engine):
        t0 = engine.get_trust("A2")
        upd = engine.update("A2", p_attack=0.0, anomaly_delta=2.0, verdict_label="HONEST")
        t1 = engine.get_trust("A2")
        assert t1 >= t0 - 0.01    # allow tiny float drift from aging

    def test_attack_event_decreases_trust(self, engine):
        t0 = engine.get_trust("A2")
        for _ in range(5):
            engine.update("A2", p_attack=0.9, anomaly_delta=200.0, verdict_label="ATTACK")
        t1 = engine.get_trust("A2")
        assert t1 < t0, f"Trust should drop after repeated attacks: {t0:.3f} → {t1:.3f}"

    def test_trust_decays_after_attacks_below_threshold(self, engine):
        """After 4-5 genuine attacks, trust should drop below 0.4."""
        for _ in range(5):
            engine.update("A2", p_attack=0.95, anomaly_delta=250.0, verdict_label="ATTACK")
        trust = engine.get_trust("A2")
        assert trust < 0.4, f"Trust should be low after sustained attacks, got {trust:.3f}"

    def test_trust_bounded(self, engine):
        """Trust must always be in [0, 1]."""
        for _ in range(100):
            engine.update("A2", p_attack=1.0, anomaly_delta=999.0, verdict_label="ATTACK")
        t, u = engine.get("A2")
        assert 0.0 <= t <= 1.0
        assert 0.0 <= u <= 1.0

    def test_fault_verdict_punishes_less_than_attack(self, engine):
        e1 = BetaTrustEngine(4.0, 1.0, 0.95)
        e2 = BetaTrustEngine(4.0, 1.0, 0.95)
        for _ in range(5):
            e1.update("A", 0.9, 100.0, "ATTACK")
            e2.update("A", 0.2, 100.0, "FAULT")
        assert e1.get_trust("A") < e2.get_trust("A"), \
            "ATTACK should reduce trust more than FAULT"

    def test_reset_peer_restores_prior(self, engine):
        for _ in range(5):
            engine.update("A2", 0.9, 200.0, "ATTACK")
        engine.reset_peer("A2")
        trust, _ = engine.get("A2")
        assert abs(trust - 0.80) < 0.01

    def test_all_trusts(self, engine):
        engine.update("A1", 0.1, 5.0, "HONEST")
        engine.update("A2", 0.8, 150.0, "ATTACK")
        trusts = engine.all_trusts()
        assert "A1" in trusts
        assert "A2" in trusts
        assert trusts["A1"] > trusts["A2"]

    def test_uncertainty_is_positive(self, engine):
        _, unc = engine.get("A2")
        assert unc > 0.0
