from __future__ import annotations

from kq_tool.regime.classifier import current_regime_snapshot


class FakeModel:
    trained = True

    def predict_current(self) -> dict:
        return {
            "current_regime": "골디락스",
            "probs": {"골디락스": 0.8, "리플레이션": 0.2},
            "next_quarter": {"골디락스": 0.7},
            "duration_avg": {"골디락스": 2.0},
            "model_type": "Fake",
        }


class BrokenModel:
    trained = True

    def predict_current(self) -> dict:
        raise RuntimeError("boom")


def test_current_regime_snapshot_normalizes_model_output(monkeypatch) -> None:
    monkeypatch.setattr("kq_tool.regime.classifier.build_payload", lambda: (_ for _ in ()).throw(RuntimeError("no ui payload")))

    snapshot = current_regime_snapshot(FakeModel())

    assert snapshot["current"] == "골디락스"
    assert snapshot["confidence"] == 0.8
    assert snapshot["model_type"] == "Fake"


def test_current_regime_snapshot_falls_back_without_model(monkeypatch) -> None:
    monkeypatch.setattr("kq_tool.regime.classifier.build_payload", lambda: (_ for _ in ()).throw(RuntimeError("no ui payload")))

    snapshot = current_regime_snapshot(None, fallback_current="디플레이션")

    assert snapshot["current"] == "디플레이션"
    assert snapshot["confidence"] == 1.0


def test_current_regime_snapshot_falls_back_on_error(monkeypatch) -> None:
    monkeypatch.setattr("kq_tool.regime.classifier.build_payload", lambda: (_ for _ in ()).throw(RuntimeError("no ui payload")))

    snapshot = current_regime_snapshot(BrokenModel(), fallback_current="리플레이션")

    assert snapshot["current"] == "리플레이션"
    assert snapshot["probs"] == {"리플레이션": 1.0}
