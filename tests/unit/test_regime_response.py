from __future__ import annotations

import pytest

from kq_tool.regime.response import build_regime_ai_payload


class FakeModel:
    trained = True
    train_data = [1, 2, 3]

    def predict_current(self) -> dict:
        return {"current_regime": "골디락스", "probs": {"골디락스": 0.9}}


class EmptyModel:
    trained = True
    train_data = None

    def predict_current(self):
        return None


def test_build_regime_ai_payload_uses_fallback_when_model_is_not_trained(monkeypatch) -> None:
    monkeypatch.setattr("kq_tool.regime.response.build_payload", lambda: (_ for _ in ()).throw(RuntimeError("no ui payload")))

    payload = build_regime_ai_payload(None, {"current": "리플레이션"})

    assert payload["current"] == "리플레이션"
    assert payload["model_type"] == "Rule-based (모델 미학습)"
    assert payload["model_status"] == "not_trained"


def test_build_regime_ai_payload_adds_model_metadata(monkeypatch) -> None:
    monkeypatch.setattr("kq_tool.regime.response.build_payload", lambda: (_ for _ in ()).throw(RuntimeError("no ui payload")))

    payload = build_regime_ai_payload(
        FakeModel(),
        {},
        regime_desc={"골디락스": "good"},
        regime_order=("골디락스", "리플레이션"),
    )

    assert payload["current_regime"] == "골디락스"
    assert payload["model_status"] == "trained"
    assert payload["sample_count"] == 3
    assert payload["regime_desc"] == {"골디락스": "good"}
    assert payload["regime_order"] == ["골디락스", "리플레이션"]


def test_build_regime_ai_payload_raises_when_prediction_fails(monkeypatch) -> None:
    monkeypatch.setattr("kq_tool.regime.response.build_payload", lambda: (_ for _ in ()).throw(RuntimeError("no ui payload")))

    with pytest.raises(RuntimeError, match="예측 실패"):
        build_regime_ai_payload(EmptyModel(), {})


def test_server_reuses_regime_ai_payload_helper() -> None:
    import server

    assert server._kq_build_regime_ai_payload is build_regime_ai_payload
