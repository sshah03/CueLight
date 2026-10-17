"""Regression test for the "presets" config key bug in Endpointer.from_pretrained.

The model config uses "presets" (matching docs/models/*.json and cuelight.js).
Previously the code looked up "policies", silently ignoring tuned per-model
presets and falling back to EndpointPolicy defaults.
"""
from types import SimpleNamespace
from cuelight.stream import Endpointer, EndpointPolicy


def _fake_det(policies_dict):
    """Simulate what hub.load returns: model.cuelight_config = cfg."""
    det = SimpleNamespace()
    det.m = SimpleNamespace(cuelight_config={"presets": policies_dict})
    det.m.eval = lambda: det.m
    det.fe = None
    return det


def test_endpointer_reads_presets_key(monkeypatch):
    """The Endpointer.from_pretrained must look up 'presets', not 'policies'."""
    policies = {"snappy": {"threshold": 0.73, "min_silence_s": 0.5, "timeout_s": 1.5}}
    det = _fake_det(policies)

    # Patch TurnDetector.from_pretrained to return our fake detector
    import cuelight.stream as sm
    orig = sm.TurnDetector.from_pretrained
    sm.TurnDetector.from_pretrained = classmethod(lambda cls, name, **kw: det)
    try:
        ep = Endpointer.from_pretrained("cuelight-nano", preset="snappy")
        assert ep.pol.threshold == 0.73, (
            "snappy preset threshold (0.73) should be applied, "
            f"got {ep.pol.threshold}"
        )
        assert ep.pol.min_silence_s == 0.5
        assert ep.pol.timeout_s == 1.5
    finally:
        sm.TurnDetector.from_pretrained = orig


def test_endpointer_falls_back_to_defaults(monkeypatch):
    """If presets is missing/empty, EndpointPolicy defaults should be used."""
    det = _fake_det({})

    import cuelight.stream as sm
    orig = sm.TurnDetector.from_pretrained
    sm.TurnDetector.from_pretrained = classmethod(lambda cls, name, **kw: det)
    try:
        ep = Endpointer.from_pretrained("cuelight-nano", preset="snappy")
        assert ep.pol == EndpointPolicy()
    finally:
        sm.TurnDetector.from_pretrained = orig
