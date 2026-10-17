"""Streaming turn detection. You get one end-of-turn probability per 20 ms of audio, and the state resets at the
start of each new user turn (on by default).

The models were trained on single user turns that start from a fresh state just before speech. Carrying their state
across lots of turns in a long call costs accuracy, so the detector resets itself when a new turn starts, meaning the
model's own speech output comes back after at least `min_silence_s` of silence. It then re-runs from a fresh state on
the last `lookback_s` of audio features, which it keeps in a small buffer. Pass `reset_on_turn_start=False` if your pipeline already
restarts the detector at each user turn.

    det = TurnDetector.from_pretrained("cuelight-nano")
    for chunk in chunks_of_320_samples:     # float32, 16 kHz mono
        p_end = det.push(chunk)
"""
from __future__ import annotations

from collections import deque
from dataclasses import dataclass

import numpy as np
import torch

from cuelight.features import FRAME, SR

HOP_S = FRAME / SR


@dataclass(frozen=True)
class ResetPolicy:
    min_silence_s: float = 2.0
    onset_frames: int = 3
    lookback_s: float = 0.3
    speech_thr: float = 0.5

    @property
    def replay(self) -> int:
        """Frames re-run after a reset: the lookback plus the onset frames that confirmed the speech."""
        return round(self.lookback_s / HOP_S) + self.onset_frames


class OnsetTracker:
    """Fires once when speech has lasted `onset_frames` frames after at least `min_silence_s` of silence."""

    def __init__(self, pol: ResetPolicy):
        self.pol, self.min_sil = pol, round(pol.min_silence_s / HOP_S)
        self.sil, self.run, self.armed = 0, 0, False

    def update(self, p_speech: float) -> bool:
        if p_speech < self.pol.speech_thr:
            self.sil, self.run = self.sil + 1, 0
            return False
        if self.run == 0:
            self.armed, self.sil = self.sil >= self.min_sil, 0
        self.run += 1
        if self.armed and self.run == self.pol.onset_frames:
            self.armed = False
            return True
        return False


class TurnDetector:
    """Wraps a CueLight model and its front end for frame-by-frame use."""

    def __init__(self, model, frontend, reset_on_turn_start: bool = True, policy: ResetPolicy = ResetPolicy()):
        self.m, self.fe = model.eval(), frontend
        self.reset_on, self.pol = reset_on_turn_start, policy
        self.n_resets = 0
        self.p_speech = 0.0
        self.clear()

    @classmethod
    def from_pretrained(cls, name_or_path: str, **kw) -> "TurnDetector":
        from cuelight.hub import load
        model, frontend = load(name_or_path)
        return cls(model, frontend, **kw)

    def clear(self):
        """Forget everything (e.g. before a new call)."""
        self.fe_state, self.state = self.fe.init_state(1), self.m.init_state(1)
        self.buf = deque(maxlen=self.pol.replay)
        self.track = OnsetTracker(self.pol)

    @torch.no_grad()
    def push(self, chunk) -> float:
        """320 new samples (20 ms at 16 kHz, float32 in [-1, 1]) -> p(the user has finished their turn)."""
        x = torch.as_tensor(np.asarray(chunk, dtype=np.float32))[None]
        assert x.shape[-1] == FRAME, f"push exactly {FRAME} samples (20 ms at 16 kHz)"
        f, self.fe_state = self.fe.step(x, self.fe_state)
        self.buf.append(f)
        logits, self.state = self.m.step(f, self.state)
        self.p_speech = float(torch.sigmoid(logits[0, 0]))
        if self.reset_on and self.track.update(self.p_speech):
            self.state = self.m.init_state(1)
            for g in self.buf:
                logits, self.state = self.m.step(g, self.state)
            self.n_resets += 1
        return float(torch.sigmoid(logits[0, 1]))


@dataclass(frozen=True)
class EndpointPolicy:
    """When to end the user's turn: once the user has been silent for at least `min_silence_s` and p_end reaches
    `threshold`, or after `timeout_s` of silence regardless."""
    threshold: float = 0.7
    min_silence_s: float = 0.6
    timeout_s: float = 2.0


class Endpointer:
    """A TurnDetector plus an end-of-turn policy: push() returns True once, on the frame where the turn ends.

    Silence is taken from the model's own speech output (p_speech < 0.5). Each model ships two tuned presets:
    "snappy" (the default: responds sooner, interrupts on about 1 in 10 pauses) and "careful" (about 1 in 20).

        ep = Endpointer.from_pretrained("cuelight-nano")              # preset="snappy"
        for chunk in chunks_of_320_samples:
            if ep.push(chunk):
                respond()
    """

    def __init__(self, detector: TurnDetector, policy: EndpointPolicy = EndpointPolicy()):
        self.det, self.pol = detector, policy
        self.silent_frames, self.fired, self.p_end = 0, True, 0.0  # nothing to end until the user has spoken

    @classmethod
    def from_pretrained(cls, name_or_path: str, preset: str = "snappy", **kw) -> "Endpointer":
        det = TurnDetector.from_pretrained(name_or_path, **kw)
        presets = getattr(det.m, "cuelight_config", {}).get("presets", {})
        return cls(det, EndpointPolicy(**presets[preset]) if preset in presets else EndpointPolicy())

    def push(self, chunk) -> bool:
        self.p_end = self.det.push(chunk)
        if self.det.p_speech >= 0.5:
            self.silent_frames, self.fired = 0, False
            return False
        self.silent_frames += 1
        silence = self.silent_frames * HOP_S
        if self.fired:  # already ended this pause (or the user hasn't spoken yet)
            return False
        if (silence >= self.pol.min_silence_s and self.p_end >= self.pol.threshold) or silence >= self.pol.timeout_s:
            self.fired = True
            return True
        return False
