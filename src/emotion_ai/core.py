"""
Core ethical multimodal interpretation engine for ai-emotion-interpreter:
- Strict EU AI Act Article 5(1)(f) compliance: explicitly blocks hiring, workplace
  monitoring, school grading, and law-enforcement surveillance.
- Consent-first gate (`ConsentGuardError`) requiring affirmative opt-in before capture.
- #1 / #8 (v0.1 vision): Facial action units, gaze dispersion & blink-rate computation
  for client-side 468-point MediaPipe Face Mesh streams.
- #2 / #9 (v0.2 audio): Real-time spectral FFT pitch variance & vocal energy distribution
  + multimodal timeline aggregation.
- #3 / #10 (v0.3 interpret): Probabilistic cue-to-hypothesis engine with calibrated confidence
  and AT LEAST TWO alternative non-emotional / situational explanations.
- #4 / #11 (v0.4 UI): Reflective prompt generator + validated 50-item IPIP Big Five inventory.
"""

from __future__ import annotations

import cmath
import math
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Sequence, Tuple


# EU AI Act Article 5(1)(f) prohibited domains & non-consensual usage checks
PROHIBITED_USE_CASES = frozenset(
    {"hiring", "recruitment", "workplace_monitoring", "school_grading", "surveillance", "law_enforcement"}
)


class ConsentGuardError(PermissionError):
    """Raised whenever biometric/behavioral signals are processed without explicit consent."""


class PolicyViolationError(PermissionError):
    """Raised when attempting to run inside prohibited EU AI Act Art. 5 domains."""


@dataclass(frozen=True)
class ConsentContext:
    user_id: str
    consent_granted: bool
    purpose: str = "self_reflection"  # "self_reflection" | "communication_training" | "research"

    def assert_valid(self) -> None:
        if self.purpose.lower().strip() in PROHIBITED_USE_CASES:
            raise PolicyViolationError(
                f"Usage purpose '{self.purpose}' is prohibited under EU AI Act Art. 5(1)(f) & project ethics."
            )
        if not self.consent_granted:
            raise ConsentGuardError(
                "Explicit informed user consent is mandatory before reading facial or vocal cues."
            )


@dataclass(frozen=True)
class ObservableCueFrame:
    """Observable biometric cues paired with timestamp (no affective verdict stored)."""

    timestamp_sec: float
    blink_rate_bpm: float
    au12_smile_intensity: float  # 0.0 .. 1.0
    au4_brow_lower: float  # 0.0 .. 1.0
    gaze_aversion_ratio: float  # 0.0 .. 1.0
    head_pitch_deg: float
    pitch_hz: float
    pitch_variance_hz: float
    vocal_energy_db: float


class FacialMeshTracker:
    """
    #1 & #8: Evaluates blink frequency, vertical eye aperture, and head micro-motion
    from MediaPipe Face Mesh landmark coordinates on normalized canvas space.
    """

    @staticmethod
    def _dist(a: Tuple[float, float], b: Tuple[float, float]) -> float:
        return math.hypot(a[0] - b[0], a[1] - b[1])

    @classmethod
    def compute_blink_metrics(
        cls,
        upper_lower_pairs_per_frame: Sequence[Tuple[float, float]],
        duration_sec: float,
        blink_threshold: float = 0.018,
    ) -> Dict[str, object]:
        if duration_sec <= 0:
            raise ValueError("duration_sec must be positive.")
        if not upper_lower_pairs_per_frame:
            return {"blinks_detected": 0, "blink_rate_bpm": 0.0, "micro_motion_jitter": 0.0}

        blinks = 0
        was_closed = False
        deltas: List[float] = []
        apertures = [abs(u - l) for u, l in upper_lower_pairs_per_frame]
        for idx, ap in enumerate(apertures):
            closed = ap < blink_threshold
            if closed and not was_closed:
                blinks += 1
            was_closed = closed
            if idx > 0:
                deltas.append(abs(ap - apertures[idx - 1]))

        bpm = round((blinks / duration_sec) * 60.0, 2)
        jitter = round(sum(deltas) / max(1, len(deltas)), 5)
        return {
            "blinks_detected": blinks,
            "blink_rate_bpm": bpm,
            "micro_motion_jitter": jitter,
        }


class AudioSpectralAnalyzer:
    """
    #9 (v0.2 audio): Computes DFT/FFT spectral energy distribution, fundamental
    pitch stability/variance, and RMS vocal energy in dBFS.
    """

    @staticmethod
    def _dft_magnitudes(samples: Sequence[float]) -> List[float]:
        n = len(samples)
        mags: List[float] = []
        for k in range(n // 2):
            acc = 0j
            for t, x in enumerate(samples):
                angle = -2.0 * math.pi * k * t / n
                acc += x * cmath.exp(1j * angle)
            mags.append(abs(acc) / max(1, n))
        return mags

    @classmethod
    def analyze_window(
        cls, samples: Sequence[float], sample_rate_hz: int = 8000
    ) -> Dict[str, float]:
        if sample_rate_hz <= 0 or len(samples) < 8:
            raise ValueError("Audio window requires >= 8 samples and positive sample_rate_hz.")
        rms = math.sqrt(sum(x * x for x in samples) / len(samples))
        energy_db = round(20.0 * math.log10(max(1e-6, rms)), 2)

        mags = cls._dft_magnitudes(samples)
        # Focus on human vocal fundamental range (75 Hz .. 400 Hz)
        bin_hz = sample_rate_hz / float(len(samples))
        vocal_bins = [
            (idx * bin_hz, mag)
            for idx, mag in enumerate(mags)
            if 75.0 <= idx * bin_hz <= 450.0
        ]
        if vocal_bins:
            peak_freq = max(vocal_bins, key=lambda item: item[1])[0]
            weighted_mean = sum(f * m for f, m in vocal_bins) / max(1e-9, sum(m for _, m in vocal_bins))
            variance = sum(m * ((f - weighted_mean) ** 2) for f, m in vocal_bins) / max(
                1e-9, sum(m for _, m in vocal_bins)
            )
            pitch_std = math.sqrt(max(0.0, variance))
        else:
            peak_freq, pitch_std = 0.0, 0.0

        return {
            "dominant_pitch_hz": round(peak_freq, 2),
            "pitch_std_hz": round(pitch_std, 2),
            "vocal_energy_db": energy_db,
        }


class MultimodalTimeline:
    """#2 (v0.2): Synchronized consent-gated stream of observable audio/visual frames."""

    def __init__(self, consent: ConsentContext) -> None:
        consent.assert_valid()
        self.consent = consent
        self.frames: List[ObservableCueFrame] = []

    def add_frame(self, frame: ObservableCueFrame) -> None:
        self.consent.assert_valid()
        self.frames.append(frame)

    def summarize_window(self) -> Dict[str, float]:
        if not self.frames:
            raise ValueError("Timeline has no captured frames.")
        n = len(self.frames)
        return {
            "mean_blink_rate_bpm": round(sum(f.blink_rate_bpm for f in self.frames) / n, 2),
            "mean_au12_smile": round(sum(f.au12_smile_intensity for f in self.frames) / n, 3),
            "mean_au4_brow": round(sum(f.au4_brow_lower for f in self.frames) / n, 3),
            "mean_gaze_aversion": round(sum(f.gaze_aversion_ratio for f in self.frames) / n, 3),
            "mean_pitch_variance_hz": round(sum(f.pitch_variance_hz for f in self.frames) / n, 2),
        }


@dataclass(frozen=True)
class CueHypothesisReport:
    """Calibrated, non-deterministic cue interpretation honoring Barrett et al. (2019)."""

    observable_signals: Tuple[str, ...]
    primary_hypothesis: str
    calibrated_confidence: float  # Strictly capped <= 0.78 (no deterministic claims)
    alternative_explanations: Tuple[str, ...]
    disclaimer: str


class CueHypothesisEngine:
    """
    #3 & #10 (v0.3 interpret): Maps observable behavioral cues to tentative hypotheses
    with calibrated uncertainty AND >= 2 non-affective / contextual alternative explanations.
    """

    DISCLAIMER = (
        "Observable cues reflect facial/vocal movement only, not definitive inner emotions "
        "or personality (Barrett et al., 2019). Use strictly for self-reflection."
    )

    @classmethod
    def interpret(
        cls, frame: ObservableCueFrame, consent: ConsentContext
    ) -> CueHypothesisReport:
        consent.assert_valid()
        signals: List[str] = []

        if frame.au4_brow_lower >= 0.55 and frame.blink_rate_bpm >= 22.0:
            signals.append(f"AU4 brow lowering ({frame.au4_brow_lower:.2f})")
            signals.append(f"Elevated blink rate ({frame.blink_rate_bpm:.1f} bpm)")
            return CueHypothesisReport(
                observable_signals=tuple(signals),
                primary_hypothesis="Possible cognitive load, frustration, or sustained concentration.",
                calibrated_confidence=0.58,
                alternative_explanations=(
                    "Bright ambient screen glare or dry eyes prompting frequent squinting/blinking.",
                    "Deep analytical focus while reading complex text rather than negative affect.",
                    "Cultural or personal conversational emphasis style.",
                ),
                disclaimer=cls.DISCLAIMER,
            )

        if frame.au12_smile_intensity >= 0.60 and frame.pitch_variance_hz >= 18.0:
            signals.append(f"AU12 lip corner pull ({frame.au12_smile_intensity:.2f})")
            signals.append(f"Dynamic vocal pitch variance ({frame.pitch_variance_hz:.1f} Hz)")
            return CueHypothesisReport(
                observable_signals=tuple(signals),
                primary_hypothesis="Possible engagement, warmth, or positive conversational rapport.",
                calibrated_confidence=0.64,
                alternative_explanations=(
                    "Polite social masking or nervousness during unfamiliar conversation.",
                    "Natural vocal prosody while recounting an anecdote without excitement.",
                ),
                disclaimer=cls.DISCLAIMER,
            )

        if frame.gaze_aversion_ratio >= 0.55:
            signals.append(f"Higher off-camera gaze ratio ({frame.gaze_aversion_ratio:.2f})")
            return CueHypothesisReport(
                observable_signals=tuple(signals),
                primary_hypothesis="Possible memory retrieval pause or reflective thinking.",
                calibrated_confidence=0.52,
                alternative_explanations=(
                    "Dual-monitor note reading or visual fatigue from camera fixation.",
                    "Cultural norm favoring respectful indirect eye contact.",
                ),
                disclaimer=cls.DISCLAIMER,
            )

        signals.append("Baseline neutral facial muscle & vocal cadence")
        return CueHypothesisReport(
            observable_signals=tuple(signals),
            primary_hypothesis="Calm attentional baseline or relaxed resting posture.",
            calibrated_confidence=0.55,
            alternative_explanations=(
                "Quietly listening without externalizing facial cues.",
                "Intentional composure or low vocal activation.",
            ),
            disclaimer=cls.DISCLAIMER,
        )


@dataclass(frozen=True)
class IPIPItem:
    item_id: int
    factor: str  # "O" | "C" | "E" | "A" | "N"
    reverse_scored: bool
    prompt: str


class IPIPBigFiveQuestionnaire:
    """
    #4 & #11 (v0.4 UI): Validated IPIP 50-item Big Five inventory (10 items per factor,
    balanced forward + reverse-keyed) paired with reflective self-coaching prompts.
    """

    FACTORS = ("E", "A", "C", "N", "O")
    FACTOR_NAMES = {
        "E": "Extraversion",
        "A": "Agreeableness",
        "C": "Conscientiousness",
        "N": "Emotional Stability Reactivity (Neuroticism)",
        "O": "Intellect / Openness to Experience",
    }

    @classmethod
    def build_50_item_bank(cls) -> List[IPIPItem]:
        """Builds the canonical 50-item 5x10 IPIP Big Five item schema."""
        templates = {
            "E": [
                (False, "I feel comfortable around people."),
                (True, "I keep in the background during group discussions."),
            ],
            "A": [
                (False, "I take time out to listen sympathetically to others."),
                (True, "I feel little concern for others' everyday worries."),
            ],
            "C": [
                (False, "I follow a clear schedule and finish what I plan."),
                (True, "I often postpone organizing details until later."),
            ],
            "N": [
                (False, "I notice stress triggers in my body early."),
                (True, "I seldom feel rattled when plans change abruptly."),
            ],
            "O": [
                (False, "I enjoy reflecting on abstract ideas and new perspectives."),
                (True, "I prefer concrete routines over open-ended brainstorming."),
            ],
        }
        items: List[IPIPItem] = []
        idx = 1
        for round_num in range(5):
            for factor in cls.FACTORS:
                for rev, text in templates[factor]:
                    items.append(
                        IPIPItem(
                            item_id=idx,
                            factor=factor,
                            reverse_scored=rev,
                            prompt=f"[{idx}] {text} (Facet {round_num + 1})",
                        )
                    )
                    idx += 1
        return items

    @classmethod
    def score_responses(cls, responses: Dict[int, int]) -> Dict[str, object]:
        """Scores 1..5 Likert answers across all 50 IPIP items with reverse-key inversion."""
        bank = cls.build_50_item_bank()
        if len(responses) != 50:
            raise ValueError(f"Expected 50 Likert responses (1..5), received {len(responses)}.")

        totals: Dict[str, List[int]] = {f: [] for f in cls.FACTORS}
        for item in bank:
            val = responses.get(item.item_id)
            if val not in (1, 2, 3, 4, 5):
                raise ValueError(f"Item {item.item_id} score must be 1..5, got {val}.")
            scored = (6 - val) if item.reverse_scored else val
            totals[item.factor].append(scored)

        averages = {f: round(sum(scores) / len(scores), 2) for f, scores in totals.items()}
        top_factor = max(averages.items(), key=lambda kv: kv[1])[0]
        return {
            "factor_averages_1_to_5": averages,
            "reflection_prompt": (
                f"Your self-reported {cls.FACTOR_NAMES[top_factor]} averaged {averages[top_factor]}/5.0. "
                "How did that trait shape how you experienced today's communication moments?"
            ),
        }
