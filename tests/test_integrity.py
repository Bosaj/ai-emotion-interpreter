"""
Automated Verification, Unit & Integration Test Suite for ai-emotion-interpreter
Covers:
- #1 / #8 (v0.1): Consent-gated Face Mesh blink frequency & local dashboard UI
- #2 / #9 (v0.2): Spectral FFT vocal pitch & energy analyzer + multimodal timeline
- #3 / #10 (v0.3): Probabilistic cue-to-hypothesis engine with >= 2 alternative explanations & EU AI Act guardrails
- #4 / #11 (v0.4): Validated 50-item IPIP Big Five reflection questionnaire
Author: @Bosaj
"""

import math
import os
import sys
import unittest

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
SRC_PATH = os.path.join(PROJECT_ROOT, "src")
if SRC_PATH not in sys.path:
    sys.path.insert(0, SRC_PATH)

from emotion_ai import (  # noqa: E402
    AudioSpectralAnalyzer,
    ConsentGuardError,
    CueHypothesisEngine,
    FacialMeshTracker,
    IPIPBigFiveQuestionnaire,
    MultimodalTimeline,
    ObservableCueFrame,
    PolicyViolationError,
)
from emotion_ai.core import ConsentContext  # noqa: E402


class TestRepositoryIntegrity(unittest.TestCase):
    def test_project_root_and_consent_dashboard(self):
        self.assertTrue(os.path.isdir(PROJECT_ROOT))
        html_path = os.path.join(PROJECT_ROOT, "web", "index.html")
        self.assertTrue(os.path.isfile(html_path))
        with open(html_path, "r", encoding="utf-8") as f:
            html = f.read()
        self.assertIn("EU AI Act", html)
        self.assertIn("consent-checkbox", html)


class TestEthicsConsentAndSignals(unittest.TestCase):
    """Tests consent gate, EU AI Act Art. 5 block, Face Mesh blink tracker, and FFT audio."""

    def test_consent_and_eu_ai_act_guardrails(self):
        no_consent = ConsentContext("u1", consent_granted=False, purpose="self_reflection")
        with self.assertRaises(ConsentGuardError):
            MultimodalTimeline(no_consent)

        hiring = ConsentContext("u1", consent_granted=True, purpose="hiring")
        with self.assertRaises(PolicyViolationError):
            MultimodalTimeline(hiring)

    def test_facial_blink_metrics_and_fft_audio_analyzer(self):
        pairs = [(0.05, 0.0), (0.01, 0.0), (0.04, 0.0), (0.008, 0.0), (0.05, 0.0)]
        blink_res = FacialMeshTracker.compute_blink_metrics(pairs, duration_sec=6.0)
        self.assertEqual(blink_res["blinks_detected"], 2)
        self.assertAlmostEqual(blink_res["blink_rate_bpm"], 20.0)

        # Pure 200 Hz sine wave sampled at 1600 Hz over 64 samples
        sr = 1600
        sine_200 = [0.5 * math.sin(2.0 * math.pi * 200.0 * i / sr) for i in range(64)]
        audio_res = AudioSpectralAnalyzer.analyze_window(sine_200, sample_rate_hz=sr)
        self.assertAlmostEqual(audio_res["dominant_pitch_hz"], 200.0, delta=1.0)
        self.assertGreater(audio_res["vocal_energy_db"], -20.0)


class TestHypothesesAndIPIP50(unittest.TestCase):
    """Tests #3, #4, #10, #11: Calibrated alternative explanations and 50-item IPIP scoring."""

    def test_cue_hypothesis_returns_alternatives_and_capped_confidence(self):
        consent = ConsentContext("u-reflect", consent_granted=True, purpose="self_reflection")
        frame = ObservableCueFrame(
            timestamp_sec=12.0,
            blink_rate_bpm=25.0,
            au12_smile_intensity=0.1,
            au4_brow_lower=0.72,
            gaze_aversion_ratio=0.2,
            head_pitch_deg=-3.0,
            pitch_hz=155.0,
            pitch_variance_hz=12.0,
            vocal_energy_db=-16.0,
        )
        timeline = MultimodalTimeline(consent)
        timeline.add_frame(frame)
        self.assertEqual(timeline.summarize_window()["mean_blink_rate_bpm"], 25.0)

        report = CueHypothesisEngine.interpret(frame, consent)
        self.assertLessEqual(report.calibrated_confidence, 0.78)
        self.assertGreaterEqual(len(report.alternative_explanations), 2)
        self.assertIn("Barrett et al.", report.disclaimer)

    def test_50_item_ipip_questionnaire_scoring(self):
        bank = IPIPBigFiveQuestionnaire.build_50_item_bank()
        self.assertEqual(len(bank), 50)
        responses = {item.item_id: 5 for item in bank}
        scored = IPIPBigFiveQuestionnaire.score_responses(responses)
        # Half reverse-scored (6-5=1) and half forward (5) => factor average = 3.0
        for val in scored["factor_averages_1_to_5"].values():
            self.assertEqual(val, 3.0)
        self.assertIn("reflection_prompt", scored)


if __name__ == "__main__":
    unittest.main()
