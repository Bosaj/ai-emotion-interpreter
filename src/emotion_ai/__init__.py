"""
ai-emotion-interpreter: Consent-gated, EU AI Act compliant multimodal cue
analyzer presenting observable facial/gaze/voice signals with calibrated confidence
and alternative explanations (never definitive emotion verdicts) plus validated
50-item IPIP Big Five reflection questionnaire.
"""

from .core import (
    PROHIBITED_USE_CASES,
    AudioSpectralAnalyzer,
    ConsentGuardError,
    CueHypothesisEngine,
    FacialMeshTracker,
    IPIPBigFiveQuestionnaire,
    MultimodalTimeline,
    ObservableCueFrame,
    PolicyViolationError,
)

__all__ = [
    "PROHIBITED_USE_CASES",
    "ConsentGuardError",
    "PolicyViolationError",
    "ObservableCueFrame",
    "FacialMeshTracker",
    "AudioSpectralAnalyzer",
    "MultimodalTimeline",
    "CueHypothesisEngine",
    "IPIPBigFiveQuestionnaire",
]

__version__ = "0.4.0"
