# 1. Architecture

1. Capture: webcam + microphone in the browser, processed locally; nothing is uploaded by default.
2. Signals: facial action units, gaze, blink rate, head pose, posture, voice pitch and energy.
3. Interpretation: cue to possible-emotion mapping with confidence and at least two alternative explanations.
4. Reflection: guided conversation and optional validated questionnaires (e.g. Big Five IPIP) for personality.

## Proposed stack
- Browser app (MediaPipe Tasks JS)
- Python research notebook (py-feat, OpenFace)
- LLM for the reflective conversation

## Open bases to fork
- [google-ai-edge/mediapipe](https://github.com/google-ai-edge/mediapipe) — Face and pose landmarks in real time.
- [cosanlab/py-feat](https://github.com/cosanlab/py-feat) — Facial expression and action-unit analysis toolkit.
- [TadasBaltrusaitis/OpenFace](https://github.com/TadasBaltrusaitis/OpenFace) — Action units, gaze and head pose.
- [serengil/deepface](https://github.com/serengil/deepface) — Face analysis framework.
