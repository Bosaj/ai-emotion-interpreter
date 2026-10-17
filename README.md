# AI Human Behavior & Emotion Interpreter

> A research-oriented tool that reads facial movements, gaze, posture and voice, and explains possible emotional cues with confidence levels — never verdicts.

**Status:** pre-alpha scaffold (no code yet). **Feasibility rating:** 8/10 (owner's rating) — strong interest; scientific and legal limits must shape the design.
**Idea:** #19 in the owner's ideas backlog. **Author:** Oussama EL HADJI. Part of the Bosaj ideas portfolio (sibling repo: `dual_portfolio_hub`).

---

## Problem

Body language and facial expressions carry emotional cues. A tool that makes those cues visible could support self-reflection, communication training and psychology research.

## Why build it (gap analysis)

Research (Barrett et al., 2019, "Emotional Expressions Reconsidered") shows that facial movements do not map reliably to inner emotions across people and cultures, and personality cannot be read from a face. The EU AI Act (Article 5(1)(f)) bans emotion recognition in workplaces and schools. So the honest design reports observable cues (action units, gaze, voice pitch), gives confidence and alternative explanations, uses validated questionnaires for personality, and runs only with the person's consent.

## Prior art & open bases to fork

- [google-ai-edge/mediapipe](https://github.com/google-ai-edge/mediapipe) — Face and pose landmarks in real time.
- [cosanlab/py-feat](https://github.com/cosanlab/py-feat) — Facial expression and action-unit analysis toolkit.
- [TadasBaltrusaitis/OpenFace](https://github.com/TadasBaltrusaitis/OpenFace) — Action units, gaze and head pose.
- [serengil/deepface](https://github.com/serengil/deepface) — Face analysis framework.

## Proposed architecture

1. Capture: webcam + microphone in the browser, processed locally; nothing is uploaded by default.
2. Signals: facial action units, gaze, blink rate, head pose, posture, voice pitch and energy.
3. Interpretation: cue to possible-emotion mapping with confidence and at least two alternative explanations.
4. Reflection: guided conversation and optional validated questionnaires (e.g. Big Five IPIP) for personality.

## Tech stack (proposed)

- Browser app (MediaPipe Tasks JS)
- Python research notebook (py-feat, OpenFace)
- LLM for the reflective conversation

## Roadmap

- v0.1 — Local face/gaze signal dashboard with consent screen.
- v0.2 — Voice features + multimodal timeline.
- v0.3 — Cue interpretation with confidence and alternatives.
- v0.4 — Reflection chat + validated personality questionnaire.

---

## For LLM agents & contributors continuing this project

This repo is a **scaffold**: README + standard files, no implementation yet. If you are an AI agent asked to "continue developing", read this section first.

- Never output a definitive emotion, personality or "life experience" judgment from appearance alone.
- Consent screen first; no use for hiring, school or surveillance (EU AI Act Art. 5).

**Ground rules:** keep this README in sync with reality as you build; prefer forking/reusing the bases above over greenfield code; write tests for anything correctness-critical; never commit secrets (see `.gitignore`); follow Conventional Commits (see `CONTRIBUTING.md`).

## Getting started

```bash
# nothing to run yet — this is a scaffold.
```

## License

[MIT](LICENSE) — © 2026 Oussama EL HADJI
