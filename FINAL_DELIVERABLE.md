# AEGIS·ID — Final Deliverable Summary

This is the navigation index for everything built across this project.
`README.md` has the full technical depth on each piece (formulas, tested
values, honesty notes); this document maps that work to the 15 points you
asked for, without duplicating the detail — follow the pointers.

---

## 1. Final architecture summary

Single Flask app (`app.py`, ~2,500 lines) serving a dashboard UI
(`templates/index.html` + `static/styles.css`). Pipeline:

```
Haar face detection → facial-landmark EAR blink detection (Haar fallback)
    → LBPH recognition → five-factor weighted trust score
    → _decide() [grant / step-up / deny] → _risk_tier() [LOW/MED/HIGH/CRITICAL]
    → conditional step-up challenge (blink/turn/nod/hand-raise, randomized)
    → conditional voice challenge-response (mandatory only at HIGH tier)
```

Registration mirrors verification's liveness gating (same challenge system,
spoof-resistance screened per-frame before any image is saved). Attack Lab
runs the *same* pipeline, labeled and logged, not a separate code path.
Everything is local — no cloud APIs, no GPU requirement, CPU-only
throughout (see §11).

## 2. Files modified

`app.py`, `templates/index.html`, `static/styles.css`, `requirements.txt`.
(Everything — every part of this project — touched these four files.
`app.py` is where essentially all the substantive logic lives.)

## 3. Files added

| File | Purpose |
|---|---|
| `test_trust_matrix.py` | Runnable 15-scenario trust-score regression test (Part 2) |
| `DEMO_SCRIPT.md` | The presentation sequence (Part 6) |
| `FINAL_DELIVERABLE.md` | This document |
| `TEST_PLAN.md` | Full test procedure (Part 10) |
| `lbfmodel.yaml` | 68-point facial landmark model (~54MB, bundled) |
| `vosk-model-small-en-us-0.15/` | Offline speech-recognition model (~40MB, bundled) |

## 4. New dependencies

Only **one** genuinely new pip package this round: **`vosk`** (offline
speech recognition, for voice challenge-response content verification —
see §7). Everything else added across this project (`librosa`,
`scikit-learn`, `sounddevice` for Voice ID) predates this session.
Hand-raise liveness and facial-landmark blink detection needed **zero** new
packages — both use `cv2` (already a dependency) plus classical CV. All
optional dependencies degrade gracefully if missing (`VOICE_AVAILABLE`,
`ASR_AVAILABLE`, `LANDMARK_AVAILABLE` flags) — the app never hard-crashes
on a missing optional package.

## 5. Hand-raise liveness

Three new challenges (`raise_left_hand`, `raise_right_hand`,
`raise_both_hands`), additive to the existing blink/turn/nod set — nothing
removed. Classical YCrCb skin-color blob detection (not MediaPipe — its
current API needs a runtime model download from an unverified domain;
zero new dependencies here instead). **Sequence-validated, not
presence-checked**: requires a sustained ≥5-frame hold in the correct
target region — tested against 7 synthetic scenarios ("wrong hand raised,"
"too-brief flash," "never raised," etc.), all correctly rejected.

**Post-deployment fix**: the original version also required the hand to be
*absent* early in the capture (only appearing partway through), intended
as anti-spoofing evidence of a genuine transition. A real user's report
of a correctly-performed hand raise being rejected led to reproducing and
fixing this — the requirement was fundamentally mismatched with the UX
(the frontend's "Get ready…" pause before capture starts means a
compliant person's hand is naturally already up by the time frames are
captured, for a *sustained-pose* challenge specifically, unlike a blink or
quick turn). Now only the sustained-hold requirement gates completion;
verified this doesn't weaken the other 7 failure-mode protections. Full
detail: `README.md` → "Hand-raise liveness."

## 6. Trust-score changes

Two categories of change, from the Part 2 audit:

- **Calibration bug fixes** (not just "make it easier"): recognition
  confidence was reusing the match-accept cutoff as its own scoring scale,
  scoring near-zero confidence for matches that had already been accepted
  as genuine. Decoupled. Sharpness/texture scales were similarly
  unreachable-in-practice. Concrete example: the same realistic capture
  scored 60.6 (denied) under the old math, 69.2 (granted) under fixed math.
- **New finding, new fix**: the audit's 15-scenario matrix caught a real
  gap — a strong face match could outweigh critically low spoof-resistance
  and grant outright, bypassing step-up. `CRITICAL_SPOOF_THRESHOLD` now
  caps the decision at step-up when spoof-resistance is below 35,
  regardless of trust score. Verified: two scenarios that previously
  granted at trust 78.3/65.4 with spoof-resistance 15/30 now correctly
  step up.

Full detail with the actual before/after numbers: `README.md` →
"Calibration" and "Risk tiers."

## 7. Voice challenge-response

Upgraded from speaker-only (MFCC+GMM against a fixed enrollment) to
requiring **both** speaker match **and** phrase-content match, independently:

- A random phrase (2 words + 1 digit, from a 24-word ASR-friendly
  vocabulary) generates fresh immediately before every attempt —
  `generate_voice_phrase()`, never a fixed sentence.
- Content verified via Vosk (genuinely offline, CPU-only). **Grammar-
  constrained** to the known vocabulary — tested directly: open-vocabulary
  transcription got 0/3 words right on real synthesized speech through the
  real model; grammar-constrained got 3/3 right on the *same* audio. Real
  accuracy improvement, not just a testing convenience.
- Two lightweight replay heuristics (near-duplicate fingerprint detection,
  spectral bandwidth) — explicitly documented as assistive, not robust;
  only the more defensible check (near-duplicate) gates the decision.

**Attack classes mitigated vs. not** — full honest table in `README.md` →
"Attack classes this mitigates — and what it doesn't." Short version: stops
cheap replay-of-unrelated-recording attacks; does **not** claim resistance
to a dedicated voice-cloning attack capable of synthesizing the exact
requested phrase on demand.

## 8. Attack Lab changes

Redesigned as a security-testing console, not a demo gimmick:

- Explicit **Live Attack Demonstration vs. Attack Simulation** distinction,
  stated in the UI, not just docs.
- New `face_manipulation` ("deepfake scenario") attack type — classical
  warp + lighting-mismatch + periodic-pattern approximation of documented
  manipulation tells. Explicitly labeled as **not** a GAN/deepfake
  generator, in both code comments and UI copy.
- New voice attack scenarios (4, live-only — no simulation mode, since
  faking a voice-degradation pipeline without real samples to validate
  against would mean faking a result).
- **Signal-by-signal breakdown** now shown per attempt: Face Match,
  Liveness, Texture, Moiré, Spatial Consistency, Trust Score, Risk Tier,
  Final Decision — previously texture/moiré/spatial were only visible
  blended into one number.
- Nothing hard-coded: every attack (live or simulated) runs through the
  identical `_score_signals()`/`_decide()` path as normal verification.

## 9. Adaptive authentication flow

`_risk_tier()` (LOW/MEDIUM/HIGH/CRITICAL) layered on top of `_decide()`'s
existing grant/step-up/deny authority — a label, not a new gate. The
behavioral change: voice used to run as an always-on bonus after *any*
grant; now it's skipped entirely for LOW/MEDIUM tiers and **mandatory**
(fail-closed if unavailable) only at HIGH tier. This is the actual
"requirements increase with risk" implementation — full detail and the
matrix validation in `README.md` → "Adaptive authentication — risk tiers."

## 10. Attack detection/mitigation mapping

See `TEST_PLAN.md` for the complete table (attack → signals involved →
expected trust-score behavior → risk tier → decision → limitations). Quick
reference:

| Attack | Primary defense | Residual risk |
|---|---|---|
| Printed photo | Liveness (no real blink) + texture/spatial | A very still, well-lit print in low motion could partially evade on a single pass — step-up's challenge gate is the real backstop |
| Screen/phone replay | Moiré + spatial-anomaly + liveness | A video *of the person blinking* passes passive liveness — spoof-resistance and the unpredictable challenge are what actually catch it |
| Video loop | Same as above | Same |
| Face manipulation (simulated) | Spatial-anomaly / patch-consistency | Heuristic, classical-CV only — not validated against real deepfake output |
| Voice replay (unrelated content) | Phrase-content check | Doesn't stop a targeted clone capable of synthesizing the exact phrase |
| Wrong speaker, correct phrase | GMM speaker check | — |
| Wrong phrase, correct speaker | ASR content check | — |

## 11. CPU performance

Every new capability benchmarked directly, no GPU anywhere:

| Operation | Measured cost |
|---|---|
| Facial-landmark fitting (68 points) | 1.6 ms/frame |
| Skin-color hand detection | 1.78 ms/frame |
| Spoof-resistance (texture+moiré+spatial) | sub-ms, part of existing per-frame budget |
| Voice: MFCC extraction | negligible, `librosa` on 3-4s clips |
| Voice: Vosk transcription | offline, streaming, real-time-capable on CPU |

None of these individually or combined meaningfully slow down capture —
the dominant cost remains Haar cascade face detection itself (30-80ms/frame,
unchanged from the original implementation).

## 12. Testing procedure

Full test plan: `TEST_PLAN.md`. Automated regression: `python
test_trust_matrix.py` (15 trust-score scenarios, re-run after any scoring
change). Everything in this session was validated with real
data/synthesized inputs where possible (real Vosk transcription on real
synthesized speech, real landmark fitting, real skin-blob pixel tests) —
`README.md` documents exactly what was tested with what, including the
honest gaps (no real human speech, no real replay footage).

## 13. Known limitations

Consolidated from the honesty notes throughout `README.md`:

- Classical CV spoof signals (texture/moiré/spatial) are heuristic, not a
  trained deepfake classifier — one signal among several, not a standalone verdict.
- Skin-color hand detection is lighting/skin-tone dependent.
- Voice replay detection threshold (0.999 cosine similarity) is reasoned,
  not empirically validated against real speech.
- Voice challenge-response does not claim resistance to targeted
  voice-cloning attacks.
- Face-manipulation attack simulation is a classical-CV approximation, not
  a real deepfake benchmark.
- No component has been tested against real, unscripted adversarial
  attempts by a third party — only against the scenarios described here.
- **This is not a claim of production-grade Aadhaar e-KYC readiness.** See
  positioning language in Part 7 of the original brief — that framing
  should be used verbatim in the presentation, not "we can detect every
  deepfake."

## 14. SIH feedback → implementation mapping

| Feedback | Addressed by |
|---|---|
| Deeper implementation plan | This document + `README.md`'s per-feature technical detail |
| Avoid expensive external APIs | Zero paid APIs anywhere; Vosk/OpenCV/scikit-learn all local |
| Prefer local models / systematic innovation | Every new capability (landmarks, hand tracking, ASR) runs fully offline |
| Explain trust-score calculation | `README.md` → "How the trust score works" + "Calibration" |
| Define risk thresholds | `README.md` → "Adaptive authentication — risk tiers" |
| Improve face-capture UI | In-browser camera preview + live analysis overlay (prior session) |
| Stronger deepfake detection research | Texture/moiré/spatial (prior session) + face-manipulation simulation (this session) |
| Investigate Vision Transformers | Evaluated and explicitly declined — see `README.md`'s ViT honesty note; classical multi-signal approach used instead, documented as the reasoned alternative |
| Metadata Analysis | EXIF analysis on Attack Lab uploads (prior session) |

## 15. Suggested live presentation sequence

`DEMO_SCRIPT.md` — four steps, ~60-90 seconds, plus a 30-second fallback
and a prepared answer for "can this detect real deepfakes?"
