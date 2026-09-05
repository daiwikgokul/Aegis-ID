# AEGIS·ID — Test Plan

Manual test procedure for local validation before presenting. Where a test
was already validated programmatically during development (synthetic data,
mocked hardware), that's noted — those confirm the *logic* is correct; you
still need to run the manual pass to confirm real-camera/real-mic behavior,
which could not be tested in the development environment (no physical
webcam/microphone access there).

For each test: **Expected result**, **signals involved**, **trust-score
behavior**, **risk tier**, **grant/step-up/deny**, and **known
limitations** where relevant.

---

## FACE

### 1. Genuine face, good conditions
- **Expected:** Granted on first pass.
- **Signals:** All five factors should score well; spoof-resistance high
  (real texture), image-quality high (good lighting/focus).
- **Trust score:** Should clear `GRANT_THRESHOLD` (65).
- **Risk tier:** LOW.
- **Result:** GRANTED.
- **Limitations:** None expected — this is the baseline case.

### 2. Wrong person (different enrolled identity, or unenrolled person)
- **Expected:** Denied, no identity assigned.
- **Signals:** `match_consistency` should be 0 — no frames should clear
  `MATCH_CONF_LIMIT` (75).
- **Trust score:** Forced to 0.0 when no frames match (see `_score_signals`).
- **Risk tier:** CRITICAL.
- **Result:** DENIED. Validated programmatically (scenario 5 in
  `test_trust_matrix.py`).
- **Limitations:** A close lookalike or an under-trained/overfit model
  could theoretically produce partial matches — this is the "borderline
  match" test below, not this one.

### 3. Poor lighting
- **Expected:** Likely step-up, not outright denial.
- **Signals:** `image_quality` drops (low weight, 0.06, by design — see
  README calibration notes); `recognition_confidence` may also drop if
  LBPH distance increases under noise.
- **Trust score:** Validated at ~65.4 in the audit's synthetic scenario —
  right at the boundary, could go either way with real conditions.
- **Risk tier:** LOW-to-MEDIUM depending on exact severity.
- **Result:** GRANTED or STEP_UP, should **not** be an outright DENY for
  lighting alone.
- **Limitations:** Very poor lighting could still push into step-up or
  even deny territory — that's correct behavior, not a bug, if the face
  genuinely can't be read reliably.

### 4. Glasses
- **Expected:** Step-up likely, not denial.
- **Signals:** Glasses glare can register as a moiré-like frequency
  artifact (false-positive-prone), and can interfere with EAR-based blink
  detection (landmark occlusion).
- **Trust score:** ~60.4 in the audit scenario.
- **Risk tier:** MEDIUM.
- **Result:** STEP_UP → should GRANT after completing the challenge.
- **Limitations:** This is a genuine, documented weak spot — glasses glare
  resembling a spoof signal is an honest limitation of the moiré heuristic,
  not something fully solved here.

### 5. Borderline face match (lighting/appearance changed since enrollment)
- **Expected:** Step-up, not an outright grant or deny.
- **Signals:** `match_consistency` and `recognition_confidence` both
  moderate; other factors can be strong.
- **Trust score:** ~57.7 in the audit scenario.
- **Risk tier:** MEDIUM.
- **Result:** STEP_UP. Validated programmatically (scenario 10).
- **Limitations:** If enrollment images are stale (old haircut, major
  weight change, etc.), consider re-enrolling rather than relying on
  step-up to compensate indefinitely.

---

## LIVENESS

### 6. Successful blink challenge
- **Expected:** Challenge completes, contributes to liveness/step-up.
- **Signals:** `_count_blinks()` via facial landmarks (EAR) — falls back to
  Haar eye cascade if landmarks unavailable.
- **Result:** `challenge_completed: true`.
- **Limitations:** Real hardware can still have a noisier eye-detection
  rate than assumed — if blink counts look inflated, the response includes
  an explicit note suggesting lighting/angle fixes.

### 7. Failed blink challenge (didn't blink twice)
- **Expected:** Hard-denied regardless of trust score.
- **Signals:** `challenge_completed: false`.
- **Trust score:** Irrelevant — the hard gate overrides even a
  near-perfect score (validated at trust 74.2, scenario 9).
- **Risk tier:** CRITICAL.
- **Result:** DENIED.

### 8. Head turn (turn_head challenge), successful
- **Expected:** Completes if a real, if slight, turn is performed.
- **Signals:** Nose-tip landmark tracking (falls back to face bbox
  centroid), excursion ≥ `CHALLENGE_MOTION_THRESHOLD` (0.06).
- **Result:** `challenge_completed: true`.
- **Limitations:** `haarcascade_frontalface_default.xml` is frontal-only —
  a real profile turn can lose face detection entirely mid-motion. The
  instruction says "slightly" turn for exactly this reason.

### 9. Head turn, failed (didn't turn far enough / turned the wrong way)
- **Expected:** Hard-denied.
- **Result:** DENIED, same hard-gate logic as blink.

### 10. Hand raise, successful (correct side, proper sequence)
- **Expected:** Completes when hand starts away, moves in, holds ≥5 frames.
- **Signals:** `_hand_regions_present()` skin-blob detection outside face
  box, at/above shoulder level.
- **Result:** `challenge_completed: true`. Validated programmatically
  end-to-end through both registration and verification pipelines with
  synthetic skin-colored test frames.
- **Limitations:** Lighting/skin-tone dependent (classical CV, not a
  trained hand detector) — stated explicitly in code and README.

### 11. Hand raise, wrong hand (e.g. right raised when left requested)
- **Expected:** Fails — the evaluator checks the *specific* requested side.
- **Result:** DENIED. Validated programmatically (7-scenario test suite
  in development, all passed including this one).

### 12. Hand raise, already raised before challenge started
- **Expected:** PASSES — this was fixed after a real false-rejection
  report. Only the sustained-hold requirement gates completion now; a
  hand already up when capture starts (very common, since people
  naturally comply during the pre-capture "get ready" pause) is expected
  behavior, not treated as suspicious.
- **Result:** GRANTED (challenge completed), assuming the hold duration
  requirement (≥5 frames) is met. Validated programmatically against the
  exact reported scenario (16 frames, hand present throughout).

### 13. Static image / no real liveness at all
- **Expected:** Zero blinks, zero motion, zero hand movement — challenge
  cannot complete regardless of which type is issued.
- **Result:** DENIED at step-up (or lands in the printed-photo attack
  scenario below for the first-pass case).

### 14. Video replay of the person performing the SAME action being asked
- **Expected:** This is the honest hard case. See "Known limitations" —
  passive liveness (blink presence, motion) genuinely can't fully
  distinguish a live person from a video of them performing a similar
  action; the *unpredictability* of which specific challenge gets issued
  (chosen fresh, not knowable in advance) is the actual defense here, not
  the liveness signal alone.
- **Limitations:** A sufficiently sophisticated attacker with a library of
  pre-recorded responses to every possible challenge, played back
  correctly in real time, is out of scope for this system. State this
  plainly if asked.

---

## PRESENTATION ATTACKS

### 15. Printed photo
- **Expected:** May pass face-match, should fail on liveness + spoof
  signals at first pass or the step-up challenge gate.
- **Signals:** `texture` low (flat print vs. real skin), `liveness` low
  (no real blink, though a "steady" eye-visible ratio can partially
  compensate — a documented finding from the Part 2 audit).
- **Trust score:** ~55.0 in the audit scenario → STEP_UP, then denied at
  the challenge gate (a static photo cannot blink/turn/raise a hand).
- **Risk tier:** MEDIUM at first pass, effectively CRITICAL once step-up's
  challenge fails.
- **Result:** DENIED (via the challenge gate, not necessarily the first
  pass's numeric score alone — this nuance is worth explaining to judges).

### 16. Phone / screen replay (static image shown)
- **Expected:** Similar to printed photo, plus moiré signal specifically.
- **Signals:** `moire` low, `texture` low.
- **Result:** DENIED, same mechanism as above.

### 17. Phone / screen replay (video WITH visible blinking)
- **Expected:** The harder case — passive liveness can look convincing.
- **Signals:** `spoof_resistance` (moiré/texture/spatial) is what actually
  catches this, not liveness.
- **Trust score:** ~65.4 with spoof=30 — **this was the exact case the
  Part 2 audit found bypassing step-up entirely before the
  `CRITICAL_SPOOF_THRESHOLD` fix.** Now correctly capped at step-up
  (HIGH tier).
- **Risk tier:** HIGH (spoof-veto triggered).
- **Result:** STEP_UP → voice becomes mandatory at this tier → DENIED if
  voice isn't enrolled/doesn't match.

### 18. Video loop replay
- **Expected:** Same family as #17.
- **Result:** Same mechanism.

### 19. Face manipulation / deepfake scenario (simulated)
- **Expected:** Measurably lower spatial-consistency score than a genuine
  capture.
- **Signals:** `spatial` specifically (patch-consistency check) — tested
  directly at 97.8 → 90.5 on synthetic texture.
- **Limitations:** **This is a classical-CV approximation of documented
  manipulation tells (warping, lighting mismatch, periodic patterns), not
  a real deepfake and not validated against real deepfake output.** State
  this explicitly if demonstrating it.

---

## VOICE

### 20. Genuine speaker + correct phrase
- **Expected:** Both speaker and content checks pass.
- **Signals:** `speaker_passed: true`, `content_passed: true`.
- **Result:** `decision: match`.

### 21. Genuine speaker + incorrect phrase (says something else)
- **Expected:** Speaker passes, content fails → overall fail.
- **Result:** `decision: no_match`. Validated programmatically via mocked
  transcription (combining-logic test in development).

### 22. Wrong speaker + correct phrase (someone else reads the displayed phrase)
- **Expected:** Content passes, speaker fails → overall fail.
- **Result:** `decision: no_match`. Validated programmatically.

### 23. Replay of a previous recording
- **Expected:** May be caught by the near-duplicate fingerprint check if
  it's literally the same audio as a recent attempt; otherwise likely
  fails content-matching anyway since phrases are randomized fresh.
- **Signals:** `replay_flags.near_duplicate_of_recent`.
- **Limitations:** The 0.999 similarity threshold is reasoned, not
  validated against real speech recordings (documented honestly in
  README — synthetic test audio was too spectrally simple to properly
  stress-test this).

### 24. Silence / no speech
- **Expected:** `error: no_speech`, doesn't proceed to scoring.

### 25. Background noise
- **Expected:** Should still work if speech is intelligible above the
  noise floor; heavy noise may reduce both ASR accuracy and speaker-match
  confidence. Not separately tuned for noisy environments — untested
  against real background noise.

### 26. Different speaking speed (genuine speaker)
- **Expected:** MFCC + GMM should tolerate normal speed variation; not
  empirically validated against real recordings at varying speeds.

---

## Regression testing

Run `python test_trust_matrix.py` after any change to `WEIGHTS`,
`_decide()`, `_score_signals()`, or the threshold constants. It exercises
15 scenarios against the real scoring functions (not hand-arithmetic) and
flags any decision that no longer matches its expected outcome.
