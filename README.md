# AEGIS·ID — Adaptive Identity Verification

Deepfake-resistant facial identity verification with a live trust-score
engine and risk-based adaptive authentication, built on OpenCV (Haar
cascade + LBPH) and Flask.

## Run it

```
pip install -r requirements.txt
python app.py
```

Then open http://127.0.0.1:5000 in your browser. A webcam is required.
The native cv2 window still opens as before, and the same live feed also
streams into a popup inside the browser (see below) — you don't need a
second monitor or to alt-tab to see what the camera sees.

**Note on the dev server:** `app.run(...)` runs with `use_reloader=False`.
Flask's debug auto-reloader watches every loaded module's file — including
site-packages, not just this project — and restarts the whole process on
any change it notices, including spurious mtime touches from things like
OneDrive syncing in the background. A restart mid-request silently kills
whatever capture was in flight (camera or microphone) with no real error
message, which is exactly the kind of failure a multi-second voice
recording is especially exposed to. If you're actively editing `app.py`,
restart the server manually after each change instead of relying on
auto-reload.

## One continuous registration, one continuous verification

Registration and verification are each a single button click that runs
everything in sequence automatically — no separate pages or manual steps
to click through mid-flow:

- **Register** (`registerUser()` in `templates/index.html`): face capture
  with a live liveness challenge, then — automatically, no separate click —
  voice enrollment (two takes). If voice recognition isn't available on the
  server, that phase is skipped with a clear note rather than blocking the
  whole registration; face enrollment still completes normally.
- **Verify** (`verify()`): face capture and trust scoring, then
  automatically a step-up challenge if the score lands in the medium-risk
  band (no manual "Run Step-Up" click needed anymore), then automatically a
  voice confirmation if this identity has voice enrolled. One combined
  message at the end reflects everything that ran — e.g. "Step-up
  verification passed. Voice confirmed (score 91). Access granted — both
  factors verified," or "Trust score cleared the threshold on the first
  pass. Access granted. (Voice not enrolled for this user, so this was
  face-only.)"
- **Voice mismatch overrides a face pass.** If face (and step-up, if it
  ran) both grant access but the automatic voice check doesn't match, the
  overall result is downgraded to denied — that's the actual point of a
  second factor, not a bug. The trust-score gauge and adaptive-authentication
  step tracker on the Verify page now show 5 stages (capture → match → risk
  → decision → voice) so the voice phase is visible in the same place as
  everything else, not off on a separate page.
- The standalone **Voice ID** page still exists for re-enrolling a voice
  independently or testing voice matching in isolation — the unified flow
  doesn't remove that option, it just means you don't *need* to visit it
  for a normal registration or verification anymore.

## In-browser camera preview

Every capture flow (Register, Verify, Step-up, and a live Attack Lab
attempt) now shows the live camera feed as a popup inside the web page
itself, not just in the separate native `cv2.imshow` window. The backend
publishes the same annotated frame (bounding boxes, match labels, challenge
instructions) that the native window shows to a thread-safe buffer, and
`/camera/stream` serves it to the browser as an MJPEG stream.

This requires the Flask dev server to run with `threaded=True` (already set
in `app.py`) — a single-threaded dev server can't serve a long-lived video
stream and the actual capture request at the same time, and the two would
block each other. If you ever change `app.run(...)`, keep `threaded=True`
or the popup will hang.

### Live analysis overlay

The same frame also draws exactly what the trust-score engine is
computing, live, per frame — this is a transparency layer, not a separate
illustration of it:

- **Patch-consistency grid** — the 4×4 grid used by the spatial-anomaly
  check, color-coded by how much each patch's noise energy deviates from
  the face's average (green = consistent, red = anomalous). It's the same
  numbers feeding `_spatial_anomaly_score`, drawn directly, not a mockup.
- **Eye-detection boxes** — small boxes wherever the eye cascade currently
  detects eyes, so blink/liveness detection isn't a black box either.
- **HUD readout** — live quality / texture / moiré / spatial scores in the
  corner, recomputed every frame regardless of whether that frame ends up
  being an accepted sample.

Implemented in `_draw_analysis_overlay()` in `app.py`. Overhead is small
(~4ms/frame measured in testing) compared to Haar cascade face detection
itself (30-80ms/frame), so it doesn't meaningfully affect capture speed.

## How the trust score works

Each verification pass keeps sampling webcam frames until it has at least
`MIN_FACE_SAMPLES` (25) usable face frames, or `MAX_CAPTURE_SEC` (12s) has
passed — a fixed short window alone can yield too few frames to score
reliably on a slower camera, which is what caused the score to swing wildly
between runs. It then scores five independent signals:

- **Match consistency** – % of sampled frames that matched the same registered identity
- **Recognition confidence** – how close the LBPH match was (trimmed mean, so a couple
  of bad-angle frames don't skew it), inverted to a 0-100 scale
- **Liveness** – a blend of blink detection and how steadily the eyes stayed visible
- **Spoof resistance** – classical anti-spoofing signals (see below)
- **Image quality** – sharpness (Laplacian variance) and exposure of the captured face

These are weighted into a single 0-100 trust score. If a run genuinely can't
gather enough face frames (bad lighting, camera too far away), the response
flags `sample_reliable: false` and the UI tells the person to move closer /
improve lighting rather than silently returning a noisy score.

### Calibration (recalibrated — scores were too harsh)

Real testing surfaced a genuine calibration bug, not just "make it easier."
The recognition-confidence formula reused `MATCH_CONF_LIMIT` (the LBPH
distance cutoff for "is this even a match") as *also* the scale for turning
that distance into a 0-100 score — so a match that just barely cleared the
accept threshold scored close to zero confidence, even though it had
already been accepted as genuine. Fixed by decoupling them:
`RECOGNITION_SCORE_SCALE` (100) is now used only for scoring, separate from
`MATCH_CONF_LIMIT` (loosened 65 → 75) which decides accept/reject.

Sharpness and LBP-texture-entropy scales were also both harsher than
achievable in practice — `_quality_score()`'s sharpness divisor loosened
400 → 200, and `_lbp_texture_score()`'s entropy is now normalized against
6.5 bits instead of the theoretical (and essentially unreachable even for a
sharp, well-lit real face) 8-bit ceiling. `image_quality`'s weight in the
overall trust score also dropped from 0.10 to 0.06 — it's the factor most
purely dependent on the camera's own hardware rather than anything about
who's in front of it, so a weak webcam shouldn't be able to single-handedly
block a real person. `GRANT_THRESHOLD` (75→65), `STEPUP_THRESHOLD` (50→40),
and `STEPUP_GRANT_THRESHOLD` (80→70) were lowered to match. None of this is
calibrated against a real multi-user dataset — they're reasoned adjustments
based on working through the formulas, not empirically tuned values, so
expect to revisit them once you've tested against your own camera.

### Low camera quality

When a pass's average `image_quality` factor comes in under
`CAMERA_QUALITY_WARN_THRESHOLD` (45), the response includes a
`camera_quality_note` explaining that the low score is likely about
lighting/resolution rather than the match itself, with concrete suggestions
(move closer, add light, clean the lens) — surfaced in the UI on both the
granted and denied paths, so a weak camera doesn't just quietly and
mysteriously lower someone's odds with no explanation.

## Challenge-response step-up

A step-up pass doesn't just re-run the same passive scan for longer. It
issues a randomized live challenge — `/verify/stepup/challenge` picks one of
**blink twice**, **turn your head**, or **nod** — which the frontend shows
*before* capture starts, so the person has to respond to something they
couldn't have known was coming. `/verify/stepup/run` then evaluates whether
that specific action was actually performed (blink timestamps, or normalized
face-center movement across frames for turn/nod) as a **hard gate**: failing
the challenge denies access regardless of the numeric trust score. This is
meaningfully harder to spoof with a static photo or a looping video than the
old approach of just requiring a higher score on more passive frames.

### Blink detection — rebuilt on facial landmarks, not Haar cascade tuning

Three rounds of tuning the Haar eye cascade's detection parameters and the
blink-counting algorithm around it all ran into the same wall: a Haar
cascade is a coarse *per-frame object detector* — it answers "is there an
eye-shaped thing here" independently on every single frame, with no memory
and no sub-object structure. That's structurally why it flickers: genuinely
open eyes get missed constantly, and there's no reliable way to tell that
apart from an actual blink using presence/absence alone. Real testing
confirmed this concretely — a report of "blinked 7 times" while not
blinking at all, reproduced almost exactly with a synthetic ~35% per-frame
miss rate.

The actual fix is a different tool, not a better-tuned version of the same
one: **facial landmark tracking + Eye Aspect Ratio (EAR)**, the standard
technique behind most real blink/drowsiness-detection systems (Soukupová &
Čech, "Real-Time Eye Blink Detection using Facial Landmarks", 2016). Instead
of asking "eye or no eye" from scratch every frame, it locates ~68 specific
points on the face — including the eyelid contour — and measures how open
the eye geometrically is. EAR stays roughly flat while eyes are open and
drops sharply when they close, because it's tracking actual eyelid shape,
not re-deciding object presence from nothing each frame.

**Implementation**: uses OpenCV's own `cv2.face.createFacemarkLBF()` —
already part of `opencv-contrib-python`, a dependency this project already
has, so this needed **no new pip package**. It's loaded with the standard
pretrained model referenced directly in [OpenCV's own
documentation](https://docs.opencv.org/3.4/d7/dec/tutorial_facemark_usage.html)
(`lbfmodel.yaml`, bundled in this folder — see `LANDMARK_MODEL_URL` in
`app.py` if you need to re-fetch it), which outputs landmarks in the same
standard 68-point ordering dlib's classic predictor uses, so the eye
indices (`RIGHT_EYE_IDX`/`LEFT_EYE_IDX`, points 36-47) are the same ones
used in essentially every EAR reference implementation.

**Validated concretely**, not just plausible-sounding:
- The EAR formula was tested against synthetic landmark geometry with a
  known open eyelid gap vs. a known closed one — correctly scored 0.300
  (open) vs. 0.050 (closed) against the classification threshold of 0.21.
- The full capture pipeline was run end-to-end on 281 frames of pure random
  noise (standing in for "no real face, no real blinking") — reported
  **0 false blinks**, where the Haar-based approach reported 7+ in
  comparable noise conditions on the same kind of input.
- Performance overhead measured directly: ~1.6ms per frame, negligible.

**Graceful fallback, not a hard requirement.** If the model file is missing,
`app.py` tries to download it once automatically from its canonical GitHub
location; if that also fails (offline, blocked network), `LANDMARK_AVAILABLE`
flips to `False` and every capture loop automatically falls back to the
Haar eye cascade approach — same pattern as `VOICE_AVAILABLE` for voice
recognition. The app still works either way; it's just meaningfully more
reliable with landmarks available; The blink counting logic itself
(`_count_blinks()`, run-length based) didn't need to change — landmark
tracking just feeds it a much cleaner boolean signal than Haar detection
did.

The live camera popup also now draws the actual eye-contour points (not
just a bounding box) when landmarks are available, and uses the nose-tip
landmark instead of the face bounding-box centroid for turn/nod motion
tracking — a single tracked point is more stable frame-to-frame than the
edges of a re-detected box, as a side benefit of already computing the
landmarks for EAR.

**One real bug found on actual hardware after shipping this**:
`_get_landmarks()` originally assumed a fixed nesting depth on
`FACEMARK.fit()`'s return value (`landmarks[0][0]`), based on what this
sandbox's OpenCV build returned. A different OpenCV build wrapped the
per-face landmark array with a different number of dimensions, which
crashed with `IndexError: index 36 is out of bounds for axis 0 with size 1`
— the code was reading a single stray point instead of the full 68. Fixed
by reshaping with `.reshape(-1, 2)` instead of hardcoded indexing, which
collapses whatever wrapping is present as long as the total element count
resolves to 68 points, plus an explicit shape check that returns `None`
(triggering the Haar fallback) rather than crashing if it ever doesn't.

### Live challenge progress (was a black box)

Step-up used to give no feedback until the very end — pass or fail, with no
visibility into why. The in-browser camera popup now shows live progress
during capture: a running blink count (`1/2`) for the blink challenge, or a
live percentage of the required motion for turn/nod — computed with the
exact same `_evaluate_challenge()` function the final decision uses, so
what's on screen can't drift from what actually gets decided.

### Turn/nod calibration

`haarcascade_frontalface_default.xml` is a **frontal-only** detector. A
real profile turn commonly makes it lose the face entirely partway through
the motion (so the peak of the turn is never captured), and even short of
that, a turning head tends to shrink the tracked bounding box more than it
translates it. The original threshold (`CHALLENGE_MOTION_THRESHOLD = 0.10`)
assumed more trackable motion than that setup can reliably deliver — lowered
to `0.06`, and the instruction text now says "slightly" turn/nod, which sets
the right expectation and keeps the motion within the range the detector can
actually track continuously. Still a tunable constant, not something
empirically calibrated against a real camera yet.

## Hand-raise liveness

Three additional challenges — `raise_left_hand`, `raise_right_hand`,
`raise_both_hands` — additive to blink/turn/nod, randomly selectable at
both registration and step-up. "Left"/"right" are frame-relative (what the
on-screen instruction says), not the person's own left/right, so there's
no camera-mirroring ambiguity.

**Classical CV, not MediaPipe**: YCrCb skin-color segmentation
(`_detect_skin_mask`, `_hand_regions_present`) — deliberately not
MediaPipe Hands, since its current API needs a runtime-downloaded model
file from a domain whose long-term reachability wasn't something this
project wanted to depend on (the same reasoning that led to OpenCV's own
Facemark for face landmarks instead of MediaPipe's face mesh). Zero new
dependencies; benchmarked at 1.78ms/frame.

**Sequence-validated, not presence-checked**: a per-frame `(left_present,
right_present)` trace is evaluated for a sustained hold (`HAND_MIN_HOLD_FRAMES`
= 5 consecutive frames) in the correct target region — ruling out "never
raised," "raised too briefly," and "raised the wrong hand/combination."
Tested against 7 synthetic trace scenarios covering exactly these failure
modes, all correctly rejected.

### Fixed: a real false-rejection found via actual use

The original design also required the hand to be *absent* early in the
capture and only appear partway through (`started_away`/`delayed_onset`),
intended to distinguish a genuine live raise from a static photo already
showing a raised hand. **A real user reported this rejecting a genuine,
correctly-performed hand raise** — reproduced directly: the frontend shows
the challenge instruction with a "Get ready…" pause *before* capture
starts. Unlike a blink or a quick head turn (naturally reactive to a "go"
cue), "raise and hold" is a *sustained* pose — a compliant person starts
performing it as soon as they read the instruction, during that pre-capture
pause, not after frame capture visibly begins. By the time the first frame
is actually captured, a fast, correctly-complying person's hand is already
up. The check was measuring reaction speed, not liveness, and rejected
exactly the behavior it should have rewarded.

**Fix**: `started_away`/`delayed_onset` no longer gate `completed` — only
the sustained-hold requirement does. They're still computed and reported
in the response detail for visibility, just not blocking. Verified against
the exact reported scenario (hand present for all 16 captured frames, i.e.
already up from frame 1) — now correctly passes — while all 7 original
failure-mode tests (too-brief flash, never raised, wrong hand, wrong
combination for both-hands) still correctly fail, confirming the fix is
precise and didn't weaken those.

**Honest trade-off**: this does weaken hand-raise's own contribution to
distinguishing a genuine live hold from a static photo of an
already-raised hand specifically — that residual defense now rests more on
the rest of the trust score (face match, spoof-resistance texture/moiré/
spatial signals) than on this challenge in isolation. Stated plainly
rather than glossed over.

## Enrollment integrity — deepfake defense doesn't start at verification

Every defense described above only matters if **enrollment** can't be
spoofed — if someone can register a printed photo or a screen replay of
another person as a "verified" identity, every downstream check is
protecting a compromised baseline. Registration is guarded the same way
step-up verification is:

- `/register/challenge` issues the same randomized live challenge (blink
  twice / turn head / nod) used for step-up, shown to the person *before*
  capture starts.
- `/register/<user>?challenge=<id>` runs the actual capture. Every frame is
  scored through the same `spoof_resistance` check (LBP texture entropy +
  FFT moiré detection) used at verification time — frames that look like a
  print or screen replay (`ENROLL_SPOOF_MIN = 55`, tune in `app.py`) are
  captured for liveness evaluation but **never saved** into the training set.
- Captures land in a staging folder first (`dataset/_staging/<user>`) and
  are only committed into the real per-user dataset folder if **both** the
  challenge was completed **and** at least `MIN_ENROLL_SAMPLES` (20) frames
  passed the spoof check. A failed or spoofed attempt is deleted from
  staging and never touches (or overwrites) a user's existing valid images —
  so a second, legitimate enrollment attempt after a failed/spoofed one is
  unaffected.
- The registration response reports `captured`, `rejected_low_spoof`,
  `avg_spoof_resistance`, and the challenge outcome, so the UI can show
  *why* an enrollment was rejected instead of a silent failure.

## Spoof resistance (classical anti-spoofing)

Three lightweight, dependency-free CV techniques feed a fifth trust-score
factor:

- **LBP texture entropy** – computes an 8-neighbor Local Binary Pattern code
  per pixel and scores the Shannon entropy of the resulting histogram. Real
  skin has rich, irregular micro-texture; a printed photo or a screen-displayed
  face tends to be texturally flatter — lower entropy.
- **FFT moiré detection** – photographing a screen (phone/monitor replay)
  tends to introduce sharp, isolated peaks in the mid-frequency band of the
  face crop's frequency spectrum that a genuine in-person face doesn't
  produce. The score falls as that peak grows.
- **Patch-consistency analysis** – splits the face into a 4×4 grid and
  compares high-frequency noise statistics across patches. This is the
  same core idea vision transformers use (tokenizing an image into
  patches) applied with classical statistics instead of a trained model.
  A real ViT-based deepfake classifier needs pretrained weights and
  GPU-class inference — impractical for a dependency-light local app with
  no internet access to a model hub — so this is the honest, feasible
  approximation of that research direction, not a claim of running an
  actual transformer. Blended/warped deepfake content tends to leave
  mismatched noise statistics between the manipulated region and its
  surroundings, even when color/lighting was blended well.

**Honesty note:** these are heuristic classical-CV signals, not a trained
deepfake classifier. They're one piece of evidence weighted alongside four
others, not a standalone verdict — busy real backgrounds or unusual lighting
can trigger false positives. Treat this as a legitimate, explainable
first line of defense (and a good talking point for judges), not a claim
of state-of-the-art deepfake detection.

## Metadata analysis

Checks EXIF data on uploaded files in the Attack Lab's simulate mode —
presence of camera Make/Model/timestamp fields, and editing-software tags
(Photoshop, GIMP, Snapseed, etc — see `KNOWN_EDITORS` in `app.py`).

This is deliberately **not** folded into the core weighted trust score.
A live webcam frame from `cv2.VideoCapture` is a raw pixel array — it was
never a file, so it has no EXIF block to check. Forcing a "metadata" factor
into the live pipeline would mean it's always a constant, uninformative
value on every real verification. Instead it's reported as a separate
signal, specific to the file-injection threat model: someone bypassing the
physical camera entirely by feeding in a pre-existing image or video file.
Live camera attempts report `metadata_confidence: null` with an explicit
"not applicable" message — that absence is itself the honest answer to
"what about metadata if the camera is bypassed."

## Audit log and lockout

Added in response to two real questions a judge is likely to ask: "how
would a compliance officer review this system's decisions" and "what
stops someone just retrying forever."

### Verification audit log

Deliberately separate from `ATTACK_LOG` (which is Attack Lab test attempts
and intentionally resets on restart — documented as such, not a bug).
`/verify` and `/verify/stepup/run` now write every **scored** decision
(identity, trust score, risk tier, decision, challenge status, timestamp)
to `verification_log.csv` — a real file, persisted, survives restarts.
Deliberately excludes technical capture failures (no face detected, camera
unavailable) from the log — those aren't access decisions, they're failed
attempts to even reach one, and including them would dilute what the log
actually represents. View recent entries at `/verify/log`, or download
the complete history as CSV via `/verify/log/export` — both from a new
**Audit Log** page in the sidebar. Tested directly: logged a real (mocked)
denied decision, confirmed the CSV has the right header and row format,
confirmed the missing-file case returns a clean "nothing logged yet"
response rather than a broken download.

### Lockout — fixed from global to two-tier, per a real limitation callout

The original version was a single **global** cooldown: 5 denials from
*anyone*, in any combination, locked out *everyone*. That was a known,
explicitly-documented limitation (there's no "claimed identity" input at
the point `/verify` is called, so a traditional per-account lockout
doesn't check cleanly *before* a capture) — but it's fixable, because
identity *is* known immediately *after* a capture, which is enough to do
this properly instead of settling for the blunt version.

**Two independent buckets now:**

- **Per-identity** (`_recent_identity_denials`, `_identity_lockout_until`):
  when a capture matches a specific enrolled identity but still ends in a
  denial (failed challenge, failed voice, spoof veto, below the step-up
  bar), that's tracked against *that identity specifically*. 5 such
  denials within the window locks out only that identity for the cooldown
  — checked and enforced right after identity is known, overriding even an
  otherwise-passing decision for the duration (that's the actual point of
  a lockout, not just tallying failures). A grant clears only that
  identity's own history.
- **Anonymous/global** (`_recent_anonymous_denials`, `_anonymous_lockout_until`):
  a denial with *no* identity matched at all (nobody recognized) can't be
  attributed to a specific enrolled person, but repeated unmatched
  attempts in a short window is exactly the pattern a brute-force/random-
  photo-spam attempt would produce — so it's still tracked, as its own
  separate, system-wide cooldown, checkable *before* a capture starts
  (unlike the per-identity case) since it doesn't depend on knowing who's
  about to attempt.

**The fix that matters most in practice**: Alice's repeated bad-lighting
failures no longer block Bob from verifying in the meantime. Verified this
exact scenario directly, end-to-end, through the real `/verify` route —
scripted Alice failing 5 times, then a 6th attempt for her that *would
have scored a genuine grant* was correctly overridden to denied (with a
specific "alice is temporarily locked out" message), while Bob's
independent attempt immediately after succeeded completely unaffected.
Also confirmed the lockout-enforced denial doesn't register as a *new*
failure itself (would otherwise let someone perpetually extend their own
lockout just by retrying during the cooldown) — Alice's denial count
stayed at 5 after the enforced 6th rejection, not 6.

`/verify/log` reports the global lockout status plus a count of how many
identities are *currently* locked out — deliberately not *which* ones,
since naming specific people as "currently failing verification" on a
page anyone can view is its own minor information leak, not something
worth trading for marginal debugging convenience.

## Attack Lab — security testing console

A dedicated page (`Attack Lab` in the sidebar), redesigned as a genuine
testing console rather than a standalone demo gimmick: **it runs
presentation attacks through the exact same scoring pipeline used during
normal verification** — nothing is hard-coded, nothing special-cased for
the "attack" framing. A blocked attempt is the real trust-score engine
doing its job, not a scripted outcome.

### Two clearly distinguished modes

- **Live attack demonstration** — present an actual printed photo, phone
  screen, or video to the webcam (or speak into the mic for voice attacks).
  Runs through `/attack/run` (face) or `/attack/voice/run` (voice) using
  the identical capture/scoring code path as `/verify`. The more convincing
  demo, since nothing is simulated — but depends on lighting, props, and a
  second device being ready.
- **Attack simulation** (face attacks only) — **self-contained by
  default**: pick an already-enrolled identity and the server sources one
  of their own registration photos automatically (`_get_enrolled_user_photo()`),
  no file to prepare beforehand. Uploading a photo still works as a
  fallback (tucked into a collapsible "Upload a photo instead" toggle) for
  bringing your own image. The server applies synthetic degradation
  matching the selected attack type, then scores the result through
  `/attack/simulate`. No physical prop needed, fully repeatable — but
  **these are approximations of attack signatures, not real deepfakes or
  real replay footage**, stated plainly in the UI, not just here.
  - **"Generate New Variant"**: all four degraders (`_degrade_printed_photo`,
    `_degrade_phone_replay`, `_degrade_video_loop`, `_degrade_face_manipulation`)
    now randomize their parameters per call within realistic ranges (blur
    strength, moiré frequency/phase, JPEG quality, warp strength) — verified
    directly that repeated calls on the identical source photo produce
    measurably different output (mean pixel diff > 0 in every case), not an
    identical result every time. A real presentation attack wouldn't look
    bit-for-bit identical on a second attempt either, so this is more
    honest, not just more convenient.
  - **Metadata analysis is skipped, honestly, for self-sourced photos.**
    Files in this app's own dataset are written by `cv2.imwrite` during
    registration and never carry real camera EXIF, regardless of whether a
    given simulation is meant to look "genuine" or "suspicious" — running
    the metadata check on them would report "no EXIF found" as a constant,
    a misleading finding dressed up as evidence. `_simulate_verification(...,
    skip_metadata=True)` reports "not applicable" instead, same honest
    framing already used for live camera attempts (which also have no file
    metadata to check, for an unrelated but analogous reason).

### Video upload support

Uploading a photo used to be the only option alongside self-contained
generation — video files were rejected outright by the file picker
(`accept="image/*"`), which was really a UX complaint in disguise: a
person reported "I can't upload videos," and the honest cause was that
the feature genuinely didn't exist yet, not that it was hidden or broken.

Added properly rather than just documented as a limitation: `/attack/simulate`
now detects a video upload (by extension or MIME type) and extracts a
representative frame via `_extract_frame_from_video()` — OpenCV's
`VideoCapture` needs a real file path, not in-memory bytes, so this writes
to a temporary file, samples several candidate positions across the clip,
and always cleans up the temp file regardless of outcome. That frame then
flows through the exact same pipeline a photo upload would.

### Real bug found via user report: rotated video frames

A person reported `"No face was detected"` on a video with a clear,
front-facing face — the first version's single-frame, single-orientation
extraction had a real gap. Root-caused by testing directly rather than
guessing: fetched a real photo with a genuinely Haar-cascade-detectable
face, confirmed detection worked, then confirmed detection dropped to
**zero** at 90/180/270 degrees rotation with the exact same cascade used
everywhere else in the app. Phone-recorded videos very commonly store raw
pixel data in landscape orientation plus a rotation metadata flag that
video *players* apply automatically — OpenCV's decoder frequently does
not, so a portrait selfie video that looks completely normal in any video
player can come out sideways here, and a frontal-face cascade fails
completely on a sideways face.

Fixed with two complementary layers:
1. **Auto-orientation requested from the decoder** (`cv2.CAP_PROP_ORIENTATION_AUTO`)
   where the backend supports it — cheap, correct when it works, silently
   ignored where it doesn't (wrapped in its own try/except).
2. **Rotation-detection fallback, always active regardless of #1**:
   `_correct_orientation_if_needed()` tries the frame as-is, then each of
   the three other rotations, and uses whichever one the *same* face
   cascade actually detects a face in. Also broadened to sample several
   candidate frame positions across the clip (not just the middle), since
   frame-count metadata and seek-by-index are separately known to be
   unreliable for some real-world encoders.

**Verified against the actual reproduced bug**, not just the isolated
logic: built a real face image, rotated it 90° (reproducing exactly what
a mis-handled phone video looks like to OpenCV), encoded it into a video,
and confirmed the fix correctly extracts and un-rotates it — face
detected. Also re-ran the non-rotated case (regression check, still
passes) and a genuinely faceless video (still honestly reports `no_face`
rather than the fallback logic masking it).

- **Verified this actually works in this environment before relying on
  it**, not just assumed: confirmed OpenCV was built with FFMPEG support,
  then round-tripped a real synthetic video (write with `cv2.VideoWriter`,
  read back with `cv2.VideoCapture`) before wiring it into the route.
- **Verified graceful failure on a corrupt/invalid file** — garbage bytes
  with a `.mp4` extension correctly return a clear `bad_video` error
  message pointing to Live demonstration mode as a fallback, not a crash
  or an unhandled exception.
- **Metadata analysis is skipped with an accurate, source-specific
  reason** — a frame extracted from a video and re-encoded as JPEG never
  carries the original file's metadata regardless of the video's own
  authenticity, same reasoning as self-contained dataset photos, but
  worded correctly for *this* source rather than reusing the dataset
  message verbatim (caught and fixed a real bug here: the first version
  wrapped everything in `bool(...)`, which would have silently discarded
  the custom message and shown the wrong explanation).
- Supported formats: MP4, MOV, WebM, AVI (by extension/MIME sniffing) —
  MP4 is the most reliably decodable across environments.

### Pre-generated content cache — no live computation needed at demo time

Every simulated attack used to degrade a source photo live, on the spot,
on every click — cheap individually, but it meant demo reliability
depended on that computation succeeding in front of judges each time, and
"Generate New Variant" had to redo the work from scratch. Now:

- **`_build_attack_cache(user)`** runs automatically right after a
  registration completes (hooked into `register_route`, best-effort — a
  cache-generation issue never fails the registration itself), generating
  `ATTACK_CACHE_VARIANTS_PER_TYPE` (3) pre-degraded variants per attack
  type from that user's own registration photos, stored under
  `attack_cache/<user>/<attack_type>/`. Measured directly: 12 variants
  (4 types × 3) generate in ~85ms — negligible.
- **`/attack/simulate` checks the cache first.** A cache hit (the normal
  case) just reads a ready-made file — measured at ~15ms, no degradation
  computed at request time at all. "Generate New Variant" picks a
  different one of the 3 cached files, still genuinely different each
  time, still no live computation.
- **Cache misses fall back gracefully, and self-heal.** A user registered
  before this feature existed has no cache yet — the route falls back to
  live generation from their registration photo (same as before), *and*
  opportunistically saves that result into the cache, so the next request
  for that user+attack-type is instant too. Verified directly: first call
  for an uncached user correctly falls back and warms the cache; the
  second call for the same user+type then correctly hits the newly-warmed
  cache.
- **A background thread also warms the cache for already-registered users
  at startup** (`_warm_attack_cache_for_existing_users`, daemon thread,
  doesn't block server startup) — covers anyone registered in an earlier
  session, so even their *first* attack attempt is instant, not just the
  second one onward via the request-time fallback.
- Uploading a photo still bypasses the cache entirely (it's a one-off
  image, not tied to a stored identity) and degrades live, same as before.

### Attack categories (face)

`printed_photo`, `phone_replay`, `video_loop`, and — new —
**`face_manipulation`** (labeled "deepfake scenario" in the UI):
`_degrade_face_manipulation()` in `app.py` applies three classical,
well-documented manipulation tells — a local geometric warp in the
jaw/mouth region, a lighting/tone mismatch in a central "swapped" ellipse
vs. its surroundings, and a faint periodic pattern at a different spatial
frequency than the screen-replay moiré simulation. **This is explicitly
not a GAN or a deepfake generator** — the docstring and UI both say so.
Tested directly: measurably lowered the spatial-consistency sub-score
(97.8 → 90.5 on synthetic test texture), confirming the existing
patch-consistency signal responds to *structural* manipulation artifacts,
not just photographing a photograph. That's the honest claim — a
demonstration that the classical signals generalize somewhat beyond simple
presentation attacks, not a benchmark against real deepfake quality.

### Attack categories (voice)

Four scenarios via `/attack/voice/types` and `/attack/voice/run`: correct
phrase + correct speaker (control), correct phrase + wrong speaker, wrong
phrase + correct speaker, and voice replay. **Deliberately live-only, no
simulation mode** — there's no synthesized-audio degradation pipeline
analogous to the image `DEGRADERS`, and building one without real
replay/synthetic-speech samples to validate against would mean faking a
result rather than honestly demonstrating one. These scenarios reuse
`voice_verify()` exactly as normal verification does; only the scenario
label for the demo log differs.

**Bug found via the challenge-secrecy audit below, fixed**: the original
`/attack/voice/run` generated the phrase and started recording in the same
synchronous call — meaning whoever needed to speak it never actually saw
what to say, since nothing displayed it first. Fine for the scenarios that
intentionally say the wrong thing, broken for the ones that need the
correct phrase read aloud. Fixed to match the same two-step "reveal, then
record" pattern used everywhere else a phrase is involved: the frontend
now fetches `/voice/challenge` first, displays it, then passes that exact
phrase to `/attack/voice/run?...&phrase=...`. Verified end-to-end that the
phrase passed through the URL is what actually gets checked against.

### Challenge/phrase secrecy audit

Traced every challenge and phrase code path end-to-end (not assumed) to
confirm nothing is generated or exposed before the moment it's actually
needed:

- `/register/challenge` and `/verify/stepup/challenge` both pick with
  `random.choice()` **at request time** — nothing precomputed, nothing
  cached from an earlier call.
- Both are only ever fetched from inside `registerUser()` / `runStepUpPhase()`,
  themselves only triggered by an explicit user action (button click, or a
  step-up decision already confirmed from a first-pass result) — never
  pre-fetched on page load, never fetched speculatively ahead of when
  they're shown.
- Same pattern confirmed for `/voice/challenge`, called from three sites
  (`runVoicePhase()`, the standalone Voice ID page's `verifyVoice()`, and
  — after the fix above — the Attack Lab's `runVoiceAttack()`) — all fetch
  immediately before display, never earlier.
- No debug/log statement anywhere prints an actual phrase or challenge
  value — checked directly (`grep` for phrase-adjacent logging calls).
  Flask's own access log records the request path, not response bodies,
  so it can't leak a phrase either.
- `random`/`np.random` are never seeded deterministically anywhere in
  `app.py` — confirmed directly, not assumed — so challenge/phrase
  selection draws from the OS-entropy default generator, not a
  reproducible sequence.
- Only one challenge is ever issued per registration or step-up attempt
  (not a multi-challenge sequence), so "never reveal the complete sequence
  in advance" is satisfied by construction — there's no sequence to leak
  in the first place.

### Signal breakdown display

Per-attempt results now show **Face Match, Liveness, Texture, Moiré,
Spatial Consistency, Trust Score, Risk Tier, and Final Decision**
individually — previously texture/moiré/spatial were only visible blended
into one `spoof_resistance` number. New `_spoof_resistance_detail()`
returns all three sub-scores alongside the blend; threaded through both
capture loops and `_simulate_verification`. Important: this is a **display
change only** — `trust_score` still only ever weights the blended
`spoof_resistance` value (see `WEIGHTS`), never the sub-components
individually. Re-ran the full `test_trust_matrix.py` after this change;
all 15 scenarios still match expected decisions.

Both face and voice attacks log to a shared running scorecard
(`/attack/log`): total attempts, how many were blocked, and the block rate.

**Honesty note on simulation (still applies):** a repeated static frame has
zero motion and no blink transitions by construction, so liveness scores
near zero for any simulated attempt — that's not a shortcut, it's a real
consequence of the input being a single static image rather than a live
feed.

## Adaptive authentication — risk tiers

`_decide()` still governs grant/step-up/deny with `GRANT_THRESHOLD` (70),
`STEPUP_THRESHOLD` (40), `STEPUP_GRANT_THRESHOLD` (70), plus the
`CRITICAL_SPOOF_THRESHOLD` (35) veto that caps a decision at step-up
regardless of trust score when spoof-resistance is critically low.

### GRANT_THRESHOLD raised from 65 to 70, with an honest note on what this does and doesn't fix

Raised after a real observation: screen-replay attempts (a phone playing
a video, held up to the camera) were getting a direct low-risk grant
roughly 20% of the time. Before changing the number, tested the actual
cause rather than guessing: fetched a real face image, confirmed a
genuine capture and 20 randomized runs of the actual `phone_replay`
degrader both scored `spoof_resistance` in a similar 60s-70s range
(median replay: 66.6; genuine: 74-76) — meaning the existing
`CRITICAL_SPOOF_THRESHOLD` veto (35) was nowhere close to firing for this
attack type, and raising it far enough to catch replay risked also
catching genuine users, since the two distributions overlap.

Checked what raising `GRANT_THRESHOLD` to 70 would actually change, with
real numbers through the real `_decide()` code, not just reasoning about
it: a "good" screen replay (match 88, recognition 82, liveness 75,
spoof 67, quality 78) scores trust 79.8 — **granted at both 65 and 70,
no change**. A "weaker" replay (worse angle/lighting) scores 67.5 —
**granted at 65, step-up at 70**. So this change is real but partial: it
closes the gap for marginal/poorly-executed replay attempts, and does
nothing for a well-executed one, because `match_consistency` (30%) and
`recognition_confidence` (28%) — 58% of the weighted score — are
inherently high for *any* video of the real person's actual face, replayed
or live. A screen replay isn't trying to fool the face match; it's trying
to fool the liveness/spoof checks specifically, and a high enough score on
the other four factors can still clear whichever bar the blended score
needs to hit.

**The deeper gap this doesn't close**: a "low risk" grant never triggers
the active challenge at all. Passive liveness (blink count from frames
already captured) can be satisfied by a video that's already showing the
person blinking, with no unpredictable real-time response required. This
threshold change reduces how often that shortcut succeeds; it doesn't
eliminate it. A more complete fix would mean requiring the active
challenge on every grant, not only risk-triggered ones — a genuine
architecture change, not a config value, and not yet implemented.

On top of that, `_risk_tier()` adds a more granular LOW / MEDIUM / HIGH /
CRITICAL label — a pure classifier, it doesn't change grant/deny authority,
only how much *additional* verification a step-up pass needs:

- **LOW** — clean first-pass grant. No second factor required.
- **MEDIUM** — step-up recovered cleanly (degraded conditions, no spoof
  concern). Grants after the challenge-gated step-up passes; voice not
  required.
- **HIGH** — step-up specifically triggered by the spoof-resistance veto,
  or still borderline even after the harder step-up capture. Voice
  confirmation becomes **mandatory**, not a bonus — if no voice is enrolled
  for this user, that's now a fail-closed **deny**, not a silent downgrade
  to face-only.
- **CRITICAL** — denied at any stage (no identity match, failed challenge,
  or trust below `STEPUP_THRESHOLD`). Stops immediately; no further factors
  requested, matching "deny rather than endlessly requesting additional
  verification."

Validated against the 15-scenario test matrix with tiers displayed: the
two spoof-veto cases (screen replay with visible blinking, strong match +
suspicious spoof) correctly show `HIGH`, while genuine degraded-condition
step-ups (poor lighting, glasses, moderate movement, borderline match)
correctly show `MEDIUM` — the system now treats a "clean video-audit"
recovery differently from a "some signal is actively suspicious" recovery,
instead of applying identical extra requirements to both.

## Voice ID — second biometric factor with challenge-response

An independent factor alongside the face. Two things are verified
independently, and **both are required**:

- **WHO is speaking** — classical MFCC (Mel-frequency cepstral coefficient)
  features and a Gaussian Mixture Model, the technique voice biometrics used
  before deep learning, still legitimate and well understood. No pretrained
  speaker-embedding model — each user's voice model is fit from scratch on
  their own enrollment recording.
- **WHAT is being said** — a fresh, random challenge phrase (`generate_voice_phrase()`:
  two random words from a 24-word vocabulary + a random digit word, e.g.
  *"copper falcon nine"*) is generated immediately before every verification
  attempt and checked with [Vosk](https://github.com/alphacep/vosk-api), a
  genuinely offline, CPU-only, no-GPU, no-cloud speech recognition toolkit.

### Why this exists

The original voice system only ever checked WHO was speaking, against a
recording of them saying anything at all — meaning anyone with **any**
recording of the enrolled user's voice would pass, regardless of what it
contained. A random phrase generated fresh per attempt closes that gap: a
stale recording has to somehow contain the right words for that specific
attempt, not just the right voice.

### Attack classes this mitigates — and what it doesn't

Being precise here matters more than sounding impressive:

| Attack | Mitigated? | Why |
|---|---|---|
| Replay of an old recording of the person saying something unrelated | **Yes** | Wrong content → fails phrase match regardless of speaker match |
| Replay of the exact same audio file across multiple attempts | **Partially** | `near_duplicate_of_recent` fingerprint check flags bit-for-bit-similar recordings |
| An attacker impersonating the voice (different speaker, guesses/hears the phrase) | **Yes** | Fails speaker (GMM) verification even with correct content |
| A sophisticated, targeted voice-cloning attack trained on the specific enrolled user, capable of synthesizing the *exact* requested phrase on demand | **No** | This system has no synthetic-speech / deepfake-audio detector. A convincing clone that can say arbitrary phrases in the target's voice would likely pass both checks. |
| Professional replay setup with high-fidelity playback (low distortion, full frequency response) | **Largely no** | The `narrow_bandwidth` heuristic is untuned and intentionally does not gate the decision (see below) |

**Do not present this as "voice cloning is solved."** It isn't. What's
genuinely gained is resistance to the *cheap, common* attack (a recording of
the person's voice, used regardless of content) — not resistance to a
dedicated adversary with real-time voice synthesis capability targeting this
specific person.

### Implementation

- `/voice/challenge` generates and returns a phrase — call this immediately
  before recording, never reuse a phrase across attempts.
- `/voice/verify/<user>?phrase=...` records a fresh take and:
  1. Transcribes it with Vosk, **grammar-constrained** to the known phrase
     vocabulary — tested directly on the same audio: open-vocabulary
     transcription got 0 of 3 expected words right, grammar-constrained got
     3 of 3 right. This isn't just a testing convenience; it's a real
     accuracy improvement built into the actual implementation, since every
     valid phrase is built entirely from a fixed word list anyway.
  2. Scores GMM speaker match, same math as before.
  3. Runs `_audio_replay_indicators()` — see the attack table above.
  4. Requires speaker match AND phrase match (when ASR is available) AND no
     replay flag. If ASR isn't available, content isn't checked and the
     response says so explicitly (`content_checked: false`) rather than
     silently treating an unchecked factor as passed.
- Called with no `phrase` parameter, `/voice/verify/<user>` falls back to
  the original speaker-only check — used by quick manual tests on the Voice
  ID page when you don't need the full flow.
- `VOICE_GRANT_THRESHOLD` (55), `VOICE_SCORE_SCALE` (1.6), and
  `PHRASE_MATCH_THRESHOLD` (0.66 — at least 2 of 3 words) are starting
  points. Speaker-verification math was validated against synthetic test
  tones (matching "voice" scored 100, unrelated scored 0, sensible gradient
  between). Content-verification was validated against real synthesized
  speech through the real model (see above). The replay-detection
  threshold (0.999 cosine similarity) is reasoned but **not** validated
  against real human speech — I found synthetic sine-wave test audio was
  too spectrally simple to properly stress-test it (two clearly "different"
  synthetic tones still measured 99.9% similar to each other, which real
  speech's much richer spectral content should not do) — flagging this
  honestly rather than claiming false confidence in the exact threshold.
- Voice is now wired into the main verification flow (see Part 4 below) —
  the "not yet fused into the trust score" limitation from earlier in this
  project is resolved.

### Enrollment robustness fix (found via a real enrollment failure)

Real usage surfaced a genuine bug: `voice_enroll()` could throw
`ValueError: Fitting the mixture model failed because some components have
ill-defined empirical covariance...` — a real `sklearn` GMM-fitting
failure, not a synthetic-audio artifact (development testing had only
triggered a superficially similar issue with robotic TTS audio, which led
to an incomplete diagnosis at the time).

Root-caused by reproducing it directly: the actual trigger is **low
spectral variety across the recording**, most commonly a beat of
near-silence before the person starts speaking (very common — recording
starts, brief pause, then speech), but also reproducible with any
low-variation audio (a held, unwavering tone). Near-identical MFCC frames
from either cause make a `GaussianMixture` component collapse during
fitting.

Fixed with three complementary layers, not just a bigger regularization
number:
1. **Silence trimming** (`librosa.effects.trim`, 30dB threshold) before
   feature extraction — removes the most common cause (lead-in silence)
   at the source.
2. **`reg_covar` raised from 1e-3 to 1e-2** for numerical headroom, and
   features cast to `float64` — both per `sklearn`'s own suggested
   mitigations in the error message.
3. **`_fit_voice_gmm()` retries with progressively fewer components**
   (4→3→2→1) if fitting still fails — verified this genuinely engages
   (not dead code) with a deliberately degenerate test case, and
   confirmed it correctly degrades to 1 component rather than raising.

Verified against the actual reproduced failure case (a pure, unwavering
tone — the most reliable trigger found while diagnosing this) and a more
realistic silence-then-natural-speech-variation case — both now enroll
successfully. Re-ran the existing speaker+content+replay-detection test
suite afterward to confirm no regression.
- Needs `librosa`, `scikit-learn`, `sounddevice`, and `vosk` (see
  `requirements.txt`), plus the bundled `vosk-model-small-en-us-0.15/`
  folder (~40MB) for content verification specifically — everything else
  keeps working without it, just speaker-only, with `content_checked: false`
  surfaced honestly in every response rather than silently skipped.

### ASR diagnostics + self-healing (found via a real "content not checked" report)

A real deployment showed `content_checked: false` with no visible reason —
`ASR_UNAVAILABLE_REASON` was only ever logged server-side
(`app.logger.warning`), never exposed anywhere the UI could show it. Fixed
two ways:

- **The actual failure reason is now exposed** via `/voice/status`
  (`asr_unavailable_reason`), and the Voice ID page shows it directly —
  either "vosk package not installed, run: pip install vosk" or the
  specific model-loading exception, instead of a silent `content_checked:
  false` with no explanation.
- **Self-healing for a partial/corrupted model folder.** The original
  check was just `os.path.isdir(VOSK_MODEL_DIR)` — true even for an empty
  or partially-extracted folder (an interrupted download, or a zip tool
  that silently skipped files), which would never trigger the auto-download
  fallback since the folder technically "exists." `_vosk_model_looks_complete()`
  now checks for the model's required subfolders (`am/`, `conf/`, `graph/`,
  `ivector/`) before trusting it, and `_ensure_asr_model()` retries once
  with a guaranteed-fresh download if the folder is missing pieces or fails
  to load outright. **Verified directly**: deliberately corrupted a real,
  working model (removed its `graph/` subfolder), confirmed the
  completeness check correctly caught it, then confirmed `_ensure_asr_model()`
  detected it, logged why, re-downloaded, and restored `ASR_AVAILABLE: True`
  automatically, with no manual cleanup needed — just a restart.
- Ruled out Windows path-length truncation (`MAX_PATH` = 260 chars) as the
  likely cause before investing in this fix — the model's deepest file
  path comes out to roughly 110 characters even under a fairly long
  OneDrive project path, well under the limit.

## Project structure

```
app.py                                   Flask app + trust score / adaptive auth logic
templates/index.html                     Dashboard UI
static/styles.css                        Styling
haarcascade_frontalface_default.xml      Face detector
haarcascade_eye.xml                      Eye detector, fallback if landmarks unavailable
lbfmodel.yaml                            68-point facial landmark model (EAR blink detection)
dataset/, trainer/                       Created at runtime (face data + model)
voiceprints/                             Created at runtime (per-user voice models)
```

## Notes for the demo

- Register at least one user, then train the model, before running Verify —
  otherwise everyone shows up as "no confident match" (correct behavior, not a bug).
- Both cascade XML files are bundled directly in this folder and loaded by
  absolute path, so the app works no matter what folder you launch
  `python app.py` from, and doesn't depend on your OpenCV install having a
  complete `cv2.data.haarcascades` bundle (a common source of broken/partial
  OpenCV installs on Windows).
- The step-up challenge and Attack Lab both need a webcam — the native
  `cv2.imshow` window still requires a display, but the in-browser popup
  preview does not depend on it (it works even if the native window fails
  to open, e.g. on a machine without a desktop compositor).
- The Attack Lab log resets whenever the Flask server restarts (in-memory,
  not persisted) — that's intentional for a demo, not a bug.
- **Avoid running this from inside a OneDrive/Dropbox/Google Drive-synced
  folder on Windows.** Those sync clients briefly lock folders right after
  batches of file writes, which can make folder cleanup during registration
  fail with `PermissionError: Access is denied`. The app retries and
  recovers automatically, but it's still worth keeping the project on a
  plain local path (e.g. `C:\dev\aegis-id`) if you can — one less thing to
  go wrong mid-demo.
- All face-cascade coordinates are cast to native Python `int` right after
  detection (see the face-box unpacking in `app.py`). OpenCV returns
  `numpy.int32`, and numpy's boolean/comparison results don't serialize to
  JSON the way `numpy.float64` happens to (it subclasses Python's `float`,
  but `numpy.bool_` does not subclass `bool`) — this was a real bug in an
  earlier version that could crash registration when a turn/nod challenge
  was in play. Fixed at the source; flagging it here in case you extend the
  motion-tracking code and reintroduce numpy scalars into a response dict.
- Everything runs locally against your own webcam — no data leaves the machine.
