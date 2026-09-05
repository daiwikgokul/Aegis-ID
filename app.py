"""
AI Identity Verification — Backend
Deepfake-resistant identity verification with a trust-score engine,
risk-based adaptive authentication, challenge-response step-up, classical
spoof-detection signals, and an Attack Lab for demoing resistance to
photo/screen replay attacks. Built on OpenCV (Haar cascades + LBPH).
"""

from flask import Flask, render_template, jsonify, request, Response, send_file
import cv2
import os
import io
import re
import csv
import json
import time
import random
import tempfile
import threading
import statistics
from datetime import datetime, timezone
import numpy as np
import PIL.Image as PILImage
import PIL.ExifTags as PILExifTags

app = Flask(__name__)

# Resolve every file path relative to this script's own folder, not the
# process's current working directory — this is what broke when Flask was
# launched from a different folder than app.py: the cascade XML failed to
# load silently (cv2.CascadeClassifier doesn't error on a bad path, it just
# creates an empty classifier), and the failure only surfaced later as a
# cryptic "!empty()" assertion inside detectMultiScale.
BASE_DIR = os.path.dirname(os.path.abspath(__file__))

DATASET_DIR = os.path.join(BASE_DIR, "dataset")
TRAINER_DIR = os.path.join(BASE_DIR, "trainer")
os.makedirs(DATASET_DIR, exist_ok=True)
os.makedirs(TRAINER_DIR, exist_ok=True)

CASCADE_PATH = os.path.join(BASE_DIR, "haarcascade_frontalface_default.xml")
EYE_CASCADE_PATH = os.path.join(BASE_DIR, "haarcascade_eye.xml")
FACE_CASCADE = cv2.CascadeClassifier(CASCADE_PATH)
# Bundled locally rather than loaded from cv2.data.haarcascades — that path
# depends on the OpenCV install being complete, which is exactly what broke
# on a machine with a partially broken / conflicting opencv-* install.
EYE_CASCADE = cv2.CascadeClassifier(EYE_CASCADE_PATH)
RECOGNIZER = cv2.face.LBPHFaceRecognizer_create()

if FACE_CASCADE.empty():
    raise SystemExit(
        f"Could not load the face cascade from:\n  {CASCADE_PATH}\n"
        "Make sure haarcascade_frontalface_default.xml sits in the same folder "
        "as app.py (not a subfolder, and not missing from the extracted zip)."
    )
if EYE_CASCADE.empty():
    raise SystemExit(
        f"Could not load the eye cascade from:\n  {EYE_CASCADE_PATH}\n"
        "Make sure haarcascade_eye.xml sits in the same folder as app.py "
        "(not missing from the extracted zip)."
    )

# ---------------------------------------------------------------------------
# Adaptive authentication configuration
# ---------------------------------------------------------------------------
# Trust score >= GRANT_THRESHOLD  -> single-factor face match is sufficient
# Trust score >= STEPUP_THRESHOLD -> risky, ask for a second, stricter pass
# Trust score <  STEPUP_THRESHOLD -> denied outright
#
# These, plus the weights and per-factor formulas below, were recalibrated
# after finding the original scoring was systematically too harsh — see the
# notes on RECOGNITION_SCORE_SCALE and the quality/texture formulas for the
# specific bugs. Still not calibrated against a real camera dataset; treat
# these as reasoned starting points, not final tuned values.
GRANT_THRESHOLD = 70
STEPUP_THRESHOLD = 40

MATCH_CONF_LIMIT = 75          # LBPH distance below this counts as a "match" —
                                # loosened from 65: blur and low-quality webcams
                                # push genuine-match distances higher, and this
                                # cutoff was rejecting real matches outright.
RECOGNITION_SCORE_SCALE = 100  # Separate from MATCH_CONF_LIMIT on purpose. The
                                # original code used MATCH_CONF_LIMIT for BOTH
                                # "is this a match at all" AND as the scale for
                                # turning distance into a 0-100 score — which
                                # meant a match that just barely cleared the
                                # accept threshold scored close to *zero*
                                # confidence, even though it was accepted as
                                # genuine. Decoupled: a distance right at the
                                # accept cutoff now scores ~25-35, not ~0.

VERIFY_DURATION_SEC = 4        # minimum capture window for a standard pass
STEPUP_DURATION_SEC = 6        # minimum capture window for a step-up pass
STEPUP_GRANT_THRESHOLD = 70    # same bar as a first pass — reaching it after a
                                # step-up still requires completing the
                                # challenge as a separate hard gate, so it's
                                # not actually an equal-or-easier requirement
                                # overall despite the equal score bar

# Spoof-resistance veto: found via the Part-2 trust-score audit
# (test_trust_matrix.py, scenarios 7 and 11) that a critically low
# spoof_resistance score could still be outweighed by a strong
# match_consistency + recognition_confidence + liveness combination,
# clearing GRANT_THRESHOLD outright and skipping step-up entirely — e.g.
# trust=78.3 with spoof_resistance=15/100 was GRANTED before this fix.
# That's the exact failure mode a weighted-average trust score is prone to:
# one severely bad signal getting diluted by several good ones. Below this
# floor, the decision is capped at step_up (never granted directly)
# regardless of how high the overall trust score is — same hard-gate
# pattern already used for challenge_completed, just for spoof evidence
# instead of challenge completion. This does NOT mean the attempt is
# automatically denied: it still has to clear the challenge gate at
# step-up, same as any other step-up case. Verified this doesn't affect
# genuine-but-imperfect scenarios (poor lighting, glasses, motion) in the
# test matrix — those score spoof_resistance 55-60, comfortably above
# this floor.
CRITICAL_SPOOF_THRESHOLD = 35

MIN_FACE_SAMPLES = 25          # keep capturing until we have at least this many
                                # frames with a detected face, so scores are computed
                                # on a stable sample instead of whatever a few seconds
                                # of variable frame rate happened to produce
MAX_CAPTURE_SEC = 12           # hard cap so a bad angle/lighting can't hang forever

# Five trust-score factors. Step-up capture additionally requires completing
# a randomized challenge (see CHALLENGES below) as a hard gate — passing the
# numeric threshold alone is not enough on a step-up pass.
#
# image_quality's weight was cut from 0.10 to 0.06: it's the factor most
# purely dependent on the camera's own hardware quality (sharpness, exposure)
# rather than anything about who's in front of it or whether they're live —
# a weak webcam shouldn't be able to single-handedly block a real person from
# ever passing. The freed weight moved to match_consistency and liveness,
# which are about behavior, not hardware.
WEIGHTS = {
    "match_consistency": 0.30,       # % of frames that matched the same identity
    "recognition_confidence": 0.28,  # inverse LBPH distance on matched frames
    "liveness": 0.21,                # blink / motion-based liveness signal
    "spoof_resistance": 0.15,        # texture + frequency-domain anti-spoofing
    "image_quality": 0.06,           # sharpness + lighting of the captured face
}
CAMERA_QUALITY_WARN_THRESHOLD = 45  # below this avg image_quality, surface a
                                     # note suggesting lighting/distance fixes
                                     # instead of leaving a low score unexplained
EYE_DETECTION_WARN_THRESHOLD = 40   # below this eye_visible_ratio, surface a
                                     # note — a persistently low ratio usually
                                     # means glasses glare, side lighting, or
                                     # an off-angle camera, not fewer blinks

# ---------------------------------------------------------------------------
# Challenge-response step-up
# ---------------------------------------------------------------------------
# A step-up pass doesn't just run longer — it requires the person to perform
# a randomly chosen action on cue. A static photo or a looping video can
# fake passive blink patterns; it can't fake responding to a challenge it
# doesn't know is coming until the moment it's issued.
#
# Instruction wording matters here for a real reason, not just tone: we're
# using haarcascade_frontalface_default.xml, a FRONTAL-only detector. A
# genuine profile turn commonly makes frontal detection drop out entirely
# partway through the motion (so the peak of the turn is never captured),
# and even short of that, a turning head tends to make the tracked
# bounding box shrink more than it translates. Asking for a big turn would
# systematically under-measure itself. "Slightly" sets the right
# expectation and keeps the head within the range the detector can
# actually track continuously.
CHALLENGES = {
    "blink_twice": {
        "label": "Blink twice",
        "instruction": "Blink twice, clearly, within the capture window.",
    },
    "turn_head": {
        "label": "Turn your head",
        "instruction": "Turn your head slightly to one side, then back to center.",
    },
    "nod": {
        "label": "Nod",
        "instruction": "Nod your head slightly down, then back up.",
    },
    "raise_left_hand": {
        "label": "Raise left-side hand",
        "instruction": "Raise the hand on the LEFT side of your video and hold it up.",
    },
    "raise_right_hand": {
        "label": "Raise right-side hand",
        "instruction": "Raise the hand on the RIGHT side of your video and hold it up.",
    },
    "raise_both_hands": {
        "label": "Raise both hands",
        "instruction": "Raise BOTH hands and hold them up.",
    },
}
# Minimum normalized excursion (fraction of frame width/height) the face
# center must sweep through for a turn/nod to count as genuinely performed.
# Lowered from an earlier 0.10 — that assumed a bigger head motion than a
# frontal-only cascade can reliably track continuously (see note above).
# Still tunable: if this feels too lenient/strict once tested against a
# real camera, adjust here.
CHALLENGE_MOTION_THRESHOLD = 0.06

# ---------------------------------------------------------------------------
# Hand-raise liveness (classical CV, not a trained hand-pose model)
# ---------------------------------------------------------------------------
# Deliberately NOT MediaPipe Hands here, even though it's the more precise
# tool: its current API needs a runtime-downloaded model file from a
# domain whose long-term reachability from an arbitrary deployment
# environment isn't something this project wants to depend on (the same
# concern that led to using OpenCV's own Facemark for face landmarks
# instead of MediaPipe's face mesh). Skin-color blob detection is
# classical, has zero new dependencies, and is honestly explainable to
# judges — consistent with every other spoof signal in this project (LBP
# texture, FFT moiré, patch consistency) being classical CV rather than a
# black-box model. The real cost is an honest one: it's lighting- and
# skin-tone-dependent in a way a trained hand detector wouldn't be. See
# the README for the explicit limitation writeup.
HAND_MIN_HOLD_FRAMES = 5   # consecutive frames the hand must stay in the
                            # target region to count as a deliberate hold,
                            # not a brief pass-through
HAND_TARGET_AREA_FRACTION = 0.15  # candidate blob area, as a fraction of
                                    # the detected face's area — scales the
                                    # "big enough to plausibly be a hand"
                                    # threshold to how close the person is
                                    # to the camera, rather than a fixed
                                    # pixel count that only works at one
                                    # distance/resolution


def _detect_skin_mask(frame_bgr):
    """YCrCb skin-color segmentation — the same classical technique used
    in most pre-deep-learning hand/skin tracking. Lighting and skin-tone
    dependent by nature; this is a known, honest limitation, not a bug."""
    ycrcb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2YCrCb)
    lower = np.array([0, 133, 77], dtype=np.uint8)
    upper = np.array([255, 173, 127], dtype=np.uint8)
    mask = cv2.inRange(ycrcb, lower, upper)
    mask = cv2.erode(mask, None, iterations=1)
    mask = cv2.dilate(mask, None, iterations=2)
    return mask


def _hand_regions_present(frame_bgr, face_box, frame_w, frame_h):
    """
    Returns (left_present, right_present): whether a skin-colored blob
    large enough to plausibly be a raised hand — outside the face box and
    at/above roughly shoulder level — is present in the left/right half of
    the frame this frame. "Left"/"right" are frame-relative (not the
    person's own left/right), which is what the on-screen instruction
    tells them, so there's no camera-mirroring ambiguity to worry about.
    """
    fx, fy, fw, fh = face_box
    mask = _detect_skin_mask(frame_bgr)

    # Exclude the face itself so it doesn't register as a "hand".
    y0, y1 = max(0, fy - 10), min(frame_h, fy + fh + 10)
    x0, x1 = max(0, fx - 10), min(frame_w, fx + fw + 10)
    mask[y0:y1, x0:x1] = 0

    # Only consider roughly shoulder-level and above — a raised hand should
    # appear at or above the face, not down at waist height where a resting
    # hand or other skin-toned clutter is more likely.
    y_cutoff = min(frame_h, fy + int(fh * 1.3))
    mask[y_cutoff:, :] = 0

    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    min_area = (fw * fh) * HAND_TARGET_AREA_FRACTION
    mid_x = frame_w / 2
    left_present, right_present = False, False
    for c in contours:
        if cv2.contourArea(c) < min_area:
            continue
        x, y, w, h = cv2.boundingRect(c)
        cx = x + w / 2
        if cx < mid_x:
            left_present = True
        else:
            right_present = True
    return left_present, right_present

# ---------------------------------------------------------------------------
# Attack Lab
# ---------------------------------------------------------------------------
# In-memory log of deliberate spoof-attempt demos, reset when the server
# restarts. Not a security log — a demo scorecard for showing judges the
# system's resistance to common presentation attacks.
ATTACK_TYPES = {
    "printed_photo": "Printed photo",
    "phone_replay": "Phone / screen replay",
    "video_loop": "Video loop",
    "face_manipulation": "Face manipulation (deepfake scenario)",
    "other": "Other attempt",
}
ATTACK_LOG = []

# ---------------------------------------------------------------------------
# Verification audit log — persistent, separate from ATTACK_LOG
# ---------------------------------------------------------------------------
# ATTACK_LOG is explicitly for Attack Lab test attempts and resets on
# restart (documented as intentional, not a bug) — it answers "did the
# demo's attacks get blocked", not "what has this system actually decided
# over time". This is a different, genuinely persistent record of real
# verification attempts (/verify and /verify/stepup/run — not Attack Lab),
# written to a CSV file that survives restarts, answering the question a
# compliance reviewer would actually ask: what did this system decide, for
# whom, and why, over its operating history.
VERIFICATION_LOG_PATH = os.path.join(BASE_DIR, "verification_log.csv")
VERIFICATION_LOG_FIELDS = [
    "timestamp", "datetime_utc", "stage", "identity", "trust_score",
    "risk_level", "risk_tier", "decision", "challenge", "challenge_completed",
    "sample_reliable",
]
VERIFICATION_LOG_RECENT_MAX = 200
VERIFICATION_LOG_RECENT = []
_verification_log_lock = threading.Lock()


def _count_existing_verification_log_entries():
    if not os.path.exists(VERIFICATION_LOG_PATH):
        return 0
    try:
        with open(VERIFICATION_LOG_PATH, "r", newline="") as f:
            return max(0, sum(1 for _ in f) - 1)  # minus header row
    except Exception:
        return 0


VERIFICATION_LOG_TOTAL_COUNT = _count_existing_verification_log_entries()


def _log_verification_attempt(result, stage):
    """
    Appends one row to the persistent audit log (CSV on disk) plus an
    in-memory recent-entries buffer for the /verify/log API. Best-effort:
    a logging failure (disk full, permissions, etc.) is caught and warned
    about, never allowed to break the actual verification response — the
    log is a record of what happened, not a gate on whether it's allowed
    to happen.
    """
    global VERIFICATION_LOG_TOTAL_COUNT
    entry = {
        "timestamp": round(time.time(), 3),
        "datetime_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "stage": stage,
        "identity": result.get("identity") or "",
        "trust_score": result.get("trust_score", ""),
        "risk_level": result.get("risk_level", ""),
        "risk_tier": result.get("risk_tier", ""),
        "decision": result.get("decision", ""),
        "challenge": result.get("challenge", ""),
        "challenge_completed": result.get("challenge_completed", ""),
        "sample_reliable": result.get("sample_reliable", ""),
    }
    with _verification_log_lock:
        try:
            file_is_new = not os.path.exists(VERIFICATION_LOG_PATH)
            with open(VERIFICATION_LOG_PATH, "a", newline="") as f:
                writer = csv.DictWriter(f, fieldnames=VERIFICATION_LOG_FIELDS)
                if file_is_new:
                    writer.writeheader()
                writer.writerow(entry)
            VERIFICATION_LOG_TOTAL_COUNT += 1
        except Exception as exc:
            app.logger.warning("Could not write to verification audit log: %s", exc)

        VERIFICATION_LOG_RECENT.append(entry)
        if len(VERIFICATION_LOG_RECENT) > VERIFICATION_LOG_RECENT_MAX:
            VERIFICATION_LOG_RECENT.pop(0)


# ---------------------------------------------------------------------------
# Lockout — rate-limiting after repeated failed verification attempts
# ---------------------------------------------------------------------------
# There's no "claimed identity" input at the point /verify is called — the
# person just looks at the camera and the system tries to match them
# against everyone enrolled — so a traditional per-account lockout can't be
# checked *before* a capture the way a login form checks a typed username.
# But identity *is* known immediately *after* a capture (it's the match
# result), and that's enough to do this properly rather than as a global
# blunt instrument: two separate buckets, tracked independently.
#
# - Per-identity bucket: a denial where a specific enrolled identity WAS
#   matched but then denied (failed challenge, failed voice, spoof veto,
#   below the step-up bar) is that person's own repeated failure — lock out
#   further attempts for THAT identity specifically, without affecting
#   anyone else trying to verify in the meantime. This is the fix for the
#   original global-lockout problem: Alice's bad lighting no longer blocks
#   Bob.
# - Anonymous bucket: a denial with NO matched identity at all (nobody
#   recognized, a genuinely unmatched face) can't be attributed to a real
#   enrolled person, so there's nothing to lock out individually — but
#   repeated unmatched attempts in a short window is exactly the pattern a
#   brute-force/random-photo-spam attempt would produce, so it's still
#   worth reacting to, just as its own separate, checkable-before-capture
#   global cooldown.
#
# A genuine grant for an identity clears THAT identity's own denial
# history (not the anonymous bucket, and not other identities' history) —
# consistent with the same reasoning as before: a real pass is evidence
# the "attack in progress" theory was wrong, for that person specifically.
LOCKOUT_THRESHOLD = 5           # denied attempts…
LOCKOUT_WINDOW_SECONDS = 120    # …within this rolling window…
LOCKOUT_DURATION_SECONDS = 60   # …triggers a cooldown this long

_recent_anonymous_denials = []       # timestamps, no identity matched
_anonymous_lockout_until = 0.0
_recent_identity_denials = {}        # identity -> [timestamps]
_identity_lockout_until = {}         # identity -> unlock timestamp
_lockout_lock = threading.Lock()


def _check_global_lockout():
    """Checks the anonymous/unmatched-attempt bucket — the only one that
    can be checked *before* a capture starts, since it doesn't depend on
    knowing an identity yet."""
    with _lockout_lock:
        now = time.time()
        if now < _anonymous_lockout_until:
            return True, round(_anonymous_lockout_until - now, 1)
        return False, 0.0


def _check_identity_lockout(identity):
    """Checks whether this specific identity is currently locked out —
    only meaningful after a capture, once identity is known."""
    if not identity:
        return False, 0.0
    with _lockout_lock:
        now = time.time()
        until = _identity_lockout_until.get(identity, 0.0)
        if now < until:
            return True, round(until - now, 1)
        return False, 0.0


def _register_verification_outcome(decision, identity=None):
    """
    Feeds a real /verify or /verify/stepup/run outcome into the lockout
    tracker — call once per attempt, after the final decision is known.
    Routes to the per-identity bucket when an identity was matched, the
    anonymous bucket otherwise. `step_up` on its own isn't a failure and
    isn't counted either way — only a final `denied` is.
    """
    global _anonymous_lockout_until
    with _lockout_lock:
        now = time.time()
        cutoff = now - LOCKOUT_WINDOW_SECONDS

        if decision == "granted":
            if identity and identity in _recent_identity_denials:
                _recent_identity_denials[identity].clear()
            return
        if decision != "denied":
            return

        if identity:
            lst = _recent_identity_denials.setdefault(identity, [])
            lst.append(now)
            while lst and lst[0] < cutoff:
                lst.pop(0)
            if len(lst) >= LOCKOUT_THRESHOLD:
                _identity_lockout_until[identity] = now + LOCKOUT_DURATION_SECONDS
        else:
            _recent_anonymous_denials.append(now)
            while _recent_anonymous_denials and _recent_anonymous_denials[0] < cutoff:
                _recent_anonymous_denials.pop(0)
            if len(_recent_anonymous_denials) >= LOCKOUT_THRESHOLD:
                _anonymous_lockout_until = now + LOCKOUT_DURATION_SECONDS

# ---------------------------------------------------------------------------
# In-browser camera preview
# ---------------------------------------------------------------------------
# The capture loops (registration, verify, step-up, live attack attempts)
# already draw bounding boxes / labels / challenge text onto each frame for
# the native cv2.imshow window. This publishes that same annotated frame to
# a thread-safe buffer that an MJPEG endpoint streams to the browser, so the
# live feed can show up as a popup inside the web UI instead of (or as well
# as) a separate desktop window. Requires the dev server to run threaded
# (see app.run at the bottom) — a single-threaded server can't serve a
# long-lived video stream and the actual capture request at the same time.
_frame_lock = threading.Lock()
_latest_frame_jpeg = None


def _placeholder_frame_jpeg(text="Camera idle"):
    frame = np.full((360, 480, 3), 24, dtype=np.uint8)
    cv2.putText(frame, text, (30, 190), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (140, 140, 140), 2)
    ok, buf = cv2.imencode(".jpg", frame, [cv2.IMWRITE_JPEG_QUALITY, 70])
    return buf.tobytes() if ok else None


def _publish_frame(frame_bgr):
    global _latest_frame_jpeg
    ok, buf = cv2.imencode(".jpg", frame_bgr, [cv2.IMWRITE_JPEG_QUALITY, 70])
    if ok:
        with _frame_lock:
            _latest_frame_jpeg = buf.tobytes()


def _reset_camera_preview(text="Camera idle"):
    global _latest_frame_jpeg
    with _frame_lock:
        _latest_frame_jpeg = _placeholder_frame_jpeg(text)


_latest_frame_jpeg = _placeholder_frame_jpeg()


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
def _quality_score(face_gray):
    """
    0-100 score from sharpness (Laplacian variance) and exposure.

    Sharpness scale was loosened from /400 to /200, and its blend weight
    from 0.6 to 0.5: a lower-end or slightly out-of-focus webcam genuinely
    produces less high-frequency detail than a studio camera even when
    everything else about the capture is perfectly legitimate, and the
    original scale needed near-professional sharpness to reach a decent
    score. This factor also carries a reduced overall weight in the trust
    score now (see WEIGHTS) since it's the most hardware-dependent, least
    identity-relevant signal of the five.
    """
    sharpness = cv2.Laplacian(face_gray, cv2.CV_64F).var()
    sharpness_score = min(100, (sharpness / 200) * 100)

    brightness = float(np.mean(face_gray))
    if 90 <= brightness <= 170:
        exposure_score = 100
    else:
        distance = min(abs(brightness - 90), abs(brightness - 170))
        exposure_score = max(0, 100 - distance * 1.5)

    return round((sharpness_score * 0.5) + (exposure_score * 0.5), 1)


def _lbp_texture_score(face_gray):
    """
    Classical texture-based anti-spoofing signal (Local Binary Patterns).
    Real skin has fine, irregular micro-texture; a printed photo or a
    screen-displayed face is texturally flatter and more regular. This
    computes an 8-neighbor LBP code per pixel, then scores the Shannon
    entropy of the resulting histogram — richer, more varied texture
    (real skin) scores higher; flat, regular texture (print/screen) scores
    lower. This is a heuristic classical-CV signal, not a trained deepfake
    classifier — treat it as assistive evidence, not a hard verdict.

    Entropy is normalized against 6.5 bits, not the theoretical max of 8.
    Even a sharp, well-lit real face rarely reaches full 8-bit LBP entropy
    in practice — 8.0 was an unreachable ceiling that made this score run
    low across the board regardless of camera quality, on top of genuinely
    penalizing blur (which also reduces achievable entropy for any image,
    real or fake) on top of that.
    """
    img = face_gray.astype(np.int16)
    h, w = img.shape
    if h < 3 or w < 3:
        return 50.0

    center = img[1:-1, 1:-1]
    code = np.zeros_like(center, dtype=np.uint8)
    shifts = [(-1, -1), (-1, 0), (-1, 1), (0, 1), (1, 1), (1, 0), (1, -1), (0, -1)]
    for i, (dy, dx) in enumerate(shifts):
        neighbor = img[1 + dy: h - 1 + dy, 1 + dx: w - 1 + dx]
        code |= ((neighbor >= center).astype(np.uint8) << i)

    hist, _ = np.histogram(code, bins=256, range=(0, 256))
    hist = hist.astype(np.float64) / (hist.sum() + 1e-9)
    nonzero = hist[hist > 0]
    entropy = float(-np.sum(nonzero * np.log2(nonzero)))  # 0-8 bits
    return round(float(np.clip((entropy / 6.5) * 100, 0, 100)), 1)


def _moire_score(face_gray):
    """
    Classical frequency-domain anti-spoofing signal. Photographing a screen
    (phone/monitor replay) tends to introduce moiré interference — sharp,
    isolated peaks in the mid-frequency band of the image's FFT — that a
    genuine in-person face doesn't produce. This measures how "peaky" the
    mid-frequency ring of the spectrum is: a clean, natural spectrum decays
    smoothly (high score); an isolated spike suggests screen replay (lower
    score). Like the texture score, this is a heuristic, not a certainty —
    busy real backgrounds/textures can also trigger it, so it's one signal
    among five, not a standalone verdict.
    """
    img = face_gray.astype(np.float32)
    h, w = img.shape
    f = np.fft.fft2(img)
    fshift = np.fft.fftshift(f)
    magnitude = np.abs(fshift)

    cy, cx = h // 2, w // 2
    Y, X = np.ogrid[:h, :w]
    dist = np.sqrt((Y - cy) ** 2 + (X - cx) ** 2)
    ring_mask = (dist > min(h, w) * 0.15) & (dist < min(h, w) * 0.45)
    ring_vals = magnitude[ring_mask]

    if ring_vals.size == 0 or ring_vals.mean() < 1e-6:
        return 100.0

    normalized = ring_vals / ring_vals.mean()
    peakiness = float(normalized.max())
    score = 100.0 if peakiness <= 6 else max(0.0, 100.0 - (peakiness - 6) * 5)
    return round(score, 1)


PATCH_GRID = 4  # shared between the spatial-anomaly score and its live overlay


def _patch_energies(face_gray, grid=PATCH_GRID):
    """Per-patch high-frequency noise-energy grid, shared by the spatial
    anomaly score and its live visual overlay so what's drawn on screen is
    exactly what's being scored, not a separate illustration of it."""
    h, w = face_gray.shape
    ph, pw = h // grid, w // grid
    if ph < 4 or pw < 4:
        return None

    energies = np.zeros((grid, grid), dtype=np.float64)
    for gy in range(grid):
        for gx in range(grid):
            patch = face_gray[gy * ph:(gy + 1) * ph, gx * pw:(gx + 1) * pw].astype(np.float32)
            residual = patch - cv2.GaussianBlur(patch, (3, 3), 0)
            energies[gy, gx] = float(np.var(residual))
    return energies


def _spatial_anomaly_score(face_gray):
    """
    Classical, ViT-inspired patch-consistency check. This is NOT a vision
    transformer — a real ViT-based deepfake detector needs pretrained
    weights and GPU-class inference, impractical for a dependency-light
    local demo. What it borrows from that family of techniques is the core
    idea: split the image into patches and reason about consistency across
    them, rather than treating the whole face as one blob.

    Blended/warped deepfake content (a face pasted or warped onto another)
    tends to leave behind mismatched high-frequency noise statistics between
    the manipulated region and its surroundings, even when the visible
    color/lighting was blended well. This splits the face into a grid,
    computes a per-patch high-frequency noise-energy estimate, and scores
    how consistent those estimates are with each other — a genuine face
    photographed as a whole tends to have fairly uniform sensor-noise
    statistics across patches; a composited one tends not to.
    """
    patch_energies = _patch_energies(face_gray)
    if patch_energies is None:
        return 50.0

    flat = patch_energies.flatten()
    mean_energy = flat.mean()
    if mean_energy < 1e-6:
        return 50.0

    # Coefficient of variation across patches — low = consistent (genuine),
    # high = inconsistent (a plausible sign of local compositing/warping).
    cv_ratio = float(flat.std() / mean_energy)
    score = max(0.0, 100.0 - cv_ratio * 60.0)
    return round(min(100.0, score), 1)


def _spoof_resistance_score(face_gray):
    """Blend of texture, frequency-domain, and patch-consistency checks."""
    texture = _lbp_texture_score(face_gray)
    moire = _moire_score(face_gray)
    spatial = _spatial_anomaly_score(face_gray)
    return round(texture * 0.45 + moire * 0.30 + spatial * 0.25, 1)


def _spoof_resistance_detail(face_gray):
    """
    Same computation as _spoof_resistance_score(), but returns the three
    sub-scores individually as well as the blend — used wherever the UI
    needs to show judges exactly which signal (texture / moiré / spatial
    consistency) drove a result, not just the combined number. The
    trust-score weighting itself is unaffected either way: WEIGHTS only
    ever multiplies the blended "spoof_resistance" factor, never these
    sub-components directly — they're diagnostic/display detail layered on
    top, not additional weighted inputs.
    """
    texture = _lbp_texture_score(face_gray)
    moire = _moire_score(face_gray)
    spatial = _spatial_anomaly_score(face_gray)
    blended = round(texture * 0.45 + moire * 0.30 + spatial * 0.25, 1)
    return {"texture": texture, "moire": moire, "spatial": spatial, "blended": blended}


def _draw_analysis_overlay(frame, x, y, w, h, face_gray, eyes, challenge=None,
                            eye_trace_so_far=None, motion_trace_so_far=None, landmarks=None,
                            hand_trace_so_far=None):
    """
    Draws exactly what the trust-score engine is looking at, live, onto the
    frame that streams into the browser popup: the patch-consistency grid
    used by the spatial-anomaly check (color-coded by how much each patch's
    noise energy deviates from the face's average — the same numbers that
    feed _spatial_anomaly_score, not a separate illustration of it), the
    actual eye-contour landmark points when facial landmark tracking is
    available (falling back to Haar eye boxes otherwise), and a small HUD
    readout of the live per-frame quality / texture / moiré / spatial
    scores. Every number shown here is computed the same way whether or
    not it's ultimately kept as an accepted sample — this is a
    transparency layer, not a preview mode.

    When a challenge is active, also shows live progress toward completing
    it (blink count so far, or turn/nod movement so far) using the exact
    same _evaluate_challenge function the final pass/fail decision uses —
    so what's on screen during capture can't drift from what actually gets
    decided at the end.
    """
    # --- patch-consistency grid, color-coded by deviation from the mean ---
    energies = _patch_energies(face_gray)
    if energies is not None:
        grid = energies.shape[0]
        mean_e = energies.mean()
        cell_w, cell_h = w / grid, h / grid
        overlay = frame.copy()
        for gy in range(grid):
            for gx in range(grid):
                dev = abs(energies[gy, gx] - mean_e) / (mean_e + 1e-6)
                # green (consistent) -> yellow -> red (anomalous)
                t = min(1.0, dev / 1.0)
                color = (int(40 + 40 * t), int(200 - 160 * t), int(40 + 200 * t))  # BGR
                px = int(x + gx * cell_w)
                py = int(y + gy * cell_h)
                cv2.rectangle(overlay, (px, py), (int(px + cell_w), int(py + cell_h)), color, -1)
        cv2.addWeighted(overlay, 0.28, frame, 0.72, 0, dst=frame)
        for i in range(grid + 1):
            gx = int(x + i * cell_w)
            gy = int(y + i * cell_h)
            cv2.line(frame, (gx, y), (gx, y + h), (200, 200, 200), 1, cv2.LINE_AA)
            cv2.line(frame, (x, gy), (x + w, gy), (200, 200, 200), 1, cv2.LINE_AA)

    # --- eye tracking visualization: real landmark contour if available,
    # otherwise the Haar-cascade boxes as a fallback ---
    if landmarks is not None:
        for idx in RIGHT_EYE_IDX + LEFT_EYE_IDX:
            px, py = int(landmarks[idx][0]), int(landmarks[idx][1])
            cv2.circle(frame, (px, py), 2, (255, 210, 60), -1, cv2.LINE_AA)
        for eye_idx in (RIGHT_EYE_IDX, LEFT_EYE_IDX):
            pts = np.array([[int(landmarks[i][0]), int(landmarks[i][1])] for i in eye_idx], dtype=np.int32)
            cv2.polylines(frame, [pts], True, (255, 210, 60), 1, cv2.LINE_AA)
    else:
        for (ex, ey, ew, eh) in eyes:
            cv2.rectangle(frame, (x + ex, y + ey), (x + ex + ew, y + ey + eh), (255, 210, 60), 1)

    # --- live HUD readout ---
    quality = _quality_score(face_gray)
    texture = _lbp_texture_score(face_gray)
    moire = _moire_score(face_gray)
    spatial = _spatial_anomaly_score(face_gray)

    lines = [
        f"quality {quality:5.1f}",
        f"texture {texture:5.1f}",
        f"moire   {moire:5.1f}",
        f"spatial {spatial:5.1f}",
    ]

    if challenge and eye_trace_so_far is not None and motion_trace_so_far is not None:
        _, live_score, _ = _evaluate_challenge(
            challenge, {"blinks": _count_blinks(eye_trace_so_far), "motion_trace": motion_trace_so_far,
                        "hand_trace": hand_trace_so_far or []}
        )
        if challenge == "blink_twice":
            lines.append(f"blinks  {_count_blinks(eye_trace_so_far)}/2")
        elif challenge in ("raise_left_hand", "raise_right_hand", "raise_both_hands"):
            lines.append(f"hand    {min(100.0, live_score):5.1f}%")
        else:
            lines.append(f"motion  {min(100.0, live_score):5.1f}%")

    hud_x, hud_y = 10, 34
    hud_w, hud_h = 150, 18 * len(lines) + 12
    hud_overlay = frame.copy()
    cv2.rectangle(hud_overlay, (hud_x, hud_y), (hud_x + hud_w, hud_y + hud_h), (20, 20, 20), -1)
    cv2.addWeighted(hud_overlay, 0.55, frame, 0.45, 0, dst=frame)
    for i, line in enumerate(lines):
        color = (140, 220, 255) if i == len(lines) - 1 and challenge else (210, 240, 210)
        cv2.putText(frame, line, (hud_x + 8, hud_y + 20 + i * 18),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.45, color, 1, cv2.LINE_AA)


def _count_blinks(eye_trace, min_open_gap=2, min_closed_run=3):
    """
    Blink counting from a raw per-frame eye-visibility boolean trace, using
    run-length encoding instead of smoothing.

    An earlier version of this smoothed the trace with a rolling window
    before applying a hysteresis threshold, to fight Haar-cascade detector
    noise. That fix over-corrected: a blink that only lasts one raw frame —
    entirely plausible, since this capture loop's actual frame rate is
    pulled down by per-frame texture/spatial/moiré analysis, often to just
    a handful of frames per second — gets averaged together with its open
    neighbors by the smoothing window and never dips low enough to register
    as "closed" at all. The fix silenced false positives by silencing real
    blinks too.

    The fix after that removed smoothing and counted every bounded closed
    run regardless of length — deliberately biased toward treating any
    isolated single-frame miss as a blink, on the reasoning that missing a
    real blink was worse than the challenge being slightly too easy. Real
    testing proved that trade wrong: on real hardware, single-frame false
    negatives on genuinely open eyes turned out to be common enough
    (confirmed by reproducing a report of 7 phantom blinks during a session
    with no actual blinking, using a ~20%-per-frame miss rate) that this
    produced wildly inflated counts, not a slightly-too-easy challenge.

    `min_closed_run` now requires a closed run to last at least this many
    frames before it counts as a blink at all. A real blink typically spans
    2+ frames except at very low frame rates, so this still catches normal
    blinking while cutting out the dominant real-world noise pattern
    (isolated single-frame misses). `min_open_gap` still debounces against
    one physical blink getting fragmented into multiple counts by a
    spurious mid-blink re-detection.
    """
    n = len(eye_trace)
    if n < 3:
        return 0

    runs = []
    cur, length = eye_trace[0], 1
    for v in eye_trace[1:]:
        if v == cur:
            length += 1
        else:
            runs.append((cur, length))
            cur, length = v, 1
    runs.append((cur, length))

    blinks = 0
    frames_since_last_blink = min_open_gap  # allow an immediate first count
    pos = 0
    for idx, (val, length) in enumerate(runs):
        if val is False and 0 < idx < len(runs) - 1 and length >= min_closed_run:
            if frames_since_last_blink >= min_open_gap:
                blinks += 1
                frames_since_last_blink = 0
        else:
            frames_since_last_blink += length
        pos += length

    # Sanity cap purely for display clarity — a real person doesn't blink
    # 7+ times in a few seconds, so a count above this almost certainly
    # means residual detector noise, not genuine blinks. Doesn't change
    # pass/fail (the blink_twice challenge only needs >=2), just avoids
    # showing a number that would confuse rather than inform.
    return min(blinks, 6)


# ---------------------------------------------------------------------------
# Facial landmark tracking (Eye Aspect Ratio blink detection)
# ---------------------------------------------------------------------------
# The proper fix for blink detection, not another round of Haar-cascade
# tuning. A Haar eye cascade is a coarse per-frame object detector — it
# answers "is there an eye-shaped thing here" independently on every frame,
# with no memory or sub-object structure, which is exactly why it flickers:
# open eyes get missed constantly, and there's no way to distinguish that
# from a real blink using presence/absence alone.
#
# Facial landmark tracking is a fundamentally different, more precise tool:
# it locates ~68 specific points on the face (eye corners, eyelid contour,
# etc.) every frame, and blink detection becomes a matter of measuring how
# open the eyelid contour actually is — the Eye Aspect Ratio (EAR) from
# Soukupová & Čech, "Real-Time Eye Blink Detection using Facial Landmarks"
# (2016), the standard technique used in most real blink/drowsiness
# detection systems. EAR stays roughly constant while eyes are open and
# drops sharply and continuously (not flickering on/off) when they close,
# because it's measuring actual eyelid geometry, not asking a binary
# detector to re-decide "eye or no eye" from scratch every frame.
#
# Uses OpenCV's own Facemark LBF module (part of opencv-contrib-python,
# already a dependency here — no new pip package needed) with the
# well-known, standard pretrained model referenced directly in OpenCV's own
# documentation: https://docs.opencv.org/3.4/d7/dec/tutorial_facemark_usage.html
# It outputs 68 points in the same standard ordering dlib's classic
# landmark predictor uses, so the eye indices below are the same ones
# used in essentially every EAR blink-detection reference implementation.
LANDMARK_MODEL_PATH = os.path.join(BASE_DIR, "lbfmodel.yaml")
LANDMARK_MODEL_URL = "https://raw.githubusercontent.com/kurnianggoro/GSOC2017/master/data/lbfmodel.yaml"

RIGHT_EYE_IDX = [36, 37, 38, 39, 40, 41]
LEFT_EYE_IDX = [42, 43, 44, 45, 46, 47]
NOSE_TIP_IDX = 30
EAR_THRESHOLD = 0.21   # below this, eyes are considered closed — standard
                        # range in the EAR literature is ~0.2-0.25; not
                        # calibrated against a real camera dataset yet

LANDMARK_AVAILABLE = False
FACEMARK = None


def _ensure_landmark_model():
    """
    Loads the facial landmark model if it's already present, or downloads
    it once from its canonical location (referenced directly in OpenCV's
    own docs) if not. Never raises — if this fails for any reason (offline,
    blocked network, corrupted download), the app falls back to the Haar
    eye cascade automatically everywhere landmarks are used, the same
    graceful-degradation pattern already used for voice recognition.
    """
    global LANDMARK_AVAILABLE, FACEMARK
    try:
        if not os.path.exists(LANDMARK_MODEL_PATH):
            app.logger.info("Downloading facial landmark model (~54MB, one-time only)...")
            import urllib.request
            urllib.request.urlretrieve(LANDMARK_MODEL_URL, LANDMARK_MODEL_PATH + ".partial")
            os.replace(LANDMARK_MODEL_PATH + ".partial", LANDMARK_MODEL_PATH)
        fm = cv2.face.createFacemarkLBF()
        fm.loadModel(LANDMARK_MODEL_PATH)
        FACEMARK = fm
        LANDMARK_AVAILABLE = True
        app.logger.info("Facial landmark model loaded — using EAR-based blink detection.")
    except Exception as exc:
        LANDMARK_AVAILABLE = False
        FACEMARK = None
        app.logger.warning(
            "Facial landmark model unavailable (%s) — falling back to Haar eye "
            "cascade for blink detection. This still works, just noisier.", exc
        )


_ensure_landmark_model()


def _eye_aspect_ratio(points, idx):
    p1, p2, p3, p4, p5, p6 = [points[i] for i in idx]
    a = float(np.linalg.norm(np.array(p2) - np.array(p6)))
    b = float(np.linalg.norm(np.array(p3) - np.array(p5)))
    c = float(np.linalg.norm(np.array(p1) - np.array(p4)))
    return (a + b) / (2.0 * c) if c > 1e-6 else 0.0


def _get_landmarks(gray, x, y, w, h):
    """
    Returns a (68, 2) landmark array for this face box, or None if the
    model isn't available or fitting failed for this particular frame.

    Uses reshape(-1, 2) rather than assuming a fixed nesting depth on the
    fit() return value — different OpenCV builds/versions have been
    observed to wrap the per-face landmark array with different extra
    dimensions (e.g. (1, 68, 2) vs (68, 2) directly), and indexing with a
    hardcoded landmarks[0][0] crashed with an IndexError on a build that
    wraps it differently than the one this was developed against. Reshape
    collapses whatever wrapping is present as long as the total element
    count is right, and the explicit shape check below rejects anything
    that doesn't actually resolve to 68 points instead of silently using
    garbage.
    """
    if not LANDMARK_AVAILABLE:
        return None
    try:
        faces_arr = np.array([[x, y, w, h]], dtype=np.int32)
        ok, landmarks = FACEMARK.fit(gray, faces_arr)
        if not ok or len(landmarks) == 0:
            return None
        pts = np.asarray(landmarks[0]).reshape(-1, 2)
        if pts.shape[0] != 68:
            return None
        return pts
    except Exception:
        pass
    return None


def _eyes_open_from_landmarks(points):
    """Returns (eyes_open: bool, ear: float) from a 68-point landmark set."""
    ear = (_eye_aspect_ratio(points, RIGHT_EYE_IDX) + _eye_aspect_ratio(points, LEFT_EYE_IDX)) / 2.0
    return ear >= EAR_THRESHOLD, ear


def _detect_eyes_in_face(gray, x, y, w, h):
    """
    Restricts eye detection to the upper ~60% of the face box (where eyes
    actually are) instead of running the eye cascade over the whole face —
    the mouth, nostrils, jawline, and chin texture in the lower face are a
    real source of false-positive "eye" detections on a Haar cascade, and
    narrowing the search region cuts a lot of that noise out directly.
    A minSize filter (proportional to face size) also drops tiny spurious
    detections that are too small to plausibly be an eye at this face scale.
    Returned coordinates stay relative to the full (x, y, w, h) face box,
    same as before, so callers don't need to change.

    minNeighbors was loosened from 6 to 4, and minSize shrunk, after real
    testing showed a very high per-frame miss rate on genuinely open eyes —
    reproduced a report of "blinked 7 times" while not blinking at all down
    to a ~30-40% per-frame miss rate on real hardware. minNeighbors=6 is on
    the strict end for an eye cascade (most examples use 3-5); the stricter
    setting was rejecting too many genuine open-eye detections outright,
    which is the actual root cause — no amount of smoothing or run-length
    filtering downstream can fully compensate for the detector itself
    missing open eyes this often.
    """
    eye_roi_h = max(1, int(h * 0.6))
    eye_roi = gray[y:y + eye_roi_h, x:x + w]
    min_size = (max(6, int(w * 0.08)), max(6, int(h * 0.05)))
    return EYE_CASCADE.detectMultiScale(eye_roi, 1.1, 4, minSize=min_size)


def _rmdir_best_effort(path, retries=5, delay=0.15):
    """
    Removes an empty directory, tolerating the transient "Access is denied"
    that Windows raises when the folder briefly sits inside a
    OneDrive/Dropbox/etc-synced path and the sync client has a fleeting
    handle open on it right after a batch of file writes. Retries with a
    short backoff; if it still can't be removed, logs and moves on rather
    than failing an otherwise-successful operation over a leftover empty
    folder.
    """
    for attempt in range(retries):
        try:
            os.rmdir(path)
            return True
        except OSError:
            if attempt == retries - 1:
                app.logger.warning(
                    "Could not remove staging folder %s after %d attempts "
                    "(likely a transient lock from OneDrive/antivirus/etc). "
                    "This does not affect the result — leftover empty folders "
                    "are harmless and get reused on the next attempt.",
                    path, retries,
                )
                return False
            time.sleep(delay)


def _safe_imshow(window, frame):
    """Display is optional — never let a headless server crash the route."""
    try:
        cv2.imshow(window, frame)
        return True
    except cv2.error:
        return False


def _open_camera():
    cam = cv2.VideoCapture(0)
    if not cam.isOpened():
        return None
    return cam


def _capture_verification_frames(duration_sec, window_title, challenge=None):
    """
    Runs the webcam loop for `duration_sec`, collecting per-frame signals:
    face matches, LBPH confidences, eye visibility (for blink/liveness),
    face crop quality, spoof-resistance texture/frequency signals, and
    (when a challenge is set) the normalized face-center trace needed to
    evaluate a turn/nod challenge. Returns a dict of raw signals for scoring.
    """
    cam = _open_camera()
    if cam is None:
        return None

    recognizer_ready = os.path.exists(os.path.join(TRAINER_DIR, "trainer.yml"))
    names = {}
    if recognizer_ready:
        RECOGNIZER.read(os.path.join(TRAINER_DIR, "trainer.yml"))
        names = np.load(os.path.join(TRAINER_DIR, "names.npy"), allow_pickle=True).item()

    total_frames = 0
    frames_with_face = 0
    frames_with_eyes = 0
    label_votes = {}
    confidences_by_label = {}
    quality_scores = []
    spoof_scores = []
    texture_scores = []
    moire_scores = []
    spatial_scores = []
    eye_visibility_trace = []
    motion_trace = []  # (elapsed_sec, normalized_cx, normalized_cy)
    hand_trace = []    # (left_present, right_present) per frame
    display_ok = True

    challenge_banner = CHALLENGES[challenge]["instruction"] if challenge in CHALLENGES else None

    start = time.time()
    # Keep capturing past the nominal duration, up to MAX_CAPTURE_SEC, until we
    # have a large enough sample of face frames — a fixed time window alone can
    # yield very few usable frames on a slow camera, which makes every score
    # below noisy and inconsistent between runs.
    while True:
        elapsed = time.time() - start
        if elapsed >= duration_sec and frames_with_face >= MIN_FACE_SAMPLES:
            break
        if elapsed >= MAX_CAPTURE_SEC:
            break

        ret, frame = cam.read()
        if not ret:
            break

        total_frames += 1
        frame_h, frame_w = frame.shape[:2]
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        faces = FACE_CASCADE.detectMultiScale(gray, 1.3, 5)

        if len(faces) > 0:
            frames_with_face += 1
            x, y, w, h = (int(v) for v in max(faces, key=lambda f: f[2] * f[3]))
            face_gray = cv2.resize(gray[y:y + h, x:x + w], (200, 200))
            quality_scores.append(_quality_score(face_gray))
            spoof_detail = _spoof_resistance_detail(face_gray)
            spoof_scores.append(spoof_detail["blended"])
            texture_scores.append(spoof_detail["texture"])
            moire_scores.append(spoof_detail["moire"])
            spatial_scores.append(spoof_detail["spatial"])

            # Landmark-based tracking when available: EAR for eye state
            # (far more reliable than Haar eye detection — see the module
            # docstring above _detect_eyes_in_face) and the nose-tip point
            # for motion tracking, more stable frame-to-frame than a Haar
            # bounding-box centroid since it's a single tracked point on
            # the face rather than the edges of a re-detected box that can
            # shift in size/position independent of actual head movement.
            landmarks = _get_landmarks(gray, x, y, w, h)
            eyes_for_overlay = []
            if landmarks is not None:
                eyes_visible, _ear = _eyes_open_from_landmarks(landmarks)
                nose = landmarks[NOSE_TIP_IDX]
                cx_norm = float(nose[0]) / frame_w
                cy_norm = float(nose[1]) / frame_h
            else:
                eyes_for_overlay = _detect_eyes_in_face(gray, x, y, w, h)
                eyes_visible = len(eyes_for_overlay) >= 1
                cx_norm = (x + w / 2) / frame_w
                cy_norm = (y + h / 2) / frame_h

            motion_trace.append((elapsed, cx_norm, cy_norm))
            eye_visibility_trace.append(eyes_visible)
            if eyes_visible:
                frames_with_eyes += 1
            hand_trace.append(_hand_regions_present(frame, (x, y, w, h), frame_w, frame_h))

            color = (0, 165, 255)
            label_text = "Scanning..."

            if recognizer_ready:
                label, conf = RECOGNIZER.predict(face_gray)
                if conf < MATCH_CONF_LIMIT:
                    label_votes[label] = label_votes.get(label, 0) + 1
                    confidences_by_label.setdefault(label, []).append(conf)
                    label_text = f"{names.get(label, 'Unknown')} ({conf:.0f})"
                    color = (0, 255, 0)
                else:
                    label_text = "Unrecognized"
                    color = (0, 0, 255)

            cv2.rectangle(frame, (x, y), (x + w, y + h), color, 2)
            cv2.putText(frame, label_text, (x, y - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)
            _draw_analysis_overlay(frame, x, y, w, h, face_gray, eyes_for_overlay,
                                    challenge=challenge,
                                    eye_trace_so_far=eye_visibility_trace,
                                    motion_trace_so_far=motion_trace,
                                    landmarks=landmarks,
                                    hand_trace_so_far=hand_trace)
        else:
            eye_visibility_trace.append(False)

        if challenge_banner:
            cv2.putText(frame, challenge_banner, (10, frame_h - 20),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 0), 2)

        _publish_frame(frame)

        if display_ok:
            display_ok = _safe_imshow(window_title, frame)
            if cv2.waitKey(1) == 27:
                break

    cam.release()
    if display_ok:
        cv2.destroyAllWindows()
    _reset_camera_preview()

    blinks = _count_blinks(eye_visibility_trace)
    eye_visible_ratio = (frames_with_eyes / frames_with_face * 100) if frames_with_face else 0.0

    return {
        "total_frames": total_frames,
        "frames_with_face": frames_with_face,
        "label_votes": label_votes,
        "confidences_by_label": confidences_by_label,
        "quality_scores": quality_scores,
        "spoof_scores": spoof_scores,
        "texture_scores": texture_scores,
        "moire_scores": moire_scores,
        "spatial_scores": spatial_scores,
        "blinks": blinks,
        "eye_visible_ratio": eye_visible_ratio,
        "motion_trace": motion_trace,
        "hand_trace": hand_trace,
        "names": names,
    }


def _trimmed_mean(values, trim_fraction=0.2):
    """Mean with the highest/lowest trim_fraction of values dropped, so a
    couple of outlier frames (motion blur, a bad angle) can't swing the
    average on a small sample the way a plain mean would."""
    if not values:
        return 0.0
    ordered = sorted(values)
    cut = int(len(ordered) * trim_fraction)
    trimmed = ordered[cut: len(ordered) - cut] if len(ordered) - 2 * cut > 0 else ordered
    return statistics.mean(trimmed)


def _liveness_score(blinks, eye_visible_ratio):
    """
    Smooth, continuous liveness signal instead of a step function.
    - Blinking is the strongest evidence of a live subject, but a raw blink
      count is quantized (0/1/2 blinks -> huge jumps), so it's capped and
      only contributes up to 60 points.
    - The eye-visible ratio is a continuous 0-100 value that fills in the
      rest, so a natural session that happens to have zero clean blink
      transitions (easy to miss with a 4-second window) doesn't collapse
      the whole factor to 0.
    """
    blink_component = min(60.0, blinks * 30.0)
    steadiness_component = min(40.0, eye_visible_ratio * 0.4)
    return round(min(100.0, blink_component + steadiness_component), 1)


def _evaluate_challenge(challenge, signals):
    """
    Checks whether the person actually performed the requested challenge
    action, using the motion/eye traces captured alongside everything else.
    Returns (completed: bool, challenge_score: 0-100, detail: str).
    completed is a hard gate on the step-up decision — a high trust score
    doesn't matter if the specific requested action wasn't detected.
    """
    if challenge not in CHALLENGES:
        return True, 0.0, "No challenge required."

    if challenge == "blink_twice":
        blinks = signals["blinks"]
        completed = blinks >= 2
        score = min(100.0, blinks * 40.0)
        detail = f"{blinks} blink(s) detected — needed at least 2."
        if blinks >= 4:
            # A person told to "blink twice" essentially never produces 4+
            # readings from real blinking alone — this many almost always
            # means detector noise on this specific camera/lighting, not
            # genuine blinking. Flag it directly rather than only relying
            # on the general eye_visible_ratio note, which can stay in a
            # "fine" range even while individual counts run high.
            detail += (" This is more than a natural blink count would produce — "
                       "likely detector noise from lighting or camera angle rather "
                       "than actual blinking. If this keeps happening, try facing "
                       "the camera more directly or improving front-facing light; "
                       "a retry may also land on a turn-head or nod challenge instead.")
        return bool(completed), float(round(score, 1)), detail

    if challenge in ("raise_left_hand", "raise_right_hand", "raise_both_hands"):
        hand_trace = signals.get("hand_trace", [])
        n = len(hand_trace)
        if n < 6:
            return False, 0.0, "Not enough frames captured to evaluate hand movement."

        def target_present(pair):
            left, right = pair
            if challenge == "raise_left_hand":
                return left
            if challenge == "raise_right_hand":
                return right
            return left and right

        # Sustained hold is the pass/fail gate — find the longest
        # contiguous run of "present in target region" anywhere in the
        # trace, and where it starts.
        best_run, cur_run, run_start, best_start = 0, 0, None, None
        for i, p in enumerate(hand_trace):
            if target_present(p):
                if cur_run == 0:
                    run_start = i
                cur_run += 1
                if cur_run > best_run:
                    best_run, best_start = cur_run, run_start
            else:
                cur_run = 0

        hold_ok = best_run >= HAND_MIN_HOLD_FRAMES

        # "Started away" / "delayed onset" are reported for visibility but
        # do NOT gate completion — deliberately removed as a hard
        # requirement after a real, reproducible false rejection: unlike a
        # blink or a quick head turn, "raise and hold" is a *sustained*
        # pose, and a compliant person naturally starts performing it as
        # soon as they read the instruction, during the pre-capture "get
        # ready" pause the frontend already shows — not after frame
        # capture visibly begins. By the time the first frame is actually
        # captured, a fast, correctly-complying person's hand is already
        # up. The original check was measuring reaction speed, not
        # liveness, and was punishing exactly the behavior it should
        # reward. This is a real, honest trade-off: hand-raise's own
        # contribution to distinguishing a live hold from a static photo
        # of an already-raised hand is weaker as a result — the rest of
        # the trust score (face match, spoof-resistance texture/moiré/
        # spatial signals) is what continues to defend against that
        # specific attack, not this challenge in isolation.
        early = hand_trace[:max(1, n // 4)]
        early_present_frac = sum(target_present(p) for p in early) / len(early)
        started_away = early_present_frac < 0.3
        delayed_onset = best_start is not None and best_start >= max(1, n // 6)

        completed = hold_ok
        score = (min(100.0, (best_run / HAND_MIN_HOLD_FRAMES) * 100) if hold_ok
                 else min(50.0, (best_run / HAND_MIN_HOLD_FRAMES) * 50))
        detail = f"Longest hold in target region: {best_run} frame(s) (needed {HAND_MIN_HOLD_FRAMES}+)."
        if not started_away:
            detail += (" (Hand was already up when capture started — expected, since people "
                       "naturally raise it during the \"get ready\" pause, not after.)")
        return bool(completed), float(round(score, 1)), detail

    trace = signals["motion_trace"]
    if len(trace) < 6:
        return False, 0.0, "Not enough face frames captured to evaluate motion."

    axis_index = 1 if challenge == "turn_head" else 2  # cx for turn, cy for nod
    values = [pt[axis_index] for pt in trace]
    baseline = statistics.median(values[: max(3, len(values) // 4)])
    excursion_pos = max(values) - baseline
    excursion_neg = baseline - min(values)
    swing = max(excursion_pos, excursion_neg)

    completed = swing >= CHALLENGE_MOTION_THRESHOLD
    score = round(min(100.0, (swing / CHALLENGE_MOTION_THRESHOLD) * 100), 1) if CHALLENGE_MOTION_THRESHOLD else 0.0
    axis_name = "horizontal" if challenge == "turn_head" else "vertical"
    detail = f"{axis_name} movement of {swing:.2f} (frame-fraction) — needed at least {CHALLENGE_MOTION_THRESHOLD:.2f}."
    # Explicit native-type cast: values in `trace` can carry numpy scalar
    # types (e.g. if a numpy int/float leaked in upstream), and comparisons
    # on those produce numpy.bool_ / numpy.float64 rather than plain Python
    # bool/float. numpy.float64 happens to subclass Python's float so it's
    # JSON-safe, but numpy.bool_ does NOT subclass bool and is NOT
    # JSON-serializable — cast explicitly here so this can't silently break
    # jsonify() again even if another numpy value sneaks in upstream.
    return bool(completed), float(score), detail


def _score_signals(signals, challenge=None):
    """Turns raw capture signals into a trust score + factor breakdown.
    If a challenge was set, its score overrides the passive liveness signal
    (it's stronger evidence), and completion is returned separately as a
    hard gate for the caller to apply."""
    total = signals["frames_with_face"]
    passive_liveness = _liveness_score(signals["blinks"], signals["eye_visible_ratio"])
    image_quality = round(statistics.mean(signals["quality_scores"]), 1) if signals["quality_scores"] else 0.0
    spoof_resistance = round(statistics.mean(signals["spoof_scores"]), 1) if signals["spoof_scores"] else 0.0

    # Sub-components of spoof_resistance, exposed for display (Attack Lab,
    # judge-facing breakdowns) — NOT separately weighted in the trust score;
    # trust_score below still only ever multiplies the blended
    # spoof_resistance value by its one WEIGHTS entry, same as before.
    texture_detail = round(statistics.mean(signals["texture_scores"]), 1) if signals.get("texture_scores") else 0.0
    moire_detail = round(statistics.mean(signals["moire_scores"]), 1) if signals.get("moire_scores") else 0.0
    spatial_detail = round(statistics.mean(signals["spatial_scores"]), 1) if signals.get("spatial_scores") else 0.0

    challenge_completed = True
    challenge_detail = None
    liveness = passive_liveness
    if challenge:
        challenge_completed, challenge_score, challenge_detail = _evaluate_challenge(challenge, signals)
        liveness = round(max(passive_liveness, challenge_score), 1)

    if total == 0 or not signals["label_votes"]:
        factors = {
            "match_consistency": 0.0,
            "recognition_confidence": 0.0,
            "liveness": liveness,
            "spoof_resistance": spoof_resistance,
            "image_quality": image_quality,
            "texture": texture_detail,
            "moire": moire_detail,
            "spatial": spatial_detail,
        }
        return 0.0, factors, None, challenge_completed, challenge_detail

    best_label = max(signals["label_votes"], key=signals["label_votes"].get)
    matched_frames = signals["label_votes"][best_label]
    match_consistency = round((matched_frames / total) * 100, 1)

    avg_conf = _trimmed_mean(signals["confidences_by_label"][best_label])
    recognition_confidence = round(max(0, 100 - (avg_conf / RECOGNITION_SCORE_SCALE) * 100), 1)

    factors = {
        "match_consistency": match_consistency,
        "recognition_confidence": recognition_confidence,
        "liveness": liveness,
        "spoof_resistance": spoof_resistance,
        "image_quality": image_quality,
        "texture": texture_detail,
        "moire": moire_detail,
        "spatial": spatial_detail,
    }

    trust_score = sum(factors[k] * WEIGHTS[k] for k in WEIGHTS)
    identity = signals["names"].get(best_label, "Unknown")

    return round(trust_score, 1), factors, identity, challenge_completed, challenge_detail


def _decide(trust_score, identity, grant_threshold=GRANT_THRESHOLD, challenge_completed=True, spoof_resistance=100.0):
    if identity is None:
        return "denied", "high"
    if not challenge_completed:
        # Failing the explicit challenge is a hard gate — no numeric score
        # can compensate for not performing the requested action.
        return "denied", "high"
    if trust_score >= grant_threshold:
        if spoof_resistance < CRITICAL_SPOOF_THRESHOLD:
            # A strong overall score must not bypass step-up when the
            # anti-spoofing signal specifically is this low — see
            # CRITICAL_SPOOF_THRESHOLD above for why. Downgrade to step_up
            # rather than deny outright: this could still be a genuine
            # person under unusual conditions, so give them the chance to
            # clear the (harder, challenge-gated) step-up pass instead of
            # a blanket rejection.
            return "step_up", "medium"
        return "granted", "low"
    if trust_score >= STEPUP_THRESHOLD:
        return "step_up", "medium"
    return "denied", "high"


# ---------------------------------------------------------------------------
# Risk tiers — Part 4: multimodal adaptive authentication
# ---------------------------------------------------------------------------
# _decide() above answers grant/step_up/deny — that authority doesn't
# change here. This adds a more granular, judge-explainable LOW / MEDIUM /
# HIGH / CRITICAL label on top of it, used to decide how much additional
# verification a step-up pass needs: a step-up that clears the bar with a
# solid trust score and clean spoof signal is genuinely lower-risk than one
# that barely squeaks through, or one that specifically triggered the
# spoof-resistance veto — those shouldn't be treated identically just
# because both technically returned "step_up". Requirements increase with
# risk, and CRITICAL is a hard stop (deny), not another factor request.
RISK_TIER_MEDIUM_HIGH_SPLIT = (STEPUP_THRESHOLD + GRANT_THRESHOLD) / 2


def _risk_tier(trust_score, decision, spoof_resistance):
    if decision == "denied":
        return "critical"
    if decision == "granted":
        return "low"
    # decision == "step_up"
    if spoof_resistance < CRITICAL_SPOOF_THRESHOLD:
        # The spoof-veto case specifically — a strong score with suspicious
        # anti-spoofing evidence is inherently higher risk regardless of
        # the numeric trust score (this is exactly the case the Part 2
        # audit found and fixed in _decide()).
        return "high"
    if trust_score >= RISK_TIER_MEDIUM_HIGH_SPLIT:
        return "medium"
    return "high"


def _run_verification(duration_sec, window_title, grant_threshold=GRANT_THRESHOLD, challenge=None):
    signals = _capture_verification_frames(duration_sec, window_title, challenge=challenge)

    if signals is None:
        return {
            "ok": False,
            "error": "camera_unavailable",
            "message": "Could not access the camera. Check that it is connected and not in use by another app.",
        }

    if signals["frames_with_face"] == 0:
        return {
            "ok": False,
            "error": "no_face",
            "message": "No face was detected during the capture window. Please try again facing the camera.",
        }

    trust_score, factors, identity, challenge_completed, challenge_detail = _score_signals(signals, challenge=challenge)
    decision, risk_level = _decide(trust_score, identity, grant_threshold, challenge_completed, factors["spoof_resistance"])
    risk_tier = _risk_tier(trust_score, decision, factors["spoof_resistance"])

    result = {
        "ok": True,
        "identity": identity,
        "trust_score": trust_score,
        "factors": factors,
        "decision": decision,
        "risk_level": risk_level,
        "risk_tier": risk_tier,
        "frames_analyzed": signals["total_frames"],
        "frames_with_face": signals["frames_with_face"],
        "blinks_detected": signals["blinks"],
        "sample_reliable": signals["frames_with_face"] >= MIN_FACE_SAMPLES,
    }

    # Low image_quality has a low weight in the trust score on purpose (see
    # WEIGHTS), so a weak camera shouldn't single-handedly block someone —
    # but it's still worth surfacing as an explanation rather than leaving
    # a lower score unexplained, since there IS something the person can
    # actually do about it (move closer, add light, clean the lens).
    if factors.get("image_quality", 100) < CAMERA_QUALITY_WARN_THRESHOLD:
        result["camera_quality_note"] = (
            "Image quality scored low this pass, likely from lighting or camera "
            "resolution rather than anything about the match itself — try moving "
            "closer to the camera, adding more front-facing light, or cleaning the lens."
        )

    # A persistently low eye-visible ratio usually means the eye cascade is
    # struggling with this specific setup (glasses glare, side lighting, an
    # off-angle camera) rather than the person blinking constantly — surface
    # it so a failed blink challenge has an explanation and a fix, not just
    # a confusing number.
    if signals.get("eye_visible_ratio", 100) < EYE_DETECTION_WARN_THRESHOLD:
        result["eye_detection_note"] = (
            "Eyes were only detected in a small fraction of frames this pass — often "
            "caused by glasses glare, side lighting, or the camera angle rather than "
            "actual blinking. Try facing the camera more directly, reducing glare on "
            "glasses, or improving front-facing light."
        )

    if challenge:
        result["challenge"] = challenge
        result["challenge_label"] = CHALLENGES[challenge]["label"]
        result["challenge_completed"] = challenge_completed
        result["challenge_detail"] = challenge_detail
    return result


# ---------------------------------------------------------------------------
# Attack Lab — simulated (upload-based) attacks
# ---------------------------------------------------------------------------
SIMULATED_FRAME_COUNT = 30  # matches roughly one MIN_FACE_SAMPLES-sized pass


def _degrade_printed_photo(bgr):
    """Approximates a printed-photo re-capture: flattened micro-texture and
    slightly reduced contrast, the way ink-on-paper loses fine detail.
    Parameters vary per call (within a realistic range) so repeated runs
    ("Generate New Variant") produce genuinely different output, not an
    identical result — a real printed photo attack wouldn't look bit-for-bit
    identical twice either."""
    d = random.randint(7, 11)
    sigma = random.randint(45, 75)
    blur_k = random.choice([3, 3, 5])
    alpha = random.uniform(0.88, 0.95)
    beta = random.randint(4, 12)
    smoothed = cv2.bilateralFilter(bgr, d=d, sigmaColor=sigma, sigmaSpace=sigma)
    blurred = cv2.GaussianBlur(smoothed, (blur_k, blur_k), 0)
    flattened = cv2.convertScaleAbs(blurred, alpha=alpha, beta=beta)
    return flattened


def _degrade_phone_replay(bgr):
    """Approximates a phone/monitor screen replay: downsample-then-upsample
    blur (re-capture softness), a faint synthetic pixel-grid overlay (moiré
    proxy), and a slight cool color-cast typical of LCD/OLED white points.
    Grid frequency/phase and cast strength vary per call for variant
    generation — different screens and angles produce different moiré
    patterns in reality, so a fixed pattern every time would be less honest,
    not more."""
    h, w = bgr.shape[:2]
    downsample = random.uniform(2.5, 4.0)
    small = cv2.resize(bgr, (max(1, int(w / downsample)), max(1, int(h / downsample))), interpolation=cv2.INTER_LINEAR)
    resampled = cv2.resize(small, (w, h), interpolation=cv2.INTER_LINEAR)

    freq = random.uniform(1.5, 2.4)
    phase = random.uniform(0, np.pi)
    amp = random.uniform(6, 14)
    yy, xx = np.indices((h, w))
    grid = (np.sin(xx * freq + phase) * np.sin(yy * freq + phase) * amp).astype(np.int16)
    grid3 = np.repeat(grid[:, :, None], 3, axis=2)
    moire = np.clip(resampled.astype(np.int16) + grid3, 0, 255).astype(np.uint8)

    cast = random.randint(4, 9)
    cool_cast = moire.astype(np.int16)
    cool_cast[:, :, 0] = np.clip(cool_cast[:, :, 0] + cast, 0, 255)  # B channel up
    cool_cast[:, :, 2] = np.clip(cool_cast[:, :, 2] - cast, 0, 255)  # R channel down
    return cool_cast.astype(np.uint8)


def _degrade_video_loop(bgr):
    """Approximates a compressed video replay: double JPEG re-encoding at a
    low quality to introduce block artifacts and softened edges. Quality
    varies per call within a "clearly degraded" range for variant
    generation."""
    quality = random.randint(22, 38)
    frame = bgr
    for _ in range(2):
        ok, enc = cv2.imencode(".jpg", frame, [cv2.IMWRITE_JPEG_QUALITY, quality])
        if not ok:
            break
        frame = cv2.imdecode(enc, cv2.IMREAD_COLOR)
    return frame


def _degrade_face_manipulation(bgr):
    """
    Heuristic approximation of common manipulated-face / face-swap tells,
    using classical image processing only.

    HONESTY NOTE, stated plainly because it matters: this is NOT a GAN, NOT
    a deepfake generator, and NOT a claim that its output is visually or
    statistically equivalent to a real manipulated-media attack. It exists
    to give the anti-spoofing signals something structurally different to
    react to than a simple photo/screen/video replay, built from three
    specific, well-documented artifacts that naive face-swap/reenactment
    output tends to leave behind:

    1. A local geometric warp concentrated in the lower face (jaw/mouth) —
       face-swap and reenactment models routinely introduce subtle warping
       there, since that's the region driven by the source performance.
    2. A lighting/tone mismatch between a central "swapped" region and its
       surroundings — a blended region frequently doesn't quite match the
       lighting of what it was composited onto, a very commonly cited
       real-world deepfake tell.
    3. A faint periodic high-frequency pattern at a different spatial
       frequency than the screen-replay moiré simulation — echoing the
       checkerboard-style upsampling artifacts some generative models
       leave in their output.

    Present this to judges as exactly what it is: a demonstration that the
    existing classical signals (spatial-anomaly / patch-consistency in
    particular) respond to *structural* manipulation artifacts, not just
    photographing a photograph — not a benchmark against real deepfake
    generation quality, which this makes no attempt to approximate.

    Warp strength, tone-shift direction, and pattern frequency vary per
    call for variant generation.
    """
    h, w = bgr.shape[:2]
    out = bgr.astype(np.float32)

    # 1. Local warp in the lower-face (jaw/mouth) region
    warp_strength = random.uniform(4.0, 9.0)
    warp_freq = random.uniform(9.0, 16.0)
    map_x, map_y = np.meshgrid(np.arange(w, dtype=np.float32), np.arange(h, dtype=np.float32))
    warp_cy, warp_cx = h * 0.68, w * 0.5
    warp_ry, warp_rx = h * 0.28, w * 0.35
    dist = ((map_y - warp_cy) / warp_ry) ** 2 + ((map_x - warp_cx) / warp_rx) ** 2
    warp_mask = np.clip(1.0 - dist, 0, 1) ** 2
    displacement = warp_strength * warp_mask * np.sin((map_x - warp_cx) / warp_freq)
    map_x_warped = np.clip(map_x + displacement, 0, w - 1)
    out = cv2.remap(out, map_x_warped, map_y, interpolation=cv2.INTER_LINEAR, borderMode=cv2.BORDER_REPLICATE)

    # 2. Central-region lighting/tone mismatch vs. surroundings
    face_cy, face_cx = h * 0.42, w * 0.5
    face_ry, face_rx = h * 0.32, w * 0.24
    yy, xx = np.indices((h, w))
    face_dist = ((yy - face_cy) / face_ry) ** 2 + ((xx - face_cx) / face_rx) ** 2
    face_mask = np.clip(1.0 - face_dist, 0, 1)[:, :, None]
    tone_shift = np.array([random.uniform(-7, -2), random.uniform(3, 9), random.uniform(6, 13)], dtype=np.float32)  # BGR
    out = out + face_mask * tone_shift

    # 3. Faint periodic pattern at a frequency distinct from the moiré simulation
    pattern_freq = random.uniform(0.4, 0.7)
    pattern_amp = random.uniform(4.0, 8.0)
    pattern = (np.sin(xx * pattern_freq) * np.sin(yy * pattern_freq) * pattern_amp).astype(np.float32)
    out = out + pattern[:, :, None]

    return np.clip(out, 0, 255).astype(np.uint8)


DEGRADERS = {
    "printed_photo": _degrade_printed_photo,
    "phone_replay": _degrade_phone_replay,
    "video_loop": _degrade_video_loop,
    "face_manipulation": _degrade_face_manipulation,
    "other": lambda bgr: bgr,  # unmodified — a plain photo, as a control case
}

KNOWN_EDITORS = ["photoshop", "gimp", "snapseed", "picsart", "lightroom", "pixlr", "canva", "faceapp"]


def _metadata_analysis_score(file_bytes):
    """
    Inspects EXIF metadata on an uploaded file — deliberately NOT part of
    the weighted trust score, because this signal only exists for file
    uploads. A live webcam frame from cv2.VideoCapture is a raw pixel array
    with no file container, so there is literally no metadata to check on a
    genuine live capture — that absence-by-design is itself the answer to
    "what about metadata if the image bypasses the camera": it's a defense
    specific to the file-injection threat model (someone feeding in a
    pre-existing image/video instead of a live feed), not a general-purpose
    signal that applies to every capture path.
    """
    try:
        img = PILImage.open(io.BytesIO(file_bytes))
        exif = img.getexif()
    except Exception:
        return 50.0, "Could not read image metadata — inconclusive.", []

    if not exif:
        return 35.0, ("No EXIF metadata found at all — consistent with a screenshot, "
                       "a web-downloaded image, or a file re-saved through another app "
                       "rather than an unmodified direct camera capture."), ["no_exif"]

    tag_map = {PILExifTags.TAGS.get(k, str(k)): v for k, v in exif.items()}
    camera_fields = ["Make", "Model", "DateTime", "DateTimeOriginal", "ExposureTime", "FNumber"]
    present = [f for f in camera_fields if f in tag_map and tag_map[f]]

    flags = []
    score = 35.0
    if len(present) >= 3:
        score = 90.0
        detail = f"Camera EXIF metadata present ({', '.join(present)}) — consistent with an unmodified camera capture."
    elif len(present) >= 1:
        score = 60.0
        detail = f"Partial camera EXIF metadata present ({', '.join(present)}) — inconclusive on its own."
    else:
        detail = "EXIF block present but no camera identification fields (Make/Model/timestamp) found."

    software = str(tag_map.get("Software", "")).lower()
    if software and any(editor in software for editor in KNOWN_EDITORS):
        score = max(0.0, score - 30.0)
        flags.append("editor_detected")
        detail += f" Editing software tag detected ('{tag_map.get('Software')}') — image may have been processed after capture."

    return round(score, 1), detail, flags


def _simulate_verification(file_bytes, attack_type, skip_metadata=False, already_degraded=False):
    """
    Builds a synthetic capture pass from a single photo instead of a live
    webcam feed: degrades it to approximate the chosen attack type, then
    repeats that one processed frame across a simulated capture window and
    runs it through the exact same scoring function as a live pass.

    A repeated static frame has zero motion and no blink transitions by
    construction — that's not a shortcut, it's the honest result: a true
    static replay genuinely can't produce those signals either, so liveness
    scoring near zero here is a real (not simulated) property of the input.

    `skip_metadata`: True, or a custom reason string, when metadata
    analysis wouldn't mean anything for this particular source — e.g. a
    photo from this app's own dataset (self-contained Attack Lab
    generation), or a frame extracted from an uploaded video and
    re-encoded as a fresh JPEG. Both cases produce a file with no original
    EXIF *by construction*, regardless of whether the underlying content
    is genuine or suspicious — running the metadata check on them would
    report "no EXIF found" every single time for a reason that has
    nothing to do with the attack being simulated, a misleading signal
    dressed up as a finding. Skipped honestly instead, with an accurate
    explanation of *why* for this specific source — same "not applicable"
    framing already used for live camera attempts.

    `already_degraded`: set when `file_bytes` came from the attack cache
    (see _get_cached_attack_variant) — the degradation was already applied
    once when the cache was built, so applying it a second time here would
    double-degrade the image rather than reuse the pre-made result.
    """
    if skip_metadata:
        default_reason = "Not applicable \u2014 sourced from this app's own dataset, which never carries camera EXIF regardless of authenticity."
        reason = skip_metadata if isinstance(skip_metadata, str) else default_reason
        metadata_score, metadata_detail, metadata_flags = None, reason, []
    else:
        metadata_score, metadata_detail, metadata_flags = _metadata_analysis_score(file_bytes)

    bgr = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)
    if bgr is None:
        return {
            "ok": False, "error": "bad_image", "message": "Could not decode the uploaded file as an image.",
            "metadata_confidence": metadata_score, "metadata_detail": metadata_detail,
        }

    if already_degraded:
        degraded = bgr
    else:
        degrader = DEGRADERS.get(attack_type, DEGRADERS["other"])
        degraded = degrader(bgr)

    gray = cv2.cvtColor(degraded, cv2.COLOR_BGR2GRAY)
    faces = FACE_CASCADE.detectMultiScale(gray, 1.3, 5)
    if len(faces) == 0:
        return {
            "ok": False,
            "error": "no_face",
            "message": "No face was detected in the uploaded (degraded) image. Try a clearer, front-facing photo.",
            "metadata_confidence": metadata_score, "metadata_detail": metadata_detail,
        }

    x, y, w, h = (int(v) for v in max(faces, key=lambda f: f[2] * f[3]))
    face_gray = cv2.resize(gray[y:y + h, x:x + w], (200, 200))

    recognizer_ready = os.path.exists(os.path.join(TRAINER_DIR, "trainer.yml"))
    names = {}
    label_votes, confidences_by_label = {}, {}
    if recognizer_ready:
        RECOGNIZER.read(os.path.join(TRAINER_DIR, "trainer.yml"))
        names = np.load(os.path.join(TRAINER_DIR, "names.npy"), allow_pickle=True).item()
        label, conf = RECOGNIZER.predict(face_gray)
        if conf < MATCH_CONF_LIMIT:
            label_votes[label] = SIMULATED_FRAME_COUNT
            confidences_by_label[label] = [conf] * SIMULATED_FRAME_COUNT

    eyes = _detect_eyes_in_face(gray, x, y, w, h)
    eye_visible_ratio = 100.0 if len(eyes) >= 1 else 0.0

    quality = _quality_score(face_gray)
    spoof_detail = _spoof_resistance_detail(face_gray)
    frame_h, frame_w = degraded.shape[:2]
    cx_norm = (x + w / 2) / frame_w
    cy_norm = (y + h / 2) / frame_h

    signals = {
        "total_frames": SIMULATED_FRAME_COUNT,
        "frames_with_face": SIMULATED_FRAME_COUNT,
        "label_votes": label_votes,
        "confidences_by_label": confidences_by_label,
        "quality_scores": [quality] * SIMULATED_FRAME_COUNT,
        "spoof_scores": [spoof_detail["blended"]] * SIMULATED_FRAME_COUNT,
        "texture_scores": [spoof_detail["texture"]] * SIMULATED_FRAME_COUNT,
        "moire_scores": [spoof_detail["moire"]] * SIMULATED_FRAME_COUNT,
        "spatial_scores": [spoof_detail["spatial"]] * SIMULATED_FRAME_COUNT,
        "blinks": 0,  # a static frame cannot produce a blink transition
        "eye_visible_ratio": eye_visible_ratio,
        "motion_trace": [(i * 0.1, cx_norm, cy_norm) for i in range(SIMULATED_FRAME_COUNT)],
        "names": names,
    }

    trust_score, factors, identity, _, _ = _score_signals(signals)
    decision, risk_level = _decide(trust_score, identity, GRANT_THRESHOLD, spoof_resistance=factors["spoof_resistance"])
    risk_tier = _risk_tier(trust_score, decision, factors["spoof_resistance"])

    return {
        "ok": True,
        "identity": identity,
        "trust_score": trust_score,
        "factors": factors,
        "decision": decision,
        "risk_level": risk_level,
        "risk_tier": risk_tier,
        "frames_analyzed": SIMULATED_FRAME_COUNT,
        "frames_with_face": SIMULATED_FRAME_COUNT,
        "blinks_detected": 0,
        "sample_reliable": True,
        "metadata_confidence": metadata_score,
        "metadata_detail": metadata_detail,
        "metadata_flags": metadata_flags,
    }


# ---------------------------------------------------------------------------
# Voice recognition — a second, independent biometric factor
# ---------------------------------------------------------------------------
# Classical MFCC + Gaussian Mixture Model speaker verification — the
# technique voice biometrics used before deep learning, and still a
# legitimate, well-understood approach. Deliberately not a deep-learning
# speaker embedding model: those need pretrained weights (unreachable
# without internet access to a model hub) and are overkill for a local
# demo. This needs no pretrained anything — each user's voice model is
# fit from scratch on their own enrollment recording.
try:
    import sounddevice as sd
    import librosa
    from sklearn.mixture import GaussianMixture
    import pickle
    VOICE_AVAILABLE = True
    VOICE_IMPORT_ERROR = None
except Exception as _voice_import_exc:  # pragma: no cover - environment dependent
    VOICE_AVAILABLE = False
    VOICE_IMPORT_ERROR = str(_voice_import_exc)

VOICE_DIR = os.path.join(BASE_DIR, "voiceprints")
os.makedirs(VOICE_DIR, exist_ok=True)

VOICE_SAMPLE_RATE = 16000
VOICE_ENROLL_SECONDS = 3.5
VOICE_VERIFY_SECONDS = 3.0
VOICE_N_MFCC = 13
VOICE_GMM_COMPONENTS = 4
VOICE_SCORE_SCALE = 1.6      # tune this if scores cluster too tightly or too spread out
VOICE_GRANT_THRESHOLD = 55   # 0-100, tune against real enrollment/verification recordings
VOICE_MIN_RMS = 0.004        # below this, treat the recording as silence/no speech

# ---------------------------------------------------------------------------
# Speech-content verification (challenge-response) — "verify WHAT was said"
# ---------------------------------------------------------------------------
# The original voice system only ever checked WHO was speaking, against a
# recording of them saying anything at all. That's genuinely weak against
# an attacker who has obtained ANY recording of the enrolled user's voice —
# it would pass speaker verification regardless of content. This adds a
# fresh, randomly generated phrase per verification attempt, checked with
# offline ASR (automatic speech recognition), as a second, independent
# requirement alongside speaker verification. Both must pass.
#
# Uses Vosk — a genuinely offline, CPU-only, no-GPU speech recognition
# toolkit (Kaldi-based), not a cloud API. The small English model (~40MB)
# is bundled in this folder (vosk-model-small-en-us-0.15/) the same way the
# facial landmark model is; if missing, app.py tries to fetch it once from
# its GitHub mirror (chosen specifically because raw.githubusercontent.com
# is a much more available fetch target in restricted network environments
# than the original host). If Vosk or the model isn't available, content
# verification is skipped and voice falls back to speaker-only checking —
# same graceful-degradation pattern as everywhere else in this module.
try:
    import vosk
    vosk.SetLogLevel(-1)  # Vosk is very chatty on stderr by default
    ASR_LIB_AVAILABLE = True
    ASR_LIB_IMPORT_ERROR = None
except Exception as _asr_import_exc:
    ASR_LIB_AVAILABLE = False
    ASR_LIB_IMPORT_ERROR = str(_asr_import_exc)

VOSK_MODEL_DIR = os.path.join(BASE_DIR, "vosk-model-small-en-us-0.15")
VOSK_MODEL_ZIP_URL = "https://raw.githubusercontent.com/kercre123/vosk-models/main/vosk-model-small-en-us-0.15.zip"
# A handful of files/folders that must exist for the model to actually be
# usable — used to tell "genuinely present" apart from "folder exists but
# is empty/partially extracted", which os.path.isdir() alone can't catch.
VOSK_MODEL_REQUIRED_PATHS = ["am", "conf", "graph", "ivector"]
ASR_AVAILABLE = False
ASR_UNAVAILABLE_REASON = None  # human-readable, exposed via /voice/status —
                                 # so "why is content checking off" is
                                 # answerable from the UI, not just a
                                 # server-log line the person may not be
                                 # watching
VOSK_MODEL = None


def _vosk_model_looks_complete(path):
    if not os.path.isdir(path):
        return False
    return all(os.path.exists(os.path.join(path, p)) for p in VOSK_MODEL_REQUIRED_PATHS)


def _download_vosk_model():
    app.logger.info("Downloading speech-recognition model (~40MB, one-time only)...")
    import urllib.request
    import zipfile
    zip_path = VOSK_MODEL_DIR + ".zip"
    urllib.request.urlretrieve(VOSK_MODEL_ZIP_URL, zip_path)
    with zipfile.ZipFile(zip_path) as zf:
        zf.extractall(BASE_DIR)
    os.remove(zip_path)


def _ensure_asr_model():
    """
    Loads the Vosk model if present and complete, downloads+extracts it if
    missing, and — this is the addition — retries once with a fresh
    download if the existing folder looks present but is actually
    incomplete or fails to load (a partial extraction, e.g. an interrupted
    download or a zip tool that silently skipped files, would otherwise
    look "present" to a plain os.path.isdir() check and never get repaired
    automatically). Never raises — content verification just becomes
    unavailable and voice falls back to speaker-only, same pattern as the
    facial landmark model and voice recognition itself. The actual failure
    reason is captured in ASR_UNAVAILABLE_REASON and surfaced via
    /voice/status, not just logged server-side where it's easy to miss.
    """
    global ASR_AVAILABLE, ASR_UNAVAILABLE_REASON, VOSK_MODEL
    if not ASR_LIB_AVAILABLE:
        ASR_UNAVAILABLE_REASON = f"vosk package not installed ({ASR_LIB_IMPORT_ERROR}). Run: pip install vosk"
        return

    attempted_redownload = False
    while True:
        try:
            if not _vosk_model_looks_complete(VOSK_MODEL_DIR):
                if os.path.isdir(VOSK_MODEL_DIR):
                    app.logger.warning(
                        "Speech-recognition model folder exists but looks incomplete "
                        "(missing expected subfolder) — re-downloading."
                    )
                    import shutil
                    shutil.rmtree(VOSK_MODEL_DIR, ignore_errors=True)
                _download_vosk_model()
            VOSK_MODEL = vosk.Model(VOSK_MODEL_DIR)
            ASR_AVAILABLE = True
            ASR_UNAVAILABLE_REASON = None
            app.logger.info("Speech-recognition model loaded — voice challenges will verify phrase content.")
            return
        except Exception as exc:
            if not attempted_redownload:
                # Could be a corrupted/partial folder that passed the
                # completeness check but still fails to load — try exactly
                # once more with a guaranteed-fresh download before giving up.
                attempted_redownload = True
                app.logger.warning("Speech-recognition model failed to load (%s) — retrying with a fresh download.", exc)
                import shutil
                shutil.rmtree(VOSK_MODEL_DIR, ignore_errors=True)
                continue
            ASR_AVAILABLE = False
            VOSK_MODEL = None
            ASR_UNAVAILABLE_REASON = str(exc)
            app.logger.warning(
                "Speech-recognition model unavailable (%s) — voice verification will "
                "check speaker identity only, not phrase content.", exc
            )
            return



_ensure_asr_model()

# Phrase vocabulary: short, common, phonetically distinct words an ASR
# model trained on general English handles well — deliberately not
# obscure/technical words, which is exactly where small ASR models are
# weakest. Random word pair + a spoken digit, generated fresh every time —
# never a fixed sentence, so a recording of a previous session's phrase
# won't match the next challenge.
VOICE_WORD_BANK = [
    "apple", "tiger", "river", "cloud", "stone", "garden", "silver", "orange",
    "winter", "copper", "forest", "planet", "yellow", "purple", "dragon",
    "castle", "harbor", "canyon", "rocket", "cotton", "maple", "willow",
    "desert", "mountain",
]
VOICE_NUMBER_WORDS = ["zero", "one", "two", "three", "four", "five", "six", "seven", "eight", "nine"]
PHRASE_MATCH_THRESHOLD = 0.66  # fraction of expected words that must appear
                                 # in the transcript — not exact match, since
                                 # ASR (even on real human speech) isn't
                                 # perfect; tolerates one word being missed
                                 # or misheard out of three

RECENT_VOICE_FINGERPRINTS = []  # in-memory, resets on restart — see
                                  # _audio_replay_indicators() for what
                                  # this is used for


def generate_voice_phrase():
    """A fresh, random challenge phrase — never a fixed sentence."""
    w1, w2 = random.sample(VOICE_WORD_BANK, 2)
    num = random.choice(VOICE_NUMBER_WORDS)
    return f"{w1} {w2} {num}"


def _transcribe_audio(y, sr=VOICE_SAMPLE_RATE):
    """
    Runs recorded audio through Vosk, constrained to the known phrase
    vocabulary (VOICE_WORD_BANK + VOICE_NUMBER_WORDS), returns the
    transcript text (or None if ASR isn't available or fails).

    Grammar-constraining matters a lot here, not just as an optimization:
    tested directly against the same recording, open-vocabulary
    recognition got 0 of 3 expected words right; constraining the grammar
    to our fixed vocabulary got 3 of 3 right. Since every valid phrase is
    built entirely from this fixed word list, there's no accuracy cost to
    constraining recognition to it — only upside, because it eliminates
    confusion with acoustically-similar words outside our vocabulary that
    could never have been the actual challenge phrase anyway.
    """
    if not ASR_AVAILABLE:
        return None
    try:
        vocab = VOICE_WORD_BANK + VOICE_NUMBER_WORDS + ["[unk]"]
        grammar = json.dumps(vocab)
        rec = vosk.KaldiRecognizer(VOSK_MODEL, sr, grammar)
        audio_int16 = (np.clip(y, -1, 1) * 32767).astype(np.int16)
        rec.AcceptWaveform(audio_int16.tobytes())
        result = json.loads(rec.FinalResult())
        return result.get("text", "")
    except Exception:
        return None


def _phrase_match_score(transcript, expected_phrase):
    """Fraction of the expected phrase's words that appear anywhere in the
    transcript — order-independent and tolerant of one miss, since ASR
    errors are normal even on genuine real speech."""
    def norm(s):
        return re.findall(r"[a-z0-9]+", s.lower())
    expected_words = norm(expected_phrase)
    if not expected_words:
        return 0.0
    transcript_words = set(norm(transcript or ""))
    matched = sum(1 for w in expected_words if w in transcript_words)
    return matched / len(expected_words)


def _audio_replay_indicators(y, sr=VOICE_SAMPLE_RATE):
    """
    Lightweight, heuristic classical-audio checks for signs of replayed
    rather than live speech. Like the visual spoof-resistance signals
    (texture, moiré), these are assistive evidence, not a robust
    anti-spoofing guarantee — explicitly NOT claimed to catch a competent
    replay/synthetic-speech attack, only two narrow, explainable patterns:

    - near_duplicate_of_recent: compares this recording's MFCC fingerprint
      against recent verification attempts. Natural speech varies at least
      slightly between repetitions, even of the same phrase — a
      near-identical match suggests literally the same audio being
      replayed. This is the more defensible of the two checks.
    - narrow_bandwidth: audio replayed through a small phone/laptop speaker
      and re-captured by a mic often has attenuated high-frequency content
      versus a live voice on a direct mic. The cutoff here (0.03) is a
      reasoned guess, NOT calibrated against real replay-vs-live recordings
      — it's reported for visibility but does not by itself gate the
      overall decision, specifically because it's unvalidated.
    """
    flags = {"narrow_bandwidth": False, "near_duplicate_of_recent": False, "likely_replay": False}
    try:
        stft = np.abs(librosa.stft(y))
        freqs = librosa.fft_frequencies(sr=sr)
        energy = stft.mean(axis=1)
        total = energy.sum() + 1e-9
        high_freq_fraction = float(energy[freqs > 4000].sum() / total)
        flags["narrow_bandwidth"] = high_freq_fraction < 0.03

        mfcc = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=13)
        fingerprint = mfcc.mean(axis=1)
        fp_norm = np.linalg.norm(fingerprint) + 1e-9
        for prev in RECENT_VOICE_FINGERPRINTS:
            sim = float(np.dot(fingerprint, prev) / (fp_norm * (np.linalg.norm(prev) + 1e-9)))
            if sim > 0.999:
                flags["near_duplicate_of_recent"] = True
                break
        RECENT_VOICE_FINGERPRINTS.append(fingerprint)
        if len(RECENT_VOICE_FINGERPRINTS) > 20:
            RECENT_VOICE_FINGERPRINTS.pop(0)
    except Exception:
        pass
    # Only the more defensible check gates the overall flag — narrow_bandwidth
    # is reported but not (yet) trusted enough to gate on, see docstring.
    flags["likely_replay"] = flags["near_duplicate_of_recent"]
    return flags


def _record_audio(duration_sec):
    """Records mono audio from the default microphone. Returns a 1-D
    float32 numpy array, or None if no microphone is available."""
    try:
        audio = sd.rec(int(duration_sec * VOICE_SAMPLE_RATE), samplerate=VOICE_SAMPLE_RATE,
                        channels=1, dtype="float32")
        sd.wait()
        return audio.flatten()
    except Exception:
        return None


def _voice_features(y):
    """
    MFCC + delta-MFCC features, frame by frame. Returns None if the
    recording looks like silence rather than actual speech.

    Trims leading/trailing silence before feature extraction. A real
    recording almost always has a beat of silence before the person starts
    speaking (recording begins, then a pause, then speech) — and those
    near-silent frames have very low-variance, near-identical MFCC values
    that can make a GMM component collapse during fitting. This is a real
    failure mode confirmed with actual human speech, not just an artifact
    of synthetic test audio used during development — trimming the silence
    out addresses the actual root cause rather than just papering over it
    with more regularization downstream.
    """
    if y is None or len(y) == 0:
        return None
    rms = float(np.sqrt(np.mean(y.astype(np.float64) ** 2)))
    if rms < VOICE_MIN_RMS:
        return None

    trimmed, _ = librosa.effects.trim(y, top_db=30)
    if len(trimmed) < VOICE_SAMPLE_RATE * 0.3:  # trimming left almost nothing — too aggressive, fall back
        trimmed = y

    mfcc = librosa.feature.mfcc(y=trimmed, sr=VOICE_SAMPLE_RATE, n_mfcc=VOICE_N_MFCC).T
    delta = librosa.feature.delta(mfcc.T).T
    # float64: sklearn's own error message for the GMM collapse failure
    # explicitly suggests this — cheap, and removes one contributing factor.
    return np.hstack([mfcc, delta]).astype(np.float64)


def _fit_voice_gmm(features, max_components=None):
    """
    Fits a GaussianMixture to voice features, retrying with progressively
    fewer components if fitting fails due to ill-defined/collapsed
    covariance. This is a real failure mode with real speech — confirmed
    by an actual enrollment failure, not hypothetical — especially on
    shorter recordings or ones with less acoustic variety than 4 components
    assumes. reg_covar is also raised from 1e-3 to 1e-2 for more numerical
    headroom, per sklearn's own suggestion in the error message.
    """
    if max_components is None:
        max_components = VOICE_GMM_COMPONENTS
    n_components = min(max_components, max(1, features.shape[0] // 10))
    last_error = None
    for n in range(n_components, 0, -1):
        try:
            gmm = GaussianMixture(n_components=n, covariance_type="diag", reg_covar=1e-2, random_state=0)
            gmm.fit(features)
            return gmm
        except ValueError as exc:
            last_error = exc
            continue
    raise last_error


def _voiceprint_path(user):
    return os.path.join(VOICE_DIR, f"{user}.pkl")


def voice_enroll(user):
    """
    Records two short takes of the same passphrase: the first trains the
    user's GMM voice model, the second is scored against that model to get
    a self-consistency reference log-likelihood (self_ll) — this is what
    later verification attempts get compared against, so the 0-100 score
    is calibrated per-user rather than relying on a shared threshold that
    would need a large multi-speaker dataset to tune properly.
    """
    if not VOICE_AVAILABLE:
        return {"ok": False, "error": "voice_unavailable", "message": _voice_unavailable_message()}

    take1 = _record_audio(VOICE_ENROLL_SECONDS)
    feats1 = _voice_features(take1)
    if feats1 is None:
        return {"ok": False, "error": "no_speech",
                "message": "No speech was detected in the first recording. Speak clearly during capture."}

    take2 = _record_audio(VOICE_ENROLL_SECONDS)
    feats2 = _voice_features(take2)
    if feats2 is None:
        return {"ok": False, "error": "no_speech",
                "message": "No speech was detected in the second recording. Speak clearly during capture."}

    try:
        gmm = _fit_voice_gmm(feats1)
    except ValueError as exc:
        return {
            "ok": False, "error": "fit_failed",
            "message": (
                f"Could not build a voice model from this recording ({exc}). "
                "Try speaking continuously for the full duration with minimal "
                "background noise, then retry."
            ),
        }
    self_ll = float(gmm.score(feats2))

    with open(_voiceprint_path(user), "wb") as f:
        pickle.dump({"gmm": gmm, "self_ll": self_ll}, f)

    return {"ok": True, "self_ll": round(self_ll, 2)}


def voice_verify(user, phrase=None):
    """
    Records a fresh take and scores it against the user's stored voice
    model. If `phrase` is given (the challenge-response path), also runs
    speech-content verification and lightweight replay checks — the
    overall decision requires speaker match AND phrase match AND no
    replay indicator, all three. Without a phrase (the standalone Voice ID
    page's simple test), behaves as before: speaker-only.
    """
    if not VOICE_AVAILABLE:
        return {"ok": False, "error": "voice_unavailable", "message": _voice_unavailable_message()}

    path = _voiceprint_path(user)
    if not os.path.exists(path):
        return {"ok": False, "error": "not_enrolled",
                "message": f'No voice model found for "{user}" — enroll their voice first.'}

    take = _record_audio(VOICE_VERIFY_SECONDS)
    feats = _voice_features(take)
    if feats is None:
        return {"ok": False, "error": "no_speech",
                "message": "No speech was detected. Speak clearly during capture."}

    with open(path, "rb") as f:
        model = pickle.load(f)
    gmm, self_ll = model["gmm"], model["self_ll"]

    test_ll = float(gmm.score(feats))
    diff = test_ll - self_ll
    voice_score = round(float(np.clip(100 + diff * VOICE_SCORE_SCALE, 0, 100)), 1)
    speaker_passed = voice_score >= VOICE_GRANT_THRESHOLD

    result = {
        "ok": True,
        "voice_score": voice_score,
        "test_ll": round(test_ll, 2),
        "self_ll": round(self_ll, 2),
        "speaker_passed": speaker_passed,
    }

    if phrase is None:
        # Standalone speaker-only check — original behavior, unchanged.
        result["decision"] = "match" if speaker_passed else "no_match"
        return result

    # Challenge-response path: also verify WHAT was said, and run the
    # lightweight replay heuristics.
    transcript = _transcribe_audio(take)
    if transcript is None and not ASR_AVAILABLE:
        content_passed = None  # ASR unavailable — content check skipped, not failed
        content_score = None
    else:
        content_score = _phrase_match_score(transcript, phrase)
        content_passed = content_score >= PHRASE_MATCH_THRESHOLD

    replay_flags = _audio_replay_indicators(take)

    # Both speaker AND content must pass (when content checking is
    # available) — neither factor alone is sufficient. If ASR is
    # unavailable, we can't require content match, so fall back to
    # speaker-only but flag that content wasn't actually checked, so the
    # caller/UI can be honest about what was and wasn't verified.
    if content_passed is None:
        overall_passed = speaker_passed and not replay_flags["likely_replay"]
    else:
        overall_passed = speaker_passed and content_passed and not replay_flags["likely_replay"]

    result.update({
        "phrase_expected": phrase,
        "transcript": transcript,
        "content_score": round(content_score, 2) if content_score is not None else None,
        "content_passed": content_passed,
        "content_checked": content_passed is not None,
        "replay_flags": replay_flags,
        "decision": "match" if overall_passed else "no_match",
    })
    return result


def _voice_unavailable_message():
    return (
        "Voice recognition dependencies aren't available on this machine "
        f"({VOICE_IMPORT_ERROR}). Install them with: "
        "pip install librosa scikit-learn sounddevice — on Linux you may also "
        "need the system package portaudio (e.g. apt install libportaudio2)."
    )


# ---------------------------------------------------------------------------
# Core actions
# ---------------------------------------------------------------------------
STAGING_DIR = os.path.join(DATASET_DIR, "_staging")
os.makedirs(STAGING_DIR, exist_ok=True)

MIN_ENROLL_SAMPLES = 20        # minimum accepted images before enrollment can succeed
ENROLL_TARGET_SAMPLES = 50     # stop early once this many accepted images are captured
ENROLL_SPOOF_MIN = 55          # per-frame spoof_resistance floor — frames below this
                                # (consistent with a print/screen replay) are captured
                                # for liveness evaluation but never saved to the dataset
MAX_ENROLL_CAPTURE_SEC = 25    # hard cap so a bad angle/lighting can't hang forever


def capture_images(user, challenge=None):
    """
    Enrolls a new user's face — but unlike a naive "save whatever the camera
    sees" loop, this requires the same liveness challenge used for step-up
    verification to be completed during capture, and screens every frame
    through the spoof-resistance check before it's allowed into the
    training set. Without this, someone could enroll a printed photo or a
    screen replay of another person as a "verified" identity on day one,
    and every downstream defense would be protecting a compromised
    baseline. Captures go to a staging folder first and are only committed
    to the real dataset if the challenge is completed — a failed or spoofed
    attempt never touches (or overwrites) a user's existing valid images.
    """
    staging_folder = os.path.join(STAGING_DIR, user)
    if os.path.exists(staging_folder):
        for f in os.listdir(staging_folder):
            os.remove(os.path.join(staging_folder, f))
    os.makedirs(staging_folder, exist_ok=True)

    cam = _open_camera()
    if cam is None:
        return {"ok": False, "error": "camera_unavailable", "captured": 0}

    accepted = 0
    rejected_low_spoof = 0
    quality_scores = []
    spoof_scores = []
    eye_visibility_trace = []
    motion_trace = []
    hand_trace = []
    display_ok = True

    challenge_banner = CHALLENGES[challenge]["instruction"] if challenge in CHALLENGES else None
    start = time.time()

    while True:
        elapsed = time.time() - start
        if accepted >= ENROLL_TARGET_SAMPLES:
            break
        if elapsed >= MAX_ENROLL_CAPTURE_SEC:
            break

        ret, frame = cam.read()
        if not ret:
            break

        frame_h, frame_w = frame.shape[:2]
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        faces = FACE_CASCADE.detectMultiScale(gray, 1.3, 5)

        if len(faces) > 0:
            x, y, w, h = (int(v) for v in max(faces, key=lambda f: f[2] * f[3]))
            face = cv2.resize(gray[y:y + h, x:x + w], (200, 200))
            quality = _quality_score(face)
            spoof = _spoof_resistance_score(face)

            landmarks = _get_landmarks(gray, x, y, w, h)
            eyes_for_overlay = []
            if landmarks is not None:
                eyes_visible, _ear = _eyes_open_from_landmarks(landmarks)
                nose = landmarks[NOSE_TIP_IDX]
                cx_norm = float(nose[0]) / frame_w
                cy_norm = float(nose[1]) / frame_h
            else:
                eyes_for_overlay = _detect_eyes_in_face(gray, x, y, w, h)
                eyes_visible = len(eyes_for_overlay) >= 1
                cx_norm = (x + w / 2) / frame_w
                cy_norm = (y + h / 2) / frame_h

            eye_visibility_trace.append(eyes_visible)
            motion_trace.append((elapsed, cx_norm, cy_norm))
            hand_trace.append(_hand_regions_present(frame, (x, y, w, h), frame_w, frame_h))

            if spoof >= ENROLL_SPOOF_MIN:
                quality_scores.append(quality)
                spoof_scores.append(spoof)
                cv2.imwrite(os.path.join(staging_folder, f"{accepted}.jpg"), face)
                accepted += 1
                color = (0, 255, 0)
                label = str(accepted)
            else:
                rejected_low_spoof += 1
                color = (0, 0, 255)
                label = "low spoof-resistance — not saved"

            cv2.rectangle(frame, (x, y), (x + w, y + h), color, 2)
            cv2.putText(frame, label, (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)
            _draw_analysis_overlay(frame, x, y, w, h, face, eyes_for_overlay,
                                    challenge=challenge,
                                    eye_trace_so_far=eye_visibility_trace,
                                    motion_trace_so_far=motion_trace,
                                    landmarks=landmarks,
                                    hand_trace_so_far=hand_trace)
        else:
            eye_visibility_trace.append(False)

        if challenge_banner:
            cv2.putText(frame, challenge_banner, (10, frame_h - 20),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 0), 2)

        _publish_frame(frame)

        if display_ok:
            display_ok = _safe_imshow("Register User (ESC to Exit)", frame)
            if cv2.waitKey(1) == 27:
                break

    cam.release()
    if display_ok:
        cv2.destroyAllWindows()
    _reset_camera_preview()

    blinks = _count_blinks(eye_visibility_trace)

    challenge_completed, challenge_score, challenge_detail = True, 0.0, None
    if challenge:
        challenge_completed, challenge_score, challenge_detail = _evaluate_challenge(
            challenge, {"blinks": blinks, "motion_trace": motion_trace, "hand_trace": hand_trace}
        )

    result = {
        "captured": accepted,
        "rejected_low_spoof": rejected_low_spoof,
        "avg_quality": round(statistics.mean(quality_scores), 1) if quality_scores else 0.0,
        "avg_spoof_resistance": round(statistics.mean(spoof_scores), 1) if spoof_scores else 0.0,
    }
    if challenge:
        result["challenge"] = challenge
        result["challenge_label"] = CHALLENGES[challenge]["label"]
        result["challenge_completed"] = challenge_completed
        result["challenge_detail"] = challenge_detail

    if accepted < MIN_ENROLL_SAMPLES:
        for f in os.listdir(staging_folder):
            os.remove(os.path.join(staging_folder, f))
        result["ok"] = False
        result["error"] = "insufficient_samples"
        result["message"] = (
            f"Only {accepted} usable frame(s) passed the spoof-resistance check "
            f"(needed at least {MIN_ENROLL_SAMPLES}). Try again with better lighting, "
            "facing the camera directly, without a photo or screen in frame."
        )
        return result

    if challenge and not challenge_completed:
        for f in os.listdir(staging_folder):
            os.remove(os.path.join(staging_folder, f))
        result["ok"] = False
        result["error"] = "challenge_failed"
        result["message"] = (
            f'The "{CHALLENGES[challenge]["label"]}" challenge was not detected during capture, '
            "so this enrollment was rejected rather than trained on unverified images."
        )
        return result

    # Committed: move staged, accepted images into the real per-user dataset
    # folder, numbering after whatever images (if any) already exist there —
    # so a second successful enrollment session adds to a user's data
    # instead of overwriting it.
    final_folder = os.path.join(DATASET_DIR, user)
    os.makedirs(final_folder, exist_ok=True)
    existing = [f for f in os.listdir(final_folder) if f.lower().endswith(".jpg")]
    next_index = len(existing)
    for i, fname in enumerate(sorted(os.listdir(staging_folder), key=lambda n: int(n.split(".")[0]))):
        os.rename(
            os.path.join(staging_folder, fname),
            os.path.join(final_folder, f"{next_index + i}.jpg"),
        )
    _rmdir_best_effort(staging_folder)

    result["ok"] = True
    return result



def train_model():
    faces = []
    labels = []
    names = {}
    idx = 0

    for person in sorted(os.listdir(DATASET_DIR)):
        path = f"{DATASET_DIR}/{person}"
        if not os.path.isdir(path):
            continue

        image_files = [f for f in os.listdir(path) if f.lower().endswith((".jpg", ".jpeg", ".png"))]
        if not image_files:
            continue

        names[idx] = person
        for img in image_files:
            image = PILImage.open(os.path.join(path, img)).convert("L")
            faces.append(np.array(image, "uint8"))
            labels.append(idx)

        idx += 1

    if not faces:
        return {"ok": False, "error": "no_data", "users": 0, "images": 0}

    RECOGNIZER.train(faces, np.array(labels))
    RECOGNIZER.save(f"{TRAINER_DIR}/trainer.yml")
    np.save(f"{TRAINER_DIR}/names.npy", names)

    return {"ok": True, "users": len(names), "images": len(faces)}


# ---------------------------------------------------------------------------
# Routes
# ---------------------------------------------------------------------------
def _safe_route(fn):
    """
    Guarantee every action route returns valid JSON, even if something inside
    throws. Without this, an unhandled exception falls through to Flask's
    debug-mode HTML error page, the frontend's res.json() call then fails to
    parse it, and the UI misreports that as "could not reach the server" —
    hiding the real error instead of showing it.
    """
    def wrapped(*args, **kwargs):
        try:
            return fn(*args, **kwargs)
        except Exception as exc:
            app.logger.exception("Error in %s", fn.__name__)
            return jsonify({
                "ok": False,
                "error": "server_error",
                "message": f"{type(exc).__name__}: {exc}",
            })
    wrapped.__name__ = fn.__name__
    return wrapped


@app.route("/")
def home():
    return render_template("index.html")


@app.route("/camera/stream")
def camera_stream_route():
    """
    MJPEG stream of whatever frame the active capture loop (registration,
    verify, step-up, or a live Attack Lab attempt) most recently published —
    the same annotated frame the native cv2 window would show, but usable
    as an <img> src inside the browser. Shows the idle placeholder when
    nothing is capturing. The generator loop exits cleanly once the browser
    stops requesting (e.g. the <img> is removed), rather than leaking a
    thread per view.
    """
    def generate():
        boundary = b"--frame\r\n"
        try:
            while True:
                with _frame_lock:
                    frame = _latest_frame_jpeg
                if frame is not None:
                    yield boundary + b"Content-Type: image/jpeg\r\n\r\n" + frame + b"\r\n"
                time.sleep(0.08)  # ~12 fps is plenty for a preview
        except GeneratorExit:
            return

    return Response(generate(), mimetype="multipart/x-mixed-replace; boundary=frame")


@app.route("/status")
@_safe_route
def status_route():
    users = [d for d in os.listdir(DATASET_DIR) if os.path.isdir(os.path.join(DATASET_DIR, d))]
    trained = os.path.exists(os.path.join(TRAINER_DIR, "trainer.yml"))
    return jsonify({
        "registered_users": len(users),
        "users": users,
        "model_trained": trained,
    })


@app.route("/register/challenge")
@_safe_route
def register_challenge_route():
    """Step 1 of registration: pick a random liveness challenge the person
    must perform during capture, same mechanism as step-up verification."""
    challenge_id = random.choice(list(CHALLENGES.keys()))
    return jsonify({
        "ok": True,
        "challenge": challenge_id,
        "label": CHALLENGES[challenge_id]["label"],
        "instruction": CHALLENGES[challenge_id]["instruction"],
    })


@app.route("/register/<user>")
@_safe_route
def register_route(user):
    """Step 2 of registration: capture against the challenge issued in step
    1. A missing/invalid challenge is rejected rather than silently allowed
    through without a liveness check."""
    challenge_id = request.args.get("challenge")
    if challenge_id not in CHALLENGES:
        return jsonify({
            "ok": False,
            "error": "bad_challenge",
            "message": "Unknown or missing challenge id. Call /register/challenge first.",
        })
    result = capture_images(user, challenge=challenge_id)
    if result.get("ok"):
        # Pre-generate Attack Lab content now, while we have fresh photos,
        # so demonstrating an attack later never needs to compute anything
        # live — see _build_attack_cache(). Best-effort: never let a cache
        # build issue affect the registration response itself.
        try:
            _build_attack_cache(user)
        except Exception as exc:
            app.logger.warning("Attack cache pre-generation failed for %s: %s", user, exc)
    return jsonify(result)


@app.route("/train")
@_safe_route
def train_route():
    result = train_model()
    return jsonify(result)


@app.route("/voice/status")
@_safe_route
def voice_status_route():
    enrolled = [f[:-4] for f in os.listdir(VOICE_DIR) if f.endswith(".pkl")] if VOICE_AVAILABLE else []
    return jsonify({
        "ok": True,
        "available": VOICE_AVAILABLE,
        "unavailable_reason": None if VOICE_AVAILABLE else _voice_unavailable_message(),
        "asr_available": ASR_AVAILABLE,
        "asr_unavailable_reason": ASR_UNAVAILABLE_REASON,
        "enrolled_users": enrolled,
    })


@app.route("/voice/enroll/<user>")
@_safe_route
def voice_enroll_route(user):
    result = voice_enroll(user)
    return jsonify(result)


@app.route("/voice/challenge")
@_safe_route
def voice_challenge_route():
    """Generates a fresh random phrase for challenge-response voice
    verification — a new one every call, never a fixed sentence."""
    phrase = generate_voice_phrase()
    return jsonify({
        "ok": True,
        "phrase": phrase,
        "content_checking_available": ASR_AVAILABLE,
    })


@app.route("/voice/verify/<user>")
@_safe_route
def voice_verify_route(user):
    """Speaker-only check (backward compatible) when called with no
    `phrase` query param — used by the standalone Voice ID page. Full
    challenge-response (speaker + content + replay checks) when a phrase
    is supplied — used by the unified verification flow, which fetches
    one from /voice/challenge first."""
    phrase = request.args.get("phrase")
    result = voice_verify(user, phrase=phrase)
    return jsonify(result)


@app.route("/verify")
@_safe_route
def verify_route():
    locked, remaining = _check_global_lockout()
    if locked:
        return jsonify({
            "ok": False, "error": "locked_out",
            "message": f"Too many unrecognized attempts. Try again in {int(remaining)} seconds.",
            "lockout_seconds_remaining": remaining,
        })
    result = _run_verification(VERIFY_DURATION_SEC, "Verify Identity (ESC to Exit)")
    if result.get("ok"):
        identity = result.get("identity")
        # Per-identity lockout is only checkable once identity is known —
        # after the capture, not before. If this specific identity is
        # currently locked out from an earlier run of failures, the
        # lockout is enforced regardless of what this attempt's own score
        # would have been — that's the actual point of a lockout, not just
        # tallying failures. Not registered as a fresh denial itself
        # (see below), so retrying during the cooldown doesn't keep
        # extending it indefinitely.
        id_locked, id_remaining = _check_identity_lockout(identity)
        overridden = False
        if id_locked and result["decision"] != "denied":
            result["decision"] = "denied"
            result["risk_level"] = "high"
            result["lockout_message"] = f'"{identity}" is temporarily locked out after repeated failed attempts. Try again in {int(id_remaining)} seconds.'
            overridden = True
        _log_verification_attempt(result, stage="first_pass")
        if not overridden:
            _register_verification_outcome(result.get("decision"), identity)
    return jsonify(result)


@app.route("/verify/stepup/challenge")
@_safe_route
def verify_stepup_challenge_route():
    """Step 1 of step-up: pick and return a random challenge so the frontend
    can display the instruction before the capture actually starts."""
    challenge_id = random.choice(list(CHALLENGES.keys()))
    return jsonify({
        "ok": True,
        "challenge": challenge_id,
        "label": CHALLENGES[challenge_id]["label"],
        "instruction": CHALLENGES[challenge_id]["instruction"],
    })


@app.route("/verify/stepup/run")
@_safe_route
def verify_stepup_run_route():
    """Step 2 of step-up: run the actual capture against the challenge the
    frontend was given in step 1. Completing the specific requested action
    is a hard gate — a high trust score alone does not grant access here."""
    locked, remaining = _check_global_lockout()
    if locked:
        return jsonify({
            "ok": False, "error": "locked_out",
            "message": f"Too many unrecognized attempts. Try again in {int(remaining)} seconds.",
            "lockout_seconds_remaining": remaining,
        })
    challenge_id = request.args.get("challenge")
    if challenge_id not in CHALLENGES:
        return jsonify({
            "ok": False,
            "error": "bad_challenge",
            "message": "Unknown or missing challenge id. Call /verify/stepup/challenge first.",
        })

    result = _run_verification(
        STEPUP_DURATION_SEC,
        f"Step-Up Verification — {CHALLENGES[challenge_id]['label']} (ESC to Exit)",
        grant_threshold=STEPUP_GRANT_THRESHOLD,
        challenge=challenge_id,
    )
    # Step-up never offers a further step-up — it's granted or denied.
    if result.get("ok") and result["decision"] == "step_up":
        result["decision"] = "denied"
        result["risk_level"] = "high"
    if result.get("ok"):
        identity = result.get("identity")
        id_locked, id_remaining = _check_identity_lockout(identity)
        overridden = False
        if id_locked and result["decision"] != "denied":
            result["decision"] = "denied"
            result["risk_level"] = "high"
            result["lockout_message"] = f'"{identity}" is temporarily locked out after repeated failed attempts. Try again in {int(id_remaining)} seconds.'
            overridden = True
        _log_verification_attempt(result, stage="step_up")
        if not overridden:
            _register_verification_outcome(result.get("decision"), identity)
    return jsonify(result)


@app.route("/verify/log")
@_safe_route
def verification_log_route():
    """
    Read-only audit trail of real verification attempts (/verify and
    /verify/stepup/run) — persisted to verification_log.csv, survives
    restarts. Deliberately separate from /attack/log, which is Attack Lab
    test attempts and intentionally resets on restart (that one answers
    "did the demo's attacks get blocked"; this one answers "what has this
    system actually decided, for whom, over its operating history" — the
    question a compliance reviewer would actually ask).
    """
    limit = min(int(request.args.get("limit", 50)), VERIFICATION_LOG_RECENT_MAX)
    recent = list(reversed(VERIFICATION_LOG_RECENT[-limit:]))
    granted = sum(1 for e in VERIFICATION_LOG_RECENT if e.get("decision") == "granted")
    denied = sum(1 for e in VERIFICATION_LOG_RECENT if e.get("decision") == "denied")
    locked, remaining = _check_global_lockout()
    now = time.time()
    identities_locked = sum(1 for until in _identity_lockout_until.values() if until > now)
    return jsonify({
        "ok": True,
        "recent": recent,
        "total_logged": VERIFICATION_LOG_TOTAL_COUNT,
        "recent_granted": granted,
        "recent_denied": denied,
        "locked_out": locked,
        "lockout_seconds_remaining": remaining,
        "identities_currently_locked_out": identities_locked,
    })


@app.route("/verify/log/export")
@_safe_route
def verification_log_export_route():
    """Downloads the full persistent audit log as CSV."""
    if not os.path.exists(VERIFICATION_LOG_PATH):
        return jsonify({"ok": False, "message": "No verification attempts logged yet."})
    return send_file(VERIFICATION_LOG_PATH, as_attachment=True,
                      download_name="verification_log.csv", mimetype="text/csv")


@app.route("/attack/types")
@_safe_route
def attack_types_route():
    return jsonify({
        "ok": True,
        "types": [{"id": k, "label": v} for k, v in ATTACK_TYPES.items()],
    })


@app.route("/attack/run")
@_safe_route
def attack_run_route():
    """
    Runs a normal verification pass, deliberately labeled and logged as an
    attempted presentation attack (printed photo, screen replay, etc). Uses
    the exact same scoring pipeline as /verify — nothing is special-cased —
    so a blocked attempt here is a genuine demonstration of the trust-score
    and spoof-resistance signals doing their job, not a scripted outcome.
    """
    attack_type = request.args.get("attack_type", "other")
    if attack_type not in ATTACK_TYPES:
        attack_type = "other"

    result = _run_verification(VERIFY_DURATION_SEC, f"Attack Lab — {ATTACK_TYPES[attack_type]} (ESC to Exit)")

    blocked = (not result.get("ok")) or result.get("decision") in ("denied", "step_up")
    entry = {
        "timestamp": time.time(),
        "attack_type": attack_type,
        "attack_label": ATTACK_TYPES[attack_type],
        "simulated": False,
        "ok": result.get("ok", False),
        "decision": result.get("decision"),
        "trust_score": result.get("trust_score"),
        "spoof_resistance": (result.get("factors") or {}).get("spoof_resistance"),
        "blocked": blocked,
    }
    ATTACK_LOG.append(entry)

    result["attack_type"] = attack_type
    result["attack_label"] = ATTACK_TYPES[attack_type]
    result["simulated"] = False
    result["blocked"] = blocked
    result["metadata_confidence"] = None
    result["metadata_detail"] = "Not applicable — a live camera frame has no file metadata to analyze."
    result["log_summary"] = _attack_log_summary()
    return jsonify(result)


def _get_enrolled_user_photo(user):
    """
    Picks a random photo from an already-enrolled user's registration
    dataset — the self-contained source for Attack Lab simulation, so
    demonstrating an attack never requires preparing or uploading a
    separate file beforehand. Returns raw JPEG bytes (same shape an
    uploaded file's bytes would be) or None if the user has no stored
    photos (e.g. registered before this feature, or dataset cleared).
    """
    folder = os.path.join(DATASET_DIR, user)
    if not os.path.isdir(folder):
        return None
    photos = [f for f in os.listdir(folder) if f.lower().endswith(".jpg")]
    if not photos:
        return None
    chosen = random.choice(photos)
    with open(os.path.join(folder, chosen), "rb") as f:
        return np.frombuffer(f.read(), dtype=np.uint8)


# ---------------------------------------------------------------------------
# Attack Lab content cache — pre-generated synthetic attack variants
# ---------------------------------------------------------------------------
# Without this, every simulated attack degrades a source photo live, on the
# spot, on every click — cheap individually, but it means demo reliability
# depends on that computation succeeding in front of judges every time, and
# "Generate New Variant" has to redo the work from scratch. Pre-generating a
# small set of degraded variants once (right after registration, when the
# person's photos already exist) and storing them means the Attack Lab just
# reads a ready-made file instead of computing anything at click time — same
# self-contained principle as before (still sourced from the user's own
# registration photos, nothing external), just moved earlier.
ATTACK_CACHE_DIR = os.path.join(BASE_DIR, "attack_cache")
os.makedirs(ATTACK_CACHE_DIR, exist_ok=True)
ATTACK_CACHE_VARIANTS_PER_TYPE = 3


def _attack_cache_dir(user, attack_type):
    return os.path.join(ATTACK_CACHE_DIR, user, attack_type)


def _build_attack_cache(user, attack_types=None):
    """
    Generates and stores ATTACK_CACHE_VARIANTS_PER_TYPE degraded variants
    per attack type for this user, sourced from their own registration
    photos. Called automatically right after registration completes; safe
    to call again later (e.g. re-registration, or a manual refresh) — it
    just overwrites the cached files. Silently does nothing (returns 0) if
    the user has no registration photos yet, rather than raising — this
    runs as a best-effort background step, not something that should be
    able to fail a registration.
    """
    if attack_types is None:
        attack_types = [t for t in ATTACK_TYPES if t != "other"]
    photos_dir = os.path.join(DATASET_DIR, user)
    if not os.path.isdir(photos_dir):
        return 0
    photos = [f for f in os.listdir(photos_dir) if f.lower().endswith(".jpg")]
    if not photos:
        return 0

    generated = 0
    for attack_type in attack_types:
        degrader = DEGRADERS.get(attack_type, DEGRADERS["other"])
        cache_dir = _attack_cache_dir(user, attack_type)
        os.makedirs(cache_dir, exist_ok=True)
        for i in range(ATTACK_CACHE_VARIANTS_PER_TYPE):
            src_name = random.choice(photos)
            try:
                with open(os.path.join(photos_dir, src_name), "rb") as f:
                    bgr = cv2.imdecode(np.frombuffer(f.read(), dtype=np.uint8), cv2.IMREAD_COLOR)
                if bgr is None:
                    continue
                degraded = degrader(bgr)
                cv2.imwrite(os.path.join(cache_dir, f"variant_{i}.jpg"), degraded)
                generated += 1
            except Exception as exc:
                app.logger.warning("Attack cache generation failed for %s/%s variant %d: %s", user, attack_type, i, exc)
    return generated


def _get_cached_attack_variant(user, attack_type):
    """Raw JPEG bytes of a random pre-generated variant for this
    (user, attack_type), or None if nothing is cached yet."""
    cache_dir = _attack_cache_dir(user, attack_type)
    if not os.path.isdir(cache_dir):
        return None
    variants = [f for f in os.listdir(cache_dir) if f.lower().endswith(".jpg")]
    if not variants:
        return None
    chosen = random.choice(variants)
    with open(os.path.join(cache_dir, chosen), "rb") as f:
        return f.read()


def _cache_generated_variant(user, attack_type, degraded_bgr):
    """Opportunistically saves a freshly (live-)generated variant into the
    cache, so a cache-miss now still leaves things warmer for next time —
    the cache converges to fully populated over normal use, not just via
    the explicit post-registration build."""
    try:
        cache_dir = _attack_cache_dir(user, attack_type)
        os.makedirs(cache_dir, exist_ok=True)
        existing = [f for f in os.listdir(cache_dir) if f.lower().endswith(".jpg")]
        idx = len(existing) % ATTACK_CACHE_VARIANTS_PER_TYPE
        cv2.imwrite(os.path.join(cache_dir, f"variant_{idx}.jpg"), degraded_bgr)
    except Exception:
        pass  # best-effort only — never let cache-warming break a real request


def _warm_attack_cache_for_existing_users():
    """
    Runs once at startup, in a background thread so it can't delay the
    server coming up: pre-generates Attack Lab content for any already-
    registered user who doesn't have it yet (registered before this
    feature existed, or the cache directory was cleared) — so the very
    first attack attempt for them is instant too, not just the second one
    onward via the request-time fallback.
    """
    try:
        if not os.path.isdir(DATASET_DIR):
            return
        users = [d for d in os.listdir(DATASET_DIR)
                 if os.path.isdir(os.path.join(DATASET_DIR, d)) and d != "_staging"]
        warmed = 0
        for user in users:
            if not os.path.isdir(os.path.join(ATTACK_CACHE_DIR, user)):
                if _build_attack_cache(user) > 0:
                    warmed += 1
        if warmed:
            app.logger.info("Attack Lab cache warmed for %d existing user(s) at startup.", warmed)
    except Exception as exc:
        app.logger.warning("Background attack-cache warming failed: %s", exc)


threading.Thread(target=_warm_attack_cache_for_existing_users, daemon=True).start()


def _correct_orientation_if_needed(frame):
    """
    Phone-recorded videos very commonly store raw pixel data in landscape
    orientation plus a rotation metadata flag that video PLAYERS apply
    automatically — but OpenCV's VideoCapture frequently does not apply
    it, so a portrait selfie video that looks completely normal in any
    video player can come out sideways here. Confirmed directly: a
    genuinely detectable frontal face goes to zero detections at 90, 180,
    or 270 degrees rotation with the same cascade used everywhere else in
    this app — this isn't a hypothetical, it's the measured behavior.

    Tries each rotation and returns the first one where a face is actually
    detected, falling back to the original orientation if none of them
    find one (so a genuinely faceless frame still honestly reports "no
    face" downstream, rather than this function silently claiming success).
    """
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    if len(FACE_CASCADE.detectMultiScale(gray, 1.3, 5)) > 0:
        return frame
    for code in (cv2.ROTATE_90_CLOCKWISE, cv2.ROTATE_180, cv2.ROTATE_90_COUNTERCLOCKWISE):
        candidate = cv2.rotate(frame, code)
        gray_c = cv2.cvtColor(candidate, cv2.COLOR_BGR2GRAY)
        if len(FACE_CASCADE.detectMultiScale(gray_c, 1.3, 5)) > 0:
            return candidate
    return frame


def _extract_frame_from_video(raw_bytes):
    """
    Extracts a single representative frame from an uploaded video file, so
    the Attack Lab's video-based attack types aren't limited to still-photo
    uploads only. OpenCV's VideoCapture needs a real file path, not
    in-memory bytes, so this writes to a temporary file, reads a frame,
    and always cleans up regardless of outcome. Returns JPEG bytes, or
    None on any failure (corrupt file, unsupported codec, zero frames) —
    callers are expected to give the person a clear message and point them
    to Live demonstration mode as a fallback, not fail silently.

    Two things this specifically guards against, both confirmed as real
    (not hypothetical) failure modes during testing:
    1. Frame-count metadata and seek-by-index are unreliable for many
       real-world video files (variable frame rate, certain encoders) —
       a single seek to "the middle" can land on a bad frame even when
       the video clearly has a detectable face elsewhere. Several
       candidate positions are tried, not just one.
    2. Phone-video rotation metadata is frequently not applied by
       OpenCV's decoder (see _correct_orientation_if_needed) — every
       candidate frame is checked for a face in all four orientations
       before giving up on it.
    """
    tmp_path = None
    try:
        with tempfile.NamedTemporaryFile(suffix=".mp4", delete=False) as tmp:
            tmp.write(raw_bytes)
            tmp_path = tmp.name

        cap = cv2.VideoCapture(tmp_path)
        if not cap.isOpened():
            return None
        # Ask the backend to apply rotation metadata automatically where
        # it's supported — cheap to request, harmless where unsupported,
        # and the rotation-detection fallback above still catches the
        # remaining cases where this flag doesn't take effect.
        try:
            cap.set(cv2.CAP_PROP_ORIENTATION_AUTO, 1)
        except Exception:
            pass

        frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        if frame_count > 0:
            candidates = sorted(set(
                max(0, min(frame_count - 1, int(frame_count * f)))
                for f in (0.5, 0.25, 0.75, 0.1, 0.9, 0.0)
            ))
        else:
            candidates = [0]

        best_frame = None
        for pos in candidates:
            cap.set(cv2.CAP_PROP_POS_FRAMES, pos)
            ok, frame = cap.read()
            if not ok or frame is None:
                continue
            if best_frame is None:
                best_frame = frame  # keep the first readable frame as a fallback
            corrected = _correct_orientation_if_needed(frame)
            gray = cv2.cvtColor(corrected, cv2.COLOR_BGR2GRAY)
            if len(FACE_CASCADE.detectMultiScale(gray, 1.3, 5)) > 0:
                cap.release()
                ok2, enc = cv2.imencode(".jpg", corrected)
                return enc.tobytes() if ok2 else None

        cap.release()
        if best_frame is None:
            return None
        # No candidate frame had a detectable face in any orientation —
        # return the first readable frame anyway (orientation-corrected if
        # that alone changes anything) so the real pipeline still gets a
        # chance and reports an honest "no face detected" rather than this
        # function silently failing before the person even sees an attempt.
        corrected = _correct_orientation_if_needed(best_frame)
        ok2, enc = cv2.imencode(".jpg", corrected)
        return enc.tobytes() if ok2 else None
    except Exception:
        return None
    finally:
        if tmp_path and os.path.exists(tmp_path):
            try:
                os.remove(tmp_path)
            except Exception:
                pass


VIDEO_EXTENSIONS = (".mp4", ".mov", ".avi", ".mkv", ".webm", ".m4v")


@app.route("/attack/simulate", methods=["POST"])
@_safe_route
def attack_simulate_route():
    """
    Runs a simulated attack from a photo or video — self-contained by
    default: pass `source_user` (an already-enrolled identity) and the
    server picks one of that user's own registration photos automatically,
    no upload needed. An uploaded `image` file still works too and takes
    priority if both are somehow provided, for backward compatibility /
    bringing your own image. Video files are accepted too — a single
    representative frame is extracted and used exactly like an uploaded
    photo from there on. The chosen image is degraded to approximate the
    selected attack type (parameters randomized per call — see the
    DEGRADERS docstrings — so repeated "Generate New Variant" runs produce
    genuinely different output, not an identical result every time), then
    scored through the exact same pipeline as a live attempt.
    """
    attack_type = request.form.get("attack_type", "other")
    if attack_type not in ATTACK_TYPES:
        attack_type = "other"

    upload = request.files.get("image")
    source_user = request.form.get("source_user", "").strip()
    file_bytes = None
    source_label = None
    already_degraded = False
    skip_metadata_for_upload = False

    if upload is not None and upload.filename != "":
        raw_upload_bytes = upload.read()
        filename_lower = upload.filename.lower()
        is_video = filename_lower.endswith(VIDEO_EXTENSIONS) or (upload.mimetype or "").startswith("video/")
        if is_video:
            extracted = _extract_frame_from_video(raw_upload_bytes)
            if extracted is None:
                return jsonify({
                    "ok": False, "error": "bad_video",
                    "message": "Could not read a frame from that video file. Try a different "
                               "file (MP4 is the most reliable format), or use Live demonstration "
                               "mode with the video playing in front of the webcam instead.",
                })
            file_bytes = np.frombuffer(extracted, dtype=np.uint8)
            source_label = "a frame extracted from the uploaded video"
            # A frame pulled from a video and re-encoded as JPEG never
            # carries the original file's metadata regardless of the
            # video's own authenticity — same reasoning as self-contained
            # dataset photos below, so the same honest skip applies, with
            # a message that accurately describes THIS source.
            skip_metadata_for_upload = "Not applicable \u2014 this is a frame extracted from the uploaded video and re-encoded as a still image, which never carries the original file's metadata regardless of the video's authenticity."
        else:
            file_bytes = np.frombuffer(raw_upload_bytes, dtype=np.uint8)
            source_label = "uploaded photo"
    elif source_user:
        # Check the pre-generated cache first — no live degradation needed
        # if content is already sitting on disk (see _build_attack_cache,
        # which runs automatically right after registration). Falls back
        # to live generation only on a cache miss (e.g. a user registered
        # before this feature existed), and opportunistically warms the
        # cache with that result so the next run for this user+type is
        # instant too.
        cached = _get_cached_attack_variant(source_user, attack_type)
        if cached is not None:
            file_bytes = np.frombuffer(cached, dtype=np.uint8)
            source_label = f"{source_user}'s pre-generated {ATTACK_TYPES.get(attack_type, 'attack')} sample"
            already_degraded = True
        else:
            raw = _get_enrolled_user_photo(source_user)
            if raw is None:
                return jsonify({
                    "ok": False, "error": "no_source_photos",
                    "message": f'"{source_user}" has no stored registration photos to use as a source. '
                               "Register this user first, or upload a photo instead.",
                })
            bgr = cv2.imdecode(raw, cv2.IMREAD_COLOR)
            if bgr is not None:
                degrader = DEGRADERS.get(attack_type, DEGRADERS["other"])
                degraded = degrader(bgr)
                _cache_generated_variant(source_user, attack_type, degraded)
                ok, enc = cv2.imencode(".jpg", degraded)
                file_bytes = np.frombuffer(enc.tobytes(), dtype=np.uint8) if ok else raw
                already_degraded = ok
            else:
                file_bytes = raw
                already_degraded = False
            source_label = f"{source_user}'s registration photo (freshly generated, now cached)"
    else:
        return jsonify({
            "ok": False, "error": "no_source",
            "message": "Select an enrolled user as the photo source, or upload an image.",
        })

    if file_bytes is None or file_bytes.size == 0:
        return jsonify({"ok": False, "error": "empty_file", "message": "No usable image data was found."})

    result = _simulate_verification(file_bytes, attack_type,
                                     skip_metadata=(skip_metadata_for_upload if skip_metadata_for_upload else bool(source_user and upload is None)),
                                     already_degraded=already_degraded)
    result["source_label"] = source_label

    blocked = (not result.get("ok")) or result.get("decision") in ("denied", "step_up")
    entry = {
        "timestamp": time.time(),
        "attack_type": attack_type,
        "attack_label": ATTACK_TYPES[attack_type],
        "simulated": True,
        "ok": result.get("ok", False),
        "decision": result.get("decision"),
        "trust_score": result.get("trust_score"),
        "spoof_resistance": (result.get("factors") or {}).get("spoof_resistance"),
        "blocked": blocked,
    }
    ATTACK_LOG.append(entry)

    result["attack_type"] = attack_type
    result["attack_label"] = ATTACK_TYPES[attack_type]
    result["simulated"] = True
    result["blocked"] = blocked
    result["log_summary"] = _attack_log_summary()
    return jsonify(result)


def _attack_log_summary():
    total = len(ATTACK_LOG)
    blocked = sum(1 for e in ATTACK_LOG if e["blocked"])
    return {
        "total_attempts": total,
        "blocked": blocked,
        "block_rate": round(blocked / total * 100, 1) if total else None,
    }


# ---------------------------------------------------------------------------
# Attack Lab — voice attack scenarios
# ---------------------------------------------------------------------------
# Unlike the face attacks above, these deliberately do NOT have a
# simulate/upload mode. There's no synthesized-audio degradation pipeline
# analogous to the image DEGRADERS — building one honestly would need
# either real replay/synthetic-speech samples to validate against (not
# available here) or a voice-cloning model (explicitly out of scope per
# the CPU-only, no-huge-models constraint). Rather than fake a result,
# these are live-only, operator-guided scenarios that run through the
# exact same /voice/challenge + voice_verify() pipeline as normal
# verification — nothing special-cased for the "attack" framing.
VOICE_ATTACK_SCENARIOS = {
    "correct_phrase_correct_speaker": {
        "label": "Correct phrase + correct speaker (control)",
        "instruction": "The enrolled user reads the displayed phrase normally. Baseline case — expected to pass.",
    },
    "correct_phrase_wrong_speaker": {
        "label": "Correct phrase + wrong speaker",
        "instruction": "Have a DIFFERENT person read the displayed phrase aloud. Content should match; speaker should not.",
    },
    "wrong_phrase_correct_speaker": {
        "label": "Wrong phrase + correct speaker",
        "instruction": "The enrolled user says something OTHER than the displayed phrase. Speaker should match; content should not.",
    },
    "voice_replay": {
        "label": "Voice replay",
        "instruction": "Play back a previous recording (e.g. from a phone) instead of speaking live. Tests the replay-detection check.",
    },
}


@app.route("/attack/voice/types")
@_safe_route
def attack_voice_types_route():
    return jsonify({
        "ok": True,
        "types": [{"id": k, "label": v["label"], "instruction": v["instruction"]} for k, v in VOICE_ATTACK_SCENARIOS.items()],
        "enrolled_users": [f[:-4] for f in os.listdir(VOICE_DIR) if f.endswith(".pkl")] if VOICE_AVAILABLE else [],
    })


@app.route("/attack/voice/run")
@_safe_route
def attack_voice_run_route():
    """
    Runs a real voice challenge-response attempt against a target user,
    labeled with which scenario the operator is demonstrating — the
    voice_verify() call is identical to a normal verification's voice step.
    Nothing about the "attack" framing changes how the audio is scored;
    the scenario label just says what the operator is about to attempt for
    the log/UI.

    Accepts an optional `phrase` — the frontend is expected to fetch one
    from /voice/challenge, show it to the operator, and pass it here, the
    same two-step "reveal, then record" pattern used everywhere else a
    phrase is involved. Without one supplied, generates a fresh phrase
    server-side as a fallback (e.g. direct API testing) — but note that
    means whoever is speaking never sees it, which only makes sense for
    the scenarios that don't require saying the right phrase in the first
    place (wrong_phrase_correct_speaker, voice_replay).
    """
    user = request.args.get("user")
    scenario = request.args.get("scenario", "correct_phrase_correct_speaker")
    if not user:
        return jsonify({"ok": False, "error": "missing_user", "message": "Select an enrolled user to target."})
    if scenario not in VOICE_ATTACK_SCENARIOS:
        scenario = "correct_phrase_correct_speaker"

    phrase = request.args.get("phrase") or generate_voice_phrase()
    result = voice_verify(user, phrase=phrase)

    blocked = (not result.get("ok")) or result.get("decision") != "match"
    entry = {
        "timestamp": time.time(),
        "attack_type": "voice_" + scenario,
        "attack_label": "Voice: " + VOICE_ATTACK_SCENARIOS[scenario]["label"],
        "simulated": False,
        "ok": result.get("ok", False),
        "decision": result.get("decision"),
        "trust_score": result.get("voice_score"),
        "spoof_resistance": None,
        "blocked": blocked,
    }
    ATTACK_LOG.append(entry)

    result["scenario"] = scenario
    result["scenario_label"] = VOICE_ATTACK_SCENARIOS[scenario]["label"]
    result["target_user"] = user
    result["blocked"] = blocked
    result["log_summary"] = _attack_log_summary()
    return jsonify(result)


@app.route("/attack/log")
@_safe_route
def attack_log_route():
    recent = list(reversed(ATTACK_LOG[-20:]))
    return jsonify({
        "ok": True,
        "summary": _attack_log_summary(),
        "recent": recent,
    })


if __name__ == "__main__":
    # threaded=True is required: the live camera preview (/camera/stream) is
    # a long-lived connection, and it needs to run at the same time as the
    # actual capture request (register/verify/etc). Flask's dev server is
    # single-threaded by default, which would make the two block each other.
    #
    # use_reloader=False is just as important: the debug reloader watches
    # every loaded module's file (including site-packages, not just this
    # project), and restarts the whole process on any change it notices —
    # including spurious mtime touches from things like OneDrive syncing in
    # the background. A restart mid-request silently kills whatever capture
    # was in flight (camera or microphone), which looks like "it just isn't
    # working" with no real error, exactly the kind of failure a multi-
    # second voice recording is especially exposed to. debug=True still
    # gives full tracebacks in the browser and terminal — only the
    # auto-restart-on-file-change behavior is disabled. If you're actively
    # editing app.py and want auto-reload back, restart manually instead
    # after each change.
    app.run(debug=True, threaded=True, use_reloader=False)
