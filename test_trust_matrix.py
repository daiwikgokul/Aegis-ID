"""
Trust-score conceptual test matrix — Part 2 audit.

Drives the REAL _score_signals() / _decide() functions from app.py with
constructed `signals` dicts representing each scenario, rather than
hand-computing weighted sums separately. This is meant to be a re-runnable
regression check: run it before and after any scoring change and diff the
decisions, not just a one-off report.

Each scenario builds a `signals` dict matching the exact contract
_score_signals() expects (see _capture_verification_frames' return value
in app.py for the canonical shape), tuned to represent the scenario
description as realistically as the five factors allow.
"""
import app as appmod

GRANT, STEPUP, DENY = "granted", "step_up", "denied"


def make_signals(match_consistency, avg_conf, blinks, eye_ratio, spoof, quality,
                  total_frames=100, has_identity=True):
    """
    Builds a signals dict that will produce approximately the given
    match_consistency / recognition_confidence (via avg_conf, on the REAL
    RECOGNITION_SCORE_SCALE) / liveness (via blinks+eye_ratio, on the REAL
    _liveness_score formula) / spoof_resistance / image_quality factors
    once run through the actual _score_signals().
    """
    matched = int(round(total_frames * match_consistency / 100.0))
    label_votes = {0: matched} if has_identity and matched > 0 else {}
    confidences_by_label = {0: [avg_conf] * matched} if matched > 0 else {}
    return {
        "total_frames": total_frames,
        "frames_with_face": total_frames,
        "label_votes": label_votes,
        "confidences_by_label": confidences_by_label,
        "quality_scores": [quality] * total_frames,
        "spoof_scores": [spoof] * total_frames,
        "blinks": blinks,
        "eye_visible_ratio": eye_ratio,
        "motion_trace": [(i * 0.1, 0.5, 0.5) for i in range(total_frames)],
        "names": {0: "enrolled_user"},
    }


# Each entry: (label, signals kwargs, challenge, challenge_completed_override, expected_decision, note)
SCENARIOS = [
    ("1. Genuine, good lighting",
     dict(match_consistency=92, avg_conf=25, blinks=2, eye_ratio=90, spoof=85, quality=80),
     None, True, GRANT, "Clean baseline case."),

    ("2. Genuine, poor lighting",
     dict(match_consistency=78, avg_conf=55, blinks=2, eye_ratio=80, spoof=55, quality=30),
     None, True, None, "Should NOT be denied outright for bad lighting alone."),

    ("3. Genuine, glasses",
     dict(match_consistency=70, avg_conf=50, blinks=1, eye_ratio=70, spoof=60, quality=70),
     None, True, None, "Glasses glare can look moire-like; degraded but shouldn't hard-fail."),

    ("4. Genuine, moderate head movement",
     dict(match_consistency=65, avg_conf=55, blinks=2, eye_ratio=80, spoof=55, quality=40),
     None, True, None, "Motion blur hits texture+quality but shouldn't hard-fail."),

    ("5. Wrong person",
     dict(match_consistency=0, avg_conf=90, blinks=2, eye_ratio=90, spoof=85, quality=80, has_identity=False),
     None, True, DENY, "No frames clear the match cutoff -> identity=None -> hard deny."),

    ("6. Printed photograph",
     dict(match_consistency=80, avg_conf=50, blinks=0, eye_ratio=100, spoof=35, quality=55),
     None, True, None, "Static: 0 blinks but eye_ratio can still look 'steady'."),

    ("7. Phone/screen replay (video w/ real blinking)",
     dict(match_consistency=75, avg_conf=45, blinks=2, eye_ratio=85, spoof=30, quality=55),
     None, True, None, "Worst case: replay WITH visible natural blinking."),

    ("8. Video replay (generic loop)",
     dict(match_consistency=72, avg_conf=48, blinks=1, eye_ratio=75, spoof=28, quality=50),
     None, True, None, "Same family as #7."),

    ("9. Genuine user failing liveness challenge",
     dict(match_consistency=90, avg_conf=20, blinks=0, eye_ratio=90, spoof=85, quality=75),
     "blink_twice", False, DENY, "Everything else perfect; challenge not completed -> hard deny."),

    ("10. Genuine, borderline face match",
     dict(match_consistency=45, avg_conf=68, blinks=2, eye_ratio=80, spoof=80, quality=65),
     None, True, None, "Should land in step-up, not an outright grant or deny."),

    ("11. STRONG match + suspicious spoof signals",
     dict(match_consistency=95, avg_conf=15, blinks=2, eye_ratio=90, spoof=15, quality=60),
     None, True, None, "THE key test: must NOT grant despite excellent match."),

    ("12. Strong match + failed challenge",
     dict(match_consistency=95, avg_conf=15, blinks=0, eye_ratio=90, spoof=85, quality=75),
     "turn_head", False, DENY, "Hard gate must override even a near-perfect trust score."),

    ("13. Genuine user + successful hand-raise challenge",
     dict(match_consistency=90, avg_conf=22, blinks=1, eye_ratio=80, spoof=82, quality=72),
     "raise_right_hand", True, GRANT, "Hand-raise treated the same as blink/turn/nod once completed."),

    ("14. Genuine user + failed hand-raise challenge",
     dict(match_consistency=90, avg_conf=20, blinks=1, eye_ratio=80, spoof=85, quality=75),
     "raise_left_hand", False, DENY, "Same hard gate as any other challenge type — no exception for hand-raise."),

    ("15. Suspicious (borderline) face + successful challenge",
     dict(match_consistency=45, avg_conf=68, blinks=2, eye_ratio=80, spoof=80, quality=65),
     "nod", True, None, "Challenge success alone must not compensate for weak match."),
]


def run():
    print(f"{'Scenario':50} {'Trust':>6} {'Decision':>9} {'Tier':>9} {'Spoof':>6}  Note")
    print("-" * 120)
    mismatches = []
    for label, kwargs, challenge, challenge_ok_override, expected, note in SCENARIOS:
        signals = make_signals(**kwargs)
        trust, factors, identity, challenge_completed, _detail = appmod._score_signals(signals, challenge=challenge)
        # Override challenge_completed for scenarios that specifically test the hard gate
        if challenge is not None:
            challenge_completed = challenge_ok_override
        decision, risk = appmod._decide(trust, identity, appmod.GRANT_THRESHOLD, challenge_completed, factors["spoof_resistance"])
        tier = appmod._risk_tier(trust, decision, factors["spoof_resistance"])
        flag = ""
        if expected is not None and decision != expected:
            flag = "  <<< MISMATCH"
            mismatches.append(label)
        print(f"{label:50} {trust:6.1f} {decision:>9} {tier:>9} {factors['spoof_resistance']:6.1f}  {note}{flag}")
    print("-" * 120)
    if mismatches:
        print(f"\n{len(mismatches)} scenario(s) did not match expected decision:")
        for m in mismatches:
            print(" -", m)
    else:
        print("\nAll scenarios with a hard-coded expectation matched.")


if __name__ == "__main__":
    run()
