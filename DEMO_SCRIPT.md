# AEGIS·ID — Live Demo Script (SIH Presentation)

Target: ~60-90 seconds. Four steps. Each step should take you *less* time
to perform than it took to read this line — practice the physical actions
(who reads the phrase, where the printed photo is, which hand to raise)
before you're in front of judges, not during.

**Positioning line to open with, if you have 10 seconds for it:**

> "We don't rely on a single deepfake detector. We combine identity,
> behavior, liveness, texture, frequency analysis, spatial consistency,
> metadata, and voice challenge-response — and use all of it together to
> calculate risk and adapt what we ask for."

---

## Step 1 — Genuine baseline (~15s)

Register/verify with the actual demo presenter beforehand so this step is
just a **Verify** click.

**Show:** Face capture → trust score → `LOW RISK` badge → `GRANTED`.

**Say:**
> "The system doesn't authenticate on face recognition alone. It combines
> identity confidence, liveness, presentation-attack signals, and image
> quality into one trust score — and this clean pass shows all of them
> agreeing."

---

## Step 2 — Printed photo or screen attack (~20s)

Use a printed photo of the same presenter, or a phone showing their photo/
video. Go to **Attack Lab → Face / presentation attacks**, pick
`Printed photo` or `Phone / screen replay`, **Live demonstration** mode,
run it.

**Show:** Face match may still score high (same identity!) — but
**Liveness ↓, Texture ↓, Moiré/Spatial anomalies ↑ → `HIGH RISK` →
`DENIED`.**

**Say:**
> "Face recognition alone would have passed this — it's the same person's
> face. What catches it is everything else: no genuine blink pattern, flat
> printed texture, and for a screen, moiré interference from the pixel
> grid. This is exactly why we don't stop at face match."

*(If you don't have a prop handy, use Attack Simulation instead — upload
any photo, same pipeline, same signals, clearly labeled as a simulation.)*

---

## Step 3 — Adaptive authentication / step-up (~20s)

Use a deliberately borderline case — off-angle lighting, or just accept
whatever step-up naturally triggers on a second **Verify** attempt.

**Show:** `MEDIUM RISK` → a randomly issued challenge appears (blink twice
/ turn head / nod / raise a hand — **whichever the system picks**, don't
pre-script which one) → person performs it → recalculated trust →
`GRANTED`.

**Say:**
> "When risk is medium, the system doesn't just ask again — it issues a
> live challenge, generated at that moment, so a recording can't have
> anticipated it. Passing it is a hard gate: no trust score, however high,
> overrides a challenge that wasn't actually completed."

---

## Step 4 — Voice challenge-response (~20s)

Only needed if the demo naturally reaches `HIGH` risk tier (spoof-veto
case), or trigger it deliberately via Attack Lab first, then pivot to a
genuine voice check on the **Voice ID** page or let it run automatically
as part of a HIGH-tier verification.

**Show:** A phrase appears on screen (e.g. *"copper falcon nine"*) →
presenter reads it → `Speaker: ✓` `Phrase: ✓` → **Voice confirmed** →
access granted with both factors verified.

**Say:**
> "The phrase is generated fresh, immediately before this attempt — never
> a fixed sentence. A recording of this person saying something else
> wouldn't satisfy it. We check *who* is speaking and *what* they said,
> independently, and both have to pass."

---

## If a judge asks "can this detect real deepfakes?"

Have this answer ready verbatim-ish — it's the honest one:

> "We don't claim to detect every deepfake, and we don't claim this is
> production-ready Aadhaar-grade authentication. What we've built is a
> multi-signal, risk-adaptive system that raises the cost of a successful
> attack significantly — an attacker needs to defeat identity matching,
> liveness, texture analysis, frequency analysis, spatial consistency, and
> potentially voice challenge-response simultaneously, not just one
> detector. Our Attack Lab is there specifically so you can see which
> signal catches which attack, live, rather than take our word for it."

## If time is short (30s version)

Just do Step 1 and Step 2. That pair alone demonstrates the core thesis —
face-alone would pass, multi-signal doesn't — in the least time.
