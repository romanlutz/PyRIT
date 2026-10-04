---
name: PyRIT Wrapped viewer
description: A chapter-colored pirate-raccoon celebration of real contributions.
colors:
  sea: "#123841"
  deep: "#0c252d"
  gold: "#ffb938"
  paper: "#fff3ce"
  muted: "#b6d4d5"
  ink: "#123039"
  teal: "#196b70"
  red: "#9b2926"
  chapter-cover: "#ffd166"
  chapter-overview: "#56d9d7"
  chapter-contributors: "#c4abff"
  chapter-prs: "#ff9788"
  chapter-reviews-people: "#91e4b9"
  chapter-issues: "#ffb8dc"
  chapter-topics: "#d8ed71"
  chapter-busiest: "#9ecbff"
  chapter-loc: "#ffd18a"
  chapter-recap: "#f4e684"
  control-stroke: "#718f92"
  disabled-text: "#8db0b4"
  divider: "#456b71"
  chart-track: "#12303920"
  dashed-ring: "#99651f"
  confetti-red: "#f03c3a"
typography:
  body: { fontFamily: '"Trebuchet MS", "Segoe UI", Arial, sans-serif' }
  display: { fontSize: "clamp(2.4rem, 4.5vw, 4.5rem)", fontWeight: 900, lineHeight: 1.04, letterSpacing: "-.035em" }
  cover: { fontSize: "clamp(3.4rem, 6vw, 6rem)", fontWeight: 900, lineHeight: 1.04, letterSpacing: "-.035em" }
  metric: { fontSize: "clamp(2.2rem, 4.2vw, 4.4rem)", fontWeight: 900, lineHeight: 1, letterSpacing: "-.04em" }
  track-title: { fontSize: "clamp(1.4rem, 2vw, 2rem)", fontWeight: 900 }
  chart-title: { fontSize: "1rem", fontWeight: 800 }
  metric-label: { fontSize: ".9rem", fontWeight: 700, lineHeight: 1.3 }
  footnote: { fontSize: ".75rem", lineHeight: 1.5 }
rounded:
  pill: "999px"
  stage: "6px"
  field: "4px"
  bar: "2px"
spacing:
  small-gap: ".5rem"
  standard-gap: "1rem"
  large-gap: "1.5rem"
  shell-gutter: "2rem"
  soundtrack-gap: "3rem"
components:
  button-primary: { backgroundColor: "{colors.gold}", textColor: "{colors.deep}", rounded: "{rounded.pill}", padding: ".55rem 1rem" }
  button-ghost: { backgroundColor: "transparent", textColor: "{colors.paper}", rounded: "{rounded.pill}", padding: ".55rem 1rem" }
  button-hover: { backgroundColor: "{colors.paper}", textColor: "{colors.deep}" }
  button-disabled: { textColor: "{colors.disabled-text}" }
  chapter-current: { backgroundColor: "{colors.chapter-cover}", textColor: "{colors.deep}", rounded: "{rounded.pill}", padding: "0", width: "44px" }
  stage: { backgroundColor: "{colors.chapter-cover}", textColor: "{colors.ink}", rounded: "{rounded.stage}", padding: "2.3rem 2.8rem 1.8rem" }
  duration-input: { backgroundColor: "{colors.deep}", textColor: "{colors.paper}", rounded: "{rounded.field}", padding: ".3rem", width: "4.5rem" }
---

# Design System: PyRIT Wrapped viewer

## Overview

**Creative North Star: "The Pirate-Raccoon Party"**

This records the built standalone viewer, not a GUI/core brand system. Keep the user-pinned pirate/raccoon world: chunky lettering, sea-dark ink, ten bright chapter fields, canonical Roakey and parrot artwork, and alternating confetti, disco, fireworks, and the canonical landing-page running sprite.

The cover names a contributor, release, or repository-year. The verified release deck has ten chapters; personal decks have nine. This refresh follows current `deck.css` and `deck.html`, not a redesign. The previously corrected fullscreen ring overflow and confirmed credit/focus captures are retained.

**Key Characteristics:**
- Distinct chapter fields against a sea-dark frame.
- Oversized truthful facts, explicit credit, and subordinate evidence.
- Canonical mascot artwork and four alternating, pauseable effects.
- Manual song cues and recording state, not embedded playback.

## Colors

The frontmatter records actual CSS values. Stage, countdown, active chapter, and fullscreen background follow `--chapter-color`; component tokens show its cover default, not a fixed global stage.
- **Primary:** cover gold, overview aqua, contributor lavender, PR coral, review/people mint, issue pink, topic lime, busiest blue, LOC apricot, and recap yellow are ten distinct fields.
- **Secondary:** Sea frames the page and fills count/addition bars; Deep supports controls. Action Gold remains on Next, Start, the clock, checkbox, and Spotify link outside fullscreen.
- **Tertiary:** Removal Red distinguishes deletions; Teal, Paper, and Confetti Red decorate the party. The dashed ring stays subordinate.
- **Neutral:** Ink labels every bright field; Paper and Muted label the dark frame. Control Stroke, Disabled Text, and Divider separate states. Chart Track is translucent ink, not opaque gold.

## Typography

**Display and Body Font:** Trebuchet MS, Segoe UI, Arial, sans-serif; no downloaded font.
- **Display / cover:** balanced heavyweight chapter headings, respectively limited to 16ch / 10ch.
- **Metrics:** tabular oversized facts above labels; long values wrap. Timers and counts also use tabular numerals.
- **Context:** descriptions use line-height (1.45) and width (65ch); summaries use (1.6), footnotes (75ch), and metric notes normal weight/spacing.

## Layout

The header/deck center within (1440px), with (2rem) side gutters. The stage uses `minmax(0, 1fr) 30%`, gap (1.5rem), and minimum height (540px), also during a take. Metrics have one/two/three-column variants; count and LOC lists use two columns; song/recording columns are (0.8fr / 1.2fr).
- **Credit:** contributors get a single-column stage without the mascot; roster groups use (1fr 2fr 1fr), with other-human names in two columns.
- **At ≤1000px:** stage padding (1.8rem), mascot column (25%), gap (1rem); metrics use `clamp(2.1rem, 4vw, 3.5rem)`.
- **At ≤640px:** wrapping header/navigation, (1rem) gutters, block stage with padding (1.4rem 1.1rem 1rem) and minimum height (440px). Mascot becomes a (122px) corner accent at right (-2.2rem), without caption; credit/song grids stack.
- **Mobile type/charts:** chapter headings (2.4rem, 13ch), cover (3rem), metrics (2.25rem; two-column 2.1rem), wrapping chart labels (0.7rem), timeline height (140px instead of 160px).
- **Runner:** reserve (135px) stage bottom padding for the sprite, including fullscreen.
- **Fullscreen:** hide header, chapter navigation, soundtrack, inventory, summaries, time details, and provenance. Remove shell gutters; stage height is `calc(100vh - 64px)` with `overflow-y: auto`, padding `clamp(2rem, 4vw, 5rem)`, and no radius. Keep a (64px) control strip; Next becomes Sea/Paper. The dashed pseudo-element is anchored at right/bottom (0), fixing overflow. Mobile uses auto height with the same viewport-based minimum; do not promise no scrolling on every viewport.

## Elevation & Depth

Flat fields, no shadows or glass. The isolated clipped stage layers the dashed ring behind content, effects at level (0), slide stack/runner at (1), and countdown at (3). Disco beams alone use a translucent conic gradient; it is decoration, not a surface treatment.

## Shapes

Pill controls, gently squared stage, compact number field, nearly square bar lanes. Thin control strokes and a cropped (170px) dashed nautical circle provide geometry without framing every fact.

## Components

### Controls and Navigation
Buttons use minimum height (44px), weight (700), and a (1px) stroke. Enabled hover uses Paper/Deep; disabled text/stroke are subdued. Focus is a current-color outline (3px), offset (4px). Header controls shrink to (36px) on mobile.
Chapter controls are (44px) wide; mobile uses (32px) width / (40px) minimum height. Preserve current-step semantics, live position, arrow keys, skip link, focusable headings, and initially disabled Previous.
The visibly labelled duration field starts at (10), bounded (5–120), with minimum height (36px); the (18px) Gold checkbox starts unchecked.

### Facts, Credit, and Charts
Only the current chapter is visible. Main PR headlines, intent, and author credit share the reporting-window merged-PR cohort; newly opened PRs stay separate. The verified release example is 207 merges, 240 newly opened PRs, and 37 human merged authors, not permanent UI constants.
Separate supplied-roster maintainers, other humans, bots, and unavailable identities. Show merged PRs / submitted reviews explicitly; complete semantic tables retain newly opened PRs, merges, reviews, comments, and issues. Never infer AI assistance from bot counts.
UTC activity charts are daily for windows ≤50 days, monthly otherwise. Ordered bars expose date/count labels and expandable exact values; zero activity remains visible as a date, not an invented bar.
Focus retains both file-count and added/removed LOC charts by topic. Language/topic churn uses explicit negative/positive values, a Removed/Added key, and (11px) diverging lanes; geometric fills are decorative.
**The Truth Before Spectacle Rule.** Keep every chapter tied to supplied facts. Never hide unknowns, inflate counts, or turn LOC into productivity. Color and motion must not be the only ways to interpret a chart.

### Mascot and Motion
Roakey and parrot retain descriptive alt text and the caption “Small paws. Big adventures.” The sway rotates (−2deg to 2deg), rises (6px), and loops over (4s), anchored at (50% 90%).
Confetti uses per-piece position/duration/delay/spin, falling from (−24px) to (680px); CSS layer opacity is (.55). Disco facets rotate over (9s), glint over (4s), and beams sweep (−20deg to 20deg) over (8s).
Three eight-ray fireworks loop over (6s), staggered by (−2s) per burst; rays rotate in (45deg) increments, move (−8px → −38px → −78px), and fade. The (151px × 120px) running sprite crosses over (9s); its four-frame stride uses (.65s, steps(4)) and object-position (0 → −604px).
Count bars reveal with `scaleX(.05 → 1)` over (.65s); timelines use `scaleY(.05 → 1)` over (.7s), both with `cubic-bezier(.16,1,.3,1)`. Diverging bars are static.
Explicit pause hides all four effects, removes count/timeline reveals, and pauses animations; hidden-page/offscreen states also pause them. Reduced motion disables all animations, hides effects, and keeps scrolling automatic. Forced colors hides effects, uses Highlight for count/diverging fills, and strengthens active/Next borders.

### Manual Soundtrack and Recording State
Song/artist cues, manual Spotify links, countdown, optional timing, pause/resume, Finish, status, and cue-sheet export support an external screen recorder, not audio playback or capture. The deck works offline; Spotify remains optional.
During a take, summaries, provenance, cue list, and timing settings hide. Fullscreen separately suppresses recording tools. Keep the note that slides are timed while music and recording remain manual.

## Do's and Don'ts

### Do:
- **Do** preserve ten chapter colors, the sea-dark frame, and chunky Trebuchet.
- **Do** reuse canonical Roakey/parrot and the landing-page running sprite.
- **Do** retain merged-cohort consistency, explicit credit, exact values, and a transcript.
- **Do** preserve visible focus, motion controls, and the fullscreen overflow correction.
- **Do** keep music and recording explicitly manual.

### Don't:
- **Don't** restore the discarded Fluent report or embedded YouTube player.
- **Don't** treat Action Gold as every chapter's background.
- **Don't** conflate newly opened work, merges, bot activity, and AI-assisted work.
- **Don't** spread this scoped viewer design into PyRIT's GUI or core.
