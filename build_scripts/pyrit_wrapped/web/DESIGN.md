---
name: PyRIT Wrapped viewer
description: A sunset-gold pirate-raccoon celebration of real contributions.
colors:
  sea: "#123841"
  deep: "#0c252d"
  gold: "#ffb938"
  paper: "#fff3ce"
  muted: "#b6d4d5"
  ink: "#123039"
  teal: "#196b70"
  red: "#9b2926"
  control-stroke: "#718f92"
  disabled-text: "#8db0b4"
  divider: "#456b71"
  chart-track: "#e89b28"
  dashed-ring: "#99651f"
  confetti-red: "#f03c3a"
typography:
  body:
    fontFamily: '"Trebuchet MS", "Segoe UI", Arial, sans-serif'
  display:
    fontSize: "clamp(2.4rem, 4.5vw, 4.5rem)"
    fontWeight: 900
    lineHeight: 1.04
    letterSpacing: "-.035em"
  metric:
    fontSize: "clamp(2.2rem, 4.2vw, 4.4rem)"
    fontWeight: 900
    lineHeight: 1
    letterSpacing: "-.04em"
  track-title:
    fontSize: "clamp(1.4rem, 2vw, 2rem)"
    fontWeight: 900
  chart-title:
    fontSize: "1rem"
    fontWeight: 800
  metric-label:
    fontSize: ".9rem"
    fontWeight: 700
    lineHeight: 1.3
  footnote:
    fontSize: ".75rem"
    lineHeight: 1.5
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
  button-primary:
    backgroundColor: "{colors.gold}"
    textColor: "{colors.deep}"
    rounded: "{rounded.pill}"
    padding: ".55rem 1rem"
  button-ghost:
    backgroundColor: "transparent"
    textColor: "{colors.paper}"
    rounded: "{rounded.pill}"
    padding: ".55rem 1rem"
  button-hover:
    backgroundColor: "{colors.paper}"
    textColor: "{colors.deep}"
  button-disabled:
    backgroundColor: "transparent"
    textColor: "{colors.disabled-text}"
  chapter-current:
    backgroundColor: "{colors.gold}"
    textColor: "{colors.deep}"
    rounded: "{rounded.pill}"
    padding: "0"
    width: "44px"
    height: "44px"
  stage:
    backgroundColor: "{colors.gold}"
    textColor: "{colors.ink}"
    rounded: "{rounded.stage}"
    padding: "2.3rem 2.8rem 1.8rem"
  duration-input:
    backgroundColor: "{colors.deep}"
    textColor: "{colors.paper}"
    rounded: "{rounded.field}"
    padding: ".3rem"
    width: "4.5rem"
---

# Design System: PyRIT Wrapped viewer

## Overview

**Creative North Star: "The Pirate-Raccoon Party"**

This records the built standalone viewer in this directory, not a new brand
system for PyRIT's frontend or core. The opening five-part contract in
`deck.html` is authoritative: a party for real contributions, a sunset-gold
stage with sea-dark ink and chunky Trebuchet, eight celebratory chapters,
manually controlled music during a take, and a user-pinned pirate/raccoon
world with bounded motion. The formal Fluent report is superseded.

The first viewport gives the headline and facts the left side, canonical
Roakey and his parrot the right, confetti overhead, and the song cue below.
Use the existing `doc\roakey.png` artwork; do not redraw or substitute it.
The prior finish review found no material desktop or mobile fixes. This is
a source-grounded record of that verified build, not another redesign pass.

**Key Characteristics:**
- A bright, flat stage against dark sea surroundings.
- Oversized truthful numbers, readable charts, and subordinate evidence.
- A familiar mascot with restrained sway and falling confetti.
- Explicit manual music cues and recording state, not embedded playback.

## Colors

Warm celebration sits inside a cool sea frame; the frontmatter records the
exact CSS colors, preserving the eight custom-property names.

### Primary
- **Sunset Gold (`gold`):** stage, current chapter, Next, start-take action,
  timer, and Spotify link.

### Secondary
- **Sea (`sea`):** page background and chart fills for counts and additions.
- **Deep (`deep`):** primary-action text, number field, and skip-link surface.
- **Teal (`teal`):** narrow confetti pieces.

### Tertiary
- **Removal Red (`red`):** deleted-line bars and the removal key.
- **Confetti Red (`confetti-red`):** circular decorative pieces only.
- **Chart Track (`chart-track`):** underlying bar lanes.
- **Dashed Ring (`dashed-ring`):** clipped nautical circle on the stage.

### Neutral
- **Paper (`paper`):** outer text, control hover fill, and pale confetti.
- **Ink (`ink`):** stage text and the diverging chart's center divider.
- **Muted (`muted`):** identity, hints, artist, status, and evidence text.
- **Control Stroke (`control-stroke`):** button and number-field outlines.
- **Disabled Text (`disabled-text`):** unavailable actions.
- **Divider (`divider`):** disabled outlines and supporting-section rules.

## Typography

**Display and Body Font:** Trebuchet MS, with Segoe UI, Arial, and sans-serif
fallbacks. No downloaded font or separate mono family is required.

**Character:** chunky, friendly, and emphatic. Headlines and facts use the
heaviest weight; context remains smaller and comfortably spaced.

### Hierarchy
- **Display:** the frontmatter's balanced chapter headline, limited to 16ch.
- **Metric:** the frontmatter's oversized facts, with tabular numerals,
  wrapping for long values, and labels underneath.
- **Track title:** a heavy, balanced secondary headline below the stage.
- **Chart title / metric label:** compact headings and explanatory labels.
- **Body:** descriptions use line-height (1.45) and a maximum width (65ch);
  expanded summaries use line-height (1.6).
- **Footnote:** supporting context, limited to 75ch. Metric notes are smaller
  (0.8rem), normal-weight, and use normal letter spacing.

Timers, chapter position, and count-chart values also use tabular numerals.
Responsive type changes are recorded in Layout rather than a second scale.

## Layout

The centered header and deck have a maximum width (1440px), desktop side
gutters from `shell-gutter`, and no fixed viewport-height lock. The stage is
a two-column grid (`minmax(0, 1fr) 30%`), with a gap (1.5rem) and minimum
height (500px). A take raises that minimum to 540px.

Metrics have three equal columns, with explicit one- and two-metric variants.
Count and language lists retain two columns. The lower song-cue / recording
grid uses fractional columns (0.8fr / 1.2fr) and `soundtrack-gap`.

- **At widths up to 1000px:** stage padding becomes 1.8rem, the mascot column
  becomes 25%, and the gap becomes 1rem. Metrics use
  `clamp(2.1rem, 4vw, 3.5rem)`; the soundtrack gap becomes 1.5rem.
- **At widths up to 640px:** header wraps; shell gutters become 1rem. The
  stage becomes block layout with padding (1.4rem 1.1rem 1rem) and minimum
  height (440px), including during a take. The mascot is a cropped corner
  accent (122px wide, top 1rem, right -2.2rem), without its caption.
  Headings reserve 55px on the right, use 2.4rem type, and a 13ch limit.
  Metrics use 2.25rem, or 2.1rem in two-metric layouts; labels use 0.72rem.
  Chart labels use 0.7rem and wrap, without removing numeric values.
  Chapters wrap; the soundtrack stacks into one column. The keyboard hint
  disappears, not the keyboard affordances.

## Elevation & Depth

There are no shadows, gradients, glass surfaces, or lifted cards. Depth comes
from sea/gold contrast and explicit stacking inside an isolated, clipped
stage: confetti behind the readable slide content, a dashed circle behind
the composition, and a solid gold countdown overlay above it.

## Shapes

Use the frontmatter's pill controls, gently squared stage, compact field
corners, and nearly square chart lanes. Thin strokes define controls and
dividers. The stage's cropped dashed circle is the nautical geometry, not
a decorative frame around every statistic.

## Components

### Buttons

Confident, compact pills. Default buttons have a minimum height (44px),
weight (700), and a thin control stroke (1px). Primary actions are Next and
Start recording mode. Hover changes enabled controls to Paper with Deep
text and a Paper border; disabled controls retain the subdued text/stroke
and a default cursor. Focus is a current-color outline (3px) offset by 4px,
not a glow. On mobile, header buttons have a minimum height (36px).

### Inputs / Fields

The duration number field uses the frontmatter's deep surface and compact
shape, a control stroke (1px), and minimum height (36px). Its initial value
is 10 seconds, with bounds (5–120). Auto-advance is initially unchecked;
its checkbox is 18px square with a gold accent. Labels remain visible.

### Navigation

Eight numbered chapter buttons expose the current step and a live position.
Desktop buttons are 44px wide with zero padding; mobile buttons are 32px
wide with a minimum height (40px). Previous begins disabled. Arrow-key
navigation, the skip link, focusable chapter headings, and fullscreen remain
part of the interface.

### Stage, Metrics, and Charts

Only one chapter is visible. Semantic definition lists put oversized facts
above their labels. Count bars retain labels and comma-formatted values;
language rows retain explicit negative deletions and positive additions
alongside a Removed / Added key. Lanes are 11px high; decorative bar geometry
is hidden from assistive technology. Expanded summaries, provenance, UTC
boundaries, limitations, and evidence/transcript links remain available.

**The Truth Before Spectacle Rule.** Keep all eight chapters tied to supplied
facts. Never hide unknowns, inflate counts, or turn LOC into productivity.
Color and motion must not be the only ways to interpret a chart.

### Mascot and Motion

Roakey's image keeps its square source proportions and descriptive alt text.
His desktop caption reads “Small paws. Big adventures.” The sway is a
four-second ease-in-out loop, rotating between -2deg and 2deg and rising at
most 6px. Confetti falls through a clipped stage; opacity is 0.55 for overview
and recap and 0.2 elsewhere. Piece durations, delays, positions, and spin
come from per-piece properties, not a fixed global timing token.

Count bars reveal once over 0.65s with `cubic-bezier(.16,1,.3,1)`, scaling
from 0.05 to 1; language bars do not share that animation. Pause animation
hides confetti, removes count-bar animation, and pauses animated elements.
The hidden-page and offscreen states also pause animations. Reduced motion
disables all animations, keeps scrolling automatic, and hides confetti.
Forced colors hides confetti, uses Highlight for chart fills, and reinforces
the current chapter and Next borders.

### Manual Soundtrack and Recording State

The song title, artist, manual Spotify search link, and expandable ordered
cue list are supporting UI, not an audio player. No selected track is an
explicit cue-list state. The page works offline; opening Spotify is optional.

Ready is the initial timer state. Starting a take presents an “All aboard
in” countdown and reminds the user to start their own recorder. Pause/resume,
Finish take, live status, completed-take timestamps, optional slide timing,
and Download cue sheet support that external workflow. Pause and Finish
begin disabled; status reserves space to prevent jumping.

During a take, full summaries, provenance, the complete cue list, and timing
settings are hidden to simplify the recording view. On mobile, the song
section gains a lower divider. The persistent note explains that the page
times slides, not music; Spotify and screen capture stay under user control.

## Do's and Don'ts

### Do:
- **Do** preserve the sunset-gold stage, sea-dark frame, and chunky Trebuchet.
- **Do** reuse canonical Roakey and his parrot, with bounded decoration.
- **Do** retain readable numeric labels, truthful context, and a transcript.
- **Do** preserve visible focus, animation pause, and reduced-motion behavior.
- **Do** keep song cues and take state explicit about manual music and capture.

### Don't:
- **Don't** restore the discarded Fluent report or embedded YouTube player.
- **Don't** imply Spotify integration, automatic audio, or in-page recording.
- **Don't** animate essential numbers away or rely on color alone.
- **Don't** spread this scoped viewer design into PyRIT's frontend or core.
