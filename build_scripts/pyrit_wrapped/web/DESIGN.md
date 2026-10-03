# Wrapped viewer design

The viewer inherits PyRIT's Fluent-style light interface rather than changing
the red-teaming GUI. It is an operating/review surface: evidence, navigation,
and playback state take priority over decorative effects.

System/Segoe typography, blue `#0f6cbd`, darker blue `#0c4b83`, white panels,
and `#f5f7fa` surroundings define the rendered interface. Added lines use blue;
removed lines use `#944314`. Values and labels remain readable without color.

The report identity and eight chapter buttons lead into one visible chapter.
Metrics use semantic definition lists; topic and language charts retain their
numeric labels. Full summaries and provenance are expandable, with separate
complete evidence files. Narrow layouts preserve navigation and use two
columns for the larger LOC values.

One YouTube player sits below the chapter and stays at least 200 by 200 pixels.
It is never obscured or used as a hidden background-audio source. Playback
requires consent, visibility, and a foreground document. Errors and browser
autoplay limits have visible recovery messages, not silent fallbacks.
