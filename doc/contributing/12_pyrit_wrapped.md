# PyRIT Wrapped

PyRIT Wrapped summarizes either a contributor's public activity or an entire
release in `microsoft/PyRIT`. Both modes produce a compact 6-8-slide story,
complete evidence, and song candidates. They do not include an HTML player,
recordings, or video. This is repository tooling, independent of the PyRIT runtime.

## Generate a recap

Use the repository's uv environment and authenticate GitHub CLI first:

```powershell
gh auth login
uv run python -m build_scripts.pyrit_wrapped summarize --contributor romanlutz --year 2026
```

The contributor can be a GitHub username, `@username`, or
`https://github.com/username`. Display names are not guessed. The year defaults
to the current UTC year, which is labeled **year to date**. Future years are rejected.

The command prints the generated report directory under `results/wrapped`.
Use `--output-dir` to choose a new directory. Existing report directories are
never overwritten. Each report contains:

| File | Contents |
|---|---|
| `snapshot.json` | Minimal source records, collection timestamps, reporting window, and effective taxonomy |
| `stats.json` | Separate-role counts, topic groups, classifications, denominators, and activity references |
| `summary.md` | Readable slide summaries, counts, and methodology |
| `activity.md` | Complete topic-grouped activity lists with GitHub source links |
| `story.json` | Supported slide types, facts, evidence references, and omission reasons |
| `songs.md` | Track/artist candidates and rationale for each emitted slide type |

Eight reusable types cover overview, PR pipeline, reviews/people, issues,
focus areas, activity peaks, LOC/languages, and recap. Review and peak slides
are omitted when there is no activity. Empty and unavailable figures are
described honestly, not turned into achievements.

## Wrap a release

```powershell
uv run python -m build_scripts.pyrit_wrapped summarize --release v1.1.0 --since-release v1.0.1
```

Omit `--since-release` to use the previous published stable release.
The selected tags are resolved to immutable commit SHAs and saved in the
snapshot. Draft releases and unordered or identical boundaries are rejected.
Release mode does not accept `--year`.

Release stories intentionally distinguish two cohorts:

- **Activity:** PRs opened/closed, merges, submitted reviews, issues, and
  comments between the two publication timestamps, across all contributors.
- **Shipped changes:** PRs whose merged commit is newly reachable in
  `base..head`, plus the complete net tree diff between the tags. PR membership
  is discovered through commit-to-PR associations and checked against the
  PR's recorded merge commit. Date-only merge counts are not shipping evidence.

Direct commits appear in the tree diff even if no PR is associated.
Contributor role counts describe identifiable PR/review/issue accounts,
not a complete census of every commit author. People and bots are shown
separately; unknown/deleted identities are not invented.

If the tags have divergent histories, the report warns that newly reachable
merge commits do not prove each patch shipped for the first time.
The tree diff remains an exact comparison of the selected snapshots.
The local Git reader may fetch missing commit/blob objects from origin;
it never checks out a tag, changes the current branch, or rebases.

## What the numbers mean

- **Authored/opened PRs:** created in the reporting window by the contributor.
- **Own closed PRs:** authored by the contributor and currently closed, with
  their last recorded closure in the window. This includes merged PRs.
- **Authored PRs that landed:** merged in the window, even if opened earlier.
- **PRs merged:** GitHub records this account in the PR's `merged_by` field.
  Own, other-author, and unavailable-author work are separated. A merge queue's
  recorded actor does not establish who clicked Merge.
- **Issues opened:** created in the window; pull requests are excluded.
- **Issues closed:** same ownership rule, using the last recorded closure.
- **PRs reviewed:** distinct PRs with a submitted review in the window.
  Repeated reviews count separately as submitted reviews, not additional PRs.
- **Comments:** inline review comments (including replies), nonempty submitted
  review bodies, PR discussion comments, and issue discussion comments are
  separate counts. Pending drafts are excluded.

GitHub author search can also return agent-associated PRs whose recorded author
is Copilot. Searches discover candidates; final authorship uses the recorded
author's stable account ID. Agent-authored candidates are excluded from the
human-authored total and reported as a collection diagnostic, not silently
credited to a person. They can still count under an independently verified
merge, review, or comment role.

Dates use UTC and a half-open interval: start of the year inclusive, next year
or the year-to-date cutoff exclusive. Comments use their recorded creation
timestamp, not the last edit timestamp. Reviews use submission timestamps,
not the PR's creation or merge date.

Roles overlap and should not be summed into a productivity score. Monthly
distinct actions deduplicate a merge appearing under both author and merger
credit or the same merge's closure and do not count a review body again as
an independent action. Release membership is not an extra activity event.
The `reviewed_prs` monthly series marks each PR's first review in the window.
The busiest month, ISO week, and calendar day use UTC and preserve all ties.
An ISO week can belong to a different year than its calendar dates.
Closures are the latest recorded timestamps, not every historical close/reopen
transition; the tool does not claim to reconstruct lifecycle timelines.
Closure discovery uses the fully paginated repository issue/PR endpoint's
updated-since window and then checks each record's actual closure timestamp.
It does not rely only on closed-date search, which can omit records that REST
and GraphQL both confirm are closed in the window.

## Lines changed

Contributor LOC is the sum of the contributor's PR diffs **merged in the
window**, not every opened/unmerged PR and not a net yearly repository diff.
The same lines can therefore be changed in multiple PRs.
Release LOC is the **net tag-to-tag tree diff**, without adding overlapping
PR diffs on top. Both are labeled and must not be used as productivity scores.

TypeScript includes `.ts` and `.tsx`; Python is `.py`; YAML includes `.yml`
and `.yaml`. Other text languages remain in the totals and full breakdown.
Notebook JSON is not counted as Python, and lockfiles have a separate bucket.
For renames, additions use the destination language and deletions use the
original language. Unchanged renamed lines do not become churn.
Git binary records have no text LOC; PR API metadata does not reliably identify
binary files, so that classification remains unavailable in contributor mode.
Incomplete file coverage produces unavailable totals, never a plausible
partial total or a silent zero.

## Topics and interpretation

Changed paths identify PR components, surfaces, and artifacts. Tests keep their
component topic, and documentation notebooks are documentation, not product
code. Python GUI-backend work can be both GUI-surface and Python-language work.
Dataset provider code is distinct from dataset content. Generated/lock files
do not outweigh substantive paths.

Issues use explicit labels and conservative title inference. Incomplete PR-file
coverage also produces explicitly inferred metadata classifications. Unknown
and Mixed remain visible. Inline-comment topics use the comment's path when
present; reviewed-PR topics describe the PR's scope, not all lines inspected.

Topic groups overlap; one PR may appear under several topics. Primary
distributions assign one category per item, so their counts reconcile to the
cohort denominator. A primary path category requires a strict majority.
"Primarily" narration additionally requires at least five opened PRs and a
confirmed strict majority. Small samples and ties remain descriptive.

Change intent recognizes prefixes such as `FIX:`, `FIX`, `FEAT:`,
`feat(scope):`, and `MAINT`, with explicit labels as a fallback. Artifact,
component, and intent are independent. A FIX touching documentation is not
automatically Python product work.

For custom rules, pass `--taxonomy path-to-taxonomy.json`. The versioned format
is defined in `build_scripts/pyrit_wrapped/taxonomy.json`; the effective rules
are saved in the snapshot for reproducible offline replay.

## Caching, replay, and completeness

Each collection has its own request cache under `.cache/pyrit_wrapped` and a
fixed reporting cutoff. Retrying an interrupted run resumes the same cutoff
and successful requests, rather than mixing older cached responses into a
newer event window. A completed collection is replayed for up to 24 hours,
with an explicit message identifying its original cutoff. Use `--refresh`
for a new live collection. Changed taxonomy rules also start a fresh collection.
No GitHub credentials, comment bodies, or diff hunks are stored. Reports show
collection completion and the oldest response used; they are not an atomic
GitHub snapshot.

Replay a complete saved snapshot without GitHub or network access:

```powershell
uv run python -m build_scripts.pyrit_wrapped summarize --snapshot .\results\wrapped\your-report\snapshot.json
```

Replay retains the original year, source timestamps, and taxonomy. Changing
the taxonomy requires a new collection, not an implicit change to a replay.

Version-1 snapshots remain replayable. Closed-activity and LOC capabilities
that were not collected are explicitly unavailable, rather than inferred from
the old subset of records. Live collection starts a version-2 cache session.

The collector follows pagination and splits searches around GitHub's
1,000-result search limit. It uses actual review/comment timestamps and
fetches older reviewed PRs rather than limiting reviews to PRs opened that year.
Release reviews use batched, independently paginated read-only GraphQL
connections, including review-only activity on old PRs. Full database review
IDs avoid GraphQL's legacy 32-bit ID limitation.
Incomplete search results, authentication failures, missing required records,
and exhausted retries fail explicitly; successful requests remain cached.

GitHub's PR-files endpoint exposes at most 3,000 files. Missing file coverage
creates a visible warning and metadata inference rather than a false complete
path distribution. Deleted or inaccessible public activity cannot be reconstructed.
Current titles, labels, item states, and paths are snapshot metadata, not
historical year-end metadata.

GitHub can expose comments whose parent PR or review now returns 404.
Comment-only parents retain explicitly unavailable metadata, not invented
authors, titles, or timestamps. Inline comments without accessible reviews
are included only after their publication is independently confirmed through
an unauthenticated public comment endpoint. Those proofs are cached without
comment bodies and linked from `activity.md`. An unavailable review is not
inferred from its comments or added to the submitted-review count. Such cases
produce collection warnings. A missing record needed for authored, merged,
or reviewed activity still fails explicitly.

## Review checkpoint and later work

Review the counts, topic groups, narrative, and omitted slide types before
implementing the local browser form or self-contained HTML story.
`songs.md` proposes two candidates per emitted slide, such as "Changes" for
LOC and "With a Little Help from My Friends" for reviews/people.
These are references to audition, not final selections or license claims.
Roughly 10-second cues are the intended format, but short duration does not
itself grant usage rights. Song selection, permissions, and how recordings will
be supplied remain user decisions. No recordings or music service are included.
