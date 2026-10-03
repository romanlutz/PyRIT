# PyRIT Wrapped

PyRIT Wrapped summarizes a contributor's public activity in `microsoft/PyRIT`.
This first milestone produces facts and slide summaries, not an HTML player,
soundtrack, or video. It is repository tooling, independent of the PyRIT runtime.

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

The catalog has 25 types, but each contributor receives only supported stories.
Sparse activity is not turned into invented achievements.

## What the numbers mean

- **Authored/opened PRs:** created in the reporting window by the contributor.
- **Authored PRs that landed:** merged in the window, even if opened earlier.
- **PRs merged:** GitHub records this account in the PR's `merged_by` field.
  Own, other-author, and unavailable-author work are separated. A merge queue's
  recorded actor does not establish who clicked Merge.
- **Issues opened:** created in the window; pull requests are excluded.
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
credit and do not count a review body again as an independent action.
The `reviewed_prs` monthly series marks each PR's first review in the window.

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

The collector follows pagination and splits searches around GitHub's
1,000-result search limit. It uses actual review/comment timestamps and
fetches older reviewed PRs rather than limiting reviews to PRs opened that year.
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
The next music workshop can suggest candidates for each supported slide type.
Song selection and how recordings will be supplied remain user decisions.
No recordings or external music service are included in this milestone.
