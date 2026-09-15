# Attack-result analytics

`AttackResultAnalytics` reads saved results through the configured memory backend.
The SDK owns cohort selection, outcome statistics, and drill-down predicates.
Memory executes aggregate/projection queries; the REST service and GUI use the
same shared contracts.

SQLite uses narrow covering indexes and a computed identifier lookup that can
reuse canonical identifiers referenced by older JSON metadata. No result IDs or
stored outcomes are changed by that lookup. The schema migration bounds the
outcome index key and adds the query indexes; it does not remove historical
duplicate rows.

## Counting and interpreting outcomes

Each distinct `attack_result_id` contributes once. Several conversations within a
result do not multiply its count. Different result IDs sharing a main conversation
remain separate. An update to an existing result ID changes its current outcome,
not the number of results.

Success rate uses `successes / (successes + failures)`. Errors and undetermined
outcomes are reported separately. A zero denominator produces `None`. The existing
`analyze_results` and `AttackStats` APIs retain their behavior.

Targeted categories and converter types are multi-valued. Their groups may
overlap, but each result contributes at most once to a given group or heatmap
cell. Metadata type names use the same case-insensitive grouping and matching
within each backend. Missing values use typed keys, rather than a string sentinel
that could collide with a real label.

An outcome restriction applies to all returned statistics and result rows.
`outcome_filter_applied` tells a renderer to mark ASR with an asterisk and explain
the restricted population. It does not change the formula. For example, a
success-only selection has a 100% rate when results exist, but is not an
unfiltered success-rate comparison.

## Querying from Python

After initializing PyRIT, use the configured memory or pass a memory instance
explicitly. Async variants do not run blocking database work on the event loop.

```python
from pyrit.analytics import AttackResultAnalytics
from pyrit.models import (
    AttackAnalyticsDimension,
    AttackAnalyticsFilter,
    AttackAnalyticsFilters,
    AttackAnalyticsQuery,
    AttackAnalyticsValue,
)

analytics = AttackResultAnalytics()
report = await analytics.query_async(
    query=AttackAnalyticsQuery(
        filters=AttackAnalyticsFilters(
            dimensions=[
                AttackAnalyticsFilter(
                    dimension=AttackAnalyticsDimension(name="operation"),
                    values=[AttackAnalyticsValue(value="operation-a")],
                )
            ]
        ),
        group_by=AttackAnalyticsDimension(name="targeted_harm_category"),
        compare_by=AttackAnalyticsDimension(name="attack_type"),
    )
)
print(report.summary.total_results, report.summary.success_rate)
```

Omit `compare_by` for a one-dimensional breakdown. Supported dimensions include
operation, operator, targeted harm category, attack type, converter type, objective
target, model, scenario, and a custom label. Label dimensions require `label_key`.
Converter dimensions can select `converter_direction="request"` or `"response"`.

Values within a filter use ANY matching by default. Converter filters also support
`match_mode="all"`. Separate predicates are always AND-combined, even when they
refer to the same multi-valued dimension.

### Drilling down without recalculating a report

Append the returned predicates, rather than replacing an existing converter
constraint or interpreting display labels:

```python
from pyrit.models import AttackAnalyticsResultsQuery

if report.cells:
    cell = report.cells[0]
    filters = report.filters.model_copy(update={"dimensions": [*report.filters.dimensions, *cell.drilldown_filters]})
    page = await analytics.results_async(query=AttackAnalyticsResultsQuery(filters=filters))
    if page.has_more:
        next_page = await analytics.results_async(
            query=AttackAnalyticsResultsQuery(filters=filters, cursor=page.next_cursor)
        )
```

Result pages contain metadata and an objective preview, not messages, scores, or
media. Their cursors are bound to the cohort and result-selection mode. Invalid or
stale cursors produce an explicit error.

`facets_async` accepts `AttackAnalyticsFacetQuery` for one dimension and a bounded
search/page. Its own dimension predicates are excluded when discovering
alternatives; the other filters still apply. Do not fetch all facets on every
interaction.

Synchronous `query`, `results`, and `facets` variants use the same contracts.
`shutdown()` stops analytics workers shared by that memory backend and should be
called outside an event loop when those workers are no longer needed.

## REST surface

| Endpoint | Request | Response |
|---|---|---|
| `POST /api/analytics/attacks/query` | `AttackAnalyticsQuery` | `AttackAnalyticsReport`, including the first result page |
| `POST /api/analytics/attacks/results` | `AttackAnalyticsResultsQuery` | `AttackAnalyticsResults`, without report aggregation |
| `POST /api/analytics/attacks/facets` | `AttackAnalyticsFacetQuery` | `AttackAnalyticsFacets` |

Reports include raw counts, rates, outcome shares, overlap/ASR annotations, typed
drill-down predicates, and freshness. A report and its first page use one short
database read view. Later pages are fresh reads, not a durable historical snapshot.
The last-updated filter must not be interpreted as an execution-time trend.

Requests are validated and bounded. Defaults show 15 groups, at most 20 values
per heatmap axis, 25 results per page, and 50 options per facet page.
The full matching cohort contributes to the statistics.

## Resource and database behavior

Report and quick-query workers have separate bounded FIFO lanes. Identical
in-flight requests can be shared only within the same operation, query, backend,
and access scope. Completed responses are not cached.

For small categorical group sets, the database returns pre-counted metadata
profiles and the SDK finishes grouping them. This path is bounded by profile
count and text size, and decodes repeated metadata once per request. Larger
profiles fall back to SQL aggregation over the full cohort. Neither path samples
results or reads their conversations.

Queue saturation produces HTTP 503; a query deadline produces HTTP 504. A cancelled
client does not free a worker while its database operation is still running.
SQLite uses a request-local progress handler on its owned connection; the handler
is removed before the connection is returned. Analytics never changes an existing
database's journaling mode.

For concurrent SQLite workloads, consider explicitly enabling WAL during database
setup on a suitable local filesystem. Rollback journaling can delay readers behind
writers, so reports expose a warning when applicable. Account for SQLite's WAL
filesystem and backup requirements when choosing this configuration.

Azure SQL has a separate SQL dialect path. Reports request SNAPSHOT isolation,
which the database must permit; analytics does not alter deployment-wide isolation
settings. Offline query coverage is not a substitute for live Azure SQL correctness
and performance verification, which is tracked as a follow-up.

## Reproducing the performance workload

From the repository root:

```powershell
uv run python build_scripts\benchmark_attack_analytics.py --journal-mode wal --output .pytest_cache\analytics-benchmark.json
```

The script creates and removes its own synthetic database in the ignored cache
directory. It never reads or modifies an existing database. It exercises the real
HTTP/ASGI routes, validation, SDK, database queries, and serialization, but excludes
network and external authentication I/O.

The default workload uses 100,000 results, five independently scoped analysts,
overlapping category/converter metadata, legacy JSON fallbacks, high-cardinality
labels, and concurrent inserts/outcome updates. Output records the environment,
data shape, cold and p95 latency, and write behavior. It exits unsuccessfully if
the configured workload misses the report/facet/page latency targets.

Display limits and correct distinct counts do not by themselves prove scalability.
Measure the intended metadata cardinality and concurrency on the deployment's
reference environment rather than assuming that a row-count target guarantees
latency.
