# Attack-result analytics

`AttackResultAnalytics` reads saved results through the configured memory backend.
The SDK owns cohort selection, outcome statistics, and drill-down predicates.
Memory executes aggregate/projection queries; the REST service and GUI use the
same shared contracts.

## Implementation map

| Layer | Entry point | Responsibility |
|---|---|---|
| Shared contracts | `pyrit.models.analytics` | Describe the cohort, typed missing values, bounded output, and cursor/freshness semantics. |
| SDK | `AttackResultAnalytics` | Snapshot requests, apply outcome-rate policy, label groups, and produce additional drill-down predicates. |
| Query execution | `AnalyticsExecution` | Bound report and quick-query workers and keep a running query's slot occupied through cancellation cleanup. |
| Memory | `AttackAnalyticsReader` / `AttackAnalyticsQueryCompiler` | Read a consistent database view and return counts or lightweight metadata projections, without loading scores or messages. |
| Small-profile grouping | `ProfileAggregation` | Finish SQLite's bounded pre-counted metadata groups using the same membership and absence rules as the SQL fallback. |
| REST | `get_analytics_service` dependency | Lazily provide the SDK object directly and release its workers at backend shutdown. |

A report selects its cohort first, then reads overall outcome counts, its chosen
chart, and the first result page. Group expansion must not multiply the overall
total. A result page and a facet lookup are separate operations so pagination and
opening a filter do not regenerate the report.

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
constraint or interpreting display labels. Check `drilldown_unavailable_reason`
first: a full-budget report remains valid, but adding another group/cell constraint
may exceed the 16-predicate or 500-value limit.

```python
from pyrit.models import AttackAnalyticsResultsQuery

if report.drilldown_unavailable_reason:
    print(report.drilldown_unavailable_reason)
elif report.cells:
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
The GUI disables chart drill-downs with the SDK-provided explanation at these
limits. It keeps the current statistics, outcome controls, and result paging
available; it never drops existing predicates to make room.

`facets_async` accepts `AttackAnalyticsFacetQuery` for one dimension and a bounded
search/page. Its own dimension predicates are excluded when discovering
alternatives; the other filters still apply. Do not fetch all facets on every
interaction.

Synchronous `query`, `results`, and `facets` variants use the same contracts.
`shutdown()` stops analytics workers shared by every analytics instance using that
memory object. Call it outside an event loop after all those callers are finished.
It waits for running work to exit before allowing a later query to create fresh
workers; it does not permanently disable the SDK object.

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
Joining an existing request inherits its original queue/execution deadline; it
does not reset the clock for the new caller.

For Python callers, `access_scope` only partitions in-flight reuse; it does not
authorize access or filter database rows. Applications must enforce their own
access policy before querying. REST supplies the authenticated caller's partition.
Each SDK response is copied so one caller cannot mutate another caller's result.

For small categorical group sets, the database returns pre-counted metadata
profiles and the SDK finishes grouping them. SQLite probes at most 4,097 profiles
and accepts at most 4,096, subject to 4,096-character source limits and a
1,000,000-character combined-text limit. Repeated metadata is decoded once per
request. Overflow triggers complete SQL aggregation, never sampling or partial
counts.

The combined-text limit is checked after fetching the bounded probe; it is not a
peak-memory or network-byte guarantee. These limits bound transferred profiles and
SDK processing, not database scans or intermediate work. SQL may still scan, join,
group, sort, or expand the full filtered cohort.

Queue saturation produces HTTP 503; a query deadline produces HTTP 504. Cancelling
an awaiting Python SDK task detaches that caller; the last caller leaving signals
cooperative cancellation. One cancelled caller does not cancel other callers
sharing the same in-flight query.

A browser abort stops the browser's wait but does not necessarily cancel the
server's awaiting Python task. These REST routes rely on their bounded queue and
execution deadlines to clean up abandoned HTTP work. In either case, a worker
slot remains occupied until its database operation actually finishes.
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

SQL Server's native padded-string comparisons can combine values that differ only
by trailing spaces; SQLite can keep those values distinct. Metadata extraction
preserves valid values up to the 4,096-character contract limit rather than relying
on `JSON_VALUE`'s 4,000-character result limit. Exact cross-backend key parity still
requires a separate, consistently length-aware comparison policy.

Legacy converter metadata must use string class names, or null/missing values.
Malformed non-string names can currently be rejected by the profile/result reader
but coerced by SQL grouping. Inventory and repair malformed stored identifiers
before relying on equivalent behavior across these query paths; analytics does
not silently rewrite stored metadata.

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
labels, and concurrent inserts/outcome updates. First-page and cursor-following
continuation latencies are reported separately. Output records the environment,
data shape, cold and p95 latency, and write behavior. It exits unsuccessfully if
the configured workload misses the report/facet/page latency targets.

Display limits and correct distinct counts do not by themselves prove scalability.
Measure the intended metadata cardinality and concurrency on the deployment's
reference environment rather than assuming that a row-count target guarantees
latency.
