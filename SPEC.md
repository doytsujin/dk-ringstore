# SRRA — Semantic Round-Robin Archive

Version 0.1 (draft). Terminology follows RRDtool where the concept is unchanged,
and departs from it where agentic use needs more than numbers.

## 1. Terms

- **Store** — one file (or one in-memory region) holding the temporal memory of a
  single entity: a dataset, a Kubernetes deployment, a cache partition, a map region,
  a cache partition.
- **Signal** (`DS:`) — one named observable within a store, with a declared type.
  RRD's data source.
- **Archive** (`SRA:`) — one retention tier over one signal: a consolidation
  function, a step count per slot, and a slot count. RRD's RRA.
- **Slot** — one consolidated bucket in an archive. Fixed size, determined by the
  consolidation function.
- **Step** — the base sampling interval of the store. Every archive's slot spans
  an integer number of steps.

`SRRA` names the concept and the store. `SRA:` is the per-archive declaration
line, mirroring RRD's `RRA:`.

## 2. Declaration is the contract

A store's footprint is computed at create time and never changes:

```
footprint = header + Σ_archives (slots × slot_size(cf, params))
```

The header carries the full declaration, so a reader can learn what a store
remembers without reading what it holds. This is the property that makes
retention part of the dataset contract rather than an operational detail.

Example declaration:

```
STEP 1s

DS:threat:GAUGE:heartbeat=5s:min=0:max=1
DS:outcome:OUTCOME:classes=permit,deny,escalate,error
DS:intent:VECTOR:dim=768:quant=int8

SRA:MEAN:threat:1:600          # 1s  × 600   = 10 min
SRA:MEAN:threat:10:2160        # 10s × 2160  = 6 h
SRA:MAX:threat:60:10080        # 1m  × 10080 = 7 d
SRA:MEAN:threat:3600:8760      # 1h  × 8760  = 1 y

SRA:COUNT:outcome:60:1440      # 1m  × 1440  = 24 h
SRA:CENTROID:intent:60:60      # 1m  × 60    = 1 h
SRA:CENTROID:intent:3600:24    # 1h  × 24    = 24 h
SRA:DRIFT:intent:3600:720      # 1h  × 720   = 30 d
```

## 3. Signal types

| Type | Meaning | Sub-slot accumulator |
|---|---|---|
| `GAUGE` | instantaneous value | Welford (n, mean, m2) |
| `COUNTER` | monotonic, rate-differenced on ingest | Welford over the rate |
| `DERIVE` | differenced, may go negative | Welford over the rate |
| `ABSOLUTE` | value since last read, divided by interval | Welford |
| `STATE` | one of K declared classes | K counters |
| `OUTCOME` | one of the four policy verdicts | 4 counters |
| `VECTOR` | embedding of declared dim | n + f64 sum vector (+ k micro-centroids) |

`STATE` and `OUTCOME` share one representation; they differ only in that
`OUTCOME`'s class set is fixed by the governance layer rather than by the author.

## 4. Consolidation functions

**Exactly mergeable** — a parent slot computed from child slots equals the value
computed from the raw samples, to floating-point rounding:

- `MEAN`, `MIN`, `MAX`, `COUNT`, `RATE`
- `VARIANCE` (Chan parallel merge over n, mean, m2)
- `CENTROID` (weighted: `c = Σ nᵢ cᵢ / Σ nᵢ`)
- `DOMINANT_STATE`, `ENTROPY`, `POLICY_OUTCOME` (counter vectors add)
- `DRIFT`, when defined as distance between a slot centroid and a pinned baseline
  centroid — it rides on `CENTROID` and inherits its exactness

**Approximate, and declared as such** — a parent cannot be recovered exactly from
children, so the archive declares a fixed-size sketch and a stated error bound:

- `P95` and other quantiles
- `DISTINCT`
- `CLUSTERS` / `OUTLIERS` over vectors

An implementation may refuse to offer an approximate CF rather than offer it
silently. What it must never do is present an approximate value with the same
type as an exact one.

## 5. Vector consolidation

Vector archives are the reason this is not just RRDtool with new labels.

A slot stores `n`, an `f64` sum vector, and optionally `k` micro-centroids with
their own counts. Merging sums and counts is exact, so the centroid ladder — 1
minute → 1 hour → 1 day — loses nothing at all. Variance follows from a
sum-of-squares scalar carried alongside.

Micro-centroids are how cluster structure survives consolidation. Merging them is
a greedy pairing on centroid distance followed by a weighted combine — a
streaming centroidal-tessellation step over a bounded codebook, applied along
the time axis rather than across embedding space. If a project already has a
CVT implementation for its index, this should reuse it rather than grow a second
one.

Storage: `dim` × `f32` is 3 KB per slot at dim 768, which is too much for wide
ladders. Quantisation is therefore part of the declaration, not an
implementation detail — `f16` halves it, `int8` with a per-slot scale quarters
it.

## 6. Ingest, unknown, and the heartbeat

Samples arrive at arbitrary times and are normalised into the store's step. A
signal declares a heartbeat: if the gap between consecutive samples exceeds it,
the intervening steps are **unknown**, not interpolated and not zero.

Each archive declares an `xff` — the fraction of unknown sub-slots a slot
tolerates before the slot itself reads unknown.

This matters more here than it did for network graphs. A gauge that reads `0`
where the truth is "no data" will be reasoned from confidently and wrongly.
Unknown must survive to the read surface as a distinct value.

## 7. The read surface

The read API is not `fetch(range) -> Vec<f64>`. Handing an agent an array of
samples recreates the problem the store exists to solve.

A read returns a **temporal context surface**: per signal, a small fixed struct
sized to sit in a prompt.

Because a store consolidates more than numbers, a window reports whatever kind
of thing it actually is. A scalar window carries mean, min, max and variance. A
categorical window carries the class counts it collapsed to, the dominant class
and the entropy. A vector window carries spread and **drift** — distance from
the coarsest permitted window's centroid — and deliberately carries no centroid
at all, since a reading is meant to be handed over whole and reporting movement
discloses strictly less than reporting position. The centroid is a separate
request, under the same grant.

Drift is absent rather than zero on the baseline window itself, and wherever no
coarser permitted window exists to measure against. A self-comparison is exactly
zero and means nothing, and a confident zero is worse than an absent number —
the same rule that makes a deviation against a flat baseline unanswerable.

```
threat:
  now         0.84
  mean_5m     0.71
  mean_1h     0.42
  mean_24h    0.38
  baseline    0.31   (30d)
  sigma       4.3
  trend       rising
  unknown     false
```

Roughly eight numbers per signal. Past about sixty signals a context surface
stops being context and becomes a dump again, so a store that needs more than
that should be split into several entities.

## 8. Governance boundary

An SRRA holds ephemeral evidence. It is deliberately lossy, and because
consolidation is irreversible the individual events cannot be reconstructed —
which makes retention policy and privacy control the same mechanism.

Two rules follow.

A decision artifact must **embed** the context surface it relied on, verbatim, at
the moment of the decision. It must never store a pointer into an archive, because
the archive will consolidate that evidence away and the audit trail will silently
break.

Resolution is a permission. A descriptor may grant an agent hourly aggregates of a
signal while withholding per-second samples. Permitted inference over history sits
alongside permitted operations in the descriptor, and the store enforces it at
read time.

## 9. Where SRRA is the wrong tool

- Retrievable content. Corpora and indexes stay exact; never consolidate
  something you are expected to return.
- General-purpose metrics. Fixed schema, no arbitrary label cardinality, no
  ad-hoc query language. If new dimensions appear at query time, use Prometheus.
- Anything with legal or audit weight. Ledger only.
- Exact quantiles or exact distinct counts.
- Low-rate signals. Below roughly one sample per minute sustained, a plain list is
  both smaller and lossless.
- Per-agent memory at fabric scale. Attach stores to regions and cells; give full
  archives only to a tracked cohort.

## 9a. Implementation status

Built and tested in `src/store.rs`: the declaration and its validation, the
footprint calculation, ingest against caller-supplied timestamps, the
consolidation ladder, unknown handling under `xff`, the resolution grant and its
two refusal forms, the context-surface read, and decision quoting.

The ladder constraint is enforced rather than assumed — each archive's span must
be a whole multiple of the one below it, because a coarse slot is built by
merging whole fine slots and a slot is the finest thing that still exists.

Not built: persistence. Everything above operates on an in-memory store, and the
header format in §2 has no encoder yet.

## 10. Open questions

- Wire format and endianness for the on-disk header; whether to make it
  mmap-friendly by construction.
- Crash safety on slot writes — fsync at consolidation boundaries versus a
  two-slot double buffer.
- Whether `DRIFT` should default to baseline-relative or previous-slot-relative.
- `no_std` + `alloc`: everything except `ENTROPY` is already free of `std`; the
  natural-log shim is the only blocker.
