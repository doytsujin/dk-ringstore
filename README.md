# ringstore

Bounded temporal memory. Several resolutions of the same signal in one store
whose footprint is computed at creation and never changes, consolidating as
observations age, with reads gated per resolution.

Pure Rust, no dependencies, no clock — timestamps are always supplied by the
caller.

```rust
use ringstore::{ArchiveSpec, Declaration, Grant, Observation, SignalSpec, Store};

// 1-second steps: per-second for 10 minutes, per-minute for 6 hours.
let decl = Declaration::new(
    1,
    vec![SignalSpec::scalar(
        "latency_p95",
        vec![ArchiveSpec::new(1, 600), ArchiveSpec::new(60, 360)],
    )],
);

println!("{} bytes, for the life of the store", decl.footprint());

let mut store = Store::new(decl)?;
store.observe("latency_p95", now_secs, Observation::Scalar(212.0))?;

// This reader sees the per-minute trend and not the per-second series.
let grant = Grant::coarser_than(store.declaration(), 60);
let reading = store.read("latency_p95", &grant)?;
```

## Why this rather than a time-series database

Because sometimes the useful property is the *ceiling*, not the query language.

A store's footprint follows arithmetically from its declaration — the sum over
archives of slots times slot size — so a component can state how much memory it
will use before it starts, and be right. That is the difference between a
control-plane process you can run in a cluster with a hard limit and one you
have to watch.

It is also a poor general-purpose metrics backend, deliberately: fixed schema,
no arbitrary label cardinality, no ad-hoc query language. If you need new
dimensions at query time, use Prometheus. This is for the case where a component
carries a small, bounded, declared memory of a handful of signals — per
deployment, per namespace, per cache region, per agent, per map region — and
needs to answer "what is normal here, and is this unusual" without an external
service.

## What is in the box

**Declared retention ladders.** Each archive has a resolution and a slot count.
Coarser archives are built by merging whole finer slots, so the declaration is
validated to be an actual ladder — each span a whole multiple of the one below.

**Exact consolidation.** Mean, variance, min and max merge exactly (Chan's
parallel form). Categorical signals carry per-class counters, so the dominant
class and the distribution's entropy survive consolidation. Embedding signals
carry a count and a componentwise sum, so a centroid merged up the ladder is
precisely the centroid of the underlying vectors — minute to hour to day loses
nothing.

**Unknown is not zero.** A step with no observation is unknown, and an archive
declares the fraction of unknown constituents it tolerates before a slot itself
reads unknown. Absence reaches the reader as absence. This matters more than it
sounds: a consumer handed a zero will treat it as a measurement.

**Reads report what the signal actually is.** A scalar window carries mean, min,
max and variance; a categorical window carries the class counts, the dominant
class and the entropy; a vector window carries spread and drift from the
coarsest permitted window. A vector window carries no centroid — reporting
movement discloses less than reporting position, and a 768-dimensional vector
does not belong in a reading meant to be handed over whole. Ask for the centroid
separately, under the same grant.

**Reads gated by resolution.** A `Grant` names the resolutions a reader may see
and is empty by default. Withheld resolutions are named in the reading and never
valued, so a refusal is distinguishable from missing data — and the current
value is gated with the finest resolution it belongs to, since otherwise a
reader denied fine history can simply poll for it.

**Quotes.** `Store::quote` freezes a reading so a record can carry the values it
relied on rather than a reference into a store built to discard them.

## What it is not

Not a general metrics backend. Not a replacement for Prometheus, InfluxDB or
Timescale. Not exact quantiles or distinct counts — those are not mergeable and
are therefore not offered rather than silently approximated. Not useful below
roughly one observation per minute sustained, where a plain list is smaller and
lossless. Not persistent yet: the store is in-memory and the on-disk format in
[SPEC.md](SPEC.md) has no encoder.

## Prior art

The core idea is [RRDtool](https://oss.oetiker.ch/rrdtool/) (Tobias Oetiker,
1997 onward) and is not claimed as new here — fixed footprint, consolidation
functions, round-robin archives, the heartbeat and the unknown fraction all come
from it, and the declaration syntax in SPEC.md deliberately echoes `DS:`/`RRA:`.
Graphite's Whisper, InfluxDB retention policies and Timescale continuous
aggregates are the same family.

The additive triple behind the embedding consolidation — count, linear sum, sum
of squares, merged by addition — is a BIRCH clustering feature (Zhang,
Ramakrishnan and Livny, SIGMOD 1996). Maintaining cluster features across a
hierarchy of time frames is CluStream (Aggarwal et al., VLDB 2003).

Restricting a consumer to a coarser temporal resolution is not new either:
[TimeCrypt](https://www.usenix.org/conference/nsdi20/presentation/burkhalter)
(NSDI 2020) does it cryptographically, with a keystream per resolution level,
and prunes aged-out raw data under a retention policy. Read it before building
anything in this space.

What this crate contributes is the combination and the small decisions that only
appear once it is built — the read shape, unknown reaching the reader, gating
the current value along with its resolution, and quoting rather than referencing.

## Status

The algebra, the store and the resolution gate are implemented and tested.
Persistence is not written. See [SPEC.md](SPEC.md) §9a for the exact split and
§10 for what is still open.

## Licence

MIT or Apache-2.0, at your option.
