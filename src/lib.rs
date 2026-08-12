//! Semantic Round-Robin Archive — bounded, progressively consolidated temporal
//! memory for agentic datasets and semantic entities.
//!
//! This crate is the seed described in `SPEC.md`. What exists here is the merge
//! algebra: the accumulators that sit in a slot, and the rules for combining
//! child slots into a parent slot as data ages up the retention ladder. The
//! on-disk store, the ingest path and the read surface are specified but not yet
//! written.
//!
//! Everything in [`Slot`] merges associatively, which is what makes
//! consolidation a parallel reduce rather than a serial pass, and what lets a
//! parent slot be computed from children instead of from raw samples. Where a
//! statistic cannot be recovered that way — quantiles, distinct counts — the
//! spec requires it to be declared approximate rather than presented alongside
//! exact values. No approximate consolidation function is implemented yet.
//!
//! Pure and dependency-free.

mod store;
pub use store::*;

use std::collections::HashMap;

/// Running scalar statistics that merge exactly.
///
/// Mean and variance use Chan's parallel form, so combining two accumulators
/// gives the same result as accumulating the underlying samples in one pass, to
/// floating-point rounding.
#[derive(Clone, Debug, PartialEq)]
pub struct Welford {
    n: u64,
    mean: f64,
    m2: f64,
    min: f64,
    max: f64,
}

/// Hand-written rather than derived: a derived `Default` would start `min` and
/// `max` at zero, so the first pushed sample would never move them and a
/// defaulted accumulator would silently report a minimum of zero.
impl Default for Welford {
    fn default() -> Self {
        Welford::new()
    }
}

impl Welford {
    pub fn new() -> Self {
        Welford {
            n: 0,
            mean: 0.0,
            m2: 0.0,
            min: f64::INFINITY,
            max: f64::NEG_INFINITY,
        }
    }

    pub fn push(&mut self, x: f64) {
        self.n += 1;
        let delta = x - self.mean;
        self.mean += delta / self.n as f64;
        self.m2 += delta * (x - self.mean);
        if x < self.min {
            self.min = x;
        }
        if x > self.max {
            self.max = x;
        }
    }

    pub fn merge(&mut self, other: &Welford) {
        if other.n == 0 {
            return;
        }
        if self.n == 0 {
            *self = other.clone();
            return;
        }
        let na = self.n as f64;
        let nb = other.n as f64;
        let total = na + nb;
        let delta = other.mean - self.mean;
        self.mean += delta * nb / total;
        self.m2 += other.m2 + delta * delta * na * nb / total;
        self.n += other.n;
        if other.min < self.min {
            self.min = other.min;
        }
        if other.max > self.max {
            self.max = other.max;
        }
    }

    pub fn count(&self) -> u64 {
        self.n
    }

    pub fn mean(&self) -> Option<f64> {
        (self.n > 0).then_some(self.mean)
    }

    /// Population variance.
    pub fn variance(&self) -> Option<f64> {
        (self.n > 0).then(|| self.m2 / self.n as f64)
    }

    pub fn min(&self) -> Option<f64> {
        (self.n > 0).then_some(self.min)
    }

    pub fn max(&self) -> Option<f64> {
        (self.n > 0).then_some(self.max)
    }
}

/// Counts over a fixed set of declared classes.
///
/// Backs both `STATE` and `OUTCOME` signals; they differ only in who fixes the
/// class set. Counter vectors add, so `DOMINANT_STATE` and `ENTROPY` survive
/// consolidation exactly.
#[derive(Clone, Debug, PartialEq)]
pub struct Categorical {
    counts: Vec<u64>,
}

impl Categorical {
    pub fn new(classes: usize) -> Self {
        Categorical {
            counts: vec![0; classes],
        }
    }

    pub fn push(&mut self, class: usize) {
        self.counts[class] += 1;
    }

    /// Merges another accumulator over the same class set.
    ///
    /// # Panics
    /// If the class counts differ — merging across declarations is a bug, not a
    /// recoverable condition.
    pub fn merge(&mut self, other: &Categorical) {
        assert_eq!(self.counts.len(), other.counts.len(), "class set mismatch");
        for (a, b) in self.counts.iter_mut().zip(&other.counts) {
            *a += b;
        }
    }

    pub fn total(&self) -> u64 {
        self.counts.iter().sum()
    }

    pub fn counts(&self) -> &[u64] {
        &self.counts
    }

    /// The most frequent class, or `None` if nothing was observed. Ties resolve
    /// to the lowest class index so the result is deterministic.
    pub fn dominant(&self) -> Option<usize> {
        let mut best: Option<(usize, u64)> = None;
        for (i, &c) in self.counts.iter().enumerate() {
            if c > 0 && best.is_none_or(|(_, b)| c > b) {
                best = Some((i, c));
            }
        }
        best.map(|(i, _)| i)
    }

    /// Shannon entropy in nats. Zero when one class holds everything, `ln(k)`
    /// when the k observed classes are uniform.
    pub fn entropy(&self) -> Option<f64> {
        let total = self.total();
        if total == 0 {
            return None;
        }
        let total = total as f64;
        let mut h = 0.0;
        for &c in &self.counts {
            if c > 0 {
                let p = c as f64 / total;
                h -= p * p.ln();
            }
        }
        Some(h)
    }
}

/// Embedding accumulator: count and componentwise sum.
///
/// The centroid ladder is exact because merging is addition. Sums are carried in
/// `f64` regardless of the storage quantisation declared for the archive, so
/// grouping order does not change the result beyond rounding.
#[derive(Clone, Debug, PartialEq)]
pub struct VecAccum {
    n: u64,
    sum: Vec<f64>,
    sumsq: f64,
}

impl VecAccum {
    pub fn new(dim: usize) -> Self {
        VecAccum {
            n: 0,
            sum: vec![0.0; dim],
            sumsq: 0.0,
        }
    }

    pub fn dim(&self) -> usize {
        self.sum.len()
    }

    /// # Panics
    /// If `v` does not match the declared dimension.
    pub fn push(&mut self, v: &[f32]) {
        assert_eq!(v.len(), self.sum.len(), "dimension mismatch");
        self.n += 1;
        for (s, &x) in self.sum.iter_mut().zip(v) {
            let x = x as f64;
            *s += x;
            self.sumsq += x * x;
        }
    }

    /// # Panics
    /// If the dimensions differ.
    pub fn merge(&mut self, other: &VecAccum) {
        assert_eq!(self.sum.len(), other.sum.len(), "dimension mismatch");
        self.n += other.n;
        for (s, o) in self.sum.iter_mut().zip(&other.sum) {
            *s += o;
        }
        self.sumsq += other.sumsq;
    }

    pub fn count(&self) -> u64 {
        self.n
    }

    pub fn centroid(&self) -> Option<Vec<f32>> {
        if self.n == 0 {
            return None;
        }
        let n = self.n as f64;
        Some(self.sum.iter().map(|s| (s / n) as f32).collect())
    }

    /// Mean squared deviation from the centroid, summed over components. Cheap
    /// because it follows from the sum of squares rather than a second pass.
    pub fn variance(&self) -> Option<f64> {
        if self.n == 0 {
            return None;
        }
        let n = self.n as f64;
        let mean_sq: f64 = self.sum.iter().map(|s| (s / n) * (s / n)).sum();
        Some((self.sumsq / n - mean_sq).max(0.0))
    }

    /// Euclidean distance from this accumulator's centroid to a pinned baseline.
    /// This is `DRIFT`; it inherits the exactness of the centroid.
    pub fn drift_from(&self, baseline: &[f32]) -> Option<f64> {
        let c = self.centroid()?;
        assert_eq!(c.len(), baseline.len(), "dimension mismatch");
        Some(
            c.iter()
                .zip(baseline)
                .map(|(a, b)| {
                    let d = (*a - *b) as f64;
                    d * d
                })
                .sum::<f64>()
                .sqrt(),
        )
    }
}

/// A bounded set of weighted micro-centroids, so cluster structure survives
/// consolidation.
///
/// Merging is a greedy pairing on centroid distance followed by a weighted
/// combine — a streaming centroidal-tessellation step over a fixed codebook.
/// This is the one place the algebra is lossy: the result depends on merge
/// order, so cluster counts derived from it are approximate and must be declared
/// so.
#[derive(Clone, Debug)]
pub struct MicroCentroids {
    k: usize,
    dim: usize,
    items: Vec<VecAccum>,
}

impl MicroCentroids {
    pub fn new(k: usize, dim: usize) -> Self {
        assert!(k > 0, "k must be positive");
        MicroCentroids {
            k,
            dim,
            items: Vec::new(),
        }
    }

    pub fn push(&mut self, v: &[f32]) {
        let mut a = VecAccum::new(self.dim);
        a.push(v);
        self.items.push(a);
        self.compact();
    }

    pub fn merge(&mut self, other: &MicroCentroids) {
        assert_eq!(self.dim, other.dim, "dimension mismatch");
        self.items.extend(other.items.iter().cloned());
        self.compact();
    }

    pub fn len(&self) -> usize {
        self.items.len()
    }

    pub fn is_empty(&self) -> bool {
        self.items.is_empty()
    }

    pub fn centroids(&self) -> Vec<(u64, Vec<f32>)> {
        self.items
            .iter()
            .filter_map(|a| a.centroid().map(|c| (a.count(), c)))
            .collect()
    }

    /// Collapses to the single overall accumulator. Exact — the total is
    /// unaffected by how the micro-centroids were paired.
    pub fn total(&self) -> VecAccum {
        let mut out = VecAccum::new(self.dim);
        for a in &self.items {
            out.merge(a);
        }
        out
    }

    fn compact(&mut self) {
        while self.items.len() > self.k {
            let (i, j) = self.closest_pair();
            let merged = {
                let mut m = self.items[i].clone();
                m.merge(&self.items[j]);
                m
            };
            // Remove the higher index first so the lower one stays valid.
            self.items.remove(j);
            self.items.remove(i);
            self.items.push(merged);
        }
    }

    fn closest_pair(&self) -> (usize, usize) {
        let cs: Vec<Vec<f32>> = self.items.iter().filter_map(|a| a.centroid()).collect();
        let mut best = (0usize, 1usize, f64::INFINITY);
        for i in 0..cs.len() {
            for j in (i + 1)..cs.len() {
                let d: f64 = cs[i]
                    .iter()
                    .zip(&cs[j])
                    .map(|(a, b)| {
                        let d = (*a - *b) as f64;
                        d * d
                    })
                    .sum();
                if d < best.2 {
                    best = (i, j, d);
                }
            }
        }
        (best.0, best.1)
    }
}

/// One consolidated bucket. Which variant a slot uses follows from the signal
/// type in the declaration.
#[derive(Clone, Debug)]
pub enum Slot {
    Scalar(Welford),
    Categorical(Categorical),
    Vector(VecAccum),
    Clusters(MicroCentroids),
}

impl Slot {
    /// # Panics
    /// If the variants differ — a declaration mismatch, not a runtime condition.
    pub fn merge(&mut self, other: &Slot) {
        match (self, other) {
            (Slot::Scalar(a), Slot::Scalar(b)) => a.merge(b),
            (Slot::Categorical(a), Slot::Categorical(b)) => a.merge(b),
            (Slot::Vector(a), Slot::Vector(b)) => a.merge(b),
            (Slot::Clusters(a), Slot::Clusters(b)) => a.merge(b),
            _ => panic!("slot type mismatch"),
        }
    }

    pub fn count(&self) -> u64 {
        match self {
            Slot::Scalar(a) => a.count(),
            Slot::Categorical(a) => a.total(),
            Slot::Vector(a) => a.count(),
            Slot::Clusters(a) => a.total().count(),
        }
    }
}

/// A ring of slots with a declared capacity. Writing past the end overwrites the
/// oldest slot, which is the whole point: the footprint never grows.
///
/// A slot may be absent, which is how "unknown" reaches the read surface as a
/// distinct value rather than as a zero an agent will reason from.
#[derive(Clone, Debug)]
pub struct Ring {
    slots: Vec<Option<Slot>>,
    cursor: usize,
    written: u64,
}

impl Ring {
    pub fn new(capacity: usize) -> Self {
        assert!(capacity > 0, "capacity must be positive");
        Ring {
            slots: vec![None; capacity],
            cursor: 0,
            written: 0,
        }
    }

    pub fn capacity(&self) -> usize {
        self.slots.len()
    }

    /// Total slots ever written, including those since overwritten.
    pub fn written(&self) -> u64 {
        self.written
    }

    pub fn push(&mut self, slot: Option<Slot>) {
        self.slots[self.cursor] = slot;
        self.cursor = (self.cursor + 1) % self.slots.len();
        self.written += 1;
    }

    /// Slots oldest first. Retained slots only; the ring's own age is
    /// `written - capacity` slots discarded.
    pub fn iter_oldest_first(&self) -> impl Iterator<Item = &Option<Slot>> {
        let n = self.slots.len();
        let start = if self.written as usize >= n {
            self.cursor
        } else {
            0
        };
        let len = if self.written as usize >= n {
            n
        } else {
            self.written as usize
        };
        (0..len).map(move |i| &self.slots[(start + i) % n])
    }

    /// The most recently written slot, if any.
    pub fn newest(&self) -> Option<&Slot> {
        if self.written == 0 {
            return None;
        }
        let n = self.slots.len();
        self.slots[(self.cursor + n - 1) % n].as_ref()
    }

    /// Folds every retained slot into one. This is what a coarser archive
    /// consumes when data ages up the ladder.
    pub fn consolidate(&self) -> Option<Slot> {
        let mut out: Option<Slot> = None;
        for s in self.iter_oldest_first().flatten() {
            match &mut out {
                Some(acc) => acc.merge(s),
                None => out = Some(s.clone()),
            }
        }
        out
    }

    /// Whether the retained window is too sparse to be trusted, against the
    /// archive's declared `xff` — the tolerated fraction of unknown slots.
    pub fn is_unknown(&self, xff: f64) -> bool {
        let n = self.slots.len();
        let retained = (self.written as usize).min(n);
        if retained == 0 {
            return true;
        }
        let unknown = self.iter_oldest_first().filter(|s| s.is_none()).count();
        unknown as f64 / retained as f64 > xff
    }
}

/// Bytes a single slot occupies, from the signal type and its parameters. The
/// store's footprint is the sum of `slots × slot_bytes` across archives, fixed
/// at create time and carried in the header.
///
/// `dim` and `classes` are ignored where they do not apply.
pub fn slot_bytes(kind: SlotKind, dim: usize, classes: usize, quant: Quant, k: usize) -> usize {
    match kind {
        // n, mean, m2, min, max
        SlotKind::Scalar => 8 + 8 * 4,
        SlotKind::Categorical => 8 * classes,
        // n, sumsq, and the quantised sum vector
        SlotKind::Vector => 8 + 8 + dim * quant.bytes(),
        SlotKind::Clusters => k * (8 + 8 + dim * quant.bytes()),
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum SlotKind {
    Scalar,
    Categorical,
    Vector,
    Clusters,
}

/// Storage quantisation for vector slots. Declared per archive, because at
/// dim 768 the difference between `F32` and `Int8` is 3 KB versus 768 B per
/// slot and decides whether a wide ladder is affordable at all.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Quant {
    F32,
    F16,
    /// One byte per component plus a per-slot scale, amortised in `slot_bytes`.
    Int8,
}

impl Quant {
    pub fn bytes(self) -> usize {
        match self {
            Quant::F32 => 4,
            Quant::F16 => 2,
            Quant::Int8 => 1,
        }
    }
}

/// The compact per-signal reading handed to an agent: the temporal context
/// surface from `SPEC.md` §7. Deliberately small — a read is not a range query.
#[derive(Clone, Debug, Default, PartialEq)]
pub struct ContextSurface {
    pub now: Option<f64>,
    pub windows: Vec<(&'static str, Option<f64>)>,
    pub baseline: Option<f64>,
    pub sigma: Option<f64>,
    pub unknown: bool,
}

impl ContextSurface {
    /// How far the current value sits from the baseline, in baseline standard
    /// deviations. `None` when either is missing or the baseline is flat, which
    /// is the honest answer — a zero-variance baseline makes every deviation
    /// infinite, and reporting that as a number invites an agent to act on it.
    pub fn sigma_from(now: f64, baseline_mean: f64, baseline_var: f64) -> Option<f64> {
        let sd = baseline_var.sqrt();
        (sd > 0.0).then(|| (now - baseline_mean) / sd)
    }
}

/// Per-signal named accumulators for one step, before consolidation into a slot.
/// A convenience for callers assembling a tick; the store will own this once it
/// exists.
#[derive(Clone, Debug, Default)]
pub struct StepBuffer {
    pub scalars: HashMap<String, Welford>,
}

impl StepBuffer {
    pub fn new() -> Self {
        Self::default()
    }

    pub fn observe(&mut self, signal: &str, value: f64) {
        self.scalars
            .entry(signal.to_string())
            .or_default()
            .push(value);
    }

    pub fn take(&mut self, signal: &str) -> Option<Welford> {
        self.scalars.remove(signal)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn batch(xs: &[f64]) -> Welford {
        let mut w = Welford::new();
        for &x in xs {
            w.push(x);
        }
        w
    }

    #[test]
    fn welford_merge_matches_single_pass() {
        let xs: Vec<f64> = (0..1000).map(|i| (i as f64 * 0.37).sin() * 100.0).collect();
        let whole = batch(&xs);

        // Split at an uneven point; consolidation never gets equal children.
        let mut left = batch(&xs[..377]);
        let right = batch(&xs[377..]);
        left.merge(&right);

        assert_eq!(left.count(), whole.count());
        assert!((left.mean().unwrap() - whole.mean().unwrap()).abs() < 1e-9);
        assert!((left.variance().unwrap() - whole.variance().unwrap()).abs() < 1e-6);
        assert_eq!(left.min(), whole.min());
        assert_eq!(left.max(), whole.max());
    }

    #[test]
    fn welford_merge_is_associative_across_a_ladder() {
        let xs: Vec<f64> = (0..600).map(|i| i as f64 % 17.0).collect();
        let whole = batch(&xs);

        // Ten children of sixty, merged pairwise up the ladder.
        let mut level: Vec<Welford> = xs.chunks(60).map(batch).collect();
        while level.len() > 1 {
            let mut next = Vec::new();
            for pair in level.chunks(2) {
                let mut a = pair[0].clone();
                if let Some(b) = pair.get(1) {
                    a.merge(b);
                }
                next.push(a);
            }
            level = next;
        }

        assert_eq!(level[0].count(), whole.count());
        assert!((level[0].mean().unwrap() - whole.mean().unwrap()).abs() < 1e-9);
        assert!((level[0].variance().unwrap() - whole.variance().unwrap()).abs() < 1e-9);
    }

    #[test]
    fn empty_welford_reports_nothing_rather_than_zero() {
        let w = Welford::new();
        assert_eq!(w.mean(), None);
        assert_eq!(w.variance(), None);
        assert_eq!(w.min(), None);
        assert_eq!(w.max(), None);
    }

    #[test]
    fn centroid_merge_equals_batch_mean() {
        let dim = 16;
        let vs: Vec<Vec<f32>> = (0..500)
            .map(|i| {
                (0..dim)
                    .map(|d| ((i * 31 + d * 7) % 97) as f32 / 97.0)
                    .collect()
            })
            .collect();

        let mut whole = VecAccum::new(dim);
        for v in &vs {
            whole.push(v);
        }

        let mut a = VecAccum::new(dim);
        for v in &vs[..123] {
            a.push(v);
        }
        let mut b = VecAccum::new(dim);
        for v in &vs[123..] {
            b.push(v);
        }
        a.merge(&b);

        assert_eq!(a.count(), whole.count());
        let (ca, cw) = (a.centroid().unwrap(), whole.centroid().unwrap());
        for (x, y) in ca.iter().zip(&cw) {
            assert!((x - y).abs() < 1e-6, "{x} vs {y}");
        }
    }

    #[test]
    fn vector_variance_is_non_negative_on_identical_input() {
        let mut a = VecAccum::new(8);
        for _ in 0..50 {
            a.push(&[0.5; 8]);
        }
        let v = a.variance().unwrap();
        assert!((0.0..1e-9).contains(&v), "{v}");
    }

    #[test]
    fn drift_is_zero_against_own_centroid() {
        let mut a = VecAccum::new(4);
        a.push(&[1.0, 2.0, 3.0, 4.0]);
        a.push(&[3.0, 2.0, 1.0, 0.0]);
        let c = a.centroid().unwrap();
        assert!(a.drift_from(&c).unwrap() < 1e-6);
        assert!(a.drift_from(&[0.0, 0.0, 0.0, 0.0]).unwrap() > 0.0);
    }

    #[test]
    fn categorical_survives_consolidation() {
        let mut a = Categorical::new(4);
        let mut b = Categorical::new(4);
        for _ in 0..30 {
            a.push(0);
        }
        for _ in 0..10 {
            a.push(1);
        }
        for _ in 0..50 {
            b.push(1);
        }
        a.merge(&b);

        assert_eq!(a.total(), 90);
        assert_eq!(a.counts(), &[30, 60, 0, 0]);
        assert_eq!(a.dominant(), Some(1));
    }

    #[test]
    fn entropy_is_zero_when_certain_and_ln_k_when_uniform() {
        let mut certain = Categorical::new(4);
        for _ in 0..10 {
            certain.push(2);
        }
        assert!(certain.entropy().unwrap().abs() < 1e-12);

        let mut uniform = Categorical::new(4);
        for i in 0..40 {
            uniform.push(i % 4);
        }
        assert!((uniform.entropy().unwrap() - 4f64.ln()).abs() < 1e-12);

        assert_eq!(Categorical::new(4).entropy(), None);
    }

    #[test]
    fn micro_centroids_stay_bounded_and_conserve_the_total() {
        let dim = 8;
        let mut m = MicroCentroids::new(4, dim);
        let mut reference = VecAccum::new(dim);

        for i in 0..200 {
            let v: Vec<f32> = (0..dim)
                .map(|d| ((i % 5) as f32) + d as f32 * 0.01)
                .collect();
            m.push(&v);
            reference.push(&v);
        }

        assert!(m.len() <= 4, "codebook grew to {}", m.len());
        let total = m.total();
        assert_eq!(total.count(), reference.count());
        for (x, y) in total
            .centroid()
            .unwrap()
            .iter()
            .zip(&reference.centroid().unwrap())
        {
            assert!((x - y).abs() < 1e-5, "{x} vs {y}");
        }
    }

    #[test]
    fn micro_centroids_separate_well_spaced_groups() {
        let dim = 2;
        let mut m = MicroCentroids::new(2, dim);
        for _ in 0..20 {
            m.push(&[0.0, 0.0]);
            m.push(&[100.0, 100.0]);
        }
        assert_eq!(m.len(), 2);
        let mut xs: Vec<f32> = m.centroids().iter().map(|(_, c)| c[0]).collect();
        xs.sort_by(|a, b| a.partial_cmp(b).unwrap());
        assert!(xs[0].abs() < 1e-3 && (xs[1] - 100.0).abs() < 1e-3, "{xs:?}");
    }

    #[test]
    fn ring_overwrites_oldest_and_never_grows() {
        let mut r = Ring::new(4);
        for i in 0..10u64 {
            let mut w = Welford::new();
            w.push(i as f64);
            r.push(Some(Slot::Scalar(w)));
        }

        assert_eq!(r.capacity(), 4);
        assert_eq!(r.written(), 10);
        assert_eq!(r.iter_oldest_first().count(), 4);

        // The four retained slots are 6, 7, 8, 9.
        let means: Vec<f64> = r
            .iter_oldest_first()
            .flatten()
            .map(|s| match s {
                Slot::Scalar(w) => w.mean().unwrap(),
                _ => unreachable!(),
            })
            .collect();
        assert_eq!(means, vec![6.0, 7.0, 8.0, 9.0]);

        match r.newest().unwrap() {
            Slot::Scalar(w) => assert_eq!(w.mean(), Some(9.0)),
            _ => unreachable!(),
        }
    }

    #[test]
    fn ring_reports_only_what_it_has_before_wrapping() {
        let mut r = Ring::new(8);
        for i in 0..3u64 {
            let mut w = Welford::new();
            w.push(i as f64);
            r.push(Some(Slot::Scalar(w)));
        }
        assert_eq!(r.iter_oldest_first().count(), 3);
        assert_eq!(r.consolidate().unwrap().count(), 3);
    }

    #[test]
    fn consolidating_a_ring_equals_merging_its_samples() {
        let mut r = Ring::new(6);
        let mut whole = Welford::new();
        for i in 0..6 {
            let mut w = Welford::new();
            for j in 0..10 {
                let x = (i * 10 + j) as f64;
                w.push(x);
                whole.push(x);
            }
            r.push(Some(Slot::Scalar(w)));
        }

        match r.consolidate().unwrap() {
            Slot::Scalar(w) => {
                assert_eq!(w.count(), whole.count());
                assert!((w.mean().unwrap() - whole.mean().unwrap()).abs() < 1e-9);
                assert!((w.variance().unwrap() - whole.variance().unwrap()).abs() < 1e-6);
            }
            _ => unreachable!(),
        }
    }

    #[test]
    fn unknown_slots_are_visible_rather_than_counted_as_zero() {
        let mut r = Ring::new(4);
        let mut w = Welford::new();
        w.push(1.0);
        r.push(Some(Slot::Scalar(w)));
        r.push(None);
        r.push(None);
        r.push(None);

        // Three of four slots unknown: fine at xff 0.9, not at 0.5.
        assert!(!r.is_unknown(0.9));
        assert!(r.is_unknown(0.5));

        // Consolidation ignores the gaps rather than treating them as zeros.
        assert_eq!(r.consolidate().unwrap().count(), 1);
        assert!(Ring::new(4).is_unknown(0.99));
    }

    #[test]
    fn footprint_follows_from_the_declaration() {
        // The ladder from SPEC.md §2 for one gauge.
        let scalar = slot_bytes(SlotKind::Scalar, 0, 0, Quant::F32, 0);
        let threat = (600 + 2160 + 10080 + 8760) * scalar;
        assert_eq!(scalar, 40);
        assert_eq!(threat, 864_000);

        // Quantisation is what makes a vector ladder affordable.
        let f32_slot = slot_bytes(SlotKind::Vector, 768, 0, Quant::F32, 0);
        let int8_slot = slot_bytes(SlotKind::Vector, 768, 0, Quant::Int8, 0);
        assert_eq!(f32_slot, 3088);
        assert_eq!(int8_slot, 784);
        assert!(f32_slot > 3 * int8_slot);
    }

    #[test]
    fn sigma_declines_to_answer_against_a_flat_baseline() {
        assert_eq!(ContextSurface::sigma_from(0.84, 0.31, 0.0), None);
        let s = ContextSurface::sigma_from(0.84, 0.31, 0.01).unwrap();
        assert!((s - 5.3).abs() < 1e-9, "{s}");
    }

    #[test]
    fn step_buffer_collects_then_yields_a_slot() {
        let mut b = StepBuffer::new();
        b.observe("threat", 0.5);
        b.observe("threat", 0.9);
        b.observe("latency", 12.0);

        let t = b.take("threat").unwrap();
        assert_eq!(t.count(), 2);
        assert!((t.mean().unwrap() - 0.7).abs() < 1e-12);
        assert!(b.take("threat").is_none());
        assert_eq!(b.take("latency").unwrap().count(), 1);
    }
}
