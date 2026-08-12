//! The store: a declaration, the archives it fixes, and the resolution gate.
//!
//! [`Declaration`] fixes what a store remembers — which signals, at which
//! resolutions, for how long — and the footprint follows from it arithmetically
//! and never changes afterwards. [`Store`] ingests observations at caller-supplied
//! timestamps, consolidates each completed slot from a finer archive into the
//! coarser one above it, and discards the finer slot when its declared retention
//! is exhausted.
//!
//! [`Grant`] is the part that is not a time-series database. A reader holds a set
//! of resolutions, and a read is answered only from archives whose resolution it
//! holds. The distinction that matters is what happens to the rest: an archive
//! past its retention has been overwritten, so an ungranted fine resolution is
//! not filtered out of the answer — after retention it is not in the store at
//! all, by any path. Retention policy and read policy are one mechanism.
//!
//! Time is always supplied by the caller. Nothing here reads a clock.

use std::collections::{BTreeMap, BTreeSet};

use crate::{slot_bytes, Categorical, Quant, Ring, Slot, SlotKind, VecAccum, Welford};

/// Bytes of fixed header, exclusive of the encoded declaration. Nominal until
/// the on-disk format is written; the footprint arithmetic does not depend on
/// the value.
pub const HEADER_BYTES: usize = 512;

// ---------------------------------------------------------------- declaration

/// One retention tier: how many steps a slot spans, and how many slots are kept.
#[derive(Clone, Debug, PartialEq)]
pub struct ArchiveSpec {
    /// Steps per slot. Must divide into the next archive's value up the ladder.
    pub steps_per_slot: u32,
    pub slots: u32,
    /// Tolerated fraction of unknown constituents before the slot itself reads
    /// unknown. RRD's `xff`.
    pub xff: f64,
}

impl ArchiveSpec {
    pub fn new(steps_per_slot: u32, slots: u32) -> Self {
        ArchiveSpec {
            steps_per_slot,
            slots,
            xff: 0.5,
        }
    }

    pub fn with_xff(mut self, xff: f64) -> Self {
        self.xff = xff;
        self
    }
}

/// One signal and its retention ladder.
#[derive(Clone, Debug, PartialEq)]
pub struct SignalSpec {
    pub name: String,
    pub kind: SlotKind,
    /// Class count for [`SlotKind::Categorical`], otherwise ignored.
    pub classes: usize,
    /// Embedding dimension for vector kinds, otherwise ignored.
    pub dim: usize,
    /// Codebook size for [`SlotKind::Clusters`], otherwise ignored.
    pub k: usize,
    pub quant: Quant,
    /// Finest first, strictly coarsening.
    pub archives: Vec<ArchiveSpec>,
}

impl SignalSpec {
    pub fn scalar(name: &str, archives: Vec<ArchiveSpec>) -> Self {
        SignalSpec {
            name: name.to_string(),
            kind: SlotKind::Scalar,
            classes: 0,
            dim: 0,
            k: 0,
            quant: Quant::F32,
            archives,
        }
    }

    pub fn categorical(name: &str, classes: usize, archives: Vec<ArchiveSpec>) -> Self {
        SignalSpec {
            name: name.to_string(),
            kind: SlotKind::Categorical,
            classes,
            dim: 0,
            k: 0,
            quant: Quant::F32,
            archives,
        }
    }

    pub fn vector(name: &str, dim: usize, quant: Quant, archives: Vec<ArchiveSpec>) -> Self {
        SignalSpec {
            name: name.to_string(),
            kind: SlotKind::Vector,
            classes: 0,
            dim,
            k: 0,
            quant,
            archives,
        }
    }

    fn empty_slot(&self) -> Slot {
        match self.kind {
            SlotKind::Scalar => Slot::Scalar(Welford::new()),
            SlotKind::Categorical => Slot::Categorical(Categorical::new(self.classes)),
            SlotKind::Vector => Slot::Vector(VecAccum::new(self.dim)),
            SlotKind::Clusters => Slot::Clusters(crate::MicroCentroids::new(self.k, self.dim)),
        }
    }

    fn slot_bytes(&self) -> usize {
        slot_bytes(self.kind, self.dim, self.classes, self.quant, self.k)
    }
}

/// What a store remembers. The contract, and the whole of the footprint
/// calculation.
#[derive(Clone, Debug, PartialEq)]
pub struct Declaration {
    pub step_secs: u64,
    pub signals: Vec<SignalSpec>,
}

impl Declaration {
    pub fn new(step_secs: u64, signals: Vec<SignalSpec>) -> Self {
        Declaration { step_secs, signals }
    }

    /// Rejects a declaration a store cannot honour.
    ///
    /// The ladder constraint is load-bearing rather than cosmetic: a coarse slot
    /// is built by merging whole fine slots, so each archive's span must be an
    /// exact multiple of the one below it. Without that, consolidation would
    /// have to split a slot, and a slot is the finest thing that still exists.
    pub fn validate(&self) -> Result<(), DeclError> {
        if self.step_secs == 0 {
            return Err(DeclError::ZeroStep);
        }
        for s in &self.signals {
            if s.archives.is_empty() {
                return Err(DeclError::NoArchives(s.name.clone()));
            }
            if matches!(s.kind, SlotKind::Categorical) && s.classes == 0 {
                return Err(DeclError::NoClasses(s.name.clone()));
            }
            if matches!(s.kind, SlotKind::Vector | SlotKind::Clusters) && s.dim == 0 {
                return Err(DeclError::NoDim(s.name.clone()));
            }
            let mut prev: Option<u32> = None;
            for a in &s.archives {
                if a.steps_per_slot == 0 || a.slots == 0 {
                    return Err(DeclError::EmptyArchive(s.name.clone()));
                }
                if !(0.0..=1.0).contains(&a.xff) {
                    return Err(DeclError::BadXff(s.name.clone()));
                }
                if let Some(p) = prev {
                    if a.steps_per_slot <= p || a.steps_per_slot % p != 0 {
                        return Err(DeclError::NotALadder {
                            signal: s.name.clone(),
                            finer: p,
                            coarser: a.steps_per_slot,
                        });
                    }
                }
                prev = Some(a.steps_per_slot);
            }
        }
        Ok(())
    }

    /// Every resolution this declaration offers, in seconds, ascending.
    pub fn resolutions(&self) -> Vec<u64> {
        let mut out: BTreeSet<u64> = BTreeSet::new();
        for s in &self.signals {
            for a in &s.archives {
                out.insert(a.steps_per_slot as u64 * self.step_secs);
            }
        }
        out.into_iter().collect()
    }

    /// Resolutions offered for one signal, ascending. Empty if unknown.
    pub fn resolutions_of(&self, signal: &str) -> Vec<u64> {
        self.signals
            .iter()
            .find(|s| s.name == signal)
            .map(|s| {
                s.archives
                    .iter()
                    .map(|a| a.steps_per_slot as u64 * self.step_secs)
                    .collect()
            })
            .unwrap_or_default()
    }

    /// Total bytes. Fixed at creation, and independent of how much is ingested.
    pub fn footprint(&self) -> usize {
        HEADER_BYTES
            + self
                .signals
                .iter()
                .map(|s| {
                    let per = s.slot_bytes();
                    s.archives
                        .iter()
                        .map(|a| a.slots as usize * per)
                        .sum::<usize>()
                })
                .sum::<usize>()
    }
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub enum DeclError {
    ZeroStep,
    NoArchives(String),
    NoClasses(String),
    NoDim(String),
    EmptyArchive(String),
    BadXff(String),
    /// A coarser archive whose span is not a whole multiple of the finer one.
    NotALadder {
        signal: String,
        finer: u32,
        coarser: u32,
    },
}

// --------------------------------------------------------------------- grants

/// The resolutions a reader may see.
///
/// Empty by default. A reader with no grant reads nothing, which is the same
/// default-deny posture the rest of the system takes toward content arriving
/// from elsewhere.
#[derive(Clone, Debug, Default, PartialEq, Eq)]
pub struct Grant {
    resolutions: BTreeSet<u64>,
}

impl Grant {
    /// Grants nothing.
    pub fn none() -> Self {
        Grant::default()
    }

    pub fn allow(mut self, resolution_secs: u64) -> Self {
        self.resolutions.insert(resolution_secs);
        self
    }

    /// Grants every declared resolution at or above `resolution_secs` — the
    /// common shape, where a reader is trusted with trends but not with the
    /// detail underneath them.
    pub fn coarser_than(decl: &Declaration, resolution_secs: u64) -> Self {
        Grant {
            resolutions: decl
                .resolutions()
                .into_iter()
                .filter(|r| *r >= resolution_secs)
                .collect(),
        }
    }

    /// Grants everything the declaration offers.
    pub fn all(decl: &Declaration) -> Self {
        Grant {
            resolutions: decl.resolutions().into_iter().collect(),
        }
    }

    pub fn permits(&self, resolution_secs: u64) -> bool {
        self.resolutions.contains(&resolution_secs)
    }

    pub fn granted(&self) -> Vec<u64> {
        self.resolutions.iter().copied().collect()
    }

    pub fn is_empty(&self) -> bool {
        self.resolutions.is_empty()
    }
}

/// Why a read was not answered.
#[derive(Clone, Debug, PartialEq, Eq)]
pub enum Refusal {
    UnknownSignal(String),
    /// The resolution exists in the declaration but is not granted to this
    /// reader.
    NotGranted {
        requested: u64,
        granted: Vec<u64>,
    },
    /// The declaration does not offer this resolution for this signal at all.
    NotDeclared {
        requested: u64,
        offered: Vec<u64>,
    },
    WrongKind {
        signal: String,
    },
}

// ---------------------------------------------------------------------- reads

/// One archive's answer.
#[derive(Clone, Debug, PartialEq)]
pub struct Window {
    pub resolution_secs: u64,
    /// The span the retained slots cover — resolution × retained slots.
    pub span_secs: u64,
    pub mean: Option<f64>,
    pub min: Option<f64>,
    pub max: Option<f64>,
    pub variance: Option<f64>,
    /// Constituent observations behind the retained slots.
    pub observations: u64,
    /// True when too few of the retained slots carry data, per the archive's
    /// declared `xff`. Distinct from a mean of zero.
    pub unknown: bool,
}

/// What a reader gets: the permitted resolutions, and an explicit note of the
/// declared ones that were withheld.
///
/// Withheld resolutions are named but never valued. The declaration is not
/// secret — a reader is entitled to know a store keeps per-second data — and
/// naming them is what makes the refusal legible instead of looking like an
/// absence of data.
#[derive(Clone, Debug, PartialEq)]
pub struct Reading {
    pub signal: String,
    /// The latest finest-resolution value, and therefore gated on the finest
    /// resolution being granted. A reader denied per-second history is not
    /// handed the current per-second sample through a different door.
    pub now: Option<f64>,
    pub windows: Vec<Window>,
    pub withheld: Vec<u64>,
}

impl Reading {
    pub fn window(&self, resolution_secs: u64) -> Option<&Window> {
        self.windows
            .iter()
            .find(|w| w.resolution_secs == resolution_secs)
    }

    /// Deviation of `now` from the coarsest permitted window, in that window's
    /// standard deviations. `None` where either is missing or the baseline is
    /// flat — a zero-variance baseline makes every deviation infinite, and
    /// reporting that as a number invites acting on it.
    pub fn sigma(&self) -> Option<f64> {
        let now = self.now?;
        let base = self.windows.last()?;
        let (mean, var) = (base.mean?, base.variance?);
        let sd = var.sqrt();
        (sd > 0.0).then(|| (now - mean) / sd)
    }
}

/// Evidence, frozen.
///
/// A decision that cites this store must carry the values themselves, because
/// the store is built to destroy what it was reading. A reference would still
/// resolve later and would no longer mean what it meant.
#[derive(Clone, Debug, PartialEq)]
pub struct DecisionQuote {
    pub taken_at: u64,
    pub reading: Reading,
}

// ---------------------------------------------------------------------- store

struct ArchiveState {
    spec: ArchiveSpec,
    /// Constituents per slot: steps for the finest archive, child slots above.
    units_per_slot: u32,
    ring: Ring,
    pending: Option<Slot>,
    seen: u32,
    unknown: u32,
}

impl ArchiveState {
    /// Absorbs one constituent. Returns the completed slot when one closes, for
    /// the archive above to consolidate in turn.
    fn feed(&mut self, unit: Option<Slot>) -> Option<Option<Slot>> {
        match unit {
            Some(s) => match &mut self.pending {
                Some(acc) => acc.merge(&s),
                None => self.pending = Some(s),
            },
            None => self.unknown += 1,
        }
        self.seen += 1;

        if self.seen < self.units_per_slot {
            return None;
        }

        let too_sparse = self.unknown as f64 / self.seen as f64 > self.spec.xff;
        let done = if too_sparse {
            None
        } else {
            self.pending.take()
        };
        self.pending = None;
        self.seen = 0;
        self.unknown = 0;
        self.ring.push(done.clone());
        Some(done)
    }
}

struct SignalState {
    spec: SignalSpec,
    step: Option<u64>,
    step_acc: Option<Slot>,
    archives: Vec<ArchiveState>,
}

/// A declared, fixed-footprint store with a resolution gate on reads.
pub struct Store {
    decl: Declaration,
    signals: BTreeMap<String, SignalState>,
}

/// One observation.
#[derive(Clone, Debug)]
pub enum Observation {
    Scalar(f64),
    Class(usize),
    Vector(Vec<f32>),
}

impl Store {
    pub fn new(decl: Declaration) -> Result<Self, DeclError> {
        decl.validate()?;
        let mut signals = BTreeMap::new();
        for spec in &decl.signals {
            let mut archives = Vec::with_capacity(spec.archives.len());
            let mut prev_span = 1u32;
            for a in &spec.archives {
                archives.push(ArchiveState {
                    units_per_slot: a.steps_per_slot / prev_span,
                    ring: Ring::new(a.slots as usize),
                    spec: a.clone(),
                    pending: None,
                    seen: 0,
                    unknown: 0,
                });
                prev_span = a.steps_per_slot;
            }
            signals.insert(
                spec.name.clone(),
                SignalState {
                    spec: spec.clone(),
                    step: None,
                    step_acc: None,
                    archives,
                },
            );
        }
        Ok(Store { decl, signals })
    }

    pub fn declaration(&self) -> &Declaration {
        &self.decl
    }

    /// Bytes, from the declaration. Constant for the life of the store.
    pub fn footprint(&self) -> usize {
        self.decl.footprint()
    }

    /// Records an observation at a caller-supplied time.
    ///
    /// Steps with no observation are carried as unknown rather than as zero or
    /// as an interpolation. A gap longer than the finest archive's whole
    /// retention is collapsed rather than replayed, since replaying it would
    /// only overwrite the same ring repeatedly.
    pub fn observe(&mut self, signal: &str, at_secs: u64, obs: Observation) -> Result<(), Refusal> {
        let step_secs = self.decl.step_secs;
        let st = self
            .signals
            .get_mut(signal)
            .ok_or_else(|| Refusal::UnknownSignal(signal.to_string()))?;
        let step = at_secs / step_secs;

        match st.step {
            None => st.step = Some(step),
            Some(cur) if step > cur => {
                Self::close_step(st);
                let max_fill =
                    st.archives[0].spec.steps_per_slot as u64 * st.archives[0].spec.slots as u64;
                let gap = (step - cur - 1).min(max_fill);
                for _ in 0..gap {
                    Self::push_step(st, None);
                }
                st.step = Some(step);
            }
            Some(cur) if step < cur => return Ok(()), // out of order: ignored
            _ => {}
        }

        let slot = st.step_acc.get_or_insert_with(|| st.spec.empty_slot());
        match (slot, obs) {
            (Slot::Scalar(w), Observation::Scalar(x)) => w.push(x),
            (Slot::Categorical(c), Observation::Class(k)) => {
                if k >= st.spec.classes {
                    return Err(Refusal::WrongKind {
                        signal: signal.to_string(),
                    });
                }
                c.push(k);
            }
            (Slot::Vector(v), Observation::Vector(x)) => {
                if x.len() != st.spec.dim {
                    return Err(Refusal::WrongKind {
                        signal: signal.to_string(),
                    });
                }
                v.push(&x);
            }
            (Slot::Clusters(m), Observation::Vector(x)) => m.push(&x),
            _ => {
                return Err(Refusal::WrongKind {
                    signal: signal.to_string(),
                })
            }
        }
        Ok(())
    }

    /// Closes the step in progress, so a read sees everything observed so far.
    /// Idempotent.
    pub fn flush(&mut self, signal: &str) -> Result<(), Refusal> {
        let st = self
            .signals
            .get_mut(signal)
            .ok_or_else(|| Refusal::UnknownSignal(signal.to_string()))?;
        Self::close_step(st);
        Ok(())
    }

    fn close_step(st: &mut SignalState) {
        if st.step.is_some() {
            let slot = st.step_acc.take();
            Self::push_step(st, slot);
        }
    }

    /// One step into the ladder: the finest archive absorbs it, and each closed
    /// slot is consolidated into the archive above.
    fn push_step(st: &mut SignalState, slot: Option<Slot>) {
        let mut carry = Some(slot);
        for a in st.archives.iter_mut() {
            match carry.take() {
                Some(unit) => carry = a.feed(unit),
                None => break,
            }
        }
    }

    /// Reads every permitted resolution, and names the withheld ones.
    pub fn read(&self, signal: &str, grant: &Grant) -> Result<Reading, Refusal> {
        let st = self
            .signals
            .get(signal)
            .ok_or_else(|| Refusal::UnknownSignal(signal.to_string()))?;

        let mut windows = Vec::new();
        let mut withheld = Vec::new();
        for a in &st.archives {
            let res = a.spec.steps_per_slot as u64 * self.decl.step_secs;
            if grant.permits(res) {
                windows.push(Self::window(a, res));
            } else {
                withheld.push(res);
            }
        }

        let finest = st.archives[0].spec.steps_per_slot as u64 * self.decl.step_secs;
        let now = grant
            .permits(finest)
            .then(|| st.archives[0].ring.newest().and_then(scalar_mean))
            .flatten();

        Ok(Reading {
            signal: signal.to_string(),
            now,
            windows,
            withheld,
        })
    }

    /// Reads one resolution, refusing where it is not granted — and separately
    /// where it was never declared, since those are different failures and a
    /// reader should be able to tell them apart.
    pub fn read_at(
        &self,
        signal: &str,
        resolution_secs: u64,
        grant: &Grant,
    ) -> Result<Window, Refusal> {
        let st = self
            .signals
            .get(signal)
            .ok_or_else(|| Refusal::UnknownSignal(signal.to_string()))?;
        let offered = self.decl.resolutions_of(signal);

        let found = st
            .archives
            .iter()
            .find(|a| a.spec.steps_per_slot as u64 * self.decl.step_secs == resolution_secs);

        let a = match found {
            Some(a) => a,
            None => {
                return Err(Refusal::NotDeclared {
                    requested: resolution_secs,
                    offered,
                })
            }
        };
        if !grant.permits(resolution_secs) {
            return Err(Refusal::NotGranted {
                requested: resolution_secs,
                granted: grant.granted(),
            });
        }
        Ok(Self::window(a, resolution_secs))
    }

    /// Reads and freezes, for a decision record to carry.
    pub fn quote(
        &self,
        signal: &str,
        grant: &Grant,
        taken_at: u64,
    ) -> Result<DecisionQuote, Refusal> {
        Ok(DecisionQuote {
            taken_at,
            reading: self.read(signal, grant)?,
        })
    }

    fn window(a: &ArchiveState, resolution_secs: u64) -> Window {
        let retained = (a.ring.written() as usize).min(a.ring.capacity());
        let folded = a.ring.consolidate();
        let (mean, min, max, variance, observations) = match folded.as_ref().and_then(scalar) {
            Some(w) => (w.mean(), w.min(), w.max(), w.variance(), w.count()),
            None => (
                None,
                None,
                None,
                None,
                folded.as_ref().map_or(0, |s| s.count()),
            ),
        };
        Window {
            resolution_secs,
            span_secs: resolution_secs * retained as u64,
            mean,
            min,
            max,
            variance,
            observations,
            unknown: a.ring.is_unknown(a.spec.xff),
        }
    }
}

fn scalar(slot: &Slot) -> Option<&Welford> {
    match slot {
        Slot::Scalar(w) => Some(w),
        _ => None,
    }
}

fn scalar_mean(slot: &Slot) -> Option<f64> {
    scalar(slot).and_then(|w| w.mean())
}

#[cfg(test)]
mod tests {
    use super::*;

    /// One signal, 1-second steps: per-second for 10s, per-10-seconds for 100s.
    fn decl() -> Declaration {
        Declaration::new(
            1,
            vec![SignalSpec::scalar(
                "threat",
                vec![ArchiveSpec::new(1, 10), ArchiveSpec::new(10, 10)],
            )],
        )
    }

    fn ingest(s: &mut Store, from: u64, to: u64, value: f64) {
        for t in from..=to {
            s.observe("threat", t, Observation::Scalar(value)).unwrap();
        }
    }

    // ------------------------------------------------------------ declaration

    #[test]
    fn a_ladder_that_does_not_divide_is_rejected() {
        let d = Declaration::new(
            1,
            vec![SignalSpec::scalar(
                "x",
                vec![ArchiveSpec::new(2, 4), ArchiveSpec::new(3, 4)],
            )],
        );
        assert_eq!(
            d.validate(),
            Err(DeclError::NotALadder {
                signal: "x".into(),
                finer: 2,
                coarser: 3
            })
        );

        // Coarser must also actually be coarser.
        let flat = Declaration::new(
            1,
            vec![SignalSpec::scalar(
                "x",
                vec![ArchiveSpec::new(10, 4), ArchiveSpec::new(10, 4)],
            )],
        );
        assert!(matches!(flat.validate(), Err(DeclError::NotALadder { .. })));

        assert!(decl().validate().is_ok());
    }

    #[test]
    fn footprint_is_fixed_by_the_declaration_and_ingest_does_not_move_it() {
        let mut s = Store::new(decl()).unwrap();
        let before = s.footprint();
        assert_eq!(before, HEADER_BYTES + (10 + 10) * 40);

        ingest(&mut s, 0, 5000, 1.0);
        s.flush("threat").unwrap();

        assert_eq!(s.footprint(), before);
    }

    // ------------------------------------------------------------ consolidation

    #[test]
    fn a_coarse_slot_is_the_merge_of_the_fine_slots_beneath_it() {
        let mut s = Store::new(decl()).unwrap();
        for t in 0..60u64 {
            s.observe("threat", t, Observation::Scalar(t as f64))
                .unwrap();
        }
        s.flush("threat").unwrap();

        let g = Grant::all(s.declaration());
        let coarse = s.read_at("threat", 10, &g).unwrap();

        // Six closed slots covering steps 0..59; mean of 0..59 is 29.5.
        assert_eq!(coarse.observations, 60);
        assert!((coarse.mean.unwrap() - 29.5).abs() < 1e-9);
        assert_eq!(coarse.min, Some(0.0));
        assert_eq!(coarse.max, Some(59.0));
        assert_eq!(coarse.span_secs, 60);
    }

    /// The limitation the whole filing rests on: past its retention the fine
    /// resolution is **gone**, not withheld — while the coarse summary of the
    /// same moment survives.
    #[test]
    fn expired_fine_detail_ceases_to_exist_while_its_coarse_summary_survives() {
        let mut s = Store::new(decl()).unwrap();

        // One distinctive spike, then a long quiet stretch that overruns the
        // 10-second fine archive many times over.
        s.observe("threat", 0, Observation::Scalar(1000.0)).unwrap();
        ingest(&mut s, 1, 60, 1.0);
        s.flush("threat").unwrap();

        let g = Grant::all(s.declaration());

        // The fine archive retains 10 seconds. The spike is not in it — not
        // filtered from the answer, absent from the store.
        let fine = s.read_at("threat", 1, &g).unwrap();
        assert_eq!(fine.span_secs, 10);
        assert_eq!(fine.max, Some(1.0));
        assert_eq!(fine.min, Some(1.0));

        // The coarse archive still carries the spike, consolidated.
        let coarse = s.read_at("threat", 10, &g).unwrap();
        assert_eq!(coarse.max, Some(1000.0));

        // And a fully granted read cannot surface it at fine resolution either.
        let r = s.read("threat", &g).unwrap();
        assert_eq!(r.window(1).unwrap().max, Some(1.0));
        assert_eq!(r.window(10).unwrap().max, Some(1000.0));
    }

    #[test]
    fn a_gap_reads_unknown_rather_than_zero() {
        let mut s = Store::new(decl()).unwrap();
        s.observe("threat", 0, Observation::Scalar(7.0)).unwrap();
        // Nothing for twenty seconds, which is twice the fine retention.
        s.observe("threat", 20, Observation::Scalar(7.0)).unwrap();

        let g = Grant::all(s.declaration());
        let fine = s.read_at("threat", 1, &g).unwrap();
        assert!(fine.unknown);
        assert_eq!(fine.mean, None);
        assert_ne!(fine.mean, Some(0.0));
    }

    // -------------------------------------------------------------- the gate

    #[test]
    fn a_reader_with_no_grant_reads_nothing() {
        let mut s = Store::new(decl()).unwrap();
        ingest(&mut s, 0, 60, 3.0);
        s.flush("threat").unwrap();

        let none = Grant::none();
        assert!(none.is_empty());

        let r = s.read("threat", &none).unwrap();
        assert!(r.windows.is_empty());
        assert_eq!(r.now, None);
        assert_eq!(r.withheld, vec![1, 10]);

        assert_eq!(
            s.read_at("threat", 10, &none),
            Err(Refusal::NotGranted {
                requested: 10,
                granted: vec![]
            })
        );
    }

    #[test]
    fn a_coarse_grant_answers_the_trend_and_refuses_the_detail() {
        let mut s = Store::new(decl()).unwrap();
        ingest(&mut s, 0, 60, 3.0);
        s.flush("threat").unwrap();

        let g = Grant::coarser_than(s.declaration(), 10);
        assert_eq!(g.granted(), vec![10]);

        let r = s.read("threat", &g).unwrap();
        assert_eq!(r.windows.len(), 1);
        assert_eq!(r.windows[0].resolution_secs, 10);
        assert!((r.windows[0].mean.unwrap() - 3.0).abs() < 1e-9);

        // The per-second archive is named as withheld — a reader may know the
        // store keeps it — but never valued.
        assert_eq!(r.withheld, vec![1]);
        assert!(r.window(1).is_none());

        assert_eq!(
            s.read_at("threat", 1, &g),
            Err(Refusal::NotGranted {
                requested: 1,
                granted: vec![10]
            })
        );
    }

    #[test]
    fn the_current_value_is_gated_with_the_resolution_it_belongs_to() {
        let mut s = Store::new(decl()).unwrap();
        ingest(&mut s, 0, 60, 42.0);
        s.flush("threat").unwrap();

        // Denied per-second history, so not handed the per-second sample either.
        let coarse = Grant::coarser_than(s.declaration(), 10);
        assert_eq!(s.read("threat", &coarse).unwrap().now, None);

        let all = Grant::all(s.declaration());
        assert_eq!(s.read("threat", &all).unwrap().now, Some(42.0));
    }

    #[test]
    fn an_undeclared_resolution_is_a_different_answer_from_an_ungranted_one() {
        let mut s = Store::new(decl()).unwrap();
        ingest(&mut s, 0, 20, 1.0);
        s.flush("threat").unwrap();

        let g = Grant::all(s.declaration());
        assert_eq!(
            s.read_at("threat", 60, &g),
            Err(Refusal::NotDeclared {
                requested: 60,
                offered: vec![1, 10]
            })
        );
        assert_eq!(
            s.read("nope", &g),
            Err(Refusal::UnknownSignal("nope".into()))
        );
    }

    // ------------------------------------------------------------- evidence

    #[test]
    fn a_quote_survives_the_consolidation_of_what_it_quoted() {
        let mut s = Store::new(decl()).unwrap();
        s.observe("threat", 0, Observation::Scalar(1000.0)).unwrap();
        ingest(&mut s, 1, 9, 1000.0);
        s.flush("threat").unwrap();

        let g = Grant::all(s.declaration());
        let quote = s.quote("threat", &g, 9).unwrap();
        let frozen = quote.clone();
        assert_eq!(quote.reading.window(1).unwrap().max, Some(1000.0));

        // Run long enough that the fine archive has been overwritten entirely.
        ingest(&mut s, 10, 200, 1.0);
        s.flush("threat").unwrap();

        let later = s.read("threat", &g).unwrap();
        assert_eq!(later.window(1).unwrap().max, Some(1.0));
        assert_ne!(later, quote.reading);

        // The record carries the values, not a reference, so it is unchanged.
        assert_eq!(quote, frozen);
        assert_eq!(quote.taken_at, 9);
    }

    #[test]
    fn sigma_declines_to_answer_against_a_flat_baseline() {
        let mut s = Store::new(decl()).unwrap();
        ingest(&mut s, 0, 60, 5.0);
        s.flush("threat").unwrap();

        let g = Grant::all(s.declaration());
        // A constant series has no spread, so there is no honest sigma.
        assert_eq!(s.read("threat", &g).unwrap().sigma(), None);

        let mut v = Store::new(decl()).unwrap();
        for t in 0..60u64 {
            v.observe("threat", t, Observation::Scalar((t % 7) as f64))
                .unwrap();
        }
        v.flush("threat").unwrap();
        assert!(v.read("threat", &g).unwrap().sigma().is_some());
    }

    // ----------------------------------------------------------- other kinds

    #[test]
    fn categorical_and_vector_signals_ingest_and_gate_the_same_way() {
        let d = Declaration::new(
            1,
            vec![
                SignalSpec::categorical(
                    "outcome",
                    4,
                    vec![ArchiveSpec::new(1, 4), ArchiveSpec::new(4, 4)],
                ),
                SignalSpec::vector(
                    "intent",
                    3,
                    Quant::Int8,
                    vec![ArchiveSpec::new(1, 4), ArchiveSpec::new(4, 4)],
                ),
            ],
        );
        let mut s = Store::new(d).unwrap();

        for t in 0..8u64 {
            s.observe("outcome", t, Observation::Class((t % 2) as usize))
                .unwrap();
            s.observe("intent", t, Observation::Vector(vec![t as f32, 0.0, 1.0]))
                .unwrap();
        }
        s.flush("outcome").unwrap();
        s.flush("intent").unwrap();

        let g = Grant::coarser_than(s.declaration(), 4);
        let r = s.read("outcome", &g).unwrap();
        assert_eq!(r.withheld, vec![1]);
        assert_eq!(r.windows.len(), 1);
        // Non-scalar kinds carry no mean, but the observation count is real.
        assert_eq!(r.windows[0].mean, None);
        assert_eq!(r.windows[0].observations, 8);

        assert_eq!(
            s.read_at("intent", 1, &g),
            Err(Refusal::NotGranted {
                requested: 1,
                granted: vec![4]
            })
        );

        assert!(matches!(
            s.observe("outcome", 9, Observation::Class(9)),
            Err(Refusal::WrongKind { .. })
        ));
        assert!(matches!(
            s.observe("intent", 9, Observation::Scalar(1.0)),
            Err(Refusal::WrongKind { .. })
        ));
    }
}
