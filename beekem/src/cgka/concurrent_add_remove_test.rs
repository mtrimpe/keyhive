//! Regression test: a concurrent add must not be destroyed by a concurrent
//! removal of the trailing member.
//!
//! # The claim
//!
//! `BeeKem::sort_leaves_and_blank_paths_for_concurrent_membership_changes` is
//! documented to make concurrent membership changes converge: "For concurrent
//! membership changes, we need to ensure that removed paths are blanked and
//! concurrently added member leaves are sorted (and their paths blanked) after
//! any other concurrent operations were applied."
//!
//! It does not hold when a concurrent `Add` and a concurrent `Remove` of the
//! **trailing** member land in the same epoch and the epoch's digest-determined
//! order applies the `Remove` first:
//!
//! 1. `BeeKem::remove_id` blanks the trailing leaf and then collects the
//!    contiguous tombstones at the end of `leaves`, decrementing
//!    `next_leaf_idx` — the removed member's slot is now free.
//! 2. The concurrent `Add` is applied with `BeeKem::push_leaf`, which reuses
//!    exactly that freed slot.
//! 3. `sort_leaves_and_blank_paths_for_concurrent_membership_changes` then
//!    re-blanks the removed member's *recorded* leaf index — which is now
//!    occupied by the **added** member.
//!
//! Expected: after the epoch the group is `{a, b, d}` (`c` removed, `d`
//! added), and every member the tree counts has a materialized leaf.
//!
//! Actual, on this commit:
//!
//! * **debug builds** trip BeeKEM's own state assertion,
//!   `debug_assert!(self.leaf(leaf_idx).is_none())` (`tree.rs`), inside
//!   `sort_leaves_and_blank_paths_for_concurrent_membership_changes`;
//! * **release builds** compile that assertion out and
//!   `blank_leaf_and_path` destroys the added member's leaf instead. The
//!   member stays in `id_to_leaf_idx` (so `contains_id` and `member_count`
//!   still count it) while its leaf is blank, and the following re-push loop —
//!   which hunts down the leaves vec for a member that no longer exists —
//!   blanks and re-packs every remaining leaf. No error is returned and no
//!   replica diverges: the intra-epoch order is digest-determined, so every
//!   replica reaches the *same* corrupted state.
//!
//! The test therefore asserts the tree invariant directly instead of merely
//! asserting that nothing panics, so that it fails in both profiles. It sweeps
//! a fixed set of seeds because the epoch order is decided by operation
//! digests: which of the two orders a given key generation produces is a coin
//! flip, and about half the seeds land on the destructive one.
//!
//! The exact count is toolchain-dependent, and a green run is not a
//! refutation: which seeds land on the destructive order depends on generated
//! keys, so a change to key generation or to `rand` moves the count, and the
//! debug and release manifestations are not guaranteed to fire on the same
//! seeds on every toolchain. Read the failure, not the number.
//!
//! Found by property-based conformance testing of a downstream vendored copy
//! of this crate.

extern crate std;

use crate::{
    cgka::Cgka,
    id::{MemberId, TreeId},
    keys::ShareKeyMap,
    operation::CgkaOperation,
};
use alloc::{format, string::String, string::ToString, sync::Arc, vec, vec::Vec};
use future_form::Local;
use keyhive_crypto::{
    share_key::{ShareKey, ShareSecretKey},
    signed::Signed,
    signer::memory::MemorySigner,
    verifiable::Verifiable,
};
use rand::{rngs::StdRng, SeedableRng};
use std::panic::{catch_unwind, AssertUnwindSafe};

/// The scenario is seeded so it is reproducible; the epoch order is decided by
/// operation digests, so the sweep covers both orders.
const SEED_BASE: u64 = 0x5107_0000;
const SEEDS: u64 = 64;

struct Member {
    id: MemberId,
    signer: MemorySigner,
    pk: ShareKey,
    sk: ShareSecretKey,
}

fn member(rng: &mut StdRng) -> Member {
    let signer = MemorySigner::generate(rng);
    let id = MemberId(signer.verifying_key());
    let sk = ShareSecretKey::generate(rng);
    let pk = sk.share_key();
    Member { id, signer, pk, sk }
}

/// Three replicas of one tree, each holding its own [`Cgka`], plus the members
/// they belong to. Operations are created once and delivered to every replica,
/// so the digest identity of an operation is shared, as it is on the wire.
struct Group {
    members: Vec<Member>,
    replicas: Vec<Cgka>,
}

impl Group {
    async fn found(n: usize, rng: &mut StdRng) -> Group {
        let doc_signer = MemorySigner::generate(rng);
        let doc_id = TreeId(doc_signer.verifying_key());
        let members: Vec<Member> = (0..n).map(|_| member(rng)).collect();

        let mut founder =
            Cgka::new::<Local, _>(doc_id, members[0].id, members[0].pk, &members[0].signer)
                .await
                .expect("founding the tree succeeds");
        founder.owner_sks.insert(members[0].pk, members[0].sk);
        let init_add_op = founder.init_add_op();

        let mut replicas = vec![founder];
        for m in &members[1..] {
            let mut sks = ShareKeyMap::new();
            sks.insert(m.pk, m.sk);
            replicas.push(
                Cgka::new_from_init_add(doc_id, members[0].id, members[0].pk, init_add_op.clone())
                    .expect("seeding a replica from the init add succeeds")
                    .with_new_owner(m.id, sks)
                    .expect("re-owning a replica succeeds"),
            );
        }

        let mut group = Group { members, replicas };
        for i in 1..n {
            let (id, pk) = (group.members[i].id, group.members[i].pk);
            let op = group.replicas[0]
                .add::<Local, _>(id, pk, &group.members[0].signer)
                .await
                .expect("authoring the founding add succeeds")
                .expect("the added member is new");
            group.broadcast(&Arc::new(op));
        }
        group
    }

    fn broadcast(&mut self, op: &Arc<Signed<CgkaOperation>>) {
        for replica in self.replicas.iter_mut() {
            replica
                .merge_concurrent_operation(op.clone())
                .expect("operations are delivered in causal order");
        }
    }
}

/// One run of the scenario. Returns the tree invariants that were violated.
async fn scenario(seed: u64) -> Vec<String> {
    let mut rng = StdRng::seed_from_u64(seed);
    let mut group = Group::found(3, &mut rng).await;

    let a = group.members[0].id;
    let b = group.members[1].id;
    let c = group.members[2].id; // the trailing leaf
    let d = member(&mut rng);

    // Concurrently, from the same three-member state:
    //   * `b` adds `d`
    //   * `a` removes the trailing member `c`
    let add_d = group.replicas[1]
        .add::<Local, _>(d.id, d.pk, &group.members[1].signer)
        .await
        .expect("authoring the add succeeds")
        .expect("the added member is new");
    let remove_c = group.replicas[0]
        .remove::<Local, _>(c, &group.members[0].signer)
        .await
        .expect("authoring the removal succeeds")
        .expect("the removed member is present");
    group.broadcast(&Arc::new(add_d));
    group.broadcast(&Arc::new(remove_c));

    // A heal that causally follows both. Merging a non-concurrent operation is
    // what makes a replica replay the concurrent epoch, so this is the point at
    // which the merge rule above runs.
    let sk = ShareSecretKey::generate(&mut rng);
    let pk = sk.share_key();
    let (_pcs_key, heal) = group.replicas[0]
        .update::<Local, _, StdRng>(pk, sk, &group.members[0].signer, &mut rng)
        .await
        .expect("authoring the covering heal succeeds");
    group.broadcast(&Arc::new(heal));

    // The tree invariant: every member the tree counts is materialized at a
    // leaf, the removed member is gone, and nobody else is.
    let tree = &group.replicas[1].tree;
    let mut violations = Vec::new();
    for (name, id) in [("a", a), ("b", b), ("d (added)", d.id)] {
        if !tree.contains_id(&id) {
            violations.push(format!("{name} is no longer a member"));
        } else if tree.node_key_for_id(id).is_err() {
            violations.push(format!(
                "{name} is counted as a member but its leaf has been blanked"
            ));
        }
    }
    if tree.contains_id(&c) {
        violations.push("c (removed) is still a member".to_string());
    }
    if tree.member_count() != 3 {
        violations.push(format!(
            "member_count is {}, expected 3 (a, b, d)",
            tree.member_count()
        ));
    }
    violations
}

/// The exact assertion this defect trips in a debug build, from
/// `sort_leaves_and_blank_paths_for_concurrent_membership_changes` via
/// `BeeKem::leaf`. Matched on the message so an *unrelated* panic on some other
/// toolchain (e.g. a sibling-resolution or out-of-bounds panic reached by a
/// different generated shape) is never reported as this one.
const EXPECTED_ASSERTION: &str = "self.leaf(leaf_idx).is_none()";

fn panic_message(payload: &alloc::boxed::Box<dyn core::any::Any + Send>) -> String {
    payload
        .downcast_ref::<String>()
        .cloned()
        .or_else(|| payload.downcast_ref::<&str>().map(|s| s.to_string()))
        .unwrap_or_else(|| "panic with a non-string payload".to_string())
}

#[test]
fn concurrent_add_and_trailing_remove_must_not_destroy_the_added_leaf() {
    let mut asserted: Vec<u64> = Vec::new();
    let mut corrupted: Vec<(u64, String)> = Vec::new();
    // Panics that are NOT this defect. They must never be counted as the
    // assertion, and their presence fails the test on its own terms.
    let mut unexpected: Vec<(u64, String)> = Vec::new();

    for seed in 0..SEEDS {
        let seed = SEED_BASE + seed;
        let outcome = catch_unwind(AssertUnwindSafe(|| {
            tokio::runtime::Builder::new_current_thread()
                .build()
                .expect("building a test runtime succeeds")
                .block_on(scenario(seed))
        }));
        match outcome {
            Err(payload) => {
                let message = panic_message(&payload);
                if message.contains(EXPECTED_ASSERTION) {
                    // Debug builds: BeeKEM's own `debug_assert!` fires first,
                    // and it is *this* assertion, not some other panic.
                    asserted.push(seed);
                } else {
                    unexpected.push((seed, message));
                }
            }
            // Release builds: the assertion is compiled out and the tree is
            // silently corrupted instead — caught by the invariant directly.
            Ok(violations) if !violations.is_empty() => {
                corrupted.push((seed, violations.join("; ")))
            }
            Ok(_) => {}
        }
    }

    let broken = asserted.len() + corrupted.len();
    let mut report = String::new();
    if !asserted.is_empty() {
        report.push_str(&format!(
            "\n  {} seed(s) tripped BeeKEM's own state assertion \
             ({EXPECTED_ASSERTION}) in \
             sort_leaves_and_blank_paths_for_concurrent_membership_changes; \
             first: {:#x}",
            asserted.len(),
            asserted[0]
        ));
    }
    if !corrupted.is_empty() {
        report.push_str(&format!(
            "\n  {} seed(s) corrupted the tree without any error; first: \
             {:#x} -> {}",
            corrupted.len(),
            corrupted[0].0,
            corrupted[0].1
        ));
    }
    if let Some((seed, message)) = unexpected.first() {
        report.push_str(&format!(
            "\n  {} seed(s) hit an UNRELATED panic (not this defect and not \
             counted as it); first: {seed:#x} -> {message}",
            unexpected.len()
        ));
    }
    assert!(
        broken == 0 && unexpected.is_empty(),
        "a concurrent add and a concurrent removal of the trailing member \
         broke the tree on {broken} of {SEEDS} seeds{report}"
    );
}
