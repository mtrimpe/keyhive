//! Regression test: a `Remove` operation's recorded leaf index must not be
//! used unchecked against the merged tree.
//!
//! # The claim
//!
//! `CgkaOperation::Remove` records the `leaf_idx` its **author** saw. When an
//! epoch contains a membership change,
//! `BeeKem::sort_leaves_and_blank_paths_for_concurrent_membership_changes`
//! re-blanks each removed member at that recorded index:
//!
//! ```ignore
//! for (id, idx) in removed_ids {
//!     let leaf_idx = LeafNodeIndex::new(idx);
//!     debug_assert!(self.leaf(leaf_idx).is_none());
//!     self.blank_leaf_and_path(leaf_idx);
//! }
//! ```
//!
//! A recorded index is only meaningful in the tree the author held. Two things
//! move a member's leaf index between replicas, and both are ordinary:
//!
//! * `BeeKem::remove_id` collects the contiguous tombstones at the end of
//!   `leaves` and decrements `next_leaf_idx`, so a later `push_leaf` reuses
//!   those slots; and
//! * the merge re-packs concurrently added leaves.
//!
//! A replica that has not yet received a removal therefore assigns a *higher*
//! leaf index to a member than the merged history does — and the merged tree's
//! `leaves` vec, which is sized to the tree it actually grew, can be shorter
//! than that index. The lookup is unchecked, so the merge panics.
//!
//! Scenario (four members `a`, `b`, `c`, `d` at leaves 0–3):
//!
//! 1. `a` removes `c`, a middle member: leaf 2 becomes a hole, `d` still holds
//!    the trailing slot 3, and the leaves vec stays 4 long.
//! 2. Concurrently, `a` removes the trailing member `d` — which collapses
//!    `next_leaf_idx` past both tombstones, down to 2 — while `b`, who has not
//!    received that removal, adds `e`. In `b`'s tree `e` goes to leaf 4 and the
//!    tree grows to 8 leaves; in the merged history `e` is packed into the hole
//!    at leaf 2 and the tree stays 4 leaves wide.
//! 3. Still unaware of the removal, `b` removes `e` again. The operation
//!    records leaf index 4.
//! 4. Concurrently, `a` adds `f`, so `b`'s removal shares an epoch with a
//!    membership change and the merge rule above runs.
//! 5. Everything reaches `a`, which replays.
//!
//! Expected: the merge resolves the removed member's position in the tree it
//! is merging into, or rejects an index it cannot honour.
//!
//! Actual, on this commit: the merge indexes `leaves` at 4 while it is 4 long.
//! Debug builds panic in `BeeKem::leaf` with `Leaf index should be in bounds`;
//! release builds compile the `debug_assert!` out and panic one line later in
//! `blank_leaf_and_path` with `index out of bounds: the len is 4 but the index
//! is 4`. The panic is not recoverable by the caller: it happens inside the
//! deterministic replay of a durable operation graph, so every replica holding
//! those operations panics the same way on every call that replays.
//!
//! The scenario is seeded and the test sweeps a fixed set of seeds, because
//! which order an epoch takes is decided by operation digests.
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
use alloc::{
    collections::BTreeMap, format, string::String, string::ToString, sync::Arc, vec, vec::Vec,
};
use future_form::Local;
use keyhive_crypto::{
    share_key::{ShareKey, ShareSecretKey},
    signed::Signed,
    signer::memory::MemorySigner,
    verifiable::Verifiable,
};
use rand::{rngs::StdRng, SeedableRng};
use std::panic::{catch_unwind, AssertUnwindSafe};

const SEED_BASE: u64 = 0x5ca1_e000;
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

/// Replicas of one tree, each holding its own [`Cgka`]. Operations are created
/// once and delivered explicitly, so a replica can lag behind by exactly the
/// operations the scenario withholds from it.
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
            let everyone: Vec<usize> = (0..n).collect();
            group.deliver(&Arc::new(op), &everyone);
        }
        group
    }

    fn deliver(&mut self, op: &Arc<Signed<CgkaOperation>>, to: &[usize]) {
        for &i in to {
            self.replicas[i]
                .merge_concurrent_operation(op.clone())
                .expect("operations are delivered in causal order");
        }
    }
}

async fn scenario(seed: u64) {
    let mut rng = StdRng::seed_from_u64(seed);
    let mut group = Group::found(4, &mut rng).await;
    let (c, d) = (group.members[2].id, group.members[3].id);
    let everyone = &[0usize, 1, 2, 3][..];

    // 1. `a` removes the middle member `c`: leaf 2 becomes a hole, the
    //    trailing slot 3 stays occupied, and the leaves vec stays 4 long.
    let remove_c = group.replicas[0]
        .remove::<Local, _>(c, &group.members[0].signer)
        .await
        .expect("authoring the removal succeeds")
        .expect("the removed member is present");
    group.deliver(&Arc::new(remove_c), everyone);

    // 2. Concurrently: `a` removes the trailing member `d`, and `b` — who has
    //    not received that removal — adds `e`. `b` therefore places `e` at
    //    leaf 4; the merged history packs it into the hole at leaf 2.
    let remove_d = group.replicas[0]
        .remove::<Local, _>(d, &group.members[0].signer)
        .await
        .expect("authoring the removal succeeds")
        .expect("the removed member is present");
    let e = member(&mut rng);
    let add_e = group.replicas[1]
        .add::<Local, _>(e.id, e.pk, &group.members[1].signer)
        .await
        .expect("authoring the add succeeds")
        .expect("the added member is new");
    group.deliver(&Arc::new(remove_d), &[0]);
    group.deliver(&Arc::new(add_e), &[0, 1]);

    // 3. Still unaware of the removal, `b` removes `e` again. The operation
    //    records the leaf index `b` sees, which is 4.
    let remove_e = group.replicas[1]
        .remove::<Local, _>(e.id, &group.members[1].signer)
        .await
        .expect("authoring the removal succeeds")
        .expect("the removed member is present");

    // 4. Concurrently, `a` adds `f`, so the removal above shares an epoch with
    //    a membership change.
    let f = member(&mut rng);
    let _add_f = group.replicas[0]
        .add::<Local, _>(f.id, f.pk, &group.members[0].signer)
        .await
        .expect("authoring the add succeeds")
        .expect("the added member is new");

    // 5. `b`'s removal reaches `a`, and `a` replays.
    group.deliver(&Arc::new(remove_e), &[0]);
    let sk = ShareSecretKey::generate(&mut rng);
    let pk = sk.share_key();
    let _ = group.replicas[0]
        .update::<Local, _, StdRng>(pk, sk, &group.members[0].signer, &mut rng)
        .await;
}

#[test]
fn recorded_leaf_index_of_a_removal_must_not_index_past_the_merged_tree() {
    let mut stale_index: Vec<(u64, String)> = Vec::new();
    // The one known co-located fault: the concurrent-add/trailing-removal
    // assertion, reported separately as its own issue. Only this exact message
    // earns the "same slot-bookkeeping family" label below.
    let mut family: BTreeMap<String, usize> = BTreeMap::new();
    // Anything else is an UNRELATED panic and must not borrow that label; its
    // presence fails the test on its own terms.
    let mut unexpected: BTreeMap<String, usize> = BTreeMap::new();

    for seed in 0..SEEDS {
        let seed = SEED_BASE + seed;
        let outcome = catch_unwind(AssertUnwindSafe(|| {
            tokio::runtime::Builder::new_current_thread()
                .build()
                .expect("building a test runtime succeeds")
                .block_on(scenario(seed))
        }));
        if let Err(payload) = outcome {
            let message = payload
                .downcast_ref::<String>()
                .cloned()
                .or_else(|| payload.downcast_ref::<&str>().map(|s| s.to_string()))
                .unwrap_or_else(|| "panic with a non-string payload".to_string());
            let message = message.lines().next().unwrap_or("").to_string();
            // Debug builds stop in `BeeKem::leaf`; release builds compile the
            // `debug_assert!` out and stop in `blank_leaf_and_path`.
            if message.contains("Leaf index should be in bounds")
                || message.contains("index out of bounds")
            {
                stale_index.push((seed, message));
            } else if message.contains("self.leaf(leaf_idx).is_none()") {
                *family.entry(message).or_insert(0) += 1;
            } else {
                *unexpected.entry(message).or_insert(0) += 1;
            }
        }
    }

    let mut report = String::new();
    if let Some((seed, message)) = stale_index.first() {
        report.push_str(&format!("\n  first seed {seed:#x}: {message}"));
    }
    for (message, count) in &family {
        report.push_str(&format!(
            "\n  ({count} further seed(s) stopped earlier, in the same \
             slot-bookkeeping family, with: {message})"
        ));
    }
    for (message, count) in &unexpected {
        report.push_str(&format!(
            "\n  ({count} seed(s) hit an UNRELATED panic, not this family: \
             {message})"
        ));
    }
    assert!(
        stale_index.is_empty() && unexpected.is_empty(),
        "a Remove operation's recorded leaf index was used unchecked against \
         the merged tree and indexed past its leaves vec on {} of {SEEDS} \
         seeds{report}",
        stale_index.len()
    );
}
