//! Regression test: concurrent removals must not wedge a group permanently.
//!
//! # The claim
//!
//! `Cgka::remove` protects group liveness at **authoring** time: it refuses to
//! remove the last member (`if self.group_size() == 1 { return
//! Err(CgkaError::RemoveLastMember) }`), and `BeeKem::remove_id` carries the
//! same guard.
//!
//! There is no equivalent protection at **replay**. Removals authored
//! concurrently are each locally valid — every author saw a larger group — but
//! the merge applies an epoch's removals one after another in digest order, so
//! the tree can be walked down through a one-member state. The next removal in
//! the epoch then hits `BeeKem::remove_id`'s `member_count() == 1` arm and the
//! whole materialization fails with `CgkaError::RemoveLastMember`.
//!
//! Because the operation graph is durable and the epoch order is derived from
//! operation digests rather than from delivery order, this is not a transient
//! failure of one replica: every replica that holds those operations re-derives
//! the same order and re-produces the same error on every subsequent call,
//! forever. The group is wedged. Nothing can be authored against it again —
//! including the obvious repair, adding a member back, because `Cgka::add`
//! replays the graph before it does anything else.
//!
//! Scenario: a three-member group `{a, b, c}` in which each member
//! concurrently removes one other member — `a` removes `b`, `b` removes `c`,
//! `c` removes `a`. Each author intends to remain in the group and each
//! removal is valid where it was authored.
//!
//! Expected: the merged history has *some* defined outcome that replicas can
//! materialize.
//!
//! Actual, on this commit: every replica errors `RemoveLastMember`, on every
//! call, permanently. The failure is order-independent, so the test does not
//! depend on which digest order an epoch happens to take: it sweeps a small
//! fixed set of seeds and every one of them wedges.
//!
//! The test opens with a sequential control over the same removals, so that
//! its claim is about concurrency rather than about emptying a group: applied
//! one at a time, the authoring guard refuses the last removal cleanly, the
//! refusal leaves the group untouched, and the group can still make progress
//! afterwards. The control passes on this commit.
//!
//! Found by property-based conformance testing of a downstream vendored copy
//! of this crate.

extern crate std;

use crate::{
    cgka::Cgka,
    error::CgkaError,
    id::{MemberId, TreeId},
    keys::ShareKeyMap,
    operation::CgkaOperation,
};
use alloc::{format, string::String, sync::Arc, vec, vec::Vec};
use future_form::Local;
use keyhive_crypto::{
    share_key::{ShareKey, ShareSecretKey},
    signed::Signed,
    signer::memory::MemorySigner,
    verifiable::Verifiable,
};
use rand::{rngs::StdRng, SeedableRng};

const SEED_BASE: u64 = 0xd1ff_0000;
const SEEDS: u64 = 8;

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
/// once and delivered to every replica, so digest identity is shared as it is
/// on the wire.
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

/// Control: the same removals applied one at a time, with every replica in
/// sync. The guard in `Cgka::remove` refuses the last one cleanly and
/// recoverably, and the group goes on working. That is the behaviour the
/// concurrent case below loses, and it is why this test is not simply
/// asserting that an emptied group must survive.
async fn sequential_control(seed: u64) -> Result<(), String> {
    let mut rng = StdRng::seed_from_u64(seed);
    let mut group = Group::found(3, &mut rng).await;
    let (a, b, c) = (
        group.members[0].id,
        group.members[1].id,
        group.members[2].id,
    );

    for victim in [b, c] {
        let op = group.replicas[0]
            .remove::<Local, _>(victim, &group.members[0].signer)
            .await
            .map_err(|e| format!("a removal that leaves members behind failed: {e:?}"))?
            .expect("the removed member is present");
        group.broadcast(&Arc::new(op));
    }

    // `a` is the last member: the authoring guard refuses, and says so.
    match group.replicas[0]
        .remove::<Local, _>(a, &group.members[0].signer)
        .await
    {
        Err(CgkaError::RemoveLastMember) => {}
        other => return Err(format!("expected the authoring guard, got {other:?}")),
    }

    // ...and the group is untouched by the refusal: it can still make progress.
    let joiner = member(&mut rng);
    group.replicas[0]
        .add::<Local, _>(joiner.id, joiner.pk, &group.members[0].signer)
        .await
        .map_err(|e| format!("the group could not make progress afterwards: {e:?}"))?;
    Ok(())
}

/// One run of the scenario. Returns the error every replica is left in, if any.
async fn scenario(seed: u64) -> Vec<(usize, CgkaError)> {
    let mut rng = StdRng::seed_from_u64(seed);
    let mut group = Group::found(3, &mut rng).await;
    let (a, b, c) = (
        group.members[0].id,
        group.members[1].id,
        group.members[2].id,
    );

    // Concurrently, from the same three-member state: each member removes one
    // other member and expects to remain.
    let mut removals = Vec::new();
    for (author, victim) in [(0usize, b), (1, c), (2, a)] {
        let op = group.replicas[author]
            .remove::<Local, _>(victim, &group.members[author].signer)
            .await
            .expect("authoring the removal succeeds")
            .expect("the removed member is present");
        removals.push(Arc::new(op));
    }
    for op in &removals {
        let op = op.clone();
        group.broadcast(&op);
    }

    // Try to make progress on every replica. Adding a member is the natural
    // repair for a group that has lost members, and it replays the graph
    // first, so it is also the shortest path to the materialization error.
    let mut wedged = Vec::new();
    for i in 0..group.replicas.len() {
        let joiner = member(&mut rng);
        let signer = &group.members[i].signer;
        if let Err(e) = group.replicas[i]
            .add::<Local, _>(joiner.id, joiner.pk, signer)
            .await
        {
            wedged.push((i, e));
        }
    }
    wedged
}

#[test]
fn concurrent_removals_must_not_permanently_wedge_the_group() {
    let runtime = || {
        tokio::runtime::Builder::new_current_thread()
            .build()
            .expect("building a test runtime succeeds")
    };

    // Sequentially, the authoring guard does its job.
    if let Err(e) = runtime().block_on(sequential_control(SEED_BASE)) {
        panic!("the sequential control did not behave as expected: {e}");
    }

    let mut wedged: Vec<(u64, String)> = Vec::new();

    for seed in 0..SEEDS {
        let seed = SEED_BASE + seed;
        let outcome = runtime().block_on(scenario(seed));
        if !outcome.is_empty() {
            let detail = outcome
                .iter()
                .map(|(i, e)| format!("replica {i}: {e:?}"))
                .collect::<Vec<_>>()
                .join(", ");
            wedged.push((seed, detail));
        }
    }

    let mut report = String::new();
    for (seed, detail) in wedged.iter().take(2) {
        report.push_str(&format!("\n  seed {seed:#x}: {detail}"));
    }
    assert!(
        wedged.is_empty(),
        "three concurrent removals in a three-member group left the tree \
         unmaterializable on {} of {SEEDS} seeds; the error is re-derived from \
         the durable operation graph on every replica and every call, so the \
         group can never make progress again{report}",
        wedged.len()
    );
}
