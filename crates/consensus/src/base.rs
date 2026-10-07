// Copyright (c) Mysten Labs, Inc.
// SPDX-License-Identifier: Apache-2.0

use std::{
    borrow::Borrow,
    collections::HashMap,
    fmt::{self, Display},
    sync::{Arc, Mutex},
};

use crate::{
    leader::LeaderElector,
    protocol::{FastPath, Protocol},
    wave::Wave,
};
use dag::{
    authority::Authority,
    block::{Block, BlockReference, RoundNumber},
    committee::Stake,
    committee::{Committee, StakeAggregator},
    consensus::{DirectCommitPath, IndirectCommitPath, LeaderStatus},
    crypto::coin,
    data::Data,
    storage::BlockReader,
};

/// The [`BaseCommitter`] contains the bare bone commit
/// logic. Once instantiated, the methods
/// `try_direct_decide` and `try_indirect_decide` can be
/// called at any time and any number of times
/// (idempotent) to determine whether a leader can be
/// committed or skipped.
pub(crate) struct BaseCommitter {
    committee: Arc<Committee>,
    block_reader: BlockReader,
    leader_elector: LeaderElector,
    direct_commit_quorum: Stake,
    direct_skip_quorum: Stake,
    certificate_quorum: Stake,
    fast_path: Option<FastPath>,
    anchor_link_size: Stake,
    pub(crate) wave: Wave,
    leader_offset: RoundNumber,
    /// Seeds already combined from a quorum of coin shares, by round. By BLS uniqueness any
    /// quorum yields the same value, so once a round has one it never changes and the
    /// (expensive) combination is done once per round instead of on every `elect_leader`.
    coin_seeds: Mutex<HashMap<RoundNumber, RoundNumber>>,
    /// Whether leaders come from the common coin (strict): a wave's leader is unknown until a
    /// quorum of coin shares exists at the wave's decision round, and `elect_leader` returns
    /// `None` until then. Off by default, in which case leaders are round-robin.
    coin_election: bool,
}

impl BaseCommitter {
    pub(crate) fn new(
        committee: Arc<Committee>,
        block_reader: BlockReader,
        leader_elector: LeaderElector,
        protocol: &Protocol,
        leader_offset: RoundNumber,
        round_offset: RoundNumber,
    ) -> Self {
        Self {
            committee,
            block_reader,
            leader_elector,
            direct_commit_quorum: protocol.direct_commit_quorum,
            direct_skip_quorum: protocol.direct_skip_quorum,
            certificate_quorum: protocol.certificate_quorum,
            fast_path: protocol.fast_path,
            anchor_link_size: protocol.anchor_link_size,
            wave: Wave::new(
                protocol.wave_length,
                round_offset,
                protocol.merged_certificates,
            ),
            leader_offset,
            coin_seeds: Mutex::new(HashMap::new()),
            coin_election: false,
        }
    }

    /// Elect leaders through the common coin instead of round-robin (see `coin_election`).
    pub(crate) fn with_coin_election(mut self) -> Self {
        self.coin_election = true;
        self
    }

    /// True iff `round` is a leader round whose leader is not yet known: coin election is on
    /// and fewer than a quorum of shares exist at the wave's decision round.
    pub(crate) fn leader_pending(&self, round: RoundNumber) -> bool {
        self.coin_election
            && self.wave.is_leader_round(round)
            && self.coin_based_seed(self.coin_round(round)).is_none()
    }

    /// The round whose shares reveal the leader of leader round `round`: the last round of
    /// its wave. An adversary cannot predict the leader before that round's blocks exist.
    fn coin_round(&self, round: RoundNumber) -> RoundNumber {
        self.wave.decision_round(self.wave.number(round))
    }

    #[cfg(test)]
    fn new_for_test(
        committee: &Arc<Committee>,
        block_reader: BlockReader,
        wave_length: RoundNumber,
        leader_offset: RoundNumber,
    ) -> Self {
        let total = committee.total_stake();
        let quorum = 2 * total / 3 + 1;
        let protocol = Protocol {
            direct_commit_quorum: quorum,
            direct_skip_quorum: quorum,
            certificate_quorum: quorum,
            quorum_threshold: quorum,
            fast_path: None,
            anchor_link_size: 1,
            wave_length,
            merged_certificates: wave_length == 2,
            leader_count: std::num::NonZeroUsize::new(1).unwrap(),
            pipeline: false,
            leader_wait: false,
            require_crypto: false,
        };
        BaseCommitter::new(
            committee.clone(),
            block_reader,
            LeaderElector::new(committee.len()),
            &protocol,
            leader_offset,
            0,
        )
    }

    /// Enable the optimistic fast path on a test committer.
    #[cfg(test)]
    fn with_fast_path(mut self, fast_path: FastPath) -> Self {
        self.fast_path = Some(fast_path);
        self
    }

    /// The leader-elect protocol is offset by `leader_offset` to ensure that different
    /// committers with different leader offsets elect different leaders for the same round number.
    /// This function returns `None` if there is no leader for the specified round, or, under
    /// coin election, if the leader is not yet known (see [`leader_pending`](Self::leader_pending)).
    ///
    /// Without coin election the seed is the round number (round-robin). With it, the seed is
    /// the common coin of the wave's decision round, and there is no fallback: until a quorum of
    /// shares exists at that round no node elects a leader, so no two nodes can disagree.
    /// `leader_offset` still shifts which authority in the cohort this committer elects.
    pub(crate) fn elect_leader(&self, round: RoundNumber) -> Option<Authority> {
        if !self.wave.is_leader_round(round) {
            return None;
        }
        let seed = if self.coin_election {
            self.coin_based_seed(self.coin_round(round))?
        } else {
            round
        };
        let offset = self.leader_offset as RoundNumber;
        Some(self.leader_elector.elect_leader(seed + offset))
    }

    /// Combine the coin shares that round `round`'s blocks carry into that round's common-coin
    /// seed. Returns `None` when fewer than a quorum of distinct authorities have a share there
    /// yet. One share per author: an equivocator's twin blocks carry the same share.
    fn coin_based_seed(&self, round: RoundNumber) -> Option<RoundNumber> {
        if let Some(seed) = self.coin_seeds.lock().unwrap().get(&round) {
            return Some(*seed);
        }
        let threshold = coin::quorum_threshold(&self.committee);
        let mut authors = std::collections::HashSet::new();
        let shares: Vec<_> = self
            .block_reader
            .get_blocks_by_round(round)
            .iter()
            .filter(|block| block.coin_share().is_some() && authors.insert(block.author()))
            .filter_map(|block| block.coin_share().cloned())
            .collect();
        if shares.len() < threshold as usize {
            return None;
        }
        let combined = coin::combine(threshold, shares.into_iter()).ok()?;
        let seed = coin::coin_seed(&combined);
        self.coin_seeds.lock().unwrap().insert(round, seed);
        Some(seed)
    }

    /// Find which block is supported at (author, round) by the given block.
    /// Blocks can indirectly reference multiple other blocks at (author, round), but only one
    /// block at (author, round) will be supported by the given block.
    fn find_support(
        &self,
        (author, round): (Authority, RoundNumber),
        from: &Data<Block>,
    ) -> Option<BlockReference> {
        if from.round() < round {
            return None;
        }
        for include in from.includes() {
            // Weak links may point to blocks with lower round numbers than strong links.
            if include.round() < round {
                continue;
            }
            if include.author_round() == (author, round) {
                return Some(*include);
            }
            let include = self
                .block_reader
                .get_block(*include)
                .expect("We should have the whole sub-dag by now");
            if let Some(support) = self.find_support((author, round), &include) {
                return Some(support);
            }
        }
        None
    }

    /// Check whether the specified block (`potential_certificate`) is a vote for
    /// the specified leader (`leader_block`).
    fn is_vote(&self, potential_vote: &Data<Block>, leader_block: &Data<Block>) -> bool {
        let (author, round) = leader_block.author_round();
        self.find_support((author, round), potential_vote) == Some(*leader_block.reference())
    }

    /// Check whether `votes` holds a `quorum` of distinct-author votes for
    /// `leader_block`, i.e. whether they form a certificate for it.
    fn is_certificate<I, B>(&self, votes: I, leader_block: &Data<Block>, quorum: Stake) -> bool
    where
        I: IntoIterator<Item = B>,
        B: Borrow<Data<Block>>,
    {
        let mut votes_stake_aggregator = StakeAggregator::new(quorum);
        for vote in votes {
            let vote = vote.borrow();
            if self.is_vote(vote, leader_block) {
                // Merged protocols never traced per vote; keep their hot path unchanged.
                if !self.wave.merged_certificates() {
                    tracing::trace!("[{self}] {vote:?} is a vote for {leader_block:?}");
                }
                if votes_stake_aggregator.add(vote.author(), &self.committee) {
                    return true;
                }
            }
        }
        false
    }

    /// Check whether the specified decision-round block references a certificate for
    /// `leader_block` (the vote-and-certify pattern).
    fn carries_certificate(
        &self,
        decision_block: &Data<Block>,
        leader_block: &Data<Block>,
    ) -> bool {
        let votes = decision_block.includes().iter().map(|reference| {
            self.block_reader
                .get_block(*reference)
                .expect("We should have the whole sub-dag by now")
        });
        if self.is_certificate(votes, leader_block, self.certificate_quorum) {
            tracing::trace!(
                "[{self}] {decision_block:?} carries a certificate for {leader_block:?}"
            );
            return true;
        }
        false
    }

    /// Decide the status of a target leader from the specified anchor, applying the
    /// graded indirect rule: commit the leader block with a certified link to the
    /// anchor; else, for fast-path protocols, commit the leader block backed by a weak
    /// quorum of anchor-linked votes; else skip the target leader.
    fn decide_leader_from_anchor(
        &self,
        anchor: &Data<Block>,
        leader: Authority,
        leader_round: RoundNumber,
    ) -> LeaderStatus {
        // Get the block(s) proposed by the leader. There could be more than one leader block
        // per round (produced by a Byzantine leader).
        let leader_blocks = self
            .block_reader
            .get_blocks_at_authority_round(leader, leader_round);

        // Get all blocks that could be potential certificates for the target leader. These blocks
        // are in the decision round of the target leader and are linked to the anchor.
        let wave = self.wave.number(leader_round);
        let decision_round = self.wave.decision_round(wave);
        let decision_blocks = self.block_reader.get_blocks_by_round(decision_round);

        // Find the leader block whose certificate is linked to the anchor: with
        // merged certificates, an `anchor_link_size` quorum of anchor-linked votes;
        // otherwise, `anchor_link_size` anchor-linked certificate-carrying blocks.
        // A weak (sub-majority) quorum lets two equivocating twins both qualify,
        // so break ties deterministically by digest.
        let certified = leader_blocks.iter().filter(|leader_block| {
            if self.wave.merged_certificates() {
                let linked_votes = decision_blocks
                    .iter()
                    .filter(|block| self.block_reader.linked(anchor, block));
                return self.is_certificate(linked_votes, leader_block, self.anchor_link_size);
            }
            let mut aggregator = StakeAggregator::new(self.anchor_link_size);
            decision_blocks.iter().any(|block| {
                self.block_reader.linked(anchor, block)
                    && self.carries_certificate(block, leader_block)
                    && aggregator.add(block.author(), &self.committee)
            })
        });
        let commit = certified.min_by_key(|leader_block| leader_block.reference().digest);

        tracing::trace!("[{self}] leader {leader} is decided by anchor {anchor:?}");
        if let Some(block) = commit {
            return LeaderStatus::IndirectCommit(block.clone(), IndirectCommitPath::Certificate);
        }

        // Second rung (fast-path protocols only): commit the leader block backed by a
        // weak quorum of votes linked to the anchor. As on the certificate rung, two
        // equivocating leader blocks can both qualify (a slow commit does not starve
        // conflicts of votes), so ties are broken deterministically by digest.
        if let Some(fast_path) = &self.fast_path {
            let voting_round = self.wave.voting_round(wave);
            let voting_blocks = self.block_reader.get_blocks_by_round(voting_round);
            let weak_commit = leader_blocks
                .into_iter()
                .filter(|leader_block| {
                    let mut aggregator = StakeAggregator::new(fast_path.weak_indirect_quorum);
                    voting_blocks.iter().any(|block| {
                        self.block_reader.linked(anchor, block)
                            && self.is_vote(block, leader_block)
                            && aggregator.add(block.author(), &self.committee)
                    })
                })
                .min_by_key(|leader_block| leader_block.reference().digest);
            if let Some(block) = weak_commit {
                return LeaderStatus::IndirectCommit(block, IndirectCommitPath::WeakQuorum);
            }
        }

        LeaderStatus::IndirectSkip(leader, leader_round)
    }

    /// Check whether the specified leader has enough blames (`direct_skip_quorum` non-votes
    /// among the voting-round blocks) to be directly skipped.
    fn enough_leader_blame(
        &self,
        voting_blocks: &[Data<Block>],
        leader: Authority,
        leader_round: RoundNumber,
    ) -> bool {
        let mut aggregator = StakeAggregator::new(self.direct_skip_quorum);
        for voting_block in voting_blocks {
            let voter = voting_block.reference().authority;
            if self
                .find_support((leader, leader_round), voting_block)
                .is_none()
            {
                tracing::trace!(
                    "[{self}] {voting_block:?} is a blame for leader {}",
                    leader.with_round(leader_round)
                );
                if aggregator.add(voter, &self.committee) {
                    return true;
                }
            }
        }
        false
    }

    /// Check whether the specified leader has enough support to be directly
    /// committed: with merged certificates, a `direct_commit_quorum` certificate of
    /// decision-round votes; otherwise, `direct_commit_quorum` certificate-carrying
    /// blocks at the decision round.
    fn enough_leader_support(
        &self,
        decision_round: RoundNumber,
        leader_block: &Data<Block>,
    ) -> bool {
        let decision_blocks = self.block_reader.get_blocks_by_round(decision_round);

        if self.wave.merged_certificates() {
            let votes = decision_blocks.iter();
            return self.is_certificate(votes, leader_block, self.direct_commit_quorum);
        }

        let mut certificate_stake_aggregator = StakeAggregator::new(self.direct_commit_quorum);
        for decision_block in &decision_blocks {
            let authority = decision_block.reference().authority;
            if self.carries_certificate(decision_block, leader_block)
                && certificate_stake_aggregator.add(authority, &self.committee)
            {
                return true;
            }
        }
        false
    }

    /// Check whether the specified leader has enough votes at the voting round to be
    /// fast-committed. Always false for protocols without a fast path.
    fn enough_fast_path_support(
        &self,
        voting_blocks: &[Data<Block>],
        leader_block: &Data<Block>,
    ) -> bool {
        let Some(fast_path) = &self.fast_path else {
            return false;
        };
        let mut votes_stake_aggregator = StakeAggregator::new(fast_path.commit_quorum);
        for voting_block in voting_blocks {
            let authority = voting_block.reference().authority;
            if self.is_vote(voting_block, leader_block)
                && votes_stake_aggregator.add(authority, &self.committee)
            {
                return true;
            }
        }
        false
    }

    /// Apply the indirect decision rule to the specified leader to see whether we can
    /// indirect-commit or indirect-skip it.
    #[tracing::instrument(
        level = "debug",
        skip_all,
        fields(leader = %leader.with_round(leader_round))
    )]
    pub(crate) fn try_indirect_decide<'a>(
        &self,
        leader: Authority,
        leader_round: RoundNumber,
        leaders: impl Iterator<Item = &'a LeaderStatus>,
    ) -> LeaderStatus {
        // The anchor is the first committed leader with round higher than the decision round of the
        // target leader. We must stop the iteration upon encountering an undecided leader.
        let anchors = leaders.filter(|x| leader_round + self.wave.length() <= x.round());

        for anchor in anchors {
            tracing::trace!(
                "[{self}] Trying to indirect-decide {} using anchor {anchor}",
                leader.with_round(leader_round),
            );
            match anchor {
                LeaderStatus::DirectCommit(anchor, _) | LeaderStatus::IndirectCommit(anchor, _) => {
                    return self.decide_leader_from_anchor(anchor, leader, leader_round);
                }
                LeaderStatus::DirectSkip(..) | LeaderStatus::IndirectSkip(..) => (),
                LeaderStatus::Undecided(..) => break,
            }
        }

        LeaderStatus::Undecided(leader, leader_round)
    }

    /// Apply the direct decision rule to the specified leader to see whether we can direct-commit
    /// or direct-skip it.
    #[tracing::instrument(
        level = "debug",
        skip_all,
        fields(leader = %leader.with_round(leader_round))
    )]
    pub(crate) fn try_direct_decide(
        &self,
        leader: Authority,
        leader_round: RoundNumber,
    ) -> LeaderStatus {
        let wave = self.wave.number(leader_round);
        let voting_round = self.wave.voting_round(wave);
        let decision_round = self.wave.decision_round(wave);

        let voting_blocks = self.block_reader.get_blocks_by_round(voting_round);

        // Check whether the leader has enough blame. That is, whether there are
        // `direct_skip_quorum` non-votes for that leader (which ensure there will never
        // be a certificate for that leader).
        if self.enough_leader_blame(&voting_blocks, leader, leader_round) {
            return LeaderStatus::DirectSkip(leader, leader_round);
        }

        // Check whether the leader(s) has enough support: `fast_path.commit_quorum`
        // votes at the voting round (fast path, when configured) or
        // `direct_commit_quorum` certificates at the decision round. Note that there
        // could be more than one leader block (created by Byzantine leaders).
        let leader_blocks = self
            .block_reader
            .get_blocks_at_authority_round(leader, leader_round);
        let mut supported = leader_blocks.into_iter().filter_map(|leader_block| {
            if self.enough_fast_path_support(&voting_blocks, &leader_block) {
                Some((leader_block, DirectCommitPath::Fast))
            } else if self.enough_leader_support(decision_round, &leader_block) {
                Some((leader_block, DirectCommitPath::Slow))
            } else {
                None
            }
        });
        let first = supported.next();
        if supported.next().is_some() {
            panic!(
                "[{self}] More than one certified block for {}",
                leader.with_round(leader_round)
            )
        }

        first
            .map(|(block, path)| LeaderStatus::DirectCommit(block, path))
            .unwrap_or_else(|| LeaderStatus::Undecided(leader, leader_round))
    }
}

impl Display for BaseCommitter {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(
            f,
            "Committer-L{}-R{}",
            self.leader_offset,
            self.wave.round_offset(),
        )
    }
}

#[cfg(test)]
mod tests {

    use std::num::NonZeroUsize;

    use dag::{
        authority::Authority,
        block::Block,
        consensus::{DirectCommitPath, IndirectCommitPath, LeaderStatus},
        crypto::{
            BlockDigest,
            coin::{self, mocked_committee_setup},
        },
        data::Data,
        storage::Storage,
        test_util::{build_dag, build_dag_layer, committee, insert_test_block},
    };

    use crate::{
        base::BaseCommitter,
        leader::LeaderElector,
        protocol::{FastPath, Protocol},
    };

    /// `elect_leader` returns `None` outside leader rounds.
    #[test]
    fn elect_leader_none_at_non_leader_round() {
        let committee = committee(4);
        let storage = Storage::new_for_test(&committee);
        let committer =
            BaseCommitter::new_for_test(&committee, storage.block_reader().clone(), 3, 0);
        // wave_length = 3, round_offset = 0 → leader rounds {0, 3, 6, ...}.
        assert!(committer.elect_leader(0).is_some());
        assert!(committer.elect_leader(1).is_none());
        assert!(committer.elect_leader(2).is_none());
        assert!(committer.elect_leader(3).is_some());
        assert!(committer.elect_leader(4).is_none());
        assert!(committer.elect_leader(6).is_some());
    }

    /// `leader_offset` shifts the elected authority within the round-robin schedule.
    #[test]
    fn elect_leader_offset_shifts_authority() {
        let committee = committee(4);
        let storage = Storage::new_for_test(&committee);
        let committer_offset_zero =
            BaseCommitter::new_for_test(&committee, storage.block_reader().clone(), 3, 0);
        let committer_offset_one =
            BaseCommitter::new_for_test(&committee, storage.block_reader().clone(), 3, 1);
        assert_ne!(
            committer_offset_zero.elect_leader(3),
            committer_offset_one.elect_leader(3),
        );
    }

    /// Insert a block per listed author at `round`, each carrying its share for that round.
    fn insert_share_blocks(
        storage: &mut Storage,
        key_shares: &[coin::CoinKeyShare],
        round: dag::block::RoundNumber,
        authors: &[usize],
    ) {
        for &i in authors {
            let block = Block::new_for_test(Authority::from(i as u64), round, vec![])
                .with_coin_share(key_shares[i].sign_round(round));
            insert_test_block(block, storage);
        }
    }

    /// Under coin election the leader of a wave is seeded by the combined coin of the wave's
    /// decision round (round 2 for wave length 3), not by round-robin.
    #[test]
    fn elect_leader_uses_coin_of_the_decision_round() {
        let committee = committee(4);
        let mut storage = Storage::new_for_test(&committee);
        let key_shares = mocked_committee_setup(&committee);
        insert_share_blocks(&mut storage, &key_shares, 2, &[0, 1, 2, 3]);

        let committer =
            BaseCommitter::new_for_test(&committee, storage.block_reader().clone(), 3, 0)
                .with_coin_election();

        let threshold = coin::quorum_threshold(&committee);
        let shares = key_shares.iter().map(|k| k.sign_round(2));
        let combined = coin::combine(threshold, shares).expect("quorum of shares present");
        let expected_seed = coin::coin_seed(&combined);
        let expected_leader = Authority::new(expected_seed % committee.len() as u64);

        assert_eq!(committer.elect_leader(0), Some(expected_leader));
        assert!(!committer.leader_pending(0));
    }

    /// Strict election: below a quorum of shares at the decision round there is no leader and
    /// no fallback. Shares at the leader round itself reveal nothing.
    #[test]
    fn elect_leader_is_strict_until_the_decision_round_has_a_quorum() {
        let committee = committee(4);
        let mut storage = Storage::new_for_test(&committee);
        let key_shares = mocked_committee_setup(&committee);

        // A full quorum of shares at the leader round (0), but only two at the decision round.
        insert_share_blocks(&mut storage, &key_shares, 0, &[0, 1, 2, 3]);
        insert_share_blocks(&mut storage, &key_shares, 2, &[0, 1]);

        let committer =
            BaseCommitter::new_for_test(&committee, storage.block_reader().clone(), 3, 0)
                .with_coin_election();
        assert_eq!(committer.elect_leader(0), None);
        assert!(committer.leader_pending(0));
        // Not a leader round: not pending either.
        assert!(!committer.leader_pending(1));

        // The third share arrives: the leader becomes known.
        insert_share_blocks(&mut storage, &key_shares, 2, &[2]);
        assert!(committer.elect_leader(0).is_some());
        assert!(!committer.leader_pending(0));
    }

    /// An equivocator's twin blocks carry the same share and count once, not twice.
    #[test]
    fn duplicate_authors_do_not_count_towards_the_coin_quorum() {
        let committee = committee(4);
        let mut storage = Storage::new_for_test(&committee);
        let key_shares = mocked_committee_setup(&committee);
        insert_share_blocks(&mut storage, &key_shares, 2, &[0, 1]);
        // A twin of author 1's block (different transactions, same share).
        let twin = Block::new_for_test(Authority::from(1u64), 2, vec![])
            .with_coin_share(key_shares[1].sign_round(2))
            .with_digest(BlockDigest::from([7u8; 32]));
        insert_test_block(twin, &mut storage);

        let committer =
            BaseCommitter::new_for_test(&committee, storage.block_reader().clone(), 3, 0)
                .with_coin_election();
        assert_eq!(committer.elect_leader(0), None);
    }

    /// Once a round's coin is combined it is cached: repeated elections agree and the round
    /// is stored once.
    #[test]
    fn combined_coin_is_cached_per_round() {
        let committee = committee(4);
        let mut storage = Storage::new_for_test(&committee);
        let key_shares = mocked_committee_setup(&committee);
        insert_share_blocks(&mut storage, &key_shares, 2, &[0, 1, 2, 3]);
        let committer =
            BaseCommitter::new_for_test(&committee, storage.block_reader().clone(), 3, 0)
                .with_coin_election();
        assert!(committer.coin_seeds.lock().unwrap().is_empty());
        let first = committer.elect_leader(0);
        assert_eq!(committer.coin_seeds.lock().unwrap().len(), 1);
        assert_eq!(committer.elect_leader(0), first);
        assert_eq!(committer.coin_seeds.lock().unwrap().len(), 1);
    }

    /// Without coin election leaders stay round-robin, whatever shares the blocks carry.
    #[test]
    fn without_coin_election_leaders_are_round_robin() {
        let committee = committee(4);
        let mut storage = Storage::new_for_test(&committee);
        let key_shares = mocked_committee_setup(&committee);
        insert_share_blocks(&mut storage, &key_shares, 2, &[0, 1, 2, 3]);
        let committer =
            BaseCommitter::new_for_test(&committee, storage.block_reader().clone(), 3, 0);
        assert_eq!(committer.elect_leader(0), Some(Authority::new(0)));
        assert!(!committer.leader_pending(0));
    }

    /// `find_support` returns the direct include when one points at the target slot.
    #[test]
    fn find_support_direct_include() {
        let committee = committee(4);
        let mut storage = Storage::new_for_test(&committee);
        build_dag(&committee, &mut storage, None, 2);
        let committer =
            BaseCommitter::new_for_test(&committee, storage.block_reader().clone(), 3, 0);

        let block_at_2 = storage
            .block_reader()
            .get_blocks_at_authority_round(Authority::from(0u64), 2)
            .into_iter()
            .next()
            .unwrap();

        let reference = committer
            .find_support((Authority::from(1u64), 1), &block_at_2)
            .expect("fully connected DAG must include round-1 (auth=1)");
        assert_eq!(reference.authority, Authority::from(1u64));
        assert_eq!(reference.round, 1);
    }

    /// `find_support` walks transitively through includes.
    #[test]
    fn find_support_transitive_walk() {
        let committee = committee(4);
        let mut storage = Storage::new_for_test(&committee);
        build_dag(&committee, &mut storage, None, 3);
        let committer =
            BaseCommitter::new_for_test(&committee, storage.block_reader().clone(), 3, 0);

        let block_at_3 = storage
            .block_reader()
            .get_blocks_at_authority_round(Authority::from(0u64), 3)
            .into_iter()
            .next()
            .unwrap();

        // The round-3 block's direct includes are at round 2; finding (1, 1) requires
        // walking through one of those round-2 blocks to locate the round-1 reference.
        let reference = committer
            .find_support((Authority::from(1u64), 1), &block_at_3)
            .expect("transitive walk must locate the round-1 (auth=1) block");
        assert_eq!(reference.authority, Authority::from(1u64));
        assert_eq!(reference.round, 1);
    }

    /// `find_support` returns `None` when the source block is below the target round.
    #[test]
    fn find_support_below_target_round_returns_none() {
        let committee = committee(4);
        let mut storage = Storage::new_for_test(&committee);
        build_dag(&committee, &mut storage, None, 1);
        let committer =
            BaseCommitter::new_for_test(&committee, storage.block_reader().clone(), 3, 0);

        let block_at_1 = storage
            .block_reader()
            .get_blocks_at_authority_round(Authority::from(0u64), 1)
            .into_iter()
            .next()
            .unwrap();

        assert!(
            committer
                .find_support((Authority::from(1u64), 2), &block_at_1)
                .is_none(),
        );
    }

    /// `is_certificate` aggregates distinct-author votes against the given quorum.
    #[test]
    fn is_certificate_counts_distinct_vote_quorum() {
        let committee = committee(4);
        let mut storage = Storage::new_for_test(&committee);
        build_dag(&committee, &mut storage, None, 3);
        let committer =
            BaseCommitter::new_for_test(&committee, storage.block_reader().clone(), 2, 0);

        let leader_block = storage
            .block_reader()
            .get_blocks_at_authority_round(Authority::from(0u64), 2)
            .into_iter()
            .next()
            .unwrap();
        let votes = storage.block_reader().get_blocks_by_round(3);

        // A fully-connected round of 4 votes reaches quorum 4 but not 5.
        assert!(committer.is_certificate(votes.iter(), &leader_block, 4));
        assert!(!committer.is_certificate(votes.iter(), &leader_block, 5));
    }

    /// At `wave_length = 3`, a fully-connected DAG produces a strong-quorum of votes
    /// under each decision-round block, so every decision-round block carries a
    /// certificate.
    #[test]
    fn carries_certificate_wave_length_three_full_dag() {
        let committee = committee(4);
        let mut storage = Storage::new_for_test(&committee);
        build_dag(&committee, &mut storage, None, 5);
        let committer =
            BaseCommitter::new_for_test(&committee, storage.block_reader().clone(), 3, 0);

        let leader_block = storage
            .block_reader()
            .get_blocks_at_authority_round(Authority::from(0u64), 3)
            .into_iter()
            .next()
            .unwrap();
        let certificate_candidate = storage
            .block_reader()
            .get_blocks_at_authority_round(Authority::from(1u64), 5)
            .into_iter()
            .next()
            .unwrap();

        assert!(committer.carries_certificate(&certificate_candidate, &leader_block));
    }

    /// `enough_leader_blame` is `true` when every non-leader voter omits the leader.
    #[test]
    fn enough_leader_blame_when_all_non_leader_voters_blame() {
        let committee = committee(4);
        let mut storage = Storage::new_for_test(&committee);

        let leader_round = 3;
        let references_at_leader = build_dag(&committee, &mut storage, None, leader_round);
        let leader = Authority::from(0u64);
        let references_without_leader: Vec<_> = references_at_leader
            .into_iter()
            .filter(|reference| reference.authority != leader)
            .collect();
        let connections = committee
            .authorities()
            .filter(|&authority| authority != leader)
            .map(|authority| (authority, references_without_leader.clone()))
            .collect();
        build_dag_layer(connections, &mut storage);

        let committer =
            BaseCommitter::new_for_test(&committee, storage.block_reader().clone(), 3, 0);
        let voting_blocks = storage.block_reader().get_blocks_by_round(4);
        assert!(committer.enough_leader_blame(&voting_blocks, leader, leader_round));
    }

    /// `enough_leader_support` is `true` for a fully-connected DAG.
    #[test]
    fn enough_leader_support_when_full_dag() {
        let committee = committee(4);
        let mut storage = Storage::new_for_test(&committee);
        build_dag(&committee, &mut storage, None, 5);
        let committer =
            BaseCommitter::new_for_test(&committee, storage.block_reader().clone(), 3, 0);

        let leader = Authority::from(0u64);
        let leader_block = storage
            .block_reader()
            .get_blocks_at_authority_round(leader, 3)
            .into_iter()
            .next()
            .unwrap();
        assert!(committer.enough_leader_support(5, &leader_block));
    }

    /// `try_direct_decide` issues `DirectCommit` on a fully-connected DAG.
    #[test]
    fn try_direct_decide_commits_on_full_dag() {
        let committee = committee(4);
        let mut storage = Storage::new_for_test(&committee);
        build_dag(&committee, &mut storage, None, 5);
        let committer =
            BaseCommitter::new_for_test(&committee, storage.block_reader().clone(), 3, 0);

        let leader = committer.elect_leader(3).unwrap();
        match committer.try_direct_decide(leader, 3) {
            LeaderStatus::DirectCommit(block, path) => {
                assert_eq!(block.author(), leader);
                assert_eq!(path, DirectCommitPath::Slow);
            }
            other => panic!("expected DirectCommit, got {other:?}"),
        }
    }

    /// `try_direct_decide` issues `DirectSkip` when the leader is unanimously blamed.
    #[test]
    fn try_direct_decide_skips_when_leader_omitted() {
        let committee = committee(4);
        let mut storage = Storage::new_for_test(&committee);

        let references_at_leader = build_dag(&committee, &mut storage, None, 3);
        // Round-robin: round 3 → authority 3.
        let leader = committee.authorities().nth(3).unwrap();
        let references_without_leader: Vec<_> = references_at_leader
            .into_iter()
            .filter(|reference| reference.authority != leader)
            .collect();
        let voting_connections = committee
            .authorities()
            .filter(|&authority| authority != leader)
            .map(|authority| (authority, references_without_leader.clone()))
            .collect();
        let references_at_voting_round = build_dag_layer(voting_connections, &mut storage);
        build_dag(
            &committee,
            &mut storage,
            Some(references_at_voting_round),
            5,
        );

        let committer =
            BaseCommitter::new_for_test(&committee, storage.block_reader().clone(), 3, 0);
        match committer.try_direct_decide(leader, 3) {
            LeaderStatus::DirectSkip(skipped, round) => {
                assert_eq!(skipped, leader);
                assert_eq!(round, 3);
            }
            other => panic!("expected DirectSkip, got {other:?}"),
        }
    }

    /// `try_direct_decide` reports `Undecided` when neither support nor blame reaches quorum.
    #[test]
    fn try_direct_decide_undecided_on_insufficient_dag() {
        let committee = committee(4);
        let mut storage = Storage::new_for_test(&committee);
        // DAG only reaches the leader round → nothing has voted yet.
        build_dag(&committee, &mut storage, None, 3);
        let committer =
            BaseCommitter::new_for_test(&committee, storage.block_reader().clone(), 3, 0);

        let leader = committer.elect_leader(3).unwrap();
        match committer.try_direct_decide(leader, 3) {
            LeaderStatus::Undecided(authority, round) => {
                assert_eq!(authority, leader);
                assert_eq!(round, 3);
            }
            other => panic!("expected Undecided, got {other:?}"),
        }
    }

    /// With a fast path, a quorum of votes at the voting round direct-commits the
    /// leader even though the decision round is absent; without a fast path the same
    /// DAG stays `Undecided`.
    #[test]
    fn try_direct_decide_fast_commits_at_voting_round() {
        let committee = committee(4);
        let mut storage = Storage::new_for_test(&committee);
        // DAG stops at the voting round: votes exist, certificates cannot.
        build_dag(&committee, &mut storage, None, 4);

        let fast_committer =
            BaseCommitter::new_for_test(&committee, storage.block_reader().clone(), 3, 0)
                .with_fast_path(FastPath {
                    commit_quorum: 3,
                    weak_indirect_quorum: 2,
                });
        let slow_committer =
            BaseCommitter::new_for_test(&committee, storage.block_reader().clone(), 3, 0);

        let leader = fast_committer.elect_leader(3).unwrap();
        match fast_committer.try_direct_decide(leader, 3) {
            LeaderStatus::DirectCommit(block, path) => {
                assert_eq!(block.author(), leader);
                assert_eq!(path, DirectCommitPath::Fast);
            }
            other => panic!("expected DirectCommit, got {other:?}"),
        }
        match slow_committer.try_direct_decide(leader, 3) {
            LeaderStatus::Undecided(authority, round) => {
                assert_eq!(authority, leader);
                assert_eq!(round, 3);
            }
            other => panic!("expected Undecided, got {other:?}"),
        }
    }

    /// With no certificate at the decision round, a weak quorum of anchor-linked votes
    /// indirect-commits the leader (second rung of the graded rule); without a fast
    /// path the same DAG indirect-skips.
    #[test]
    fn decide_leader_from_anchor_weak_quorum_commits() {
        let committee = committee(4);
        let mut storage = Storage::new_for_test(&committee);

        let leader_round = 3;
        let references_at_leader = build_dag(&committee, &mut storage, None, leader_round);
        // Round-robin: round 3 → authority 3.
        let leader = committee.authorities().nth(3).unwrap();
        let references_without_leader: Vec<_> = references_at_leader
            .iter()
            .copied()
            .filter(|reference| reference.authority != leader)
            .collect();

        // Voting round: authorities 0 and 1 vote for the leader; 2 and 3 blame it.
        // Two votes clear the weak quorum (2) but not the certificate quorum (3).
        let voters = [
            committee.authorities().next().unwrap(),
            committee.authorities().nth(1).unwrap(),
        ];
        let voting_connections = committee
            .authorities()
            .map(|authority| {
                let parents = if voters.contains(&authority) {
                    references_at_leader.clone()
                } else {
                    references_without_leader.clone()
                };
                (authority, parents)
            })
            .collect();
        let references_at_voting_round = build_dag_layer(voting_connections, &mut storage);

        // Decision and anchor rounds are fully connected: every decision block carries
        // only the two votes (no certificate) and the anchor links every voting block.
        build_dag(
            &committee,
            &mut storage,
            Some(references_at_voting_round),
            6,
        );
        let anchor = storage
            .block_reader()
            .get_blocks_at_authority_round(committee.authorities().next().unwrap(), 6)
            .into_iter()
            .next()
            .unwrap();

        let fast_committer =
            BaseCommitter::new_for_test(&committee, storage.block_reader().clone(), 3, 0)
                .with_fast_path(FastPath {
                    commit_quorum: 3,
                    weak_indirect_quorum: 2,
                });
        let slow_committer =
            BaseCommitter::new_for_test(&committee, storage.block_reader().clone(), 3, 0);

        match fast_committer.decide_leader_from_anchor(&anchor, leader, leader_round) {
            LeaderStatus::IndirectCommit(block, path) => {
                assert_eq!(block.author(), leader);
                assert_eq!(path, IndirectCommitPath::WeakQuorum);
            }
            other => panic!("expected IndirectCommit, got {other:?}"),
        }
        match slow_committer.decide_leader_from_anchor(&anchor, leader, leader_round) {
            LeaderStatus::IndirectSkip(authority, round) => {
                assert_eq!(authority, leader);
                assert_eq!(round, leader_round);
            }
            other => panic!("expected IndirectSkip, got {other:?}"),
        }
    }

    #[test]
    fn decide_leader_from_anchor_commits_min_digest_twin() {
        let committee = committee(6);
        let mut storage = Storage::new_for_test(&committee);

        // Round 1: fully connected layer over genesis.
        let round_1 = build_dag(&committee, &mut storage, None, 1);

        // Round 2 (leader round): the leader equivocates with two twins carrying
        // distinct digests; the other authorities propose normal blocks.
        let leader = Authority::from(5u64);
        let normal_round_2 = build_dag_layer(
            (0..5u64)
                .map(|authority| (Authority::from(authority), round_1.clone()))
                .collect(),
            &mut storage,
        );
        let twin_a_digest = BlockDigest::from([1u8; 32]);
        let twin_b_digest = BlockDigest::from([2u8; 32]);
        let twin_a = insert_test_block(
            Block::new_for_test(leader, 2, round_1.clone()).with_digest(twin_a_digest),
            &mut storage,
        );
        let twin_b = insert_test_block(
            Block::new_for_test(leader, 2, round_1).with_digest(twin_b_digest),
            &mut storage,
        );

        // Round 3 (decision round, which coincides with the voting round at
        // `wave_length == 2`): authorities 0, 1, 2 vote for twin A; 3, 4, 5 for twin B.
        // Each side is a weak quorum (3), so both twins pass the certificate rung.
        let mut parents_a = normal_round_2.clone();
        parents_a.push(twin_a);
        let mut parents_b = normal_round_2;
        parents_b.push(twin_b);
        let round_3 = build_dag_layer(
            (0..6u64)
                .map(|authority| {
                    let parents = if authority < 3 {
                        parents_a.clone()
                    } else {
                        parents_b.clone()
                    };
                    (Authority::from(authority), parents)
                })
                .collect(),
            &mut storage,
        );

        // Round 4 (anchor round): fully connected, so the anchor links every vote.
        build_dag(&committee, &mut storage, Some(round_3), 4);
        let anchor = storage
            .block_reader()
            .get_blocks_at_authority_round(leader, 4)
            .into_iter()
            .next()
            .unwrap();

        let protocol = Protocol::blue_bottle_partially_synchronous(
            committee.total_stake(),
            NonZeroUsize::new(1).unwrap(),
        );
        let committer = BaseCommitter::new(
            committee.clone(),
            storage.block_reader().clone(),
            LeaderElector::new(committee.len()),
            &protocol,
            0,
            0,
        );

        match committer.decide_leader_from_anchor(&anchor, leader, 2) {
            LeaderStatus::IndirectCommit(block, path) => {
                assert_eq!(block.author(), leader);
                assert_eq!(block.reference().digest, twin_a_digest);
                assert_eq!(path, IndirectCommitPath::Certificate);
            }
            other => panic!("expected IndirectCommit, got {other:?}"),
        }
    }

    /// Async Blue Bottle committer over a 6-validator committee (strong quorum 5,
    /// weak quorum 3), with round-robin leaders standing in for the coin.
    fn blue_bottle_async_committer(
        committee: &std::sync::Arc<dag::committee::Committee>,
        storage: &Storage,
    ) -> BaseCommitter {
        let protocol = Protocol::blue_bottle_asynchronous(
            committee.total_stake(),
            NonZeroUsize::new(1).unwrap(),
        );
        BaseCommitter::new(
            committee.clone(),
            storage.block_reader().clone(),
            LeaderElector::new(committee.len()),
            &protocol,
            0,
            0,
        )
    }

    /// Async Blue Bottle: a fully-connected DAG holds a strong certificate (5 of 6
    /// decision-round votes) at round `r + 2`, so the leader direct-commits.
    #[test]
    fn blue_bottle_async_direct_commits_on_full_dag() {
        let committee = committee(6);
        let mut storage = Storage::new_for_test(&committee);
        build_dag(&committee, &mut storage, None, 5);
        let committer = blue_bottle_async_committer(&committee, &storage);

        let leader = committer.elect_leader(3).unwrap();
        match committer.try_direct_decide(leader, 3) {
            LeaderStatus::DirectCommit(block, path) => {
                assert_eq!(block.author(), leader);
                assert_eq!(path, DirectCommitPath::Slow);
            }
            other => panic!("expected DirectCommit, got {other:?}"),
        }
    }

    /// Async Blue Bottle: 5 of 6 decision-round blocks without a path to the leader
    /// form the slot-blame witness, so the leader direct-skips.
    #[test]
    fn blue_bottle_async_direct_skips_when_leader_omitted() {
        let committee = committee(6);
        let mut storage = Storage::new_for_test(&committee);

        let references_at_leader = build_dag(&committee, &mut storage, None, 3);
        // Round-robin: round 3 → authority 3.
        let leader = committee.authorities().nth(3).unwrap();
        let references_without_leader: Vec<_> = references_at_leader
            .into_iter()
            .filter(|reference| reference.authority != leader)
            .collect();
        let boost_connections = committee
            .authorities()
            .map(|authority| (authority, references_without_leader.clone()))
            .collect();
        let references_at_boost_round = build_dag_layer(boost_connections, &mut storage);
        build_dag(&committee, &mut storage, Some(references_at_boost_round), 5);

        let committer = blue_bottle_async_committer(&committee, &storage);
        match committer.try_direct_decide(leader, 3) {
            LeaderStatus::DirectSkip(skipped, round) => {
                assert_eq!(skipped, leader);
                assert_eq!(round, 3);
            }
            other => panic!("expected DirectSkip, got {other:?}"),
        }
    }

    /// Async Blue Bottle DAG where only the first `supporter_count` authorities keep
    /// a path to the round-3 leader through the decision round, under a fully-linked
    /// anchor at round 6.
    fn blue_bottle_async_partial_support(
        committee: &std::sync::Arc<dag::committee::Committee>,
        supporter_count: usize,
    ) -> (Storage, Data<Block>, Authority) {
        let mut storage = Storage::new_for_test(committee);

        let leader_round = 3;
        let references_at_leader = build_dag(committee, &mut storage, None, leader_round);
        // Round-robin: round 3 → authority 3.
        let leader = committee.authorities().nth(3).unwrap();
        let references_without_leader: Vec<_> = references_at_leader
            .iter()
            .copied()
            .filter(|reference| reference.authority != leader)
            .collect();

        // Boost round: only the supporters keep a path to the leader.
        let supporters: Vec<_> = committee.authorities().take(supporter_count).collect();
        let boost_connections = committee
            .authorities()
            .map(|authority| {
                let parents = if supporters.contains(&authority) {
                    references_at_leader.clone()
                } else {
                    references_without_leader.clone()
                };
                (authority, parents)
            })
            .collect();
        let references_at_boost_round = build_dag_layer(boost_connections, &mut storage);
        let (supporting_boost, blaming_boost): (Vec<_>, Vec<_>) = references_at_boost_round
            .into_iter()
            .partition(|reference| supporters.contains(&reference.authority));

        // Decision round: exactly `supporter_count` votes, the rest blames.
        let decision_connections = committee
            .authorities()
            .map(|authority| {
                let parents = if supporters.contains(&authority) {
                    supporting_boost.clone()
                } else {
                    blaming_boost.clone()
                };
                (authority, parents)
            })
            .collect();
        let references_at_decision_round = build_dag_layer(decision_connections, &mut storage);

        // Anchor round: fully connected, so the anchor links every decision block.
        build_dag(
            committee,
            &mut storage,
            Some(references_at_decision_round),
            6,
        );
        let anchor = storage
            .block_reader()
            .get_blocks_at_authority_round(committee.authorities().next().unwrap(), 6)
            .into_iter()
            .next()
            .unwrap();

        (storage, anchor, leader)
    }

    /// Async Blue Bottle: 3 decision-round votes miss the strong quorum (5) but form
    /// a weak certificate (3) in the anchor's history, so the leader stays undecided
    /// directly and indirect-commits.
    #[test]
    fn blue_bottle_async_weak_certificate_indirect_commits() {
        let committee = committee(6);
        let (storage, anchor, leader) = blue_bottle_async_partial_support(&committee, 3);

        let committer = blue_bottle_async_committer(&committee, &storage);
        match committer.try_direct_decide(leader, 3) {
            LeaderStatus::Undecided(authority, round) => {
                assert_eq!(authority, leader);
                assert_eq!(round, 3);
            }
            other => panic!("expected Undecided, got {other:?}"),
        }
        match committer.decide_leader_from_anchor(&anchor, leader, 3) {
            LeaderStatus::IndirectCommit(block, path) => {
                assert_eq!(block.author(), leader);
                assert_eq!(path, IndirectCommitPath::Certificate);
            }
            other => panic!("expected IndirectCommit, got {other:?}"),
        }
    }

    /// Async Blue Bottle: 2 decision-round votes stay below the weak certificate
    /// quorum (3), so the anchor indirect-skips the leader.
    #[test]
    fn blue_bottle_async_below_weak_certificate_indirect_skips() {
        let committee = committee(6);
        let (storage, anchor, leader) = blue_bottle_async_partial_support(&committee, 2);

        let committer = blue_bottle_async_committer(&committee, &storage);
        match committer.try_direct_decide(leader, 3) {
            LeaderStatus::Undecided(authority, round) => {
                assert_eq!(authority, leader);
                assert_eq!(round, 3);
            }
            other => panic!("expected Undecided, got {other:?}"),
        }
        match committer.decide_leader_from_anchor(&anchor, leader, 3) {
            LeaderStatus::IndirectSkip(authority, round) => {
                assert_eq!(authority, leader);
                assert_eq!(round, 3);
            }
            other => panic!("expected IndirectSkip, got {other:?}"),
        }
    }

    /// `try_indirect_decide` returns `Undecided` when the anchor iterator is empty.
    #[test]
    fn try_indirect_decide_undecided_on_empty_anchors() {
        let committee = committee(4);
        let mut storage = Storage::new_for_test(&committee);
        build_dag(&committee, &mut storage, None, 3);
        let committer =
            BaseCommitter::new_for_test(&committee, storage.block_reader().clone(), 3, 0);

        let leader = committer.elect_leader(3).unwrap();
        match committer.try_indirect_decide(leader, 3, std::iter::empty()) {
            LeaderStatus::Undecided(authority, round) => {
                assert_eq!(authority, leader);
                assert_eq!(round, 3);
            }
            other => panic!("expected Undecided, got {other:?}"),
        }
    }

    /// `try_indirect_decide` short-circuits when an undecided anchor appears in the iterator.
    #[test]
    fn try_indirect_decide_breaks_on_undecided_anchor() {
        let committee = committee(4);
        let mut storage = Storage::new_for_test(&committee);
        build_dag(&committee, &mut storage, None, 3);
        let committer =
            BaseCommitter::new_for_test(&committee, storage.block_reader().clone(), 3, 0);

        let leader = committer.elect_leader(3).unwrap();
        let undecided_anchor = LeaderStatus::Undecided(Authority::from(0u64), 6);
        let anchors = [undecided_anchor];
        match committer.try_indirect_decide(leader, 3, anchors.iter()) {
            LeaderStatus::Undecided(authority, round) => {
                assert_eq!(authority, leader);
                assert_eq!(round, 3);
            }
            other => panic!("expected Undecided after undecided anchor, got {other:?}"),
        }
    }
}
