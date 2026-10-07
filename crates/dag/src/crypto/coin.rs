// Copyright (c) Mysten Labs, Inc.
// SPDX-License-Identifier: Apache-2.0

//! Common-coin primitives for leader unpredictability.
//!
//! Mahi-Mahi and Blue Bottle's asynchronous variant both require each block to carry a share
//! of a "global perfect coin": a threshold BLS signature over the round number. Once 2f+1
//! shares for a round are available, any validator can combine them into the same signature
//! regardless of which 2f+1 contributed (the uniqueness property of BLS), hash it, and use
//! the result to pick that round's leader(s) *after* the round's blocks are already fixed —
//! so an adversary cannot predict or steer the choice in advance.
//!
//! **Status:** this module wires up signing, verifying, and combining shares using
//! `fastcrypto-tbls`'s ready-made BLS12-381 instantiation, and both ends are connected to real
//! code, not just tests:
//! [`Core::try_new_block`](crate::core::Core::try_new_block) calls
//! [`CoinKeyShare::sign_round`] to attach a share to each proposed block (when the replica has
//! a key share at all), and `BaseCommitter::elect_leader` (in the `consensus` crate) combines a
//! round's shares and uses the result as the leader seed, falling back to plain round-robin
//! whenever fewer than a quorum of blocks at that round carry a share.
//!
//! There is **no distributed key generation (DKG) protocol**. Replicas are instead booted with
//! their key share next to their signing key: `PrivateReplicaConfig` carries an optional
//! [`CoinKeyShare`], and [`mocked_setup_for_size`] produces the shares for benchmark and
//! simulator committees. That setup is insecure by construction (see its docs), so the coin is
//! only as unpredictable as a seeded benchmark key. Also still open: the coin share on
//! [`Block`](crate::block::Block) is optional and not covered by the block's digest/signature,
//! shares are not verified when a block arrives, and the seed is the raw leading bytes of the
//! signature rather than a hash of it.

use fastcrypto::error::FastCryptoResult;
use fastcrypto_tbls::{
    mocked_dkg,
    nodes::{Node, Nodes, PartyId},
    tbls::ThresholdBls,
    types::{PublicBlsKey, PublicVssKey, Share, ThresholdBls12381MinSig},
};
use rand::thread_rng;
use serde::{Deserialize, Serialize};

use crate::{block::RoundNumber, committee::Committee};

/// A validator's share of a signature on a given round; this is what a block carries as its
/// coin share. Two shares for the same round from the same validator are the same value
/// (BLS signing is deterministic), so this is safe to recompute rather than cache.
pub type CoinShare =
    fastcrypto_tbls::tbls::PartialSignature<<ThresholdBls12381MinSig as ThresholdBls>::Signature>;

/// The combined coin value for a round: a standard (non-partial) BLS signature, unique for
/// that round's message regardless of which 2f+1 shares were used to reconstruct it.
pub type CoinValue = <ThresholdBls12381MinSig as ThresholdBls>::Signature;

/// One validator's long-lived key material for the coin: its private share of the threshold
/// key, the committee-wide public commitment to the secret-sharing polynomial (needed to
/// verify anyone's share), and the party index `fastcrypto-tbls` uses internally.
#[derive(Clone, Serialize, Deserialize)]
pub struct CoinKeyShare {
    party: PartyId,
    private_share: Share,
    public_vss_key: PublicVssKey,
}

impl CoinKeyShare {
    /// Sign `round` with this validator's private share. The message is just the round
    /// number's big-endian bytes — the papers only need the coin to be unpredictable per
    /// round, not bound to any other block content.
    pub fn sign_round(&self, round: RoundNumber) -> CoinShare {
        ThresholdBls12381MinSig::partial_sign(&self.private_share, &round.to_be_bytes())
    }

    /// This party's index into the committee's share set (not the same as the
    /// [`Authority`](crate::authority::Authority) index: `fastcrypto-tbls` assigns its own
    /// 1-based party/share numbering).
    pub fn party(&self) -> PartyId {
        self.party
    }
}

/// Verify that `share` is a legitimate coin share for `round` under the committee's public
/// VSS key. Callers should reject blocks whose coin share fails this.
pub fn verify_share(
    public_vss_key: &PublicVssKey,
    round: RoundNumber,
    share: &CoinShare,
) -> FastCryptoResult<()> {
    ThresholdBls12381MinSig::partial_verify(public_vss_key, &round.to_be_bytes(), share)
}

/// Combine `threshold` (i.e. 2f+1) shares for the same round into the round's coin value.
/// Returns an error if fewer than `threshold` distinct-index shares are given — see
/// [`ThresholdBls::aggregate`] for exact semantics (extra shares beyond the threshold are
/// accepted but ignored, so this tolerates a few invalid shares mixed in with enough good
/// ones).
pub fn combine(
    threshold: u16,
    shares: impl Iterator<Item = CoinShare> + Clone,
) -> FastCryptoResult<CoinValue> {
    ThresholdBls12381MinSig::aggregate(threshold, shares)
}

/// Hash a combined coin value down to a `u64`, for picking a leader index
/// (`seed % committee_len`) the same way every validator will, since everyone who combines
/// the same 2f+1-or-more shares gets the same [`CoinValue`] and thus the same seed.
///
/// Uses the first 8 bytes of the value's canonical byte encoding. This is a seed, not a
/// cryptographic commitment on its own — it is only unpredictable because the *coin value*
/// it is derived from was unpredictable before enough shares existed to combine it.
pub fn coin_seed(value: &CoinValue) -> u64 {
    use fastcrypto::serde_helpers::ToFromByteArray;
    let bytes = value.to_byte_array();
    u64::from_be_bytes(bytes[..8].try_into().expect("BLS signatures are > 8 bytes"))
}

/// **Test/bootstrap-only.** Generate a committee-wide threshold key and hand each authority
/// its private share, using `fastcrypto-tbls`'s insecure mocked DKG rather than a real
/// distributed protocol. The returned `Vec` is ordered the same as `committee.authorities()`.
///
/// See [`mocked_setup_for_size`] for what "insecure" means here.
pub fn mocked_committee_setup(committee: &Committee) -> Vec<CoinKeyShare> {
    mocked_setup_for_size(committee.len())
}

/// **Test/benchmark-only.** Hand out one [`CoinKeyShare`] per authority for a committee of
/// `committee_size` equally staked validators, in authority order, without running a DKG.
///
/// This is a stand-in for key distribution: replicas are booted with their share next to their
/// signing key (see `PrivateReplicaConfig` in the `replica` crate). It is **not** secure: the
/// underlying secret polynomial is a fixed constant from `fastcrypto-tbls`'s mocked DKG, so
/// anyone who reads the code can compute every share and thus the coin. That is the same
/// status as the seeded Ed25519 keys benchmarks already use. The output is deterministic, so
/// separate calls with the same `committee_size` return the same shares.
pub fn mocked_setup_for_size(committee_size: usize) -> Vec<CoinKeyShare> {
    let threshold = quorum_threshold_for_size(committee_size) as u32;
    let nodes = Nodes::new((0..committee_size).map(mocked_node).collect())
        .expect("committee produces a valid node set");

    // A fixed, arbitrary "full private key" for the mocked setup. It must only be consistent
    // across all parties' calls below, not secret: this whole function is explicitly not for
    // production use.
    let full_private_key: u128 = 0xC01_C01;

    (0..committee_size)
        .map(|index| {
            let party = index as PartyId;
            let output = mocked_dkg::generate_mocked_output::<PublicBlsKey, PublicBlsKey>(
                nodes.clone(),
                threshold,
                full_private_key,
                party,
            );
            CoinKeyShare {
                party,
                private_share: output
                    .shares
                    .expect("mocked output always carries shares")
                    .into_iter()
                    .next()
                    .expect("mocked setup gives exactly one share per unweighted party"),
                public_vss_key: output.vss_pk,
            }
        })
        .collect()
}

/// The 2f+1-out-of-3f+1 quorum size for a (uniformly-staked) committee, in share count.
/// Matches the quorum convention used throughout `consensus::Protocol` (see e.g.
/// `Protocol::mysticeti`), expressed here in shares rather than stake since
/// [`mocked_committee_setup`] gives every authority exactly one share.
pub fn quorum_threshold(committee: &Committee) -> u16 {
    (2 * committee.total_stake() / 3 + 1) as u16
}

/// [`quorum_threshold`] for a committee of `committee_size` authorities with stake 1 each.
fn quorum_threshold_for_size(committee_size: usize) -> u16 {
    (2 * committee_size / 3 + 1) as u16
}

/// Build the unweighted `fastcrypto-tbls` [`Node`] record for the authority at `index`.
/// Weighting (giving some validators more than one share) isn't used here: every authority
/// gets weight 1, so `total_stake() == committee size` is assumed. A real deployment with
/// non-uniform stake would need a stake-to-share-count mapping, which is out of scope for this
/// mocked setup. The encryption key below is only there to satisfy `Node`'s field; shares
/// aren't actually encrypted by the mocked DKG, so the matching private key is thrown away
/// immediately.
fn mocked_node(index: usize) -> Node<PublicBlsKey> {
    let id = index as PartyId;
    let throwaway_private_key =
        fastcrypto_tbls::ecies_v1::PrivateKey::<PublicBlsKey>::new(&mut thread_rng());
    Node {
        id,
        pk: fastcrypto_tbls::ecies_v1::PublicKey::from_private_key(&throwaway_private_key),
        weight: 1,
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::test_util::committee;

    /// Any 2f+1 (here: all 4, in a 4-validator test committee with f=0 in the mocked setup's
    /// threshold) shares for the same round combine to the same coin value, and that value
    /// verifies independently of which shares were used to reconstruct it.
    #[test]
    fn combined_coin_is_stable_across_quorums() {
        let committee = committee(4);
        let key_shares = mocked_committee_setup(&committee);
        let threshold = quorum_threshold(&committee);
        let round: RoundNumber = 7;

        let shares: Vec<CoinShare> = key_shares.iter().map(|k| k.sign_round(round)).collect();

        // Every individual share verifies against the shared public VSS key.
        let vss_pk = &key_shares[0].public_vss_key;
        for share in &shares {
            verify_share(vss_pk, round, share).expect("each share should verify");
        }

        // Combining the first `threshold` shares and combining the last `threshold` shares
        // (a different quorum) must agree, since BLS signatures are unique per message/key.
        let first_quorum = combine(threshold, shares[..threshold as usize].iter().cloned())
            .expect("enough shares to combine");
        let last_quorum = combine(
            threshold,
            shares[shares.len() - threshold as usize..].iter().cloned(),
        )
        .expect("enough shares to combine");
        assert_eq!(
            first_quorum, last_quorum,
            "different quorums of shares must combine to the same coin value",
        );
        assert_eq!(coin_seed(&first_quorum), coin_seed(&last_quorum));
    }

    /// Fewer than `threshold` shares must not combine.
    #[test]
    fn combine_fails_below_threshold() {
        let committee = committee(4);
        let key_shares = mocked_committee_setup(&committee);
        let threshold = quorum_threshold(&committee);
        let round: RoundNumber = 1;

        let shares: Vec<CoinShare> = key_shares
            .iter()
            .take(threshold as usize - 1)
            .map(|k| k.sign_round(round))
            .collect();

        assert!(combine(threshold, shares.into_iter()).is_err());
    }

    /// A key share survives a YAML round trip (it is stored in the private replica config) and
    /// still signs the same share.
    #[test]
    fn key_share_round_trips_through_yaml() {
        let key_share = mocked_setup_for_size(4).remove(1);
        let yaml = serde_yaml::to_string(&key_share).expect("key share serializes");
        let restored: CoinKeyShare = serde_yaml::from_str(&yaml).expect("key share deserializes");

        assert_eq!(restored.party(), key_share.party());
        let round: RoundNumber = 3;
        assert_eq!(restored.sign_round(round), key_share.sign_round(round));
        assert_eq!(restored.public_vss_key, key_share.public_vss_key);
    }

    /// Shares generated by separate calls for the same committee size are interchangeable,
    /// which is what lets every replica build its own share from the same configuration.
    #[test]
    fn mocked_setup_is_deterministic() {
        let first = mocked_setup_for_size(4);
        let second = mocked_setup_for_size(4);
        let round: RoundNumber = 9;
        for (a, b) in first.iter().zip(&second) {
            assert_eq!(a.sign_round(round), b.sign_round(round));
        }
    }
}
