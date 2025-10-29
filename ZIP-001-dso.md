# ZIP-001: Zoo DSO (Decentralized Semantic Optimization)

**Status**: Active  
**Type**: Technical Specification  
**Created**: 2025-10-28  
**Authors**: Zoo Labs Foundation  
**Builds On**: [HIP-002 (ASO)](https://github.com/hanzoai/papers/blob/main/hips/HIP-002-aso.md), [HIP-004 (HMM)](https://github.com/hanzoai/papers/blob/main/hips/HIP-004-hmm.md)

## Abstract

This ZIP specifies **Decentralized Semantic Optimization (DSO)**, a protocol for sharing and aggregating experiential priors across distributed language model agents without parameter updates. DSO builds on Hanzo's Active Semantic Optimization (ASO) by adding Byzantine-robust consensus and decentralized storage.

## Motivation

While ASO (HIP-002) enables training-free adaptation for individual agents, real-world deployments require:

1. **Shared Learning**: Agents learning from each other's experiences
2. **Byzantine Robustness**: Protection against malicious or faulty agents
3. **Decentralized Storage**: No single point of failure for experience libraries
4. **Incentive Alignment**: Economic rewards for quality contributions

DSO extends ASO's semantic advantages with:
- Stake-weighted median voting for Byzantine fault tolerance
- IPFS/Arweave storage via ExperienceRegistry smart contract
- P2P gossip protocol for prior synchronization
- Quality scoring and slashing mechanisms

## Specification

### Core Components

#### 1. Experience Aggregation (Builds on ASO)

**Input**: Multiple agent experiences using ASO's advantage extraction
```
A_i^(agent_k) = (g_i^(k) - mean(g^(k))) / (std(g^(k)) + ε)
```

**Byzantine-Robust Aggregation**:
```
A_consensus = weighted_median({A_i^(k)}, {stake_k})
```

Key insight: Median is robust to up to 49% Byzantine agents.

#### 2. ExperienceRegistry Smart Contract

**Storage Commitment**:
```solidity
struct Experience {
    bytes32 advantageHash;      // BitDelta compressed (from ASO)
    string storageURI;          // IPFS/Arweave
    uint256 stake;
    address contributor;
    uint256 qualityScore;
}

function submitExperience(
    bytes32 advantageHash,
    string storageURI,
    bytes proof
) external payable;
```

**Quality Verification**:
- Uses PoAI attestations from Hanzo Network (HIP-004)
- Slashing if experience degrades performance
- Rewards proportional to usage and quality

#### 3. P2P Gossip Protocol

**Prior Synchronization**:
- Agents broadcast new experiences to peers
- Stake-weighted gossip (higher stake = more propagation)
- DHT for discovery

**Network Integration**:
- Uses Hanzo Network's HMM (HIP-004) for compute pricing
- Settles payments via PoAI attestations

### Implementation Path

**Phase 1 - Single Agent (Covered by ASO)**:
- ✅ TF-GRPO advantage extraction
- ✅ PoE decoding
- ✅ BitDelta compression

**Phase 2 - Multi-Agent (This ZIP)**:
- Byzantine-robust aggregation
- ExperienceRegistry deployment
- P2P gossip implementation

**Phase 3 - Economic Layer (HIP-004 Integration)**:
- HMM pricing for experience storage
- PoAI rewards for quality contributions
- Slashing mechanism activation

## Relationship to Hanzo Infrastructure

DSO is implemented **on top of** Hanzo's infrastructure:

1. **ASO Foundation** (HIP-002):
   - DSO uses ASO's TF-GRPO for advantage extraction
   - Shares BitDelta compression (29.5× savings)
   - Compatible with PoE decoding

2. **HMM Integration** (HIP-004):
   - Experience storage priced via Hamiltonian invariants
   - Compute credits for aggregation jobs
   - Liquidity routing to high-quality experiences

3. **PoAI Verification** (HIP-004):
   - Quality attestations via TEE proofs
   - Slashing for fraudulent experiences
   - Bonus rewards for verified improvements

## Co-Development History

**2024 Q1-Q2**: Hanzo develops ASO (HIP-002) for single-agent adaptation  
**2024 Q3**: Zoo extends ASO with Byzantine-robust aggregation  
**2024 Q4**: Integration with Hanzo Network's HMM (HIP-004) for economics  
**2025 Q1**: Production deployment on Hanzo L1 infrastructure  

This ZIP represents **years of co-development** between Hanzo AI Inc and Zoo Labs Foundation, building on shared primitives (BitDelta, TF-GRPO, PoE) while adding decentralization.

## Related Resources

- **Paper**: https://github.com/zooai/papers/blob/main/zoo-dso.pdf
- **LaTeX Source**: https://github.com/zooai/papers/blob/main/zoo-dso.tex
- **Shared Sections**:
  - `sections/dso-core.tex`: DSO protocol specification
  - `sections/bitdelta.tex`: 1-bit compression (shared with ASO)
- **Dependencies**:
  - [HIP-002 (ASO)](https://github.com/hanzoai/papers/blob/main/hips/HIP-002-aso.md): Foundation for advantage extraction
  - [HIP-004 (HMM)](https://github.com/hanzoai/papers/blob/main/hips/HIP-004-hmm.md): Economic layer and compute pricing

## Performance

**Improvements over isolated ASO**:
- **15.2% better** in multi-agent tasks vs isolated operation
- **Byzantine tolerance**: Up to 49% faulty agents
- **Storage efficiency**: 29.5× compression (inherited from ASO's BitDelta)
- **Network overhead**: < 5% additional latency vs centralized

---

*ZIP-001 Created: October 28, 2025*  
*Status: Active*  
*Contact: foundation@zoo.ai*
