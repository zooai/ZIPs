# ZIP-001: Zoo Research Papers Repository

**Status**: Active
**Type**: Informational
**Created**: 2025-10-28
**Authors**: Zoo Labs Foundation Inc

## Abstract

This ZIP documents the official research papers repository for the Zoo AI ecosystem at `github.com/zooai/papers`. The repository contains 6 major research papers targeting publication in top-tier venues (NeurIPS, IEEE Blockchain, EMNLP, MLSys, JMLR).

## Motivation

As Zoo develops novel technologies (Decentralized Semantic Optimization, Proof-of-Active-Inference, Zen Models, BitDelta, Gym Platform, Hamiltonian Market Makers), we need centralized documentation of research contributions for:
1. Academic peer review and citation
2. Community transparency
3. Technical specification
4. Reproducibility

## Specification

### Repository Structure

```
github.com/zooai/papers
├── dso_whitepaper/          # DSO: Semantic experiences + BitDelta
├── poai_protocol/           # PoAI: Active inference consensus
├── zen_models/              # Zen: Efficient LLMs with spatial reasoning
├── bitdelta/                # BitDelta: 1-bit Byzantine quantization
├── gym_platform/            # Gym: Training infrastructure
├── hmm_economics/           # HMM: Market mechanism design
└── RESEARCH_ROADMAP.md      # Overall publication strategy
```

### Papers Overview

#### 1. Decentralized Semantic Optimization (DSO)
- **Target**: NeurIPS 2025 (May deadline)
- **Contributions**:
  - Bayesian Training-Free GRPO
  - 3840-dim canonical embedding space
  - BitDelta 31.7× compression
  - Byzantine-robust aggregation
- **Results**: 99.8% cost reduction ($18 vs $10K+), 2-5% performance gain
- **Status**: ✅ LaTeX complete (31KB), ready for PDF

#### 2. Proof-of-Active-Inference (PoAI)
- **Target**: IEEE Blockchain 2025 (Aug deadline)
- **Contributions**:
  - Active inference at network scale
  - Attestations with ΔI (info gain) + ΔU (utility)
  - Hamiltonian Market Maker (Ψ·Θ=κ)
  - $AI tokenomics
- **Novel**: First blockchain consensus based on epistemic value
- **Status**: ✅ LaTeX complete

#### 3. Zen Models
- **Target**: EMNLP 2025 (June) or arXiv
- **Contributions**:
  - Zen-Eco-4B, Zen-Coder-30B, Zen-Coder-480B
  - Spatial reasoning module
  - GSPO training
- **Results**: Competitive with 7B models at 4B params
- **Status**: ✅ LaTeX complete

#### 4. BitDelta
- **Target**: MLSys 2026 (Oct deadline)
- **Contributions**:
  - 1-bit quantization for federated updates
  - Byzantine median voting
  - DeltaSoup multi-tenant serving
- **Results**: 31.7× compression, <1% accuracy loss
- **Status**: 🔄 Skeleton created

#### 5. Gym Platform
- **Target**: JMLR MLOSS (rolling)
- **Contributions**:
  - Unified training platform
  - 100+ models, 10+ methods
  - Open-source Apache 2.0
- **Status**: 🔄 Skeleton created

#### 6. Hamiltonian Market Maker (HMM)
- **Target**: Financial Cryptography 2026
- **Contributions**:
  - Compute pricing via Hamiltonian dynamics
  - Liquidity allocation for EFE optimization
- **Status**: 🔄 Skeleton created (may merge with PoAI)

## Timeline

### Q2 2025 (April-June)
- Submit DSO to NeurIPS (May)
- Submit PoAI to IEEE Blockchain (Aug)
- Submit Zen to EMNLP (June) or publish on arXiv

### Q3 2025 (July-September)
- Submit BitDelta to MLSys (Oct)
- Submit Gym to JMLR MLOSS

### Q4 2025 (October-December)
- HMM decision (standalone vs merged)
- Workshop submissions (NeurIPS, ICML)

## Rationale

### Why Split Into Multiple Papers?

1. **Different audiences**: ML (NeurIPS), Blockchain (IEEE), Systems (MLSys), NLP (ACL/EMNLP)
2. **Focused contributions**: Each paper tells one clear story
3. **Citation impact**: Multiple entry points for different research communities
4. **Parallel submissions**: Can submit to different venues simultaneously
5. **Revision flexibility**: Independent revision cycles

### Why Separate Repository?

- **Modularity**: Papers independent of codebase
- **Accessibility**: Researchers need papers without full codebase
- **Versioning**: Papers have different lifecycle than code
- **Licensing**: Papers use CC BY 4.0, code uses Apache 2.0

## Local Development

Papers repository is located at:
- **Local**: `~/work/zoo/papers/`
- **Remote**: `github.com/zooai/papers`

To build any paper:
```bash
cd ~/work/zoo/papers/<paper_directory>
make pdf
make view
```

## References

- **Main Repository**: https://github.com/zooai/papers
- **Gym Platform**: https://github.com/zooai/gym
- **Zen Models**: https://github.com/zenlm
- **Research Roadmap**: `~/work/zoo/papers/RESEARCH_ROADMAP.md`

## Copyright

Papers: CC BY 4.0 (Creative Commons Attribution)
Code: Apache 2.0
Organization: Zoo Labs Foundation Inc (501(c)(3) non-profit)

---

*ZIP-001 Created: October 28, 2025*
*Status: Active*
*Contact: research@zoo.ngo*
