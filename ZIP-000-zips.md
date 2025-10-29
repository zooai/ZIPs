# ZIP-000: Zoo Improvement Proposals

**Status**: Active  
**Type**: Meta  
**Created**: 2025-10-28  
**Authors**: Zoo Labs Foundation

## Abstract

This ZIP describes the Zoo Improvement Proposal (ZIP) process. ZIPs document technical specifications, research contributions, and protocol standards for Zoo Labs Foundation projects.

## Motivation

Zoo Labs Foundation (501c3 non-profit) collaborates with Hanzo AI Inc on shared AI infrastructure. While Hanzo focuses on infrastructure (ASO, HMM, network), Zoo focuses on decentralized learning protocols (DSO, federated training).

ZIPs provide:
1. **Clear Attribution**: Separate Zoo contributions from Hanzo infrastructure
2. **Research Continuity**: Document multi-year co-development
3. **Interoperability**: Show dependencies on Hanzo protocols (HIPs)
4. **Open Research**: Public specifications for reproducibility

## ZIP Structure

Each ZIP contains:
- **Abstract**: Brief overview
- **Motivation**: Problem being solved
- **Specification**: Technical details
- **Relationship to Hanzo**: Dependencies on HIPs
- **Co-Development History**: Timeline showing collaboration
- **Related Resources**: Links to papers, code, HIPs

## ZIP Numbering

- **ZIP-000**: This meta-document
- **ZIP-001**: DSO (Decentralized Semantic Optimization)
- **ZIP-002+**: Future proposals

## Relationship to HIPs

Zoo Improvement Proposals (ZIPs) often **build on** Hanzo Improvement Proposals (HIPs):

- **HIP-002 (ASO)**: Foundation for ZIP-001 (DSO)
- **HIP-004 (HMM)**: Economic layer for ZIP-001 (DSO)

This reflects the partnership:
- **Hanzo AI Inc**: Infrastructure, compute markets, network (HIPs)
- **Zoo Labs Foundation**: Decentralized learning, semantic optimization (ZIPs)

## Co-Development Model

**Years of Collaboration** (2024-2025):
1. Shared primitives (BitDelta, TF-GRPO, PoE)
2. Hanzo develops base infrastructure (ASO, HMM)
3. Zoo extends with decentralization (DSO)
4. Integration on Hanzo Network (L1)

**No Duplication**: Protocols reference each other, build incrementally

## Related Resources

- **Zoo Papers**: https://github.com/zooai/papers
- **Hanzo Papers**: https://github.com/hanzoai/papers
- **Hanzo HIPs**: ~/work/hanzo/hips
- **Zoo ZIPs**: ~/work/zoo/zips

---

*ZIP-000 Created: October 28, 2025*  
*Status: Active*  
*Contact: foundation@zoo.ai*
