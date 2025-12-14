---
zip: 005
title: Zoo AI Mining Integration
description: Integration of Hanzo AI mining rewards with Zoo ecosystem DeFi and governance
author: Zoo Labs Foundation (@zoolabs)
status: Draft
type: Standards Track
created: 2024-11-30
requires: LP-2000, HIP-006
---

# ZIP-005: Zoo AI Mining Integration

## Abstract

This ZIP specifies how Zoo ecosystem integrates with the Hanzo AI Mining Protocol, enabling AI mining rewards to flow into Zoo DeFi protocols, governance systems, and research funding mechanisms. Zoo EVM (Chain ID: 200200) natively supports AI and ZOO tokens through the Teleport bridge. Lux C-Chain (Chain ID: 96369) also supports native AI tokens.

## Motivation

Zoo Labs Foundation focuses on decentralized AI (DeAI) and decentralized science (DeSci). Native AI mining integration provides:

1. **Research Funding**: AI mining rewards fund open research
2. **DeFi Integration**: AI tokens in Zoo liquidity pools
3. **Governance Power**: Miners participate in Zoo governance
4. **Compute Incentives**: Research projects access compute via mining network

## Architecture

```
┌──────────────────────────────────────────────────────────────────┐
│                    Hanzo Networks L1 (Mining)                    │
│                              │                                   │
│                      Teleport Bridge                             │
│                              │                                   │
└──────────────────────────────┼───────────────────────────────────┘
                               │
                               ▼
┌──────────────────────────────────────────────────────────────────┐
│                     Zoo EVM (Chain ID: 200200)                   │
│                                                                  │
│  ┌──────────────┐    ┌──────────────┐    ┌──────────────┐       │
│  │  AI Token    │    │  ZOO Token   │    │   Research   │       │
│  │  Contract    │    │  Contract    │    │   Treasury   │       │
│  └──────┬───────┘    └──────┬───────┘    └──────┬───────┘       │
│         │                   │                   │                │
│         ▼                   ▼                   ▼                │
│  ┌──────────────────────────────────────────────────────┐       │
│  │                    Zoo DeFi Hub                       │       │
│  │  ┌─────────┐  ┌─────────┐  ┌─────────┐  ┌─────────┐ │       │
│  │  │ AI/ZOO  │  │  Yield  │  │ Research│  │  Gover- │ │       │
│  │  │  Pool   │  │ Farming │  │ Grants  │  │  nance  │ │       │
│  │  └─────────┘  └─────────┘  └─────────┘  └─────────┘ │       │
│  └──────────────────────────────────────────────────────┘       │
│                                                                  │
└──────────────────────────────────────────────────────────────────┘
```

## Specification

### 1. Token Integration

#### AI Token on Zoo EVM

AI tokens teleported to Zoo EVM are represented as native tokens (not wrapped):

```solidity
// AI Token interface on Zoo EVM
interface IAIToken {
    function totalSupply() external view returns (uint256);
    function balanceOf(address account) external view returns (uint256);
    function transfer(address to, uint256 amount) external returns (bool);
    function allowance(address owner, address spender) external view returns (uint256);
    function approve(address spender, uint256 amount) external returns (bool);
    function transferFrom(address from, address to, uint256 amount) external returns (bool);

    // Mining-specific
    function miningOrigin(address holder) external view returns (bytes32 teleportId);
    function isMiner(address account) external view returns (bool);
}
```

**Contract Address:** `0xAI00...` (TBD)

#### ZOO Token Integration

ZOO tokens gain mining utility:

| Function | Description |
|----------|-------------|
| Staking | Stake ZOO to boost mining rewards |
| Governance | Vote on mining parameters |
| Fees | Pay Teleport fees in ZOO |
| Research | Fund research with ZOO |

### 2. DeFi Protocols

#### AI/ZOO Liquidity Pool

```solidity
interface IAIZooPool {
    /// @notice Add liquidity to AI/ZOO pool
    function addLiquidity(
        uint256 aiAmount,
        uint256 zooAmount,
        uint256 minLPTokens,
        address recipient,
        uint256 deadline
    ) external returns (uint256 lpTokens);

    /// @notice Swap AI for ZOO
    function swapAIForZOO(
        uint256 aiAmount,
        uint256 minZooOut,
        address recipient,
        uint256 deadline
    ) external returns (uint256 zooAmount);

    /// @notice Get current exchange rate
    function getAIToZOORate() external view returns (uint256);
}
```

#### Mining Yield Aggregator

```solidity
interface IMiningYieldAggregator {
    /// @notice Deposit AI mining rewards for auto-compounding
    function depositMiningRewards(uint256 amount) external;

    /// @notice Claim accumulated yields
    function claimYield() external returns (uint256 aiAmount, uint256 zooAmount);

    /// @notice Get projected APY
    function getProjectedAPY() external view returns (uint256 aiAPY, uint256 zooAPY);
}
```

### 3. Research Treasury

A portion of AI mining rewards funds Zoo research initiatives:

```solidity
interface IResearchTreasury {
    /// @notice Mining rewards automatically contribute to treasury
    function miningContribution(uint256 aiAmount) external;

    /// @notice Propose research grant
    function proposeGrant(
        string calldata title,
        string calldata description,
        uint256 requestedAmount,
        address recipient
    ) external returns (uint256 proposalId);

    /// @notice Vote on grant proposal
    function voteOnGrant(uint256 proposalId, bool support) external;

    /// @notice Execute approved grant
    function executeGrant(uint256 proposalId) external;
}
```

**Treasury Parameters:**

| Parameter | Value |
|-----------|-------|
| Mining Contribution | 2% of teleported rewards |
| Grant Threshold | 10,000 ZOO voting power |
| Voting Period | 7 days |
| Execution Delay | 2 days |

### 4. Governance Integration

AI miners gain governance rights:

```solidity
interface IZooMiningGovernance {
    /// @notice Get governance power from mining activity
    function miningVotingPower(address miner) external view returns (uint256);

    /// @notice Delegate mining voting power
    function delegateMiningPower(address delegate) external;

    /// @notice Propose mining parameter change
    function proposeMiningChange(
        bytes32 parameterHash,
        uint256 newValue
    ) external returns (uint256 proposalId);
}
```

**Governance Parameters:**

| Parameter | Description | Current Value |
|-----------|-------------|---------------|
| `miningRewardRate` | Base reward per compute unit | 100 AI |
| `teleportFee` | Fee for cross-chain transfer | 0.1% |
| `treasuryRate` | Research treasury contribution | 2% |
| `minStakeBoost` | Minimum ZOO stake for boost | 1000 ZOO |

### 5. NVTrust Chain-Binding (Double-Spend Prevention)

Zoo integrates with the NVTrust chain-binding mechanism to prevent AI work from being claimed on multiple chains.

#### Design Principle

We avoid double-spend by **binding each unit of AI work to Zoo's chain ID (200200) BEFORE the compute runs**, and having the GPU's NVTrust enclave sign an attested receipt including that chain ID.

#### Work Context for Zoo Mining

```solidity
/// Zoo-specific work context for AI mining
struct WorkContext {
    uint32 chainId;        // 200200 (Zoo EVM)
    bytes32 jobId;         // Workload identifier
    bytes32 modelHash;     // Model identity
    bytes32 inputHash;     // Prompt/data identity
    bytes32 deviceId;      // GPU hardware ID
    bytes32 nonce;         // Unique per job
    uint64 timestamp;      // Unix timestamp
}

/// Attested receipt from NVTrust enclave
struct AttestedReceipt {
    WorkContext context;
    bytes32 resultHash;    // Hash of AI output
    uint64 flops;          // Compute units
    uint64 tokensProcessed;// LLM tokens
    bytes nvtrustSignature;// NVIDIA hardware attestation
}
```

#### Zoo Receipt Verification

```solidity
interface IZooMiningVerifier {
    /// @notice Verify NVTrust attested receipt and mint reward
    /// @dev Prevents double-spend via spent key check
    function verifyAndMint(
        AttestedReceipt calldata receipt
    ) external returns (uint256 reward);

    /// @notice Check if work has already been minted
    /// @dev Key = keccak256(deviceId || nonce || chainId)
    function isSpent(bytes32 spentKey) external view returns (bool);

    /// @notice Verify NVTrust signature against NVIDIA root
    function verifyNVTrust(
        bytes calldata receipt,
        bytes calldata signature
    ) external view returns (bool);
}
```

#### Spent Set (Double-Spend Prevention)

```solidity
contract ZooMiningVerifier is IZooMiningVerifier {
    // Spent set: keccak256(deviceId || nonce || chainId) => minted
    mapping(bytes32 => bool) private _spentSet;

    uint32 constant ZOO_CHAIN_ID = 200200;

    function verifyAndMint(
        AttestedReceipt calldata receipt
    ) external override returns (uint256 reward) {
        // 1. Verify NVTrust signature
        require(verifyNVTrust(
            abi.encode(receipt),
            receipt.nvtrustSignature
        ), "Invalid attestation");

        // 2. Verify chain_id is Zoo (200200)
        require(
            receipt.context.chainId == ZOO_CHAIN_ID,
            "Wrong chain: expected Zoo (200200)"
        );

        // 3. Compute unique spent key
        bytes32 spentKey = keccak256(abi.encodePacked(
            receipt.context.deviceId,
            receipt.context.nonce,
            receipt.context.chainId
        ));

        // 4. Check spent set (DOUBLE-SPEND PREVENTION)
        require(!_spentSet[spentKey], "Already minted");

        // 5. Mark as spent and calculate reward
        _spentSet[spentKey] = true;
        reward = calculateReward(receipt.flops, receipt.tokensProcessed);

        // 6. Mint AI tokens to miner
        aiToken.mint(msg.sender, reward);

        emit WorkMinted(
            receipt.context.deviceId,
            receipt.context.jobId,
            reward
        );
    }
}
```

#### Multi-Chain Mining Protection

The same GPU can mine for Zoo, Hanzo, and Lux, but:
- Zoo only accepts receipts with `chainId == 200200`
- Each chain maintains its own spent set
- Same `(deviceId, nonce)` pair can exist on multiple chains with different `chainId`

| GPU | Zoo Receipt | Hanzo Receipt | Lux Receipt |
|-----|-------------|---------------|-------------|
| H100-001 | chainId: 200200 | chainId: 36963 | chainId: 96369 |
| H100-001 | Valid on Zoo | Invalid on Zoo | Invalid on Zoo |

**Key Invariant:** A receipt with `chainId: 36963` (Hanzo) cannot be minted on Zoo (200200) - the chain ID check rejects it before the spent set is even consulted.

**Reference Implementation:**
- [`lux/ai/pkg/attestation/nvtrust.go`](https://github.com/luxfi/ai/blob/main/pkg/attestation/nvtrust.go)
- [`shinkai/hanzo-node/hanzo-libs/hanzo-mining/src/ledger.rs`](https://github.com/hanzoai/node/blob/main/hanzo-libs/hanzo-mining/src/ledger.rs)

### 6. Teleport Receiver

Zoo EVM receives teleported AI via the precompile:

```solidity
interface ITeleportReceiver {
    /// @notice Called when AI is teleported from Hanzo L1
    event AIReceived(
        bytes32 indexed teleportId,
        address indexed recipient,
        uint256 amount,
        bytes mldaPublicKey
    );

    /// @notice Process incoming teleport
    function receiveTeleport(
        bytes32 teleportId,
        address recipient,
        uint256 amount,
        bytes calldata mldaSignature
    ) external returns (bool);

    /// @notice Verify teleport signature (ML-DSA)
    function verifyTeleportSignature(
        bytes32 teleportId,
        bytes calldata publicKey,
        bytes calldata signature
    ) external view returns (bool);
}
```

## Rationale

### Why Zoo EVM?

1. **Research Focus**: Zoo's DeSci mission aligns with AI compute
2. **Community Governance**: Miners participate in research decisions
3. **DeFi Depth**: Existing liquidity for AI/ZOO pairs
4. **Foundation Support**: Zoo Labs provides ongoing development

### Why Native Tokens?

Teleport creates native tokens rather than wrapped versions:

1. **No Bridge Risk**: Assets don't depend on bridge security
2. **Full Fungibility**: No wrapping/unwrapping overhead
3. **Direct Governance**: Native tokens have full voting rights

## Security Considerations

### Teleport Verification

All incoming teleports verified via ML-DSA:

1. Signature validation against sender public key
2. Teleport ID uniqueness check (no replays)
3. Amount bounds verification
4. Rate limiting per sender

### Treasury Protection

Research treasury secured by:

1. Multi-sig execution (3 of 5)
2. Time-locked proposals
3. Grant amount limits
4. Audited smart contracts

### Oracle Risks

AI/ZOO exchange rates use:

1. TWAP (Time-Weighted Average Price)
2. Multiple oracle sources
3. Circuit breakers on extreme moves

## Test Cases

### Integration Tests

```typescript
describe("Zoo AI Mining Integration", () => {
    it("should receive teleported AI tokens", async () => {
        const teleportId = await teleportFromHanzo(miner, 1000);
        const balance = await aiToken.balanceOf(miner);
        expect(balance).to.equal(1000);
    });

    it("should contribute to research treasury", async () => {
        await teleportFromHanzo(miner, 10000);
        const treasuryBalance = await treasury.balance();
        expect(treasuryBalance).to.equal(200); // 2%
    });

    it("should grant governance power to miners", async () => {
        await teleportFromHanzo(miner, 10000);
        const power = await governance.miningVotingPower(miner);
        expect(power).to.be.gt(0);
    });
});
```

## Reference Implementation

| Component | Repository | Path |
|-----------|------------|------|
| AI Token | zoo-contracts | `contracts/tokens/AIToken.sol` |
| ZOO Integration | zoo-contracts | `contracts/mining/MiningIntegration.sol` |
| Teleport Receiver | zoo-contracts | `contracts/bridge/TeleportReceiver.sol` |
| Research Treasury | zoo-contracts | `contracts/governance/ResearchTreasury.sol` |
| DeFi Hub | zoo-contracts | `contracts/defi/AIZooHub.sol` |

### Source Code References

| Protocol Component | Hanzo Implementation |
|--------------------|---------------------|
| Mining Wallet | [`hanzo-mining/src/wallet.rs`](https://github.com/hanzoai/node/blob/main/hanzo-libs/hanzo-mining/src/wallet.rs) |
| Teleport Bridge | [`hanzo-mining/src/bridge.rs`](https://github.com/hanzoai/node/blob/main/hanzo-libs/hanzo-mining/src/bridge.rs) |
| EVM Integration | [`hanzo-mining/src/evm.rs`](https://github.com/hanzoai/node/blob/main/hanzo-libs/hanzo-mining/src/evm.rs) |
| Global Ledger | [`hanzo-mining/src/ledger.rs`](https://github.com/hanzoai/node/blob/main/hanzo-libs/hanzo-mining/src/ledger.rs) |

## Related Proposals

- **LP-2000**: AI Mining Standard (Lux ecosystem-wide specification)
- **HIP-006**: Hanzo AI Mining Protocol (L1 implementation)
- **LP-0004**: Quantum Resistant Cryptography
- **LP-0005**: Quantum Safe Wallets
- **ZIP-001**: Zoo Token Standard
- **ZIP-003**: Zoo Genesis Parameters

## Copyright

Copyright 2024 Zoo Labs Foundation. Released under CC0.
