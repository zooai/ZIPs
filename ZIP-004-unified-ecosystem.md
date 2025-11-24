# ZIP-004: Zoo Unified Ecosystem Architecture

**Status**: Implemented
**Type**: Ecosystem Architecture
**Created**: 2025-10-31
**Updated**: 2025-10-31

## Abstract

This ZIP documents the unified Zoo Network ecosystem architecture, consolidating all Zoo applications, contracts, and infrastructure into a single monorepo for improved developer experience and ecosystem coherence.

## Motivation

The Zoo Network previously consisted of multiple disconnected repositories:
- Separate repos for each web application
- Fragmented smart contract deployments
- Inconsistent tooling and dependencies
- Difficult to run and test integrated flows

This resulted in:
- Poor developer experience
- Inconsistent user experience across properties
- Difficulty maintaining shared components
- Complex deployment processes

## Specification

### 1. Unified Monorepo Structure

```
zoo/
├── app/                    # zoolabs.io (Next.js)
├── foundation/             # zoo.ngo (Next.js)
├── network/                # zoo.network (Next.js)
├── dao-governance/app/     # zoo.vote (Next.js + React)
├── fund/                   # zoo.fund (Next.js)
├── computer/               # zoo.computer (Vite + React)
├── exchange/               # zoo.exchange (React + Uniswap fork)
├── contracts/              # Smart contracts (Hardhat)
├── ui/                     # @zoo/ui shared components
├── dev-all.sh              # Unified development script
└── package.json            # Monorepo configuration
```

### 2. Application Domains

| Domain | Purpose | Port | Framework |
|--------|---------|------|-----------|
| **zoolabs.io** | AI research, ZenLM models, desktop app downloads | 3000 | Next.js 14 |
| **zoo.ngo** | 501(c)(3) foundation, ai.zoo.ngo subdomain | 3002 | Next.js 15 |
| **zoo.network** | Blockchain explorer, network statistics | 3003 | Next.js 15 |
| **zoo.vote** | DAO governance, ZK staking, proposals | 3004 | Next.js 14 + React |
| **zoo.fund** | Conservation DAO discovery & fundraising | 3005 | Next.js 15 |
| **zoo.computer** | AI hardware sales (NVIDIA DGX Sparks) | 3007 | Vite 6 + React 19 |
| **zoo.exchange** | Decentralized exchange (Uniswap V3 fork) | 3008 | React + Craco |

### 3. Unified Tooling

**Package Manager**: pnpm (v8.15+)
- Faster than npm/yarn
- Better disk space usage
- Proper monorepo support

**Node.js**: v20.x or higher
- Required for all applications
- Managed via nvm

**TypeScript**: v5.x
- Consistent across all apps
- Shared tsconfig.json configurations

### 4. Shared Component Library

**@zoo/ui** package provides:
- Unified Header component
- Unified Footer component
- RainbowKit wallet integration
- Common UI primitives (Radix UI)
- Consistent theming (TailwindCSS 4.x)

### 5. Smart Contract Integration

**ZK Governance System**:
- ZOO Token (1B supply) - Base token
- KEEPER Token (1B supply) - Governance enabler
- ZK Staking Contract - ZOO+KEEPER → ZK conversion
- ZooGovernor - OpenZeppelin Governor implementation
- Timelock Controller - 2-day execution delay

**Deployment**:
- Local: `npx hardhat node` (port 8545)
- Testnet: TBD
- Mainnet: TBD

### 6. Development Workflow

**Single Command Start**:
```bash
./dev-all.sh
```

This script:
1. Starts local blockchain (port 8545)
2. Deploys ZK governance contracts
3. Launches all 7 web applications in parallel
4. Creates log files for each service
5. Displays live URLs and contract addresses

**Individual App Start**:
```bash
# Example for zoolabs.io
cd app && pnpm dev
```

**Monorepo Commands**:
```bash
pnpm install -r          # Install all dependencies
pnpm -r build            # Build all apps
pnpm -r test             # Run all tests
pnpm dev:zoolabs         # Start zoolabs.io only
pnpm dev:foundation      # Start zoo.ngo only
```

### 7. Token Economics Integration

**Three-Token System**:
1. **ZOO** - Base governance token
2. **KEEPER** - Governance enabler
3. **ZK** - Voting power (deflationary)

**ZK Staking Formula**:
```
ZK = (ZOO + KEEPER) × Time Bonus
```

| Lock Duration | Multiplier |
|---------------|------------|
| 0 days        | 1.0x       |
| 1 year        | 1.9x       |
| 2 years       | 2.8x       |
| 5 years       | 5.5x       |
| 10 years      | 10.0x      |

### 8. Cross-App Navigation

All applications share:
- Common header with logo, search, menu, connect wallet
- Common footer with links to all Zoo properties
- Consistent design language (dark theme, monochrome)
- Unified wallet connection (RainbowKit)

### 9. Documentation Structure

**Core Documents**:
- `README.md` - Quick start guide
- `STAKING.md` - ZK staking system documentation
- `ZOO_ECOSYSTEM_ARCHITECTURE.md` - Full technical architecture
- `ECOSYSTEM_STATUS.md` - Current deployment status

**Per-App Documentation**:
- Each app has its own README
- Specific setup instructions
- API documentation where applicable

## Implementation

### Phase 1: Monorepo Setup ✅
- [x] Create monorepo structure
- [x] Migrate all apps to pnpm
- [x] Update dependencies to latest versions
- [x] Configure workspaces

### Phase 2: App Consolidation ✅
- [x] Copy and configure zoolabs.io
- [x] Copy and configure zoo.ngo
- [x] Copy and configure zoo.network
- [x] Copy and configure zoo.vote
- [x] Copy and configure zoo.fund
- [x] Copy and configure zoo.computer
- [x] Copy and configure zoo.exchange

### Phase 3: Shared Components ✅
- [x] Create @zoo/ui package
- [x] Implement unified Header
- [x] Implement unified Footer
- [x] Integrate RainbowKit

### Phase 4: Development Tools ✅
- [x] Create dev-all.sh script
- [x] Configure individual dev scripts
- [x] Setup logging
- [x] Add contract deployment

### Phase 5: Documentation ✅
- [x] Write comprehensive README
- [x] Document staking system
- [x] Document architecture
- [x] Create status reports

### Phase 6: Integration & Testing 🔄
- [ ] Test full ecosystem flow
- [ ] Verify wallet connections across apps
- [ ] Test contract interactions
- [ ] Performance optimization

### Phase 7: Production Deployment 📋
- [ ] Configure production builds
- [ ] Setup CI/CD pipelines
- [ ] Deploy to testnet
- [ ] Deploy to mainnet

## Benefits

**For Developers**:
- Single `git clone` for entire ecosystem
- One command to run everything
- Consistent tooling and dependencies
- Shared components reduce duplication
- Easier testing of integrated flows

**For Users**:
- Consistent experience across all Zoo properties
- Seamless navigation between applications
- Unified wallet connection
- Single sign-on experience

**For Maintainers**:
- Centralized dependency management
- Easier to enforce standards
- Simplified deployment process
- Better code reuse

## Security Considerations

1. **Smart Contract Security**:
   - All contracts based on audited OpenZeppelin implementations
   - 2-day timelock on governance actions
   - Non-custodial (users control keys)

2. **Application Security**:
   - Environment variables for secrets
   - No private keys in source code
   - Secure RPC endpoints

3. **Development Security**:
   - `.gitignore` configured properly
   - No sensitive data in logs
   - Localhost-only default configuration

## Backwards Compatibility

This ZIP represents a new unified architecture. Existing deployments are not affected, but new deployments should follow this structure.

## References

- [ZIP-001: Zoo DSO](./ZIP-001-dso.md)
- [ZIP-002: ZenLM Reranker](./ZIP-002-zen-reranker.md)
- [ZIP-003: Genesis Protocol](./ZIP-003-genesis.md)
- [OpenZeppelin Governor Docs](https://docs.openzeppelin.com/contracts/governance)
- [pnpm Workspaces](https://pnpm.io/workspaces)

## Copyright

Copyright and related rights waived via CC0.
