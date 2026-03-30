import type { Block } from '../types/blocks'

export interface PaperSection {
  readonly id: string
  readonly number: string
  readonly title: string
  readonly description: string
  readonly blocks: readonly Block[]
}

export const PAPER_METADATA = {
  title: 'Geography Drives Blockchain Centralization',
  subtitle: 'An editorial reading layer over Yang, Oz, Wu, and Zhang (2025).',
  citation: 'Yang, Oz, Wu, Zhang (2025) · arXiv:2509.21475',
  abstract: 'The paper models validators as geographically mobile agents who optimize for MEV capture and consensus latency. Even under simplified assumptions, both SSP and MSP push validators toward a small set of low-latency regions, with different mechanisms and sensitivities.',
  keyClaims: [
    'Both SSP and MSP make geography part of block-building economics, not just network background.',
    'The same attestation-threshold change pushes SSP toward more concentration and MSP toward less.',
    'Starting validator geography can matter as much as, or more than, paradigm choice.',
  ],
  references: [
    { label: 'arXiv paper', url: 'https://arxiv.org/abs/2509.21475' },
    { label: 'Simulation repository', url: 'https://github.com/syang-ng/geographical-decentralization-simulation' },
  ],
} as const

export const PAPER_SECTIONS: readonly PaperSection[] = [
  {
    id: 'system-model',
    number: '§3',
    title: 'System Model',
    description: 'Geography, consensus, MEV extraction, and information sources.',
    blocks: [
      {
        type: 'insight',
        emphasis: 'key-finding',
        title: 'Two-layer geographic game',
        text: 'Validators choose GCP regions to minimize latency on two critical paths: (1) value capture — proximity to MEV information sources (relays in SSP, signal sources in MSP), and (2) consensus — proximity to other validators for fast attestation propagation. The tension between these two forces shapes geographic equilibrium.',
      },
      {
        type: 'table',
        title: 'Latency-Critical Paths by Paradigm',
        headers: ['Path', 'SSP', 'MSP'],
        rows: [
          ['Value capture', 'Proposer -> Relay', 'Sources -> Proposer'],
          ['Consensus', 'Relay -> Attesters', 'Proposer -> Attesters'],
          ['Hops', '2 (via relay)', '1 (direct)'],
          ['Optimization target', 'Single best relay', 'Sum over all sources'],
        ],
        highlight: [3],
      },
    ],
  },
  {
    id: 'simulation-design',
    number: '§4.1',
    title: 'Simulation Design',
    description: 'Agent-based model setup, migration dynamics, and metric definitions.',
    blocks: [
      {
        type: 'stat',
        value: '40',
        label: 'GCP Regions',
        sublabel: 'Real inter-region latency measurements',
      },
      {
        type: 'stat',
        value: '10,000',
        label: 'Reported Slots',
        sublabel: 'Paper-facing public datasets',
      },
      {
        type: 'stat',
        value: '1,000',
        label: 'Validators',
        sublabel: 'Reference population in the paper runs',
      },
      {
        type: 'insight',
        text: 'Each slot, validators compare expected rewards across all 40 regions and migrate if the net benefit exceeds migration cost. The paper-facing runs are typically shown at 10,000 slots with 1,000 validators and published dataset families centered on a 0.002 ETH migration cost. The MEV function is deterministic and linear in latency, which the paper treats as a modeling limitation rather than a claim about production Ethereum.',
      },
    ],
  },
  {
    id: 'baseline-results',
    number: '§4.2',
    title: 'Baseline Results',
    description: 'Convergence analysis with the homogeneous initial distribution.',
    blocks: [
      {
        type: 'comparison',
        title: 'Baseline Convergence: SSP vs MSP',
        left: {
          label: 'SSP (External)',
          items: [
            { key: 'Convergence speed', value: 'Slower rise from the neutral baseline' },
            { key: 'Cost sensitivity', value: 'More sensitive to migration friction' },
            { key: 'Reward variance', value: 'Lower than MSP in the same setup' },
            { key: 'Geographic pull', value: 'North America remains a recurring focal hub' },
            { key: 'With migration costs', value: 'More persistence away from the tightest hubs' },
          ],
        },
        right: {
          label: 'MSP (Local)',
          items: [
            { key: 'Convergence speed', value: 'Faster rise from the same neutral baseline' },
            { key: 'Cost sensitivity', value: 'Less sensitive than SSP' },
            { key: 'Reward variance', value: 'Higher than SSP in the same setup' },
            { key: 'Geographic pull', value: 'North America dominates more quickly' },
            { key: 'Mechanism', value: 'Many-source value adds to consensus pressure' },
          ],
        },
        verdict: 'Both paradigms centralize from the homogeneous start, but MSP does so faster and usually more strongly in the paper baseline family.',
      },
    ],
  },
  {
    id: 'se1-source-placement',
    number: '§4.4',
    title: 'SE1: Information-Source Placement',
    description: 'Latency-aligned vs misaligned sources and their paradigm-specific effects.',
    blocks: [
      {
        type: 'table',
        title: 'Source Placement Effects',
        headers: ['Configuration', 'SSP Effect', 'MSP Effect'],
        rows: [
          ['Latency-aligned', 'Usually softer than the misaligned SSP case', 'Usually stronger centralization than the homogeneous MSP case'],
          ['Latency-misaligned', 'Stronger co-location pressure around the poorly connected relay', 'Can soften reward variance because source and attester pulls diverge'],
        ],
        highlight: [1],
      },
      {
        type: 'insight',
        emphasis: 'surprising',
        title: 'Opposite paradigm sensitivities',
        text: 'The same infrastructure change has opposite effects depending on the paradigm. MSP benefits from aligned sources (reinforces geographic advantages), while SSP suffers more from misaligned sources (creates a large co-location premium for the poorly-connected relay).',
      },
    ],
  },
  {
    id: 'se2-distribution',
    number: '§4.5',
    title: 'SE2: Heterogeneous Validator Distribution',
    description: 'Starting from real Ethereum validator distribution (Chainbound/Dune data).',
    blocks: [
      {
        type: 'insight',
        emphasis: 'key-finding',
        title: 'Initial conditions dominate',
        text: 'When validators start from today’s already concentrated Ethereum geography, both paradigms inherit a large share of the outcome immediately. The starting distribution matters more than the paradigm label for the first-order result, because the system begins near the eventual attractor. SSP still shows stronger amplification relative to its own neutral baseline.',
      },
    ],
  },
  {
    id: 'se3-joint',
    number: '§4.5 + App. E',
    title: 'SE3: Joint Heterogeneity',
    description: 'Combined source placement + distribution effects, including transient decentralization.',
    blocks: [
      {
        type: 'insight',
        emphasis: 'surprising',
        title: 'Transient decentralization - the only case where Gini decreases',
        text: 'The temporary dip in concentration under today’s heterogeneous validator start shows up in SSP when relay placement is poorly connected to that starting geography. Remote relays briefly pull some validators away from the existing hubs, but the effect is transient rather than a new steady state.',
      },
      {
        type: 'caveat',
        text: 'This transient decentralization effect is fragile and parameter-dependent. It should not be interpreted as a decentralization mechanism or a mitigation recipe.',
      },
    ],
  },
  {
    id: 'se4a-attestation',
    number: 'App. E.3',
    title: 'SE4a: Attestation Threshold (gamma)',
    description: 'The paper\'s most surprising finding — opposite effects across paradigms.',
    blocks: [
      {
        type: 'table',
        title: 'Directional Effect of Gamma',
        headers: ['Gamma move', 'SSP', 'MSP'],
        rows: [
          ['Lower gamma', 'Looser timing reduces relay-latency pressure', 'Less pressure to balance attesters against sources'],
          ['Higher gamma', 'Tighter timing increases centralization pressure', 'Tighter timing can disperse equilibrium by sharpening the source vs attester trade-off'],
        ],
      },
      {
        type: 'insight',
        emphasis: 'surprising',
        title: 'The only protocol parameter with opposite effects',
        text: 'SSP: higher gamma -> more centralization (tighter timing amplifies relay latency importance). MSP: higher gamma -> less centralization (forces balance between attester proximity for quorum and signal proximity for value — these point in different geographic directions). This makes attestation threshold the most paradigm-sensitive protocol parameter.',
      },
    ],
  },
  {
    id: 'se4b-slots',
    number: 'App. E.4',
    title: 'SE4b: Shorter Slot Times (EIP-7782)',
    description: 'Impact of 6-second slots vs the current 12-second slots.',
    blocks: [
      {
        type: 'insight',
        emphasis: 'key-finding',
        title: 'Same trajectories, higher reward variance',
        text: 'Centralization trajectories (Gini, HHI, LC) remain largely unchanged under 6s slots. But CV_g (reward variance) is higher — the same latency advantage becomes a larger fraction of the shortened timing window. Implication: further slot time reductions amplify inequality without changing the geographic equilibrium.',
      },
    ],
  },
  {
    id: 'discussion',
    number: '§5',
    title: 'Discussion & Mitigations',
    description: 'Policy implications and potential mitigation strategies.',
    blocks: [
      {
        type: 'table',
        title: 'Potential Mitigation Directions',
        headers: ['Strategy', 'Mechanism', 'Trade-off'],
        rows: [
          ['Geographic diversity incentives', 'Reward validators in underrepresented regions', 'May reduce overall network efficiency'],
          ['Relay/source decentralization', 'Encourage broader geographic placement of infrastructure', 'Harder to coordinate in practice'],
          ['Latency equalization', 'Protocol-level latency compensation', 'Complex to implement fairly'],
          ['Migration-friction tuning', 'Increase switching costs to slow centralization', 'Reduces validator flexibility'],
        ],
      },
      {
        type: 'caveat',
        text: 'The authors do not advocate for specific mitigations. These are research directions, not recommendations. The paper\'s contribution is diagnostic (measuring the problem) rather than prescriptive (solving it).',
      },
    ],
  },
  {
    id: 'limitations',
    number: '§5',
    title: 'Limitations',
    description: 'Acknowledged assumptions and their potential impact.',
    blocks: [
      {
        type: 'table',
        title: 'Paper Limitations',
        headers: ['Assumption', 'Impact', 'Extension'],
        rows: [
          ['GCP-only latency', 'Other providers may differ', 'Multi-cloud dataset'],
          ['Deterministic linear MEV', 'Real MEV is stochastic', 'Stochastic model'],
          ['Fungible sources', 'Real suppliers differ', 'Heterogeneous values'],
          ['Full information', 'Proposers may not know all latencies', 'Partial info model'],
          ['Constant migration cost', 'Real costs vary', 'Time-varying costs'],
          ['No strategic behavior', 'Coalitions may form', 'Game-theoretic model'],
          ['No multi-paradigm coexistence', 'SSP and MSP may coexist', 'Hybrid model'],
        ],
      },
    ],
  },
] as const
