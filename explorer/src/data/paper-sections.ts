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
    'Both SSP and MSP centralize geographically, but through different latency-critical paths.',
    'MSP concentrates faster in baseline conditions, while SSP becomes especially sensitive to relay placement and attestation timing.',
    'Initial validator distribution and infrastructure placement can matter as much as, or more than, paradigm choice.',
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
        text: 'Validators choose GCP regions to minimize latency on two critical paths: (1) value capture — proximity to MEV information sources (relays in SSP, orderflow providers in MSP), and (2) consensus — proximity to other validators for fast attestation propagation. The tension between these two forces shapes geographic equilibrium.',
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
    number: '§4',
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
        value: '1,000',
        label: 'Simulation Slots',
        sublabel: 'Default convergence window',
      },
      {
        type: 'stat',
        value: '100',
        label: 'Validators',
        sublabel: 'Default population size',
      },
      {
        type: 'insight',
        text: 'Each slot, validators compare expected rewards across all 40 regions and migrate if the net benefit exceeds the migration cost (default 0.0001 ETH). The MEV function is deterministic and linear in latency — a simplifying assumption acknowledged as a limitation. Fast mode uses expected-value calculations instead of Monte Carlo sampling.',
      },
    ],
  },
  {
    id: 'baseline-results',
    number: '§5.1',
    title: 'Baseline Results',
    description: 'Convergence analysis with the homogeneous initial distribution.',
    blocks: [
      {
        type: 'comparison',
        title: 'Baseline Convergence: SSP vs MSP',
        left: {
          label: 'SSP (External)',
          items: [
            { key: 'Final Gini_g', value: '~0.40' },
            { key: 'Final HHI_g', value: '~0.06' },
            { key: 'Final LC_g', value: '~8 regions' },
            { key: 'Convergence speed', value: 'Gradual' },
            { key: 'Locus', value: 'NA + Middle East' },
          ],
        },
        right: {
          label: 'MSP (Local)',
          items: [
            { key: 'Final Gini_g', value: '~0.55' },
            { key: 'Final HHI_g', value: '~0.10' },
            { key: 'Final LC_g', value: '~4 regions' },
            { key: 'Convergence speed', value: 'Rapid' },
            { key: 'Locus', value: 'NA + EU' },
          ],
        },
        verdict: 'MSP centralizes 37-80% more across all metrics, reaching liveness threshold faster.',
      },
    ],
  },
  {
    id: 'se1-source-placement',
    number: '§5.2.1',
    title: 'SE1: Information-Source Placement',
    description: 'Latency-aligned vs misaligned sources and their paradigm-specific effects.',
    blocks: [
      {
        type: 'table',
        title: 'Source Placement Effects',
        headers: ['Configuration', 'SSP Effect', 'MSP Effect'],
        rows: [
          ['Latency-aligned', 'Moderate centralization', 'HIGHER centralization'],
          ['Latency-misaligned', 'HIGHER centralization', 'Lower CV_g (balanced trade-offs)'],
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
    number: '§5.2.2',
    title: 'SE2: Heterogeneous Validator Distribution',
    description: 'Starting from real Ethereum validator distribution (Chainbound/Dune data).',
    blocks: [
      {
        type: 'insight',
        emphasis: 'key-finding',
        title: 'Initial conditions dominate',
        text: 'When validators start already concentrated in US + EU (matching real Ethereum), both paradigms converge rapidly. The starting distribution matters more than the paradigm choice — metrics start elevated and quickly reach co-location equilibrium. SSP shows stronger amplification of reward disparities relative to its own baseline.',
      },
    ],
  },
  {
    id: 'se3-joint',
    number: '§5.2.3',
    title: 'SE3: Joint Heterogeneity',
    description: 'Combined source placement + distribution effects, including transient decentralization.',
    blocks: [
      {
        type: 'insight',
        emphasis: 'surprising',
        title: 'Transient decentralization - the only case where Gini decreases',
        text: 'MSP + misaligned sources + heterogeneous distribution is the only configuration where the Gini coefficient temporarily decreases (early slots show lower concentration). This happens because misaligned sources create competing geographic pulls that temporarily scatter validators. But it is NOT a steady state — eventually re-centralizes.',
      },
      {
        type: 'caveat',
        text: 'This transient decentralization effect is fragile and parameter-dependent. It should not be interpreted as a decentralization mechanism — it is a temporary artifact of competing forces before equilibrium.',
      },
    ],
  },
  {
    id: 'se4a-attestation',
    number: '§5.3.1',
    title: 'SE4a: Attestation Threshold (gamma)',
    description: 'The paper\'s most surprising finding — opposite effects across paradigms.',
    blocks: [
      {
        type: 'chart',
        title: 'Centralization vs Attestation Threshold',
        data: [
          { label: 'gamma = 1/3', value: 30, category: 'SSP' },
          { label: 'gamma = 1/2', value: 45, category: 'SSP' },
          { label: 'gamma = 2/3', value: 65, category: 'SSP' },
          { label: 'gamma = 4/5', value: 80, category: 'SSP' },
        ],
        unit: '% centralization index',
        chartType: 'bar',
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
    number: '§5.3.2',
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
    number: '§6',
    title: 'Discussion & Mitigations',
    description: 'Policy implications and potential mitigation strategies.',
    blocks: [
      {
        type: 'table',
        title: 'Potential Mitigation Directions',
        headers: ['Strategy', 'Mechanism', 'Trade-off'],
        rows: [
          ['Geographic diversity incentives', 'Reward validators in underrepresented regions', 'May reduce overall network efficiency'],
          ['Relay/source decentralization', 'Mandate geographic distribution of infrastructure', 'Harder to enforce in practice'],
          ['Latency equalization', 'Protocol-level latency compensation', 'Complex to implement fairly'],
          ['Migration cost tuning', 'Increase switching costs to slow centralization', 'Reduces validator flexibility'],
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
    number: '§7',
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
