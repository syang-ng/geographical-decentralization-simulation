import type { Block } from '../types/blocks'
import { GCP_REGIONS } from './gcp-regions'

// 9 blocks covering all 9 block types — the "executive summary" of the paper.
// Zero API calls. Hand-crafted from Yang et al. (2025) arXiv:2509.21475.

export const DEFAULT_BLOCKS: readonly Block[] = [
  // Row 1: Key stats (3-up grid)
  {
    type: 'stat',
    value: '2',
    label: 'Paradigms Compared',
    sublabel: 'SSP (external) vs MSP (local) block building',
  },
  {
    type: 'stat',
    value: '40',
    label: 'GCP Regions Simulated',
    sublabel: 'across 7 macro-regions worldwide',
  },
  {
    type: 'stat',
    value: '7',
    label: 'Experiments Analyzed',
    sublabel: 'baseline + 6 sensitivity evaluations',
  },

  // Row 2: Core finding (key-finding emphasis)
  {
    type: 'insight',
    emphasis: 'key-finding',
    title: 'Both paradigms centralize, but differently',
    text: 'Both SSP and MSP push toward geographic concentration but through **opposite mechanisms** with different protocol sensitivities. SSP validators cluster at relay locations (US-East, EU-West) to minimize proposer-relay latency. MSP validators cluster along the Atlantic corridor where proximity to many distributed signals AND many attesters overlaps. Under baseline conditions, **MSP centralizes faster and more severely** than SSP.',
  },

  // Row 3: Head-to-head comparison
  {
    type: 'comparison',
    title: 'SSP vs MSP: Baseline Centralization',
    left: {
      label: 'SSP (External)',
      items: [
        { key: 'Gini_g', value: 'Moderate ↑' },
        { key: 'HHI_g', value: 'Moderate ↑' },
        { key: 'CV_g', value: 'Moderate ↑' },
        { key: 'LC_g', value: 'Moderate ↓' },
        { key: 'Locus', value: 'NA + Middle East' },
      ],
    },
    right: {
      label: 'MSP (Local)',
      items: [
        { key: 'Gini_g', value: 'Higher ↑' },
        { key: 'HHI_g', value: 'Higher ↑' },
        { key: 'CV_g', value: 'Higher ↑' },
        { key: 'LC_g', value: 'Sharper ↓' },
        { key: 'Locus', value: 'NA + EU' },
      ],
    },
    verdict: 'MSP centralizes faster in baseline conditions — higher Gini/HHI, lower LC, and consistently larger reward disparities (CV_g).',
  },

  // Row 4: Surprising finding
  {
    type: 'insight',
    emphasis: 'surprising',
    title: 'Attestation threshold has opposite effects',
    text: 'Higher γ (attestation threshold) → SSP centralizes **MORE** but MSP centralizes **LESS**. In SSP, tighter timing amplifies latency sensitivity — reducing proposer-relay latency yields larger marginal MEV. In MSP, a higher threshold forces proposers to balance attester proximity (quorum) vs signal proximity (value), and these point in **different geographic directions**, dispersing rather than concentrating validators. This is the only protocol parameter with opposite effects across paradigms.',
  },

  // Row 5: Geographic canvas (MapBlock with all 40 regions)
  {
    type: 'map',
    title: 'Simulation Geographic Canvas — 40 GCP Regions',
    regions: GCP_REGIONS.map(r => ({
      name: r.id,
      lat: r.lat,
      lon: r.lon,
      value: 25, // homogeneous initial distribution
      label: r.city.split(',')[0],
    })),
    colorScale: 'binary',
  },

  // Row 6a: Caveat
  {
    type: 'caveat',
    text: 'These findings are derived from agent-based simulation using GCP-only latency data. Real validator behavior involves additional factors: stochastic MEV, heterogeneous migration costs, non-fungible information sources, and incomplete latency information. The deterministic linear MEV function is a simplifying assumption.',
  },

  // Row 6b: Sources
  {
    type: 'source',
    refs: [
      {
        label: 'arXiv:2509.21475',
        section: 'Full paper',
        url: 'https://arxiv.org/abs/2509.21475',
      },
      {
        label: 'GitHub: syang-ng/geographical-decentralization-simulation',
        section: 'Source code + data',
        url: 'https://github.com/syang-ng/geographical-decentralization-simulation',
      },
      {
        label: 'Yang, Oz, Wu, Zhang (2025)',
        section: 'Authors',
      },
    ],
  },
] as const

// The 8 pre-rendered topic cards for Tier 1 (zero API cost)
export interface TopicCard {
  readonly id: string
  readonly title: string
  readonly description: string
  readonly prompts: readonly string[]
  readonly blocks: readonly Block[]
}

export const OVERVIEW_CARD: TopicCard = {
  id: 'overview',
  title: 'Main findings',
  description: 'A curated overview of the paper’s central results and caveats.',
  prompts: [
    'What are the main findings?',
    'Give me a summary of the paper.',
    'What does the paper find?',
    'What is the main takeaway?',
    'Give me the key findings.',
  ],
  blocks: DEFAULT_BLOCKS,
}

export const TOPIC_CARDS: readonly TopicCard[] = [
  {
    id: 'ssp-vs-msp',
    title: 'SSP vs MSP: Which centralizes more?',
    description: 'Head-to-head comparison of external vs local block building under baseline conditions.',
    prompts: [
      'How does SSP compare to MSP?',
      'Which paradigm centralizes more?',
      'SSP vs MSP',
      'Compare external and local block building.',
    ],
    blocks: [
      {
        type: 'comparison',
        title: 'SSP vs MSP: Baseline Centralization Metrics',
        left: {
          label: 'SSP (External)',
          items: [
            { key: 'Mechanism', value: 'Co-locate with relay' },
            { key: 'Path', value: 'Proposer→Relay→Attesters (2 hops)' },
            { key: 'Centralizing force', value: 'Relay latency dominates' },
            { key: 'Convergence locus', value: 'NA + Middle East' },
          ],
        },
        right: {
          label: 'MSP (Local)',
          items: [
            { key: 'Mechanism', value: 'Optimize signal+attester proximity' },
            { key: 'Path', value: 'Proposer→Attesters (1 hop)' },
            { key: 'Centralizing force', value: 'Distributed pull to many sources' },
            { key: 'Convergence locus', value: 'NA primary, EU secondary' },
          ],
        },
        verdict: 'Both centralize, but MSP is faster and more severe under baseline homogeneous conditions.',
      },
      {
        type: 'insight',
        emphasis: 'key-finding',
        title: 'Key mechanical difference',
        text: 'SSP evaluates all (region, relay) pairs and picks the single best. MSP sums all signal offers per region — the value function is additive over sources, creating a fundamentally different optimization landscape.',
      },
    ],
  },
  {
    id: 'geographic-convergence',
    title: 'Where do validators end up?',
    description: 'Geographic convergence patterns across experiments and paradigms.',
    prompts: [
      'Where do validators concentrate geographically?',
      'Where do validators end up?',
      'Where do they converge?',
      'What regions win out?',
    ],
    blocks: [
      {
        type: 'table',
        title: 'Convergence Loci by Paradigm and Experiment',
        headers: ['Experiment', 'SSP Convergence', 'MSP Convergence'],
        rows: [
          ['Baseline (Uniform)', 'NA + Middle East', 'NA primary, SA/Africa→EU'],
          ['SE1: Aligned sources', 'Low-latency hubs', 'Same hubs (reinforced)'],
          ['SE1: Misaligned sources', 'Peripheral relay region', 'Balanced (lower CV_g)'],
          ['SE2: Real ETH distribution', 'US + EU (amplified)', 'US + EU (rapid convergence)'],
          ['SE4a: Higher γ', 'Stronger NA/EU', 'More dispersed'],
        ],
        highlight: [4],
      },
      {
        type: 'insight',
        text: 'The convergence locus depends on where **information sources** are placed (SE1), but **real Ethereum validator distribution** already concentrates in US+EU, so both paradigms converge there rapidly when starting from realistic conditions (SE2).',
      },
    ],
  },
  {
    id: 'source-placement',
    title: 'Does source placement matter?',
    description: 'SE1: How aligned vs misaligned information sources affect centralization.',
    prompts: [
      'Does source placement matter?',
      'What happens with latency-aligned sources?',
      'What happens with latency-misaligned sources?',
      'How do source locations change centralization?',
    ],
    blocks: [
      {
        type: 'insight',
        emphasis: 'key-finding',
        title: 'Opposite paradigm sensitivities to source placement',
        text: 'MSP: latency-**aligned** sources centralize MORE (low-latency regions benefit both value capture and propagation). SSP: latency-**misaligned** sources centralize MORE (poorly connected relays create a large proposer-relay gap, making co-location extremely valuable). The same infrastructure change has **opposite effects** depending on the paradigm.',
      },
      {
        type: 'caveat',
        text: 'Exception: MSP + misaligned sources produces LOWER CV_g (reward variance) than baseline — the trade-off between signal proximity and attester proximity creates more balanced rewards even as geographic concentration increases.',
      },
    ],
  },
  {
    id: 'initial-distribution',
    title: 'What if validators start concentrated?',
    description: 'SE2: Using real Ethereum validator distribution from Chainbound/Dune data.',
    prompts: [
      'What if validators start concentrated?',
      'What happens with real Ethereum validator distribution?',
      'What if validators begin in US and Europe?',
      'How does heterogeneous validator distribution change things?',
    ],
    blocks: [
      {
        type: 'insight',
        emphasis: 'key-finding',
        title: 'No substantial difference between paradigms',
        text: 'When starting from the real Ethereum distribution (already concentrated in US + EU), metrics start elevated and both paradigms converge rapidly to co-location equilibrium. **The starting distribution matters more than the paradigm choice** when validators are already concentrated.',
      },
      {
        type: 'insight',
        text: 'SSP shows stronger **amplification** of reward disparities relative to its baseline — the gap between best and worst regions grows faster under SSP when starting concentrated. This suggests SSP is more sensitive to initial conditions.',
      },
    ],
  },
  {
    id: 'attestation-threshold',
    title: 'The attestation threshold surprise',
    description: 'SE4a: Why higher γ centralizes SSP more but MSP less.',
    prompts: [
      'Why does attestation threshold have opposite effects?',
      'What happens when gamma changes?',
      'Explain the attestation threshold result.',
      'How does higher gamma affect SSP and MSP?',
    ],
    blocks: [
      {
        type: 'chart',
        title: 'Attestation Threshold (γ) Effect on Centralization',
        data: [
          { label: 'γ = 1/3', value: 30, category: 'SSP' },
          { label: 'γ = 1/2', value: 45, category: 'SSP' },
          { label: 'γ = 2/3', value: 65, category: 'SSP' },
          { label: 'γ = 4/5', value: 80, category: 'SSP' },
        ],
        unit: '% centralization index',
        chartType: 'bar',
      },
      {
        type: 'insight',
        emphasis: 'surprising',
        title: 'Opposite protocol lever',
        text: 'In SSP, tighter timing (higher γ) amplifies latency sensitivity — reducing proposer-relay latency yields **larger marginal MEV gains**. In MSP, higher γ forces proposers to balance attester proximity (quorum) vs signal proximity (value). These point in **different geographic directions**, so tightening threshold disperses rather than concentrates. This is the paper\'s most surprising finding.',
      },
    ],
  },
  {
    id: 'shorter-slots',
    title: 'Shorter slot times (EIP-7782)',
    description: 'SE4b: What happens with 6-second slots instead of 12.',
    prompts: [
      'What happens with shorter slots?',
      'What about EIP-7782?',
      'What changes with 6-second slots?',
      'How do shorter slot times affect centralization?',
    ],
    blocks: [
      {
        type: 'insight',
        emphasis: 'key-finding',
        title: 'Trajectories unchanged, reward variance higher',
        text: 'Centralization trajectories (Gini, HHI, LC) remain largely **unchanged** under 6s slots. But CV_g (reward variance) is **higher** for both paradigms — the same latency advantage becomes a larger fraction of the shortened timing window, amplifying reward disparities.',
      },
      {
        type: 'caveat',
        text: 'Implication: further slot time reductions (beyond EIP-7782) may strengthen migration incentives without changing the geographic equilibrium, creating a more unequal but similarly centralized network.',
      },
    ],
  },
  {
    id: 'metrics-explained',
    title: 'Key metrics explained',
    description: 'Understanding Gini_g, HHI_g, CV_g, and LC_g.',
    prompts: [
      'Explain the paper metrics.',
      'What are Gini_g, HHI_g, CV_g, and LC_g?',
      'How should I read the metrics?',
      'What metrics does the paper use?',
    ],
    blocks: [
      {
        type: 'table',
        title: 'Paper Metrics — Geographic Concentration Measures',
        headers: ['Metric', 'Range', 'Interpretation', 'Ideal (decentralized)'],
        rows: [
          ['Gini_g', '0 → 1', 'Stake inequality across regions', '→ 0 (even distribution)'],
          ['HHI_g', '1/|R| → 1', 'Herfindahl-Hirschman Index', '→ 1/40 = 0.025'],
          ['CV_g', '0 → ∞', 'Coefficient of variation of payoffs', '→ 0 (equal rewards)'],
          ['LC_g', '1 → |R|', 'Min regions to break liveness (Nakamoto coeff.)', '→ 40 (max resilience)'],
        ],
        highlight: [3],
      },
      {
        type: 'caveat',
        text: 'These are NOT the same as measure.py\'s metrics (NNI, Moran\'s I, Geary\'s C). The paper uses custom geographic concentration metrics; the Dash visualization uses spatial statistics metrics.',
      },
    ],
  },
  {
    id: 'limitations',
    title: 'Limitations & what\'s next',
    description: 'What the authors acknowledge and open research directions.',
    prompts: [
      'What are the limitations?',
      'What caveats should I keep in mind?',
      'What assumptions does the paper make?',
      'What are the next research directions?',
    ],
    blocks: [
      {
        type: 'table',
        title: 'Paper Limitations',
        headers: ['Limitation', 'Impact', 'Possible Extension'],
        rows: [
          ['GCP-only latency data', 'Other providers may differ', 'Multi-cloud latency dataset'],
          ['Deterministic linear MEV', 'Real MEV is stochastic', 'Stochastic MEV model'],
          ['Fungible info sources', 'Real suppliers differ in value', 'Heterogeneous source values'],
          ['Full-information assumption', 'Proposers may not know all latencies', 'Partial information model'],
          ['Constant migration cost', 'Real costs vary over time', 'Time-varying cost functions'],
        ],
      },
      {
        type: 'source',
        refs: [
          { label: 'Section 7 — Limitations', section: 'Full discussion of assumptions' },
          { label: 'arXiv:2509.21475', url: 'https://arxiv.org/abs/2509.21475' },
        ],
      },
    ],
  },
] as const
