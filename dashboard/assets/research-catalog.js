// Frozen researcher-facing dataset manifest for the local MVP dashboard.
window.RESEARCH_CATALOG = {
  introBlurb: `This project provides a lightweight interface for the simulations of the geographical decentralization of Ethereum validators that we developed as part of our research.
Our motivation is to better understand the dynamics of validator placement, latency, and performance in decentralized networks.

This frozen dashboard is for inspecting reported simulation outputs without re-running the experiments. You can switch among the checked-in published datasets and adjust viewer controls that change presentation, but you are not varying the model itself here. For new parameter choices or fresh runs, use the Simulation Lab.`,
  defaultSelection: {
    evaluation: "Baseline",
    paradigm: "Local",
    result: "cost_0.002",
    path: "simulations/baseline/MSP/cost_0.002/data.json"
  },
  datasets: [
    {
      evaluation: "Test",
      paradigm: "External",
      result: "data",
      path: "simulations/test/SSP/data/data.json",
      sourceRole: "supplier",
      metadata: {
        v: 100,
        cost: 0,
        delta: 12000,
        cutoff: 4000,
        gamma: 0.6667,
        description: "This shows the simulation results for a test setting with homogeneous validators and information sources. There are 100 validators, and we run the simulation for 1,000 slots."
      }
    },
    {
      evaluation: "Baseline",
      paradigm: "Local",
      result: "cost_0.0",
      path: "simulations/baseline/MSP/cost_0.0/data.json",
      sourceRole: "signal",
      metadata: {
        v: 1000,
        cost: 0,
        delta: 12000,
        cutoff: 4000,
        gamma: 0.6667,
        description: "These are the simulation results for the baseline configuration with homogeneous validator and information source distributions (Local Block Building). Compared to the baseline, the only difference in this experiment is the migration cost: we vary the migration cost to study its effect on validators' relocation behavior, while keeping all other parameters unchanged."
      }
    },
    {
      evaluation: "Baseline",
      paradigm: "Local",
      result: "cost_0.001",
      path: "simulations/baseline/MSP/cost_0.001/data.json",
      sourceRole: "signal",
      metadata: {
        v: 1000,
        cost: 0.001,
        delta: 12000,
        cutoff: 4000,
        gamma: 0.6667,
        description: "These are the simulation results for the baseline configuration with homogeneous validator and information source distributions (Local Block Building). Compared to the baseline, the only difference in this experiment is the migration cost: we vary the migration cost to study its effect on validators' relocation behavior, while keeping all other parameters unchanged."
      }
    },
    {
      evaluation: "Baseline",
      paradigm: "Local",
      result: "cost_0.002",
      path: "simulations/baseline/MSP/cost_0.002/data.json",
      sourceRole: "signal",
      metadata: {
        v: 1000,
        cost: 0,
        delta: 12000,
        cutoff: 4000,
        gamma: 0.6667,
        description: "These are the simulation results for the baseline configuration with homogeneous validator and information source distributions (Local Block Building)."
      }
    },
    {
      evaluation: "Baseline",
      paradigm: "Local",
      result: "cost_0.003",
      path: "simulations/baseline/MSP/cost_0.003/data.json",
      sourceRole: "signal",
      metadata: {
        v: 1000,
        cost: 0.003,
        delta: 12000,
        cutoff: 4000,
        gamma: 0.6667,
        description: "These are the simulation results for the baseline configuration with homogeneous validator and information source distributions (Local Block Building). Compared to the baseline, the only difference in this experiment is the migration cost: we vary the migration cost to study its effect on validators' relocation behavior, while keeping all other parameters unchanged."
      }
    },
    {
      evaluation: "Baseline",
      paradigm: "External",
      result: "cost_0.0",
      path: "simulations/baseline/SSP/cost_0.0/data.json",
      sourceRole: "supplier",
      metadata: {
        v: 1000,
        cost: 0,
        delta: 12000,
        cutoff: 4000,
        gamma: 0.6667,
        description: "These are the simulation results for the baseline configuration with homogeneous validator and information source distributions (External Block Building). Compared to the baseline, the only difference in this experiment is the migration cost: we vary the migration cost to study its effect on validators' relocation behavior, while keeping all other parameters unchanged."
      }
    },
    {
      evaluation: "Baseline",
      paradigm: "External",
      result: "cost_0.001",
      path: "simulations/baseline/SSP/cost_0.001/data.json",
      sourceRole: "supplier",
      metadata: {
        v: 1000,
        cost: 0.001,
        delta: 12000,
        cutoff: 4000,
        gamma: 0.6667,
        description: "These are the simulation results for the baseline configuration with homogeneous validator and information source distributions (External Block Building). Compared to the baseline, the only difference in this experiment is the migration cost: we vary the migration cost to study its effect on validators' relocation behavior, while keeping all other parameters unchanged."
      }
    },
    {
      evaluation: "Baseline",
      paradigm: "External",
      result: "cost_0.002",
      path: "simulations/baseline/SSP/cost_0.002/data.json",
      sourceRole: "supplier",
      metadata: {
        v: 1000,
        cost: 0.002,
        delta: 12000,
        cutoff: 4000,
        gamma: 0.6667,
        description: "These are the simulation results for the baseline configuration with homogeneous validator and information source distributions (External Block Building)."
      }
    },
    {
      evaluation: "Baseline",
      paradigm: "External",
      result: "cost_0.003",
      path: "simulations/baseline/SSP/cost_0.003/data.json",
      sourceRole: "supplier",
      metadata: {
        v: 1000,
        cost: 0.003,
        delta: 12000,
        cutoff: 4000,
        gamma: 0.6667,
        description: "These are the simulation results for the baseline configuration with homogeneous validator and information source distributions (External Block Building). Compared to the baseline, the only difference in this experiment is the migration cost: we vary the migration cost to study its effect on validators' relocation behavior, while keeping all other parameters unchanged."
      }
    },
    {
      evaluation: "SE1-Information-Source-Placement-Effect",
      paradigm: "Local",
      result: "latency-aligned",
      path: "simulations/heterogeneous_info/MSP/cost_0.002_latency_latency-aligned/data.json",
      sourceRole: "signal",
      metadata: {
        v: 1000,
        cost: 0.002,
        delta: 12000,
        cutoff: 4000,
        gamma: 0.6667,
        description: "These are the simulation results for SE 1: Information-Source Placement Effect — latency-aligned setting (Local Block Building)."
      }
    },
    {
      evaluation: "SE1-Information-Source-Placement-Effect",
      paradigm: "Local",
      result: "latency-misaligned",
      path: "simulations/heterogeneous_info/MSP/cost_0.002_latency_latency-misaligned/data.json",
      sourceRole: "signal",
      metadata: {
        v: 1000,
        cost: 0.002,
        delta: 12000,
        cutoff: 4000,
        gamma: 0.6667,
        description: "These are the simulation results for SE 1: Information-Source Placement Effect — latency-misaligned setting (Local Block Building)."
      }
    },
    {
      evaluation: "SE1-Information-Source-Placement-Effect",
      paradigm: "External",
      result: "latency-aligned",
      path: "simulations/heterogeneous_info/SSP/cost_0.002_latency_latency-aligned/data.json",
      sourceRole: "supplier",
      metadata: {
        v: 1000,
        cost: 0.002,
        delta: 12000,
        cutoff: 4000,
        gamma: 0.6667,
        description: "These are the simulation results for SE 1: Information-Source Placement Effect — latency-aligned setting (External Block Building)."
      }
    },
    {
      evaluation: "SE1-Information-Source-Placement-Effect",
      paradigm: "External",
      result: "latency-misaligned",
      path: "simulations/heterogeneous_info/SSP/cost_0.002_latency_latency-misaligned/data.json",
      sourceRole: "supplier",
      metadata: {
        v: 1000,
        cost: 0.002,
        delta: 12000,
        cutoff: 4000,
        gamma: 0.6667,
        description: "These are the simulation results for SE 1: Information-Source Placement Effect — latency-misaligned setting (External Block Building)."
      }
    },
    {
      evaluation: "SE2-Validator-Distribution-Effect",
      paradigm: "Local",
      result: "cost_0.0",
      path: "simulations/heterogeneous_validators/MSP/cost_0.0_validators_heterogeneous/data.json",
      sourceRole: "signal",
      metadata: {
        v: 1000,
        cost: 0,
        delta: 12000,
        cutoff: 4000,
        gamma: 0.6667,
        description: "These are the simulation results for SE 2: Validator Distribution Effect - heterogeneous validator distribution (Local Block Building)."
      }
    },
    {
      evaluation: "SE2-Validator-Distribution-Effect",
      paradigm: "Local",
      result: "cost_0.002",
      path: "simulations/heterogeneous_validators/MSP/cost_0.002_validators_heterogeneous/data.json",
      sourceRole: "signal",
      metadata: {
        v: 1000,
        cost: 0.002,
        delta: 12000,
        cutoff: 4000,
        gamma: 0.6667,
        description: "These are the simulation results for SE 2: Validator Distribution Effect - heterogeneous validator distribution (Local Block Building)."
      }
    },
    {
      evaluation: "SE2-Validator-Distribution-Effect",
      paradigm: "External",
      result: "cost_0.0",
      path: "simulations/heterogeneous_validators/SSP/cost_0.0_validators_heterogeneous/data.json",
      sourceRole: "supplier",
      metadata: {
        v: 1000,
        cost: 0.002,
        delta: 12000,
        cutoff: 4000,
        gamma: 0.6667,
        description: "These are the simulation results for SE 2: Validator Distribution Effect - heterogeneous validator distribution (External Block Building)."
      }
    },
    {
      evaluation: "SE2-Validator-Distribution-Effect",
      paradigm: "External",
      result: "cost_0.002",
      path: "simulations/heterogeneous_validators/SSP/cost_0.002_validators_heterogeneous/data.json",
      sourceRole: "supplier",
      metadata: {
        v: 1000,
        cost: 0.002,
        delta: 12000,
        cutoff: 4000,
        gamma: 0.6667,
        description: "These are the simulation results for SE 2: Validator Distribution Effect - heterogeneous validator distribution (External Block Building)."
      }
    },
    {
      evaluation: "SE3-Joint-Heterogeneity",
      paradigm: "Local",
      result: "latency-aligned",
      path: "simulations/heterogeneous_both/MSP/cost_0.002_latency_latency-aligned/data.json",
      sourceRole: "signal",
      metadata: {
        v: 1000,
        cost: 0.002,
        delta: 12000,
        cutoff: 4000,
        gamma: 0.6667,
        description: "These are the simulation results for SE 3: Joint Heterogeneity — latency-aligned setting (Local Block Building)."
      }
    },
    {
      evaluation: "SE3-Joint-Heterogeneity",
      paradigm: "Local",
      result: "latency-misaligned",
      path: "simulations/heterogeneous_both/MSP/cost_0.002_latency_latency-misaligned/data.json",
      sourceRole: "signal",
      metadata: {
        v: 1000,
        cost: 0.002,
        delta: 12000,
        cutoff: 4000,
        gamma: 0.6667,
        description: "These are the simulation results for SE 3: Joint Heterogeneity — latency-misaligned setting (Local Block Building)."
      }
    },
    {
      evaluation: "SE3-Joint-Heterogeneity",
      paradigm: "External",
      result: "latency-aligned",
      path: "simulations/heterogeneous_both/SSP/cost_0.002_latency_latency-aligned/data.json",
      sourceRole: "supplier",
      metadata: {
        v: 1000,
        cost: 0.002,
        delta: 12000,
        cutoff: 4000,
        gamma: 0.6667,
        description: "These are the simulation results for SE 3: Joint Heterogeneity — latency-aligned setting (External Block Building)."
      }
    },
    {
      evaluation: "SE3-Joint-Heterogeneity",
      paradigm: "External",
      result: "latency-misaligned",
      path: "simulations/heterogeneous_both/SSP/cost_0.002_latency_latency-misaligned/data.json",
      sourceRole: "supplier",
      metadata: {
        v: 1000,
        cost: 0.002,
        delta: 12000,
        cutoff: 4000,
        gamma: 0.6667,
        description: "These are the simulation results for SE 3: Joint Heterogeneity — latency-misaligned setting (External Block Building)."
      }
    },
    {
      evaluation: "SE4-Attestation-Threshold",
      paradigm: "Local",
      result: "gamma_0.3333",
      path: "simulations/different_gammas/MSP/cost_0.002_gamma_0.3333/data.json",
      sourceRole: "signal",
      metadata: {
        v: 1000,
        cost: 0.002,
        delta: 12000,
        cutoff: 4000,
        gamma: 0.3333,
        description: "These are the simulation results for Experiment SE 4: Consensus Parameter Effects — Attestation Threshold Effect (Local Block Building). The attestation threshold γ is set to 0.3333."
      }
    },
    {
      evaluation: "SE4-Attestation-Threshold",
      paradigm: "Local",
      result: "gamma_0.5",
      path: "simulations/different_gammas/MSP/cost_0.002_gamma_0.5/data.json",
      sourceRole: "signal",
      metadata: {
        v: 1000,
        cost: 0.002,
        delta: 12000,
        cutoff: 4000,
        gamma: 0.5,
        description: "These are the simulation results for Experiment SE 4: Consensus Parameter Effects — Attestation Threshold Effect (Local Block Building). The attestation threshold γ is set to 0.5."
      }
    },
    {
      evaluation: "SE4-Attestation-Threshold",
      paradigm: "Local",
      result: "gamma_0.6667",
      path: "simulations/different_gammas/MSP/cost_0.002_gamma_0.6667/data.json",
      sourceRole: "signal",
      metadata: {
        v: 1000,
        cost: 0.002,
        delta: 12000,
        cutoff: 4000,
        gamma: 0.6667,
        description: "These are the simulation results for Experiment SE 4: Consensus Parameter Effects — Attestation Threshold Effect (Local Block Building). The attestation threshold γ is set to 0.6667."
      }
    },
    {
      evaluation: "SE4-Attestation-Threshold",
      paradigm: "Local",
      result: "gamma_0.8",
      path: "simulations/different_gammas/MSP/cost_0.002_gamma_0.8/data.json",
      sourceRole: "signal",
      metadata: {
        v: 1000,
        cost: 0.002,
        delta: 12000,
        cutoff: 4000,
        gamma: 0.8,
        description: "These are the simulation results for Experiment SE 4: Consensus Parameter Effects — Attestation Threshold Effect (Local Block Building). The attestation threshold γ is set to 0.8."
      }
    },
    {
      evaluation: "SE4-Attestation-Threshold",
      paradigm: "External",
      result: "gamma_0.3333",
      path: "simulations/different_gammas/SSP/cost_0.002_gamma_0.3333/data.json",
      sourceRole: "supplier",
      metadata: {
        v: 1000,
        cost: 0.002,
        delta: 12000,
        cutoff: 4000,
        gamma: 0.3333,
        description: "These are the simulation results for Experiment SE 4: Consensus Parameter Effects — Attestation Threshold Effect (External Block Building). The attestation threshold γ is set to 0.3333."
      }
    },
    {
      evaluation: "SE4-Attestation-Threshold",
      paradigm: "External",
      result: "gamma_0.5",
      path: "simulations/different_gammas/SSP/cost_0.002_gamma_0.5/data.json",
      sourceRole: "supplier",
      metadata: {
        v: 1000,
        cost: 0.002,
        delta: 12000,
        cutoff: 4000,
        gamma: 0.5,
        description: "These are the simulation results for Experiment SE 4: Consensus Parameter Effects — Attestation Threshold Effect (External Block Building). The attestation threshold γ is set to 0.5."
      }
    },
    {
      evaluation: "SE4-Attestation-Threshold",
      paradigm: "External",
      result: "gamma_0.6667",
      path: "simulations/different_gammas/SSP/cost_0.002_gamma_0.6667/data.json",
      sourceRole: "supplier",
      metadata: {
        v: 1000,
        cost: 0.002,
        delta: 12000,
        cutoff: 4000,
        gamma: 0.6667,
        description: "These are the simulation results for Experiment SE 4: Consensus Parameter Effects — Attestation Threshold Effect (External Block Building). The attestation threshold γ is set to 0.6667."
      }
    },
    {
      evaluation: "SE4-Attestation-Threshold",
      paradigm: "External",
      result: "gamma_0.8",
      path: "simulations/different_gammas/SSP/cost_0.002_gamma_0.8/data.json",
      sourceRole: "supplier",
      metadata: {
        v: 1000,
        cost: 0.002,
        delta: 12000,
        cutoff: 4000,
        gamma: 0.8,
        description: "These are the simulation results for Experiment SE 4: Consensus Parameter Effects — Attestation Threshold Effect (External Block Building). The attestation threshold γ is set to 0.8."
      }
    },
    {
      evaluation: "SE4-EIP7782",
      paradigm: "Local",
      result: "delta_6000_cutoff_3000",
      path: "simulations/eip7782/MSP/cost_0.002_delta_6000_cutoff_3000/data.json",
      sourceRole: "signal",
      metadata: {
        v: 1000,
        cost: 0.002,
        delta: 6000,
        cutoff: 3000,
        gamma: 0.6667,
        description: "These are the simulation results for Experiment SE 4: Consensus Parameter Effects — Shorter Slot Time (Local Block Building). The slot time Δ is set to 6000 ms, and the cutoff time is set to 3000 ms."
      }
    },
    {
      evaluation: "SE4-EIP7782",
      paradigm: "External",
      result: "delta_6000_cutoff_3000",
      path: "simulations/eip7782/SSP/cost_0.002_delta_6000_cutoff_3000/data.json",
      sourceRole: "supplier",
      metadata: {
        v: 1000,
        cost: 0.002,
        delta: 6000,
        cutoff: 3000,
        gamma: 0.6667,
        description: "These are the simulation results for Experiment SE 4: Consensus Parameter Effects — Shorter Slot Time (External Block Building). The slot time Δ is set to 6000 ms, and the cutoff time is set to 3000 ms."
      }
    }
  ]
};
