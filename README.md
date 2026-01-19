# 🌍 Geographical Decentralization Simulation

This repository contains the simulation and evaluation code for the paper "Geographical Centralization Resilience in Ethereum's Block-Building Paradigms".

## Installation

Please install Python and the dependencies by running the following command:
```bash
pip install -r requirements.txt
```

## Evaluations

### Baseline

Run the simulation with homogeneous validators and homogeneous information sources.

```bash
cd evaluations
fab run-baseline
```

Plot the results.
```bash
cd plot
fab plot-baseline
```

### SE 1: Information-Source Placement Effect

Run the simulation with homogeneous validators but heterogeneous information sources. Specifically, we focus on two cases:
- `latency-aligned`: Information sources are placed in regions with low latency (Asia, Europe, and North America).
- `latency-misaligned`: Information sources are placed in regions with high latency (Africa, Oceania, South America).

```bash
cd evaluations
fab run-heterogeneous-information-sources
```

Plot the results.
```bash
cd evaluations
fab plot-heterogeneous-information-sources
```

### SE 2: Validator Distribution Effect
Run the simulation with homogeneous information sources but heterogeneous validators. Specifically, the validators are sampled from the [real-world distribution](https://dune.com/data/dune.rig_ef.validator_metadata).

```bash
cd evaluations
fab run-heterogeneous-validators
```

Plot the results.
```bash
cd plot
fab plot-heterogeneous_validators
```

### SE 3: Joint Heterogeneity
Run the simulation with heterogeneous validators and heterogeneous information sources.

```bash
cd evaluations
fab run-hetero-both
```

Polt the results.
```bash
cd plot
fab plot-hetero-both
```

### SE 4: Consensus-Parameter Effect

We also test other settings to further understand how consensus changes would affect geographical decentralization.

#### Attestation Threshold Effect

```bash
# test different \gamma (consensus threshold)
cd evaluations
fab run-different-gammas

# plot different \gamma (consensus threshold)
fab plot-different-gammas
```

#### Shorter Slot Time Effect

```bash
# test eip-7782
cd evaluations
fab run-eip7782 

# plot eip-7782
cd plot
fab plot-eip7782
```

### Other Figures

#### Validator Distribution and Inter-region Internet Latencies 

```bash
cd plot
python3 country_density_plus_continent_latency.py
```

#### Heatmap of Median Latency

```bash
cd plot
python3 latency_heatmap.py
```

#### Marginal Benefit Distribution

```bash
cd plot
python3 marginal_benefit.py
```

#### Experiments with Different Scales

```bash
# plot different-scale
cd plot
fab plot-different-scale
```

#### Migration Costs

```bash
# plot cost
cd plot
fab plot-cost
```

#### Validator Convergence Locus

Two figures on validator convergence are also generated when running `fab plot-baseline` and `fab plot-hetero-both`.