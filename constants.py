from dataclasses import dataclass
# Constants
SLOT_DURATION_MS = 12000  # Duration of an Ethereum slot in milliseconds
TIME_GRANULARITY_MS = 100  # Simulation time step in milliseconds
ATTESTATION_TIME_MS = (
    4000  # Default time for Attesters to attest (can be varied per Attester)
)

# Network Latency Model Parameters: latency = BASE_NETWORK_LATENCY_MS + (distance_ratio * MAX_ADDITIONAL_NETWORK_LATENCY_MS)
BASE_NETWORK_LATENCY_MS = 50  # Minimum network latency regardless of distance
MAX_ADDITIONAL_NETWORK_LATENCY_MS = (
    2000  # Max additional latency for max distance on sphere
)

# MEV yield model (simulating Builder bids increasing over time)
## The number is set randomly now.
BASE_MEV_AMOUNT = 0.2  # Initial MEV in ETH
MEV_INCREASE_PER_SECOND = 0.08  # MEV increase per second (ETH/sec)

# Consensus related constants
ATTESTATION_THRESHOLD = (
    2 / 3
)  # Threshold for attestation to be valid (2/3 of validators)
# Consensus reward parameters
TIMELY_HEAD_REWARD = 14
TIMELY_SOURCE_REWARD = 14
TIMELY_TARGET_REWARD = 14
PROPOSER_REWARD = 4
SYNC_COMMITTEE_REWARD = 2

# Percentage of validators that are cloud-based
CLOUD_VALIDATOR_PERCENTAGE = 1
# Percentage of validators that are non-compliant (i.e., willing to use noncensoring relay)
NON_COMPLIANT_VALIDATOR_PERCENTAGE = 1


@dataclass(frozen=True)
class LinearMEVUtility:
    base_mev: float
    mev_increase: float
    multiplier: float = 1.0

    def __call__(self, x: float) -> float:
        return (self.base_mev * self.multiplier) + (x * self.mev_increase * self.multiplier)
