from constants import (
    ATTESTATION_THRESHOLD,
    ATTESTATION_TIME_MS,
    SLOT_DURATION_MS,
    TIME_GRANULARITY_MS,
    TIMELY_HEAD_REWARD,
    TIMELY_SOURCE_REWARD,
    TIMELY_TARGET_REWARD,
    PROPOSER_REWARD,
    SYNC_COMMITTEE_REWARD,
)

class ConsensusSettings:
    def __init__(
            self,
            slot_duration_ms=SLOT_DURATION_MS,
            time_granularity_ms=TIME_GRANULARITY_MS,
            attestation_time_ms=ATTESTATION_TIME_MS,
            attestation_threshold=ATTESTATION_THRESHOLD,
            timely_head_reward=TIMELY_HEAD_REWARD,
            timely_source_reward=TIMELY_SOURCE_REWARD,
            timely_target_reward=TIMELY_TARGET_REWARD,
            proposer_reward=PROPOSER_REWARD,
            sync_committee_reward=SYNC_COMMITTEE_REWARD,
        ):
        self.slot_duration_ms = slot_duration_ms
        self.time_granularity_ms = time_granularity_ms
        self.attestation_time_ms = attestation_time_ms
        self.attestation_threshold = attestation_threshold
        self.timely_head_reward = timely_head_reward
        self.timely_source_reward = timely_source_reward
        self.timely_target_reward = timely_target_reward
        self.proposer_reward = proposer_reward
        self.sync_committee_reward = sync_committee_reward

    def __repr__(self):
        return (
            f"ConsensusSettings(\n"
            f"  slot_duration_ms={self.slot_duration_ms},\n"
            f"  time_granularity_ms={self.time_granularity_ms},\n"
            f"  attestation_time_ms={self.attestation_time_ms},\n"
            f"  attestation_threshold={self.attestation_threshold},\n"
            f"  timely_head_reward={self.timely_head_reward},\n"
            f"  timely_source_reward={self.timely_source_reward},\n"
            f"  timely_target_reward={self.timely_target_reward},\n"
            f"  proposer_reward={self.proposer_reward},\n"
            f"  sync_committee_reward={self.sync_committee_reward}\n"
            f")"
        )
