from constants import (
    ATTESTATION_THRESHOLD,
    ATTESTATION_TIME_MS,
    SLOT_DURATION_MS,
    TIME_GRANULARITY_MS,
)

class ConsensusSettings:
    def __init__(
            self,
            slot_duration_ms=SLOT_DURATION_MS,
            time_granularity_ms=TIME_GRANULARITY_MS,
            attestation_time_ms=ATTESTATION_TIME_MS,
            attestation_threshold=ATTESTATION_THRESHOLD
        ):
        self.slot_duration_ms = slot_duration_ms
        self.time_granularity_ms = time_granularity_ms
        self.attestation_time_ms = attestation_time_ms
        self.attestation_threshold = attestation_threshold
