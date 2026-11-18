from dataclasses import dataclass, asdict
from pathlib import Path
import json

ROOT = Path(__file__).resolve().parents[1]
FEATURE_NAMES = ["distance_m", "log_distance", "heading_cos", "heading_sin_abs",
                 "log_length", "fraction", "endpoint_fraction", "speed_mps",
                 "log_dt", "gap_flag", "rank_fraction", "heading_missing"]
SEED = 352
@dataclass(frozen=True)
class Settings:
    candidates: int = 8
    sample_spacing_m: float = 30.0
    max_candidate_distance_m: float = 250.0
    heading_min_step_m: float = 2.0
    max_heading_gap_s: float = 5.0
    decoder_gap_s: float = 30.0
    speed_cap_mps: float = 55.0
    transition_beta_m: float = 25.0
    emission_weight: float = 1.0
    training_observations_per_trip: int = 320
    tuning_observations_per_trip: int = 140
    folds: int = 3
    tuning_trials: int = 16

    def save(self, path):
        Path(path).write_text(json.dumps(asdict(self), indent=2))

    @classmethod
    def load(cls, path):
        return cls(**json.loads(Path(path).read_text()))
