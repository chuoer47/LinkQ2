from dataclasses import dataclass
import math

GREEDY = 1e-6


@dataclass(slots=True)
class SamplingParams:
    max_tokens: int = 128
    temperature: float = GREEDY
    ignore_eos: bool = False

    def __post_init__(self):
        if not math.isfinite(self.temperature) or self.temperature < 0:
            raise ValueError('temperature must be finite and nonnegative')
        if self.max_tokens < 1:
            raise ValueError('max_tokens must be positive')
        if self.temperature <= 0:
            self.temperature = GREEDY
