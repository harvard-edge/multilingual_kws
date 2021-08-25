
from dataclasses import dataclass, field, asdict
import os
from typing import List, Optional, Tuple

@dataclass
class WavTranscript:
    wav: os.PathLike
    transcript: str
    keyword: Optional[str] = None
    occurences_s: List[Tuple[float, float]] = field(default_factory=list)
    tgfile: Optional[os.PathLike] = None
