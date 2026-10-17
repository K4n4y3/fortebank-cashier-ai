from dataclasses import dataclass, field
from typing import Callable

from app import rules

# progress(fraction 0..1, stage label)
ProgressFn = Callable[[float, str], None]


@dataclass
class EvidenceFrame:
    timestamp_sec: float
    image_base64: str


@dataclass
class Violation:
    type_id: str
    timecode_sec: float
    observation: str  # что именно видно на видео, без предположений
    confidence: float
    evidence: list[EvidenceFrame] = field(default_factory=list)
    sources_hint: list[str] = field(default_factory=list)  # какие слои подтвердили: cv, vlm

    def description(self) -> str:
        r = rules.rule(self.type_id)
        parts = [self.observation.strip().rstrip(".") + ".", r["why"]]
        src = rules.source_text(self.type_id)
        if src:
            parts.append(src)
        return " ".join(parts)

    def to_api(self) -> dict:
        return {
            "type": rules.rule(self.type_id)["name"],
            "timecode_sec": round(self.timecode_sec, 1),
            "description": self.description(),
            "evidence_frames": [
                {"timestamp_sec": round(e.timestamp_sec, 2), "image_base64": e.image_base64}
                for e in self.evidence[:3]
            ],
        }

    def to_ui(self) -> dict:
        d = self.to_api()
        r = rules.rule(self.type_id)
        d.update(
            type_id=self.type_id,
            confidence=round(self.confidence, 2),
            observation=self.observation,
            why=r["why"],
            sources=r.get("sources") or [],
            detected_by=self.sources_hint,
        )
        return d


@dataclass
class AnalysisResult:
    violations: list[Violation]
    duration_sec: float
    processing_sec: float = 0.0
    notes: list[str] = field(default_factory=list)  # служебные заметки (деградация по времени и т.п.)

    @property
    def verdict(self) -> str:
        return "violation" if self.violations else "no_violation"
