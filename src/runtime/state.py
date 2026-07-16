from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime, timezone
from enum import Enum
from typing import Any


class RuntimeState(str, Enum):
    IDLE = "IDLE"
    LISTENING = "LISTENING"
    PROCESSING = "PROCESSING"
    DISPATCHING = "DISPATCHING"
    COOLDOWN = "COOLDOWN"
    STOPPED = "STOPPED"
    ERROR = "ERROR"


@dataclass(frozen=True)
class RuntimeEvent:
    kind: str
    state: RuntimeState
    data: dict[str, Any] = field(default_factory=dict)
    timestamp: str = field(
        default_factory=lambda: datetime.now(timezone.utc).isoformat()
    )
