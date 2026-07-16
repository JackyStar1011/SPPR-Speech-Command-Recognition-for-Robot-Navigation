from __future__ import annotations

import json
import socket
from typing import Any


DEFAULT_WEBOTS_TELEMETRY_HOST = "127.0.0.1"
DEFAULT_WEBOTS_TELEMETRY_PORT = 5006
REQUIRED_TELEMETRY_FIELDS = {
    "timestamp",
    "x",
    "y",
    "z",
    "yaw",
    "motion_state",
    "left_velocity",
    "right_velocity",
}


class WebotsTelemetryReceiver:
    """Receive and validate wheelchair pose telemetry from Webots."""

    def __init__(
        self,
        host: str = DEFAULT_WEBOTS_TELEMETRY_HOST,
        port: int = DEFAULT_WEBOTS_TELEMETRY_PORT,
    ) -> None:
        if not host:
            raise ValueError("host must not be empty")
        if not 1 <= port <= 65535:
            raise ValueError("port must be between 1 and 65535")

        self.host = host
        self.port = port
        self._socket: socket.socket | None = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        self._socket.bind((host, port))

    def receive(self, timeout_seconds: float = 0.25) -> dict[str, Any] | None:
        if self._socket is None:
            raise RuntimeError("Webots telemetry receiver is closed")
        if timeout_seconds <= 0.0:
            raise ValueError("timeout_seconds must be positive")

        self._socket.settimeout(timeout_seconds)
        try:
            packet, sender = self._socket.recvfrom(4096)
        except socket.timeout:
            return None

        try:
            payload = json.loads(packet.decode("utf-8"))
        except (UnicodeDecodeError, json.JSONDecodeError) as error:
            raise ValueError(f"invalid telemetry packet from {sender}: {error}") from error
        if not isinstance(payload, dict):
            raise ValueError(f"telemetry packet from {sender} must contain a JSON object")

        missing = REQUIRED_TELEMETRY_FIELDS - payload.keys()
        if missing:
            raise ValueError(
                f"telemetry packet from {sender} is missing: {', '.join(sorted(missing))}"
            )

        for name in (
            "timestamp",
            "x",
            "y",
            "z",
            "yaw",
            "left_velocity",
            "right_velocity",
        ):
            payload[name] = float(payload[name])
        payload["motion_state"] = str(payload["motion_state"])
        payload["sender"] = f"{sender[0]}:{sender[1]}"
        return payload

    def close(self) -> None:
        if self._socket is not None:
            self._socket.close()
            self._socket = None

    def __enter__(self) -> WebotsTelemetryReceiver:
        return self

    def __exit__(self, exc_type, exc_value, traceback) -> None:
        self.close()
