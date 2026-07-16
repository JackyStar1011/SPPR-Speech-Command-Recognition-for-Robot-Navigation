from __future__ import annotations

import socket

from src.robot.actions import VALID_WEBOTS_ACTIONS


DEFAULT_WEBOTS_HOST = "127.0.0.1"
DEFAULT_WEBOTS_COMMAND_PORT = 5005


class WebotsUDPClient:
    """Send validated wheelchair actions to a Webots controller over UDP."""

    def __init__(
        self,
        host: str = DEFAULT_WEBOTS_HOST,
        port: int = DEFAULT_WEBOTS_COMMAND_PORT,
        *,
        stop_on_close: bool = False,
    ) -> None:
        if not host:
            raise ValueError("host must not be empty")
        if not 1 <= port <= 65535:
            raise ValueError("port must be between 1 and 65535")

        self.host = host
        self.port = port
        self.stop_on_close = stop_on_close
        self._socket: socket.socket | None = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)

    @property
    def is_closed(self) -> bool:
        return self._socket is None

    def send_action(self, action: str) -> None:
        normalized_action = action.strip().upper()
        if normalized_action not in VALID_WEBOTS_ACTIONS:
            allowed = ", ".join(sorted(VALID_WEBOTS_ACTIONS))
            raise ValueError(f"unsupported Webots action {action!r}; expected one of: {allowed}")
        if self._socket is None:
            raise RuntimeError("Webots UDP client is closed")

        payload = normalized_action.encode("utf-8")
        self._socket.sendto(payload, (self.host, self.port))

    def stop(self) -> None:
        self.send_action("STOP")

    def close(self, *, send_stop: bool | None = None) -> None:
        if self._socket is None:
            return

        should_stop = self.stop_on_close if send_stop is None else send_stop
        try:
            if should_stop:
                self.stop()
        finally:
            self._socket.close()
            self._socket = None

    def __enter__(self) -> WebotsUDPClient:
        return self

    def __exit__(self, exc_type, exc_value, traceback) -> None:
        self.close()
