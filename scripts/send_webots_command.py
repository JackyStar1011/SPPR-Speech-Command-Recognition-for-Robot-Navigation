from __future__ import annotations

import argparse
import sys
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.robot.actions import VALID_WEBOTS_ACTIONS  # noqa: E402
from src.robot.webots_udp import (
    DEFAULT_WEBOTS_COMMAND_PORT,
    DEFAULT_WEBOTS_HOST,
    WebotsUDPClient,
)  # noqa: E402



def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Send a UDP movement command to the Webots wheelchair controller."
    )
    parser.add_argument(
        "command",
        type=str.upper,
        choices=sorted(VALID_WEBOTS_ACTIONS),
        help="Movement command to send.",
    )
    parser.add_argument("--host", default=DEFAULT_WEBOTS_HOST)
    parser.add_argument("--port", type=int, default=DEFAULT_WEBOTS_COMMAND_PORT)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    with WebotsUDPClient(host=args.host, port=args.port) as client:
        client.send_action(args.command)

    print(f"Sent UDP command to {args.host}:{args.port}: {args.command}")


if __name__ == "__main__":
    main()
