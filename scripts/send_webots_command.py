from __future__ import annotations

import argparse
import socket


UDP_HOST = "127.0.0.1"
UDP_PORT = 5005
VALID_COMMANDS = (
    "MOVE_FORWARD",
    "MOVE_BACKWARD",
    "TURN_LEFT",
    "TURN_RIGHT",
    "STOP",
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Send a UDP movement command to the Webots wheelchair controller."
    )
    parser.add_argument(
        "command",
        type=str.upper,
        choices=VALID_COMMANDS,
        help="Movement command to send.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    payload = args.command.encode("utf-8")

    with socket.socket(socket.AF_INET, socket.SOCK_DGRAM) as udp_socket:
        udp_socket.sendto(payload, (UDP_HOST, UDP_PORT))

    print(f"Sent UDP command to {UDP_HOST}:{UDP_PORT}: {args.command}")


if __name__ == "__main__":
    main()
