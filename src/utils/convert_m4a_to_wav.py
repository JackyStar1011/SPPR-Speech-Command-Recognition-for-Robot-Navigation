from __future__ import annotations

import argparse
import shutil
import subprocess
from pathlib import Path


def convert_m4a_to_wav(
    input_path: str | Path,
    output_path: str | Path | None = None,
    sample_rate: int = 16000,
    mono: bool = True,
    overwrite: bool = False,
) -> Path:
    input_path = Path(input_path)
    if not input_path.exists():
        raise FileNotFoundError(f"Input file not found: {input_path}")
    if input_path.suffix.lower() != ".m4a":
        raise ValueError(f"Expected a .m4a file, got: {input_path.suffix}")

    ffmpeg_path = shutil.which("ffmpeg")
    if ffmpeg_path is None:
        raise RuntimeError("ffmpeg was not found in PATH. Install ffmpeg before converting .m4a files.")

    output_path = Path(output_path) if output_path else input_path.with_suffix(".wav")
    output_path.parent.mkdir(parents=True, exist_ok=True)

    command = [
        ffmpeg_path,
        "-hide_banner",
        "-loglevel",
        "error",
        "-y" if overwrite else "-n",
        "-i",
        str(input_path),
        "-ar",
        str(sample_rate),
        "-sample_fmt",
        "s16",
    ]
    if mono:
        command.extend(["-ac", "1"])
    command.append(str(output_path))

    try:
        subprocess.run(command, check=True)
    except subprocess.CalledProcessError as error:
        raise RuntimeError(f"Failed to convert {input_path} to WAV: {error}") from error

    return output_path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Convert a .m4a audio file to .wav.")
    parser.add_argument("--input", required=True, help="Path to the input .m4a file.")
    parser.add_argument("--output", default=None, help="Path to the output .wav file.")
    parser.add_argument("--sample-rate", type=int, default=16000, help="Output sample rate.")
    parser.add_argument("--stereo", action="store_true", help="Keep stereo instead of converting to mono.")
    parser.add_argument("--overwrite", action="store_true", help="Overwrite the output file if it exists.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    output_path = convert_m4a_to_wav(
        input_path=args.input,
        output_path=args.output,
        sample_rate=args.sample_rate,
        mono=not args.stereo,
        overwrite=args.overwrite,
    )
    print(f"converted={output_path}")


if __name__ == "__main__":
    main()
