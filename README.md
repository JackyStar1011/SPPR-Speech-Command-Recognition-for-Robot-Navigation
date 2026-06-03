# Speech Command Recognition for Robot Navigation

Baseline supervised speech command recognition project for mapping short voice clips to robot navigation intents.

Target labels:

- `forward`
- `backward`
- `left`
- `right`
- `stop`
- `unknown`

Robot action mapping:

| Command | Action |
| --- | --- |
| `forward` | `MOVE_FORWARD` |
| `backward` | `MOVE_BACKWARD` |
| `left` | `TURN_LEFT` |
| `right` | `TURN_RIGHT` |
| `stop` | `STOP` |
| `unknown` | `IGNORE` |

## Setup

```bash
python -m venv .venv
.venv\Scripts\activate
python -m pip install -r requirements.txt
```

The project uses `torchaudio.datasets.SPEECHCOMMANDS`. The first training or evaluation run downloads the dataset into `data/raw/`.

## Configuration

Default configuration lives in `configs/baseline.yaml`.

Important defaults:

- sample rate: `16000`
- clip duration: `1.0` second
- waveform length: `16000` samples
- Log-Mel: `n_fft=400`, `win_length=400`, `hop_length=160`, `n_mels=64`
- batch size: `64`
- optimizer: Adam with learning rate `1e-3`
- epochs: `20`
- checkpoint: `outputs/checkpoints/best_cnn.pt`

## Train

```bash
python -m src.training.train --config configs/baseline.yaml
```

Training uses the official Speech Commands train/validation/test splits. The five main commands are kept as explicit classes. All other words are mapped to `unknown`, with the unknown class sampled to roughly match the average size of a main command class.

The best checkpoint is saved by validation accuracy. After training, the script evaluates the best checkpoint on the test split.

## Evaluate

```bash
python -m src.training.evaluate --config configs/baseline.yaml --checkpoint outputs/checkpoints/best_cnn.pt
```

Saved artifacts:

- `outputs/metrics/test_metrics.json`
- `outputs/metrics/classification_report.txt`
- `outputs/figures/test_confusion_matrix.png`

Metrics include accuracy, macro F1, and per-class precision/recall/F1.

## WAV Inference

```bash
python -m src.inference.infer_wav --file path\to\audio.wav
```

Optional arguments:

```bash
python -m src.inference.infer_wav ^
  --file path\to\audio.wav ^
  --checkpoint outputs/checkpoints/best_cnn.pt ^
  --threshold 0.70 ^
  --device auto
```

The script prints:

- predicted command
- raw model command
- confidence
- mapped robot action

If confidence is below the threshold, the output command becomes `unknown` and the robot action becomes `IGNORE`.

## Microphone Inference

```bash
python -m src.inference.infer_mic --seconds 1.0 --threshold 0.70
```

This records from the default microphone using `sounddevice`, then runs the same preprocessing and classifier as WAV inference.

## Streamlit Demo

```bash
streamlit run app/streamlit_app.py
```

The app lets you upload a WAV file, inspect its waveform and Log-Mel spectrogram, and view the predicted command, confidence, and robot action.

## Project Structure

```text
configs/baseline.yaml          Training and preprocessing config
src/data/dataset.py            Speech Commands dataset wrapper with balanced unknown sampling
src/data/preprocess.py         Resample, mono conversion, fixed-length waveform, amplitude normalization
src/features/logmel.py         Log-Mel feature extraction
src/models/cnn.py              Small CNN classifier
src/training/train.py          Training loop and best checkpoint saving
src/training/evaluate.py       Evaluation script and artifact generation
src/inference/infer_wav.py     WAV file inference CLI
src/inference/infer_mic.py     Microphone inference CLI
src/inference/predictor.py     Shared checkpoint loading and prediction logic
src/robot/actions.py           Command-to-action mapping
app/streamlit_app.py           Upload-based demo UI
```

Generated data, checkpoints, figures, and metrics are ignored by Git except for `.gitkeep` placeholders.
