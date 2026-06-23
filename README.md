# Speech Command Recognition with CNN-GRU

Nhận dạng sáu lớp lệnh giọng nói cho robot:

- `forward`, `backward`, `left`, `right`, `stop`
- `unknown`

Pipeline chính:

```text
waveform 16 kHz, 1 giây
  -> Log-Mel Spectrogram (64 mel bins)
  -> CNN trích xuất đặc trưng tần số-thời gian
  -> GRU mô hình hóa chuỗi frame
  -> temporal mean pooling
  -> bộ phân loại 6 lớp
```

## Cài đặt

```powershell
python -m venv .venv
.venv\Scripts\Activate.ps1
pip install -r requirements.txt
```

Dataset Speech Commands được tải tự động vào `data/raw` khi chạy lần đầu.

## Cấu hình

Cấu hình mặc định: `configs/cnn_gru.yaml`.

- Audio: 16 kHz, 1 giây
- Log-Mel: `n_fft=400`, `hop_length=160`, `n_mels=64`
- CNN channels: `16 -> 32 -> 64`
- GRU: 2 tầng, hidden size 128, một chiều
- Optimizer: AdamW, learning rate `1e-3`
- Checkpoint: `outputs/checkpoints/best_cnn_gru.pt`

## Train

```powershell
python -m src.training.train --config configs/cnn_gru.yaml
```

Model tốt nhất trên validation được lưu vào đường dẫn checkpoint trong config. Sau
khi train xong, chương trình tự đánh giá trên test set.

## Evaluate

```powershell
python -m src.training.evaluate `
  --config configs/cnn_gru.yaml `
  --checkpoint outputs/checkpoints/best_cnn_gru.pt
```

Kết quả gồm accuracy, macro-F1, precision/recall/F1 từng lớp và confusion matrix.

## Tune confidence threshold

Không dùng threshold của model cũ. Sau khi train CNN-GRU, chọn threshold chỉ trên
validation rồi mới áp dụng lên test:

```powershell
python -m src.training.tune_threshold `
  --config configs/cnn_gru.yaml `
  --checkpoint outputs/checkpoints/best_cnn_gru.pt
```

Trước khi tune, `inference.threshold: 0.0` tương đương với dự đoán argmax.

## Inference

File WAV:

```powershell
python -m src.inference.infer_wav --file path\to\audio.wav
```

Microphone:

```powershell
python -m src.inference.infer_mic --seconds 1.0
```

Streamlit:

```powershell
streamlit run app/streamlit_app.py
```

## Kiểm thử

```powershell
python -m unittest discover -s tests -v
```

Ứng dụng Streamlit thu lệnh trực tiếp từ microphone, hiển thị waveform và Log-Mel
spectrogram, áp dụng lệnh dự đoán vào mô phỏng robot và cho phép xuất báo cáo HTML.

## Cấu trúc chính

```text
configs/cnn_gru.yaml          Cấu hình dữ liệu, đặc trưng và huấn luyện
src/data/                     Dataset và tiền xử lý waveform
src/features/logmel.py        Trích xuất Log-Mel Spectrogram
src/models/cnn_gru.py         Kiến trúc CNN-GRU
src/training/                 Train, evaluate và tune threshold
src/inference/                Inference WAV và microphone
src/robot/                    Ánh xạ hành động và mô phỏng robot
app/streamlit_app.py          Giao diện demo
tests/                        Unit tests
```

Để so sánh công bằng với model khác, phải giữ nguyên Speech Commands split, danh
sách lớp, preprocessing và các chỉ số: accuracy, macro-F1, recall lớp `unknown`,
số tham số và thời gian inference.
