# Speech Command Recognition for Robot Navigation

Nhận dạng lệnh giọng nói cho robot bằng các mô hình CNN, CNN-GRU và MFCC-CNN.

Các nhãn chính: `forward`, `backward`, `left`, `right`, `stop` và `unknown`.

## Cài đặt

```powershell
python -m venv .venv
.venv\Scripts\Activate.ps1
pip install -r requirements.txt
```

Dataset Speech Commands được tải tự động vào `data/raw` trong lần chạy đầu tiên.

## Huấn luyện và đánh giá

CNN-GRU:

```powershell
python -m src.training.train --config configs/models/cnn_gru.yaml
python -m src.training.evaluate `
  --config configs/models/cnn_gru.yaml `
  --checkpoint outputs/checkpoints/best_cnn_gru.pt
```

CNN baseline:

```powershell
python -m src.training.train --config configs/models/cnn.yaml
```

MFCC-CNN:

```powershell
python -m src.training.train_mfcc --config configs/models/mfcc_cnn.yaml
python -m src.training.evaluate_mfcc `
  --config configs/models/mfcc_cnn.yaml `
  --checkpoint outputs/checkpoints/best_mfcc_cnn.pt
```

Tune confidence threshold:

```powershell
python -m src.training.tune_threshold `
  --config configs/models/cnn_gru.yaml `
  --checkpoint outputs/checkpoints/best_cnn_gru.pt
```

## Chạy inference

```powershell
python -m src.inference.infer_wav --file path\to\audio.wav
python -m src.inference.infer_mic --seconds 1.0
python -m src.inference.infer_wav_mfcc --file path\to\audio.wav
streamlit run app/streamlit_app.py
```

Ứng dụng Streamlit hiển thị waveform, Log-Mel spectrogram, kết quả dự đoán và mô phỏng
chuyển động robot. Trước khi thực thi, lệnh đi qua lớp kiểm tra an toàn trong `src/robot`.

## Kiểm thử

```powershell
python -m pytest
```

## Cấu trúc repository

```text
app/                 Giao diện Streamlit
configs/models/      Cấu hình cho từng mô hình
data/                 Dữ liệu cục bộ; chỉ .gitkeep được commit
docs/reports/         Tài liệu và biểu mẫu dự án
docs/results/         Báo cáo kết quả thí nghiệm
notebooks/            Notebook phân tích và huấn luyện Colab
outputs/              Checkpoint, metrics và hình sinh ra
scripts/              Công cụ tạo báo cáo và sơ đồ
src/data/             Dataset và tiền xử lý waveform
src/features/         Log-Mel và MFCC feature extraction
src/models/           Kiến trúc mô hình và model factory
src/training/         Huấn luyện, đánh giá và tune threshold
src/inference/        Inference WAV, microphone và predictor
src/robot/            Safety layer, ánh xạ hành động và simulator
tests/                Bộ kiểm thử tự động
```

Kết quả CNN-GRU chi tiết nằm tại `docs/results/cnn_gru.md`.
