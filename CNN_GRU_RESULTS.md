# Kết quả Log-Mel + CNN-GRU

## Cấu hình model

| Thuộc tính | Giá trị |
| --- | ---: |
| Input | mono, 16 kHz, 1 giây |
| Log-Mel | 64 mel bins, `n_fft=400`, `hop_length=160` |
| CNN channels | 16 → 32 → 64 |
| GRU | 2 tầng, hidden size 128, một chiều |
| Temporal aggregation | Mean pooling |
| Số lớp | 6 |
| Số tham số | 370,038 |
| Checkpoint tốt nhất | Epoch 30/40 |
| Kích thước checkpoint | 4.26 MiB |

Các lớp: `forward`, `backward`, `left`, `right`, `stop`, `unknown`.

## Validation

| Chỉ số | Kết quả |
| --- | ---: |
| Loss | 0.1216 |
| Accuracy | 97.62% |
| Macro-F1 | 97.37% |

Checkpoint được chọn theo validation macro-F1.

## Official test — Speech Commands v0.02

| Chỉ số | Kết quả |
| --- | ---: |
| Số mẫu | 1,847 |
| Loss | 0.1090 |
| Accuracy | **97.46%** |
| Macro-F1 | **97.14%** |

| Lớp | Precision | Recall | F1 | Support |
| --- | ---: | ---: | ---: | ---: |
| forward | 98.04% | 96.77% | 97.40% | 155 |
| backward | 96.89% | 94.55% | 95.71% | 165 |
| left | 97.13% | 98.54% | 97.83% | 412 |
| right | 96.56% | 99.24% | 97.88% | 396 |
| stop | 100.00% | 99.51% | 99.76% | 411 |
| unknown | 95.65% | 92.86% | 94.23% | 308 |

Kết quả chuẩn dùng argmax (`inference.threshold=0.0`). Threshold 0.42 được chọn
trên validation nhưng làm test accuracy giảm còn 97.40% và macro-F1 còn 97.08%,
vì vậy không được chọn làm kết quả cuối.

## Tốc độ CPU

Benchmark batch size 1, PyTorch CPU, 4 threads, 200 lượt sau warm-up:

| Thành phần | Độ trễ trung bình |
| --- | ---: |
| Chỉ CNN-GRU | 4.59 ms/mẫu |
| Log-Mel + CNN-GRU | 4.84 ms/mẫu |

Con số tốc độ chỉ nên so sánh với model khác khi chạy trên cùng máy, cùng batch
size và cùng cách benchmark.

## Thử nghiệm giọng thật hiện tại

Inference có căn chỉnh vùng giọng nói trước khi lấy cửa sổ 1 giây.

| Lớp | Đúng | Accuracy |
| --- | ---: | ---: |
| forward | 5/6 | 83.33% |
| stop | 10/10 | 100.00% |
| left | 9/10 | 90.00% |
| right | 5/7 | 71.43% |
| Tổng tạm thời | 29/33 | 87.88% |

Đây không phải benchmark chính thức: tập chỉ có một người nói, mới có bốn lớp,
số mẫu ít và `forward03.wav` đang chờ thay thế. Chỉ dùng tập này để so sánh các
model nếu mọi model được chạy trên đúng cùng danh sách file và cùng preprocessing.
