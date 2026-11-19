# Deterministic TinyML Keyword Spotting Benchmark on ESP32-S3

An I/O-decoupled, microsecond-precision benchmarking harness for evaluating TensorFlow Lite Micro (`esp-tflite-micro`) inference latency on the **ESP32-S3** microcontroller using **ESP-IDF v5.3.1**.

## 1. Overview & Scope
- **Target Hardware:** Heltec Wireless Tracker (ESP32-S3 @ 240 MHz, 8 MB QIO Flash, 320 KB Internal SRAM, no onboard microphone)
- **Framework:** ESP-IDF v5.3.1 (`espressif/esp-tflite-micro` v1.4.1, `espressif/esp-nn` v1.4.1)
- **Model:** Espressif `micro_speech` example (`TinyConv` 4-op graph: `DepthwiseConv2D`, `FullyConnected`, `Softmax`, `Reshape`; 30 KB static `tensor_arena`). *Note: This is Espressif's default KWS example, distinct from the larger MLPerf Tiny DS-CNN reference model.*
- **Latency Isolation:** Replaces blocking I2S microphone reads (`i2s_readsamples`) with a deterministic **Dummy Audio Provider** (`+20 ms` stride + `vTaskDelay(20 ms)` to keep Native USB alive) and wraps `esp_timer_get_time()` (1-us resolution) strictly around `interpreter->Invoke()`.

## 2. Experimental Results: Dummy Baseline vs. Live INMP441 (`real-mic-v1`)

| Metric (ESP32-S3 @ 240 MHz) | `baseline-v1` (Dummy, Pure Inf, $N=1{,}243$) | Dummy Full-Pipeline ($N=325$) | `real-mic-v1` (Live INMP441, $N=3{,}350$) | Interpretation |
| :--- | :--- | :--- | :--- | :--- |
| **Cold-Start Feature Extraction (49 slices)** | — | 60,553 µs (60.55 ms) | **68,549 µs (68.55 ms)** | One-time initial spectrogram fill |
| **Cold-Start Inference (Run #1)** | 3,761 µs (3.76 ms) | 3,782 µs (3.78 ms) | **3,776 µs (3.78 ms)** | SPI Flash (DROM) → 32 KB Data Cache warm-up |
| **Steady-State Feature Extraction (1 slice)** | — | 1,261.85 ± 3.74 µs | **1,363.67 ± 6.20 µs** | +101.8 µs for active non-zero log-mel / PCAN math |
| **Steady-State Pure Inference (`Invoke`)** | 3,121.43 ± 1.59 µs | 3,150.94 ± 1.67 µs | **3,132.52 ± 0.75 µs** | Sub-µs jitter — Core 1 I2S pinning isolates Core 0 |
| **End-to-End Latency (`Feat + Inf + Post`)** | — | 4,412.79 µs (4.41 ms) | **4,571.34 ± 6.38 µs (4.57 ms)** | Full live pipeline including `RecognizeCommands` |
| **20 ms Frame Duty Cycle** | 15.61% (Inf only) | 22.06% (`Feat + Inf`) | **22.86% (Full E2E)** | Leaves **77.14% (15.43 ms/frame)** CPU headroom |
| **Configured / Peak Tensor Arena (SRAM)** | 30,720 B / — | 30,720 B / 12,052 B | **30,720 B / 12,048 B (11.77 KB)** | 39.2% actual utilization of static 30 KB arena |
| **App Binary Size (Flash)** | 436,160 B (425.94 KB) | 436,480 B (426.25 KB) | **464,592 B (453.70 KB)** | +28.43 KB for `i2s_std` driver + live telemetry |

> **Live Accuracy Status:** Preliminary ambient-room testing confirmed 9/10 recall on both `"yes"` and `"no"` (`0` false `"yes"` triggers), while identifying low-frequency room hum / DC offset causing false `"no"` triggers during silence. Controlled quiet-room accuracy benchmarks with a Core 1 80 Hz DC-blocking high-pass filter are pending in `real-mic-v2`.

## 3. How to Reproduce

```powershell
idf.py fullclean
idf.py set-target esp32s3
idf.py -p COM7 flash monitor
```

To parse the serial log into CSV and compute summary statistics:
```powershell
python scripts/parse_latency_log.py results/baseline_serial_log.txt results/baseline_latency.csv
```

## 4. Visual Plots & Documentation
- Raw Serial Monitor Screenshot: `images/terminal_screenshot.png`
- Cold-Start vs. Steady-State Stability Plot: `images/cache_and_stability.jpg`
- Cache Warm-Up Callout Plot: `images/cache_callout_alt.jpg`
- Real-Time Duty Cycle & Benchmark Plot: `images/duty_cycle_benchmarks.jpg`
- Full Methodology Report (Word): `docs/TinyML_ESP32S3_Baseline_Report.docx`
- 12-Slide Defense Presentation (PPTX): `docs/TinyML_ESP32S3_Baseline_Presentation.pptx`

## 5. References

1. C. Banbury et al., "MLPerf Tiny Benchmark," in *Proceedings of the Neural Information Processing Systems Track on Datasets and Benchmarks (NeurIPS Datasets and Benchmarks)*, vol. 1, 2021. [Online]. Available: https://datasets-benchmarks-proceedings.neurips.cc/paper_files/paper/2021/hash/da4fb5c6e93e74d3df8527599fa62642-Abstract-round1.html
2. R. David et al., "TensorFlow Lite Micro: Embedded Machine Learning for TinyML Systems," in *Proceedings of Machine Learning and Systems (MLSys)*, vol. 3, pp. 800–811, 2021. [Online]. Available: https://proceedings.mlsys.org/paper_files/paper/2021/hash/6c44dc73014d66ba49b28d483a8f8b0d-Abstract.html
3. J. Lin, W.-M. Chen, Y. Lin, J. Cohn, C. Gan, and S. Han, "MCUNet: Tiny Deep Learning on IoT Devices," in *Advances in Neural Information Processing Systems (NeurIPS)*, vol. 33, pp. 11711–11722, 2020. [Online]. Available: https://proceedings.neurips.cc/paper/2020/hash/86c51678350f656dcc7f490a43946ee5-Abstract.html
