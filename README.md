# Deterministic TinyML Keyword Spotting Benchmark on ESP32-S3

An I/O-decoupled, microsecond-precision benchmarking harness for evaluating TensorFlow Lite Micro (`esp-tflite-micro`) inference latency on the **ESP32-S3** microcontroller using **ESP-IDF v5.3.1**.

## 1. Overview & Scope
- **Target Hardware:** Heltec Wireless Tracker (ESP32-S3 @ 240 MHz, 8 MB QIO Flash, 320 KB Internal SRAM, no onboard microphone)
- **Framework:** ESP-IDF v5.3.1 (`espressif/esp-tflite-micro` v1.4.1, `espressif/esp-nn` v1.4.1)
- **Model:** Espressif `micro_speech` example (`TinyConv` 4-op graph: `DepthwiseConv2D`, `FullyConnected`, `Softmax`, `Reshape`; 30 KB static `tensor_arena`). *Note: This is Espressif's default KWS example, distinct from the larger MLPerf Tiny DS-CNN reference model.*
- **Latency Isolation:** Replaces blocking I2S microphone reads (`i2s_readsamples`) with a deterministic **Dummy Audio Provider** (`+20 ms` stride + `vTaskDelay(20 ms)` to keep Native USB alive) and wraps `esp_timer_get_time()` (1-us resolution) strictly around `interpreter->Invoke()`.

## 2. Baseline Experimental Results (`baseline-v1`)

| Metric | Measured Value | Interpretation |
| :--- | :--- | :--- |
| **Cold-Start Inference (Run #1)** | **3,761 us (3.76 ms)** | Includes SPI Flash (DROM) -> 32 KB SRAM Data Cache fetch penalty |
| **Steady-State Mean (N = 1,243)** | **3,121.43 us (3.12 ms)** | Deterministic band: 3,119 - 3,124 us (+/-2.5 us jitter, std = 1.59 us) |
| **Cache Warm-Up Penalty** | **+639.57 us (+20.49%)** | First-run cache miss overhead |
| **20 ms Frame Duty Cycle** | **15.61% Active** | Leaves 84.39% (16.88 ms/frame) CPU slack for sleep/peripherals |
| **App Binary Size** | **436,160 bytes (0x6A7C0)** | ~426 KB (78% of OTA partition free) |
| **Configured Tensor Arena** | **30,720 bytes (30 KB)** | Static SRAM allocation |

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
1. C. Banbury et al., *MLPerf Tiny Benchmark*, NeurIPS 2021 ([arXiv:2106.07597](https://arxiv.org/abs/2106.07597))
2. R. David et al., *TensorFlow Lite Micro: Embedded Machine Learning for TinyML Systems*, MLSys 2021 ([arXiv:2010.08678](https://arxiv.org/abs/2010.08678))
3. J. Lin et al., *MCUNet: Tiny Deep Learning on IoT Devices*, NeurIPS 2020 ([arXiv:2007.10319](https://arxiv.org/abs/2007.10319))
