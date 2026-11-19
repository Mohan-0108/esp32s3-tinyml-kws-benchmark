# Real-Time TinyML Keyword Spotting Benchmark on ESP32-S3

An end-to-end, microsecond-precision benchmarking harness and live I2S deployment of TensorFlow Lite Micro (`esp-tflite-micro`) Keyword Spotting (`micro_speech`) on the ESP32-S3 microcontroller using ESP-IDF v5.3.1.

## 1. Overview & Architecture

* **Target Hardware:** Heltec Wireless Tracker (ESP32-S3 Dual-Core Xtensa LX7 @ 240 MHz, 8 MB QIO Flash, 320 KB Internal SRAM)
* **Audio Sensor:** External INMP441 Omnidirectional I2S MEMS Microphone (24-bit word in 32-bit slot, 16 kHz Mono Left)
  * **Pinout:** `VDD → 3V3`, `GND → GND`, `L/R → GND`, `WS → GPIO 17`, `SCK → GPIO 7`, `SD → GPIO 6`
* **Framework:** ESP-IDF v5.3.1 (`espressif/esp-tflite-micro` v1.4.1, `espressif/esp-nn` v1.4.1 SIMD acceleration)
* **Model:** Quantized INT8 `micro_speech` TinyConv CNN (4-op graph: `DepthwiseConv2D`, `FullyConnected`, `Softmax`, `Reshape`; classes: `silence`, `unknown`, `yes`, `no`). *Note: This is Espressif's compact KWS model, distinct from the larger MLPerf Tiny DS-CNN reference model.*
* **Dual-Core Real-Time Pipeline (`real-mic-v2`):**
  * **Core 1 (`inmp41_capture`, Priority 5):** Continuously reads 20 ms DMA chunks (320 samples @ 16 kHz) via `i2s_std`, scales 24-bit words to native 16-bit PCM (`kAudioGainShift = 16`), applies a 1st-order integer **80 Hz DC-blocking high-pass filter** ($y[n] = (x[n] - x[n-1]) + \lfloor 31\,y[n-1] / 32 \rfloor$), and writes into a lock-protected 32,768-sample (~2.05 s) ring buffer with a 512-sample (32 ms) timestamp safety lag.
  * **Core 0 (`app_main`):** Pulls 512-sample frames every 20 ms (50 Hz inference rate), computes 40-bin Log-Mel + PCAN spectrogram slices (`FeatureProvider`), runs INT8 `interpreter->Invoke()`, and applies temporal smoothing via `RecognizeCommands(300 ms, 0.72, 1200 ms, 3)`.

## 2. Latency & Resource Benchmarks (`baseline-v1` → `real-mic-v2`)

| Metric (ESP32-S3 @ 240 MHz) | `baseline-v1` (Dummy, Pure Inf, $N=1{,}243$) | `real-mic-v1` (Live INMP441, No HPF, $N=3{,}350$) | `real-mic-v2` (Live + 80 Hz HPF, $N=8{,}550$) | Interpretation |
| :--- | :--- | :--- | :--- | :--- |
| **Cold-Start Feature Extraction (49 slices)** | — | 68,549 µs (68.55 ms) | **67,364 µs (67.36 ms)** | One-time initial spectrogram fill |
| **Cold-Start Inference (Run #1)** | 3,761 µs (3.76 ms) | 3,776 µs (3.78 ms) | **3,781 µs (3.78 ms)** | SPI Flash (DROM) → 32 KB Data Cache warm-up |
| **Steady Feature Extraction (1 slice)** | — | 1,363.67 ± 6.20 µs | **1,359.19 ± 3.64 µs** | Deterministic band: 1,353–1,370 µs |
| **Steady Pure Inference (`Invoke`)** | 3,121.43 ± 1.59 µs | 3,132.52 ± 0.75 µs | **3,140.71 ± 0.78 µs** | Sub-µs jitter — Core 1 I2S pinning isolates Core 0 |
| **End-to-End Pipeline (`Feat + Inf + Post`)** | — | 4,571.34 ± 6.38 µs | **4,576.59 ± 3.99 µs (4.58 ms)** | Zero Core 0 latency penalty from Core 1 HPF |
| **20 ms Frame Duty Cycle** | 15.61% (Inf only) | 22.86% (Full E2E) | **22.88% (Full E2E)** | Leaves **77.12% (15.42 ms/frame)** CPU headroom |
| **Peak / Configured Tensor Arena (SRAM)** | — / 30,720 B | 12,048 / 30,720 B | **12,048 / 30,720 B (11.77 KB)** | 39.2% actual utilization of static 30 KB arena |
| **App Binary Size (Flash)** | 436,160 B (425.94 KB) | 464,592 B (453.70 KB) | **464,624 B (453.73 KB)** | +32 bytes Flash overhead for 80 Hz HPF |

## 3. Live Quiet-Room Evaluation & Filter Ablation (`real-mic-v2`)

Evaluated continuously over **171 seconds** (8,550 steady-state frames) in a low-ambient indoor environment (~15 cm speaking distance): 35 s silence + 20 × `"yes"` + 20 × `"no"` + 10 × non-target (`"unknown"`) words.

### 50-Utterance Confusion Matrix

| Ground Truth \ Predicted | `"yes"` | `"no"` | Rejected (`sil` / `unk`) | Total Trials |
| :--- | :---: | :---: | :---: | :---: |
| **`"yes"`** | **18** | 0 | 2 | **20** |
| **`"no"`** | 0 | **20** | 0 | **20** |
| **`"unknown"`** | 0 | 3 *(4 raw triggers)* | **7** | **10** |
| **Steady Silence (`1.5–35.0 s`)** | 0 | **0** *(+1 boot transient @ `0.75 s`)* | — | **33.5 s** |

### Classification Metrics & Filter Impact

| Metric | Utterance-Level (Steady-State) | Strict Event-Level (All 43 Triggers) |
| :--- | :---: | :---: |
| **`"yes"` Precision / Recall / F1** | **100.0% / 90.0% / 94.7%** | **100.0% / 90.0% / 94.7%** |
| **`"no"` Precision / Recall / F1** | **87.0% / 100.0% / 93.0%** | **80.0% / 100.0% / 88.9%** |
| **Target Keyword Macro-F1** | **93.9%** | **91.8%** |
| **50-Utterance 3-Class Accuracy** | **45 / 50 (90.0%)** | — |
| **Unknown Word Rejection Rate** | **7 / 10 (70.0%)** | — |
| **Silence False-Alarm Rate (FPR/min)** | **0.00 FP/min** (`real-mic-v1`: 30.0 FP/min) | **1.71 FP/min** (incl. `0.75 s` boot transient) |

* **Ablation Finding:** Unfiltered `real-mic-v1` (`kAudioGainShift = 15`, no HPF) suffered from INMP441 DC bias and low-frequency room hum energizing the lower Mel bins, causing **30.0 FP/min** false `"no"` triggers during silence. Adding the 80 Hz integer HPF on Core 1 (+32 B Flash, 0 µs Core 0 penalty) eliminated steady-state silence false alarms (**0.00 FP/min**) and `"yes"`/`"no"` cross-talk (**0%**).

## 4. How to Reproduce

```powershell
idf.py fullclean
idf.py set-target esp32s3
idf.py -p COM7 flash monitor
