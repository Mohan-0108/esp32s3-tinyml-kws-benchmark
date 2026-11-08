import csv
import math
import re
import sys
from pathlib import Path

LOG_PATTERN = re.compile(
    r"W\s*\((\d+)\)\s*THESIS_DATA:\s*Pure Inference Latency:\s*(\d+)\s*ms\s*\((\d+)\s*us\)"
)
AUDIO_STRIDE_US = 20000.0

def parse_log(input_path: Path, output_csv: Path):
    records = []
    with input_path.open("r", encoding="utf-8", errors="ignore") as f:
        for line in f:
            match = LOG_PATTERN.search(line)
            if match:
                rtos_time_ms = int(match.group(1))
                latency_ms = int(match.group(2))
                latency_us = int(match.group(3))
                records.append((len(records) + 1, rtos_time_ms, latency_ms, latency_us))

    if not records:
        print(f"[ERROR] No THESIS_DATA lines found in {input_path}")
        sys.exit(1)

    output_csv.parent.mkdir(parents=True, exist_ok=True)
    with output_csv.open("w", newline="", encoding="utf-8") as csvfile:
        writer = csv.writer(csvfile)
        writer.writerow(["inference_index", "rtos_timestamp_ms", "latency_ms", "latency_us", "is_cold_start"])
        for idx, rtos_ms, lat_ms, lat_us in records:
            writer.writerow([idx, rtos_ms, lat_ms, lat_us, 1 if idx == 1 else 0])

    cold_start_us = records[0][3]
    steady_us = [r[3] for r in records[1:]] if len(records) > 1 else [cold_start_us]
    n_steady = len(steady_us)
    mean_us = sum(steady_us) / n_steady
    variance = sum((x - mean_us) ** 2 for x in steady_us) / (n_steady - 1) if n_steady > 1 else 0.0
    std_us = math.sqrt(variance)
    min_us = min(steady_us)
    max_us = max(steady_us)
    cache_penalty_us = cold_start_us - mean_us
    cache_penalty_pct = (cache_penalty_us / mean_us) * 100.0
    duty_cycle_pct = (mean_us / AUDIO_STRIDE_US) * 100.0

    print("=" * 60)
    print(" ESP32-S3 TINYML INFERENCE LATENCY SUMMARY")
    print("=" * 60)
    print(f" Total Inferences Parsed : {len(records)}")
    print(f" Cold-Start (Run #1)     : {cold_start_us} us ({cold_start_us / 1000:.3f} ms)")
    print(f" Steady-State Count      : {n_steady} runs")
    print(f" Steady-State Mean       : {mean_us:.2f} us ({mean_us / 1000:.3f} ms)")
    print(f" Steady-State Std Dev    : {std_us:.2f} us")
    print(f" Steady-State Min / Max  : {min_us} us / {max_us} us")
    print(f" Cache Warm-Up Penalty   : +{cache_penalty_us:.2f} us (+{cache_penalty_pct:.2f}%)")
    print(f" 20 ms Frame Duty Cycle  : {duty_cycle_pct:.2f}% active ({100.0 - duty_cycle_pct:.2f}% slack)")
    print(f" CSV Exported To         : {output_csv}")
    print("=" * 60)

if __name__ == "__main__":
    log_file = Path(sys.argv[1]) if len(sys.argv) > 1 else Path("results/baseline_serial_log.txt")
    csv_file = Path(sys.argv[2]) if len(sys.argv) > 2 else Path("results/baseline_latency.csv")
    parse_log(log_file, csv_file)
