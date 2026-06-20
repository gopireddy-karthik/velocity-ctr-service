import time
import statistics
from typing import List

import numpy as np
import requests


ENDPOINT = "http://127.0.0.1:8001/predict_ctr"
TOTAL_REQUESTS = 200
TIMEOUT_SECONDS = 5


def build_payloads() -> List[dict]:
    payloads = []
    user_ids = ["user_101", "user_102", "user_999"]
    ad_positions = ["banner", "native", "interstitial"]

    for i in range(TOTAL_REQUESTS):
        payloads.append(
            {
                "user_id": user_ids[i % len(user_ids)],
                "ad_id": f"ad_{1000 + i}",
                "ad_position": ad_positions[i % len(ad_positions)],
            }
        )

    return payloads


def run_benchmark():
    payloads = build_payloads()
    latencies_ms = []
    success_count = 0

    print(f"Starting benchmark: {TOTAL_REQUESTS} sequential requests to {ENDPOINT}\n")

    for idx, payload in enumerate(payloads, start=1):
        start_time = time.time()
        try:
            response = requests.post(ENDPOINT, json=payload, timeout=TIMEOUT_SECONDS)
            elapsed_ms = (time.time() - start_time) * 1000.0
            latencies_ms.append(elapsed_ms)

            if response.status_code == 200:
                success_count += 1
            else:
                print(f"[{idx}] Non-200 response: {response.status_code} - {response.text}")

        except requests.RequestException as exc:
            elapsed_ms = (time.time() - start_time) * 1000.0
            latencies_ms.append(elapsed_ms)
            print(f"[{idx}] Request failed: {exc}")

        if idx % 50 == 0:
            print(f"Completed {idx}/{TOTAL_REQUESTS} requests...")

    return latencies_ms, success_count


def print_results(latencies_ms: List[float], success_count: int) -> None:
    total = len(latencies_ms)
    avg_latency = np.mean(latencies_ms)
    median_latency = np.percentile(latencies_ms, 50)
    p99_latency = np.percentile(latencies_ms, 99)
    min_latency = float(np.min(latencies_ms))
    max_latency = float(np.max(latencies_ms))
    std_latency = statistics.pstdev(latencies_ms)
    total_time = sum(latencies_ms) / 1000.0

    rows = [
        ("Total Requests Processed", f"{total}"),
        ("Successful Responses", f"{success_count}"),
        ("Average Latency (ms)", f"{avg_latency:.2f}"),
        ("Median Latency (p50, ms)", f"{median_latency:.2f}"),
        ("p99 Latency (ms)", f"{p99_latency:.2f}"),
        ("Min Latency (ms)", f"{min_latency:.2f}"),
        ("Max Latency (ms)", f"{max_latency:.2f}"),
        ("Std Dev (ms)", f"{std_latency:.2f}"),
        ("Total Client Time (s)", f"{total_time:.2f}"),
    ]

    label_width = max(len(label) for label, _ in rows)
    value_width = max(len(value) for _, value in rows)
    total_width = label_width + value_width + 7

    print("\nBenchmark Summary")
    print("+" + "=" * (total_width - 2) + "+")
    print(f"| {'Metric'.ljust(label_width)} | {'Value'.ljust(value_width)} |")
    print("+" + "=" * (total_width - 2) + "+")

    for label, value in rows:
        print(f"| {label.ljust(label_width)} | {value.rjust(value_width)} |")

    print("+" + "=" * (total_width - 2) + "+")


if __name__ == "__main__":
    latencies, success = run_benchmark()
    print_results(latencies, success)
