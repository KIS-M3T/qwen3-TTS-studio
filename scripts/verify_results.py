import csv
from collections import Counter

with open("bench/results_benchmark.csv", "r", encoding="utf-8") as f:
    rows = list(csv.DictReader(f))

print(f"Total rows in results_benchmark.csv: {len(rows)}")
counts = Counter((r["model_id"], r["run_type"]) for r in rows)
for k, v in sorted(counts.items()):
    print(f"  {k}: {v}")

print("\nSample REAL_INFERENCE rows:")
for r in rows:
    if r["run_type"] == "REAL_INFERENCE":
        print(f"  [{r['model_id']}] {r['test_id']}: gen_time={r['gen_time_sec']}s, rtf={r['rtf']}, vram={r['peak_vram_mb']}MB, dur={r['duration_sec']}s")
        if r["test_id"] in ["NUM_01", "LEN_01", "LEN_04"]:
            pass

print("\nSample CACHE_RECOVERY_UNMEASURED rows:")
cached_count = 0
for r in rows:
    if r["run_type"] == "CACHE_RECOVERY_UNMEASURED":
        if cached_count < 3:
            print(f"  [{r['model_id']}] {r['test_id']}: gen_time={r['gen_time_sec']!r}, rtf={r['rtf']!r}, vram={r['peak_vram_mb']!r}, dur={r['duration_sec']}s, peak_dbfs={r['peak_dbfs']}dBFS")
            cached_count += 1
