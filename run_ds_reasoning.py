"""Валидация: deepseek с включённым reasoning (max 12000), 4 профиля x3 = 12 запросов."""

import csv
import sys
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))
from run_full import FIELDS, META, run_one

MODEL = "deepseek/deepseek-v4.1-flash"

jobs = []
for pid in ["P0", "P1", "P4", "P7"]:
    for r in range(3):
        jobs.append((pid, r, r % 6))

print(f"12 запросов, reasoning ON, max_tokens=12000", flush=True)
rows = []
with ThreadPoolExecutor(max_workers=3) as ex:
    futs = [ex.submit(run_one, MODEL, None, 12000, pid, r, p) for pid, r, p in jobs]
    for i, f in enumerate(futs):
        try:
            rows.append(f.result())
            print(f"  готово {i+1}/12 parse={rows[-1]['parse_ok']}", flush=True)
        except Exception as e:
            print(f"  FAIL {jobs[i]}: {str(e)[:150]}", flush=True)

with open("valid_ds_reasoning.csv", "w", newline="", encoding="utf-8") as fo:
    w = csv.DictWriter(fo, fieldnames=FIELDS)
    w.writeheader()
    w.writerows(rows)
cost = sum(float(r["cost_rub"]) for r in rows if r["cost_rub"] not in ("", None))
print(f"ГОТОВО: parse={sum(r['parse_ok'] for r in rows)}/12 cost={cost:.2f} руб.", flush=True)
