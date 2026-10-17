"""Полный прогон: P0 x50 + P1–P8 x30 = 290 запросов на модель.

Запуск: nohup python3 run_full.py ALIAS > logs/full_ALIAS.log 2>&1 &
ALIAS: deepseek | luna | qwen

Поля CSV — как в пилоте (§6.8 + X*/IDF/beta/флаги).
"""

import csv
import datetime
import json
import re
import subprocess
import sys
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent / "prompts"))
from build_request import build_messages  # noqa: E402

BASE = Path(__file__).parent

CONFIGS = {
    # alias: (model_id, reasoning_param, max_tokens)
    "deepseek": ("deepseek/deepseek-v4.1-flash", {"enabled": False}, 1500),
    "luna": ("openai/gpt-6-luna", None, 1500),
    "qwen": ("qwen/qwen3.8-flash", {"enabled": False}, 3000),
}

N_P0 = 50
N_PX = 30
MAX_WORKERS = 5
TEMPERATURE = 1.0

X_GRID = [75, 70, 65, 60, 50, 40, 30]
Y_VAL = 80
LISTS = ["today_vs_1m", "today_vs_6m", "6m_vs_7m"]

META = {
    "P0": ("not_specified", "not_specified", "", "baseline"),
    "P1": ("female", "college", 25, "young"),
    "P2": ("female", "college", 48, "older"),
    "P3": ("female", "no_college", 25, "young"),
    "P4": ("female", "no_college", 48, "older"),
    "P5": ("male", "college", 25, "young"),
    "P6": ("male", "college", 48, "older"),
    "P7": ("male", "no_college", 25, "young"),
    "P8": ("male", "no_college", 48, "older"),
}

FIELDS = [
    "timestamp", "model_name", "profile_id", "gender", "education", "age", "age_group",
    "run_id", "list_order", "n_attempts", "parse_ok", "multiple_switch",
    "X1_star", "X2_star", "X3_star", "parsed_decision",
    "IDF1", "IDF3", "beta_hat", "present_biased", "future_biased", "consistent",
    "censor_flag", "reasoning_length", "reasoning_text", "reasoning_native",
    "raw_response", "prompt_tokens", "completion_tokens", "cost", "cost_rub",
    "temperature", "reasoning_enabled", "ai_disclaimer_flag",
]


def get_key() -> str:
    for line in open(BASE / "api_key.yaml", encoding="utf-8"):
        if "API_KEY" in line:
            return line.split(":", 1)[1].strip()
    raise RuntimeError("API_KEY not found")


class RateLimitError(RuntimeError):
    pass


def call_api_once(messages, model, reasoning, max_tokens):
    body = {"model": model, "messages": messages, "temperature": TEMPERATURE, "max_tokens": max_tokens}
    if reasoning is not None:
        body["reasoning"] = reasoning
    out = subprocess.run(
        ["curl", "-s", "--max-time", "150", "https://polza.ai/api/v1/chat/completions",
         "-H", f"Authorization: Bearer {get_key()}", "-H", "Content-Type: application/json",
         "-d", json.dumps(body)],
        capture_output=True, text=True, timeout=180,
    )
    if out.returncode != 0:
        raise RuntimeError(f"curl failed: {out.stderr[:200]}")
    data = json.loads(out.stdout)
    if "error" in data:
        err = json.dumps(data["error"])
        if "TOO_MANY_REQUESTS" in err or "429" in err:
            raise RateLimitError(err[:300])
        raise RuntimeError(f"API error: {err[:300]}")
    msg = data["choices"][0]["message"]
    usage = data.get("usage", {})
    return (msg.get("content") or "", msg.get("reasoning") or "",
            usage.get("prompt_tokens", 0), usage.get("completion_tokens", 0),
            usage.get("cost_rub", usage.get("cost", "")))


def call_api(messages, model, reasoning, max_tokens):
    """Транспорт с backoff на 429 (до ~2 мин ожидания суммарно)."""
    for wait in (5, 10, 20, 30, 45, 60):
        try:
            return call_api_once(messages, model, reasoning, max_tokens)
        except RateLimitError:
            time.sleep(wait)
    return call_api_once(messages, model, reasoning, max_tokens)


def extract_json(text: str):
    text = re.sub(r"```(?:json)?", "", text).strip()
    m = re.search(r"\{.*\}", text, re.DOTALL)
    if not m:
        return None
    try:
        return json.loads(m.group(0)[: m.group(0).rfind("}") + 1])
    except Exception:
        return None


def switch_star(seq):
    if any(s not in ("A", "B") for s in seq):
        return None, False, "invalid"
    if all(s == "B" for s in seq):
        return 77.5, False, "right"
    if all(s == "A" for s in seq):
        return 25.0, False, "left"
    seen_b = False
    multi = False
    for s in seq:
        if s == "B":
            seen_b = True
        elif seen_b:
            multi = True
            break
    if multi:
        for i in range(6):
            if seq[i] == "A" and seq[i + 1] == "B":
                return (X_GRID[i] + X_GRID[i + 1]) / 2, True, "interval_multi"
        # только B->A переходы: reverse
        for i in range(6):
            if seq[i] == "B" and seq[i + 1] == "A":
                return (X_GRID[i] + X_GRID[i + 1]) / 2, True, "reverse"
        return None, True, "invalid"
    for i in range(6):
        if seq[i] == "A" and seq[i + 1] == "B":
            return (X_GRID[i] + X_GRID[i + 1]) / 2, False, "interval"
    return None, False, "invalid"


def run_one(model, reasoning, max_tokens, profile_id, run_id, perm_index):
    gender, education, age, age_group = META[profile_id]
    messages, list_order = build_messages(profile_id, perm_index)
    raw, native_rs, ptok, ctok, cost_rub, n_att, parsed, p_ok = "", "", 0, 0, "", 0, None, False
    for att in range(1, 3):
        n_att = att
        raw, native_rs, ptok, ctok, cost_rub = call_api(messages, model, reasoning, max_tokens)
        parsed = extract_json(raw)
        if (parsed and isinstance(parsed.get("choices"), dict)
                and all(isinstance(parsed["choices"].get(k), list)
                        and len(parsed["choices"][k]) == 7
                        and all(v in ("A", "B") for v in parsed["choices"][k]) for k in LISTS)):
            p_ok = True
            break
    row = {
        "timestamp": datetime.datetime.now(datetime.timezone.utc).isoformat(),
        "model_name": model, "profile_id": profile_id, "gender": gender,
        "education": education, "age": age, "age_group": age_group, "run_id": run_id,
        "list_order": "|".join(list_order), "n_attempts": n_att, "parse_ok": int(p_ok),
        "raw_response": raw, "reasoning_text": (parsed.get("reasoning", "") if isinstance(parsed, dict) else ""),
        "reasoning_native": native_rs, "prompt_tokens": ptok, "completion_tokens": ctok,
        "cost": cost_rub, "cost_rub": cost_rub, "temperature": TEMPERATURE,
        "reasoning_enabled": int(reasoning is None or reasoning.get("enabled", True)),
    }
    if p_ok:
        stars, multis, censors = {}, {}, {}
        for k in LISTS:
            xs, mu, ce = switch_star(parsed["choices"][k])
            stars[k], multis[k], censors[k] = xs, mu, ce
        x1, x2, x3 = stars["today_vs_1m"], stars["today_vs_6m"], stars["6m_vs_7m"]
        row.update({
            "X1_star": x1, "X2_star": x2, "X3_star": x3, "parsed_decision": x1,
            "multiple_switch": int(any(multis.values())), "censor_flag": censors["today_vs_1m"],
            "IDF1": (x1 / Y_VAL) if x1 is not None else "",
            "IDF3": (x3 / Y_VAL) if x3 is not None else "",
            "beta_hat": (x1 / x3) if x1 and x3 else "",
            "present_biased": int(x1 < x3) if x1 is not None and x3 is not None else "",
            "future_biased": int(x1 > x3) if x1 is not None and x3 is not None else "",
            "consistent": int(x1 == x3) if x1 is not None and x3 is not None else "",
        })
    else:
        row.update({k: "" for k in ("X1_star", "X2_star", "X3_star", "parsed_decision",
                    "multiple_switch", "IDF1", "IDF3", "beta_hat",
                    "present_biased", "future_biased", "consistent")})
        row["censor_flag"] = "parse_fail"
    row["reasoning_length"] = len(str(row["reasoning_text"]).split())
    row["ai_disclaimer_flag"] = int(bool(re.search(r"as an ai|language model", raw, re.I)))
    return row


def main():
    alias = sys.argv[1]
    workers = int(sys.argv[2]) if len(sys.argv) > 2 else MAX_WORKERS
    model, reasoning, max_tokens = CONFIGS[alias]
    jobs = []
    for pid in ["P0", "P1", "P2", "P3", "P4", "P5", "P6", "P7", "P8"]:
        n = N_P0 if pid == "P0" else N_PX
        for r in range(n):
            jobs.append((pid, r, (r + int(pid[1:]) if pid != "P0" else r) % 6))
    print(f"[{alias}] {len(jobs)} запросов, модель {model}, reasoning={reasoning}, workers={workers}", flush=True)
    rows, fails = [], 0
    with ThreadPoolExecutor(max_workers=workers) as ex:
        futs = [ex.submit(run_one, model, reasoning, max_tokens, pid, r, p) for pid, r, p in jobs]
        for i, f in enumerate(futs):
            try:
                rows.append(f.result())
            except Exception as e:
                fails += 1
                print(f"[{alias}] FAIL job {jobs[i]}: {str(e)[:150]}", flush=True)
            if (i + 1) % 25 == 0:
                print(f"[{alias}] прогресс: {i+1}/{len(jobs)}", flush=True)
    ts = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    out = BASE / f"full_{alias}_{ts}.csv"
    with open(out, "w", newline="", encoding="utf-8") as fo:
        w = csv.DictWriter(fo, fieldnames=FIELDS)
        w.writeheader()
        w.writerows(rows)
    ok = sum(r["parse_ok"] for r in rows)
    cost = sum(float(r["cost_rub"]) for r in rows if r["cost_rub"] not in ("", None))
    print(f"[{alias}] ГОТОВО: {len(rows)} строк, fails={fails}, parse_ok={ok}/{len(rows)}, cost={cost:.2f} руб. -> {out}", flush=True)


if __name__ == "__main__":
    main()
