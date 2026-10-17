"""Сборка запроса messages=[system, user] с рандомизацией порядка списков.

Использование:
    from build_request import build_messages
    msgs, list_order = build_messages("P1", perm_index=0)

Порядок списков рандомизируется (6 перестановок), логируется в поле `list_order`.
См. 01_task_context.md §6.6, 02_llm_profiles.md §7.
"""

import itertools
import json
from pathlib import Path

BASE = Path(__file__).parent

with open(BASE / "user_blocks.json", encoding="utf-8") as f:
    BLOCKS = json.load(f)

with open(BASE.parent / "profiles.json", encoding="utf-8") as f:
    PROFILES = json.load(f)

SYSTEM_FILES = {
    "P0": "system/P0_baseline.txt",
    "P1": "system/P1_female_college_25.txt",
    "P2": "system/P2_female_college_48.txt",
    "P3": "system/P3_female_no_college_25.txt",
    "P4": "system/P4_female_no_college_48.txt",
    "P5": "system/P5_male_college_25.txt",
    "P6": "system/P6_male_college_48.txt",
    "P7": "system/P7_male_no_college_25.txt",
    "P8": "system/P8_male_no_college_48.txt",
}

PERMUTATIONS = list(itertools.permutations(["today_vs_1m", "today_vs_6m", "6m_vs_7m"]))
TEMPERATURE = 1.0


def load_system(profile_id: str) -> str:
    return (BASE / SYSTEM_FILES[profile_id]).read_text(encoding="utf-8").strip()


def build_user(list_order: tuple[str, str, str]) -> str:
    list_lines = "\n".join(BLOCKS["lists"][k] for k in list_order)
    return "\n\n".join(
        [BLOCKS["intro"], list_lines, BLOCKS["rows"], BLOCKS["payment"], BLOCKS["anonymity"], BLOCKS["format"]]
    )


def build_messages(profile_id: str, perm_index: int):
    """Возвращает (messages, list_order). perm_index 0..5."""
    list_order = PERMUTATIONS[perm_index % 6]
    messages = [
        {"role": "system", "content": load_system(profile_id)},
        {"role": "user", "content": build_user(list_order)},
    ]
    return messages, list(list_order)


if __name__ == "__main__":
    # Самопроверка: все 9 профилей x 6 перестановок собираются
    for pid in ["P0"] + [f"P{i}" for i in range(1, 9)]:
        for i in range(6):
            msgs, order = build_messages(pid, i)
            assert msgs[0]["role"] == "system" and msgs[1]["role"] == "user"
            assert set(order) == {"today_vs_1m", "today_vs_6m", "6m_vs_7m"}
            for k in ["today_vs_1m", "today_vs_6m", "6m_vs_7m"]:
                assert k in msgs[1]["content"]
    print("OK: 9 profiles x 6 permutations, temperature =", TEMPERATURE)
