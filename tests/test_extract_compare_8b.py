"""Quick test of Extract-Then-Compare structured prompting on Llama-3.1-8B-Instruct."""

from __future__ import annotations

import json
import sys
import urllib.request
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from core_shared.key_loader import load_provider_key


def main() -> None:
    url, model, key = load_provider_key("cloudflare")
    cases = [
        (
            "TP_Newsweek",
            True,
            "A: Newsweek to end print edition, go all-digital in 2013 | Newsweek magazine stops printing after 80 years",
            "B: Newsweek ends 80 years of print publication | Last print issue of Newsweek set for Dec 31",
        ),
        (
            "FP_Google_vs_Newsweek",
            False,
            "A: Google stock drops 9% after premature Q3 earnings release | Trading halted on GOOG",
            "B: Newsweek to end print edition, go all-digital in 2013 | Last print issue Dec 31",
        ),
        (
            "FP_FBI_vs_Google",
            False,
            "A: FBI arrests Bangladeshi man in NYC Federal Reserve bomb plot | Man arrested trying to blow up NY Fed",
            "B: Google stock plunges after accidental earnings leak | GOOG trading halted",
        ),
        (
            "FP_BoyScouts_vs_Savile",
            False,
            "A: Boy Scouts release secret perversion files on alleged sex abusers in Oregon court",
            "B: Scotland Yard launches formal criminal inquiry into Jimmy Savile BBC abuse scandal",
        ),
        (
            "FP_Chelsea_vs_Rooney",
            False,
            "A: Juan Mata scores twice as Chelsea beat Tottenham 4-2 at White Hart Lane",
            "B: Wayne Rooney scores twice for Manchester United against Stoke City at Old Trafford",
        ),
        (
            "FP_Beirut_vs_Kuwait",
            False,
            "A: Thousands attend funeral of assassinated Lebanese intelligence chief Wissam al-Hassan in Beirut",
            "B: Kuwaiti police clash with opposition protesters marching over election law changes",
        ),
        (
            "FP_Lance_vs_Yelp",
            False,
            "A: Nike drops Lance Armstrong over doping allegations; steps down from Livestrong",
            "B: Just checked in at 24 Hour Fitness on Yelp; great morning gym workout",
        ),
    ]

    sys_msg = (
        "You adjudicate whether two social-media post groups report the EXACT SAME specific real-world event. "
        "Different incidents, different countries, different sports matches, or different organizations are NOT the same event, "
        "even if they share a broad topic (e.g., two different abuse scandals, two different football matches, or two different Middle East news stories). "
        "First concisely summarize Group A's specific event (event_a) and Group B's specific event (event_b) in 5-10 words each, "
        "then set same_event to true ONLY if both refer to the exact same specific incident."
    )

    correct = 0
    for name, truth, a, b in cases:
        user_msg = (
            f"Group A:\n{a}\n\nGroup B:\n{b}\n\n"
            'Return strict JSON: {"event_a": "<5-10 words>", "event_b": "<5-10 words>", "same_event": true or false}'
        )
        payload = {
            "model": model,
            "messages": [
                {"role": "system", "content": sys_msg},
                {"role": "user", "content": user_msg},
            ],
            "temperature": 0.0,
            "max_tokens": 120,
            "response_format": {"type": "json_object"},
        }
        req = urllib.request.Request(
            f"{url}/chat/completions",
            data=json.dumps(payload).encode("utf-8"),
            headers={"Content-Type": "application/json", "Authorization": f"Bearer {key}"},
            method="POST",
        )
        with urllib.request.urlopen(req, timeout=20) as r:
            body = json.loads(r.read().decode("utf-8"))
        raw = body["choices"][0]["message"]["content"]
        parsed = json.loads(raw)
        ok = bool(parsed.get("same_event")) == truth
        correct += int(ok)
        print(
            f"[{'PASS' if ok else 'FAIL'}] {name:24s} truth={str(truth):5s} pred={str(parsed.get('same_event')):5s} "
            f"tok={body['usage']['completion_tokens']:2d} | A={parsed.get('event_a')!r} | B={parsed.get('event_b')!r}"
        )
    print(f"Accuracy: {correct}/{len(cases)}")


if __name__ == "__main__":
    main()
