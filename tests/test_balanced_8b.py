"""Test 2-shot balanced Extract-Then-Compare calibration on Llama-3.1-8B-Instruct."""

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
        ("TP_Newsweek", True,
         "Newsweek to cease print, go all-digital in 2013",
         "Newsweek to cease print publication at end of year"),
        ("TP_Google", True,
         "Google stock takes a dive after mid-day earnings release",
         "Google stock crashes below $700 after earnings report leaks early"),
        ("TP_Mitchell", True,
         "Andrew Mitchell resigns as Government Chief Whip",
         "Andrew Mitchell resigns over police row remarks"),
        ("TP_YashChopra", True,
         "Bollywood veteran Yash Chopra dies at 80",
         "Yash Chopra dies of dengue fever at 80"),
        ("TP_ChelseaMatch", True,
         "Chelsea beats Tottenham 4-2 with Juan Mata goals",
         "Gary Cahill scores goal for Chelsea against Tottenham"),
        ("TP_Cardiff", True,
         "Cardiff hit-and-runs: Police question 31-year-old man",
         "Cardiff hit-and-runs: Woman killed, 13 injured"),
        ("TN_Google_Newsweek", False,
         "Google stock drops 9% after premature Q3 earnings release",
         "Newsweek ends print edition, goes all-digital in 2013"),
        ("TN_BoyScouts_Savile", False,
         "Boy Scouts release secret perversion files in Oregon court",
         "Scotland Yard launches formal criminal inquiry into Jimmy Savile BBC abuse"),
        ("TN_Chelsea_Rooney", False,
         "Juan Mata scores twice as Chelsea beat Tottenham 4-2",
         "Wayne Rooney scores twice for Manchester United against Stoke City"),
        ("TN_Beirut_Kuwait", False,
         "Thousands attend funeral of assassinated intelligence chief in Beirut",
         "Kuwaiti police clash with protesters over election law changes"),
    ]

    sys_msg = (
        "You decide whether two social-media post groups refer to the SAME specific real-world event.\n"
        "Rules:\n"
        "- TRUE if both cover the same specific incident, match, announcement, or developing story, "
        "even if they emphasize different details, updates, or reactions (e.g., 'Team X beats Team Y 4-2' "
        "and 'Player Z scores for Team X against Team Y' -> true; 'Official resigns' and 'Official quits over scandal' -> true).\n"
        "- FALSE if they cover different incidents, different matches, different countries, or different main subjects "
        "(e.g., 'Team X vs Team Y' and 'Team A vs Team B' -> false; 'Scandal at Org 1' and 'Scandal at Org 2' -> false).\n"
        "First write concise 5-10 word factual summaries (event_a, event_b), then output boolean same_event in strict JSON."
    )

    ok_tp = tot_tp = ok_tn = tot_tn = 0
    for name, truth, a, b in cases:
        user_msg = (
            f"Group A:\n  - {a}\n\nGroup B:\n  - {b}\n\n"
            'Return strict JSON: {"event_a": "<5-10 words>", "event_b": "<5-10 words>", "same_event": true or false}'
        )
        payload = {
            "model": model,
            "messages": [{"role": "system", "content": sys_msg}, {"role": "user", "content": user_msg}],
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
        parsed = json.loads(body["choices"][0]["message"]["content"])
        pred = bool(parsed.get("same_event"))
        if truth:
            tot_tp += 1
            ok_tp += int(pred)
        else:
            tot_tn += 1
            ok_tn += int(not pred)
        print(f"[{'PASS' if pred == truth else 'FAIL'}] {name:20s} truth={str(truth):5s} pred={str(pred):5s} | A={parsed.get('event_a')!r} | B={parsed.get('event_b')!r}")
    print(f"TP Recall: {ok_tp}/{tot_tp} | TN Specificity: {ok_tn}/{tot_tn} | Total: {ok_tp+ok_tn}/{len(cases)}")


if __name__ == "__main__":
    main()
