"""One table over every processed clip: outputs/summary.csv + outputs/summary.md

    python -m tools.summary
"""
import csv
import json
import sys
from pathlib import Path

COLS = [("clip", "Clip"), ("minutes", "Min"), ("left_share_baseline", "L found MP %"),
        ("left_share_final", "L found ours %"), ("right_share_baseline", "R found MP %"),
        ("right_share_final", "R found ours %"), ("both_hands_baseline", "Both MP %"),
        ("both_hands_final", "Both ours %"), ("jitter_baseline", "Jitter MP"), ("jitter_final", "Jitter ours"),
        ("baseline_label_flips", "Label flips fixed"), ("rejected_coworker_hands", "Co-worker hands rejected"),
        ("grabs", "Grabs"), ("fixed_contacts", "Surface contacts"), ("fps_video_equiv", "Video fps")]


def main(out="outputs"):
    rows = []
    for f in sorted(Path(out).glob("*/*.results.json")):
        d = json.load(open(f))
        m = dict(d["metrics"])
        m["clip"] = d["meta"]["clip"].rsplit(".", 1)[0]
        m["minutes"] = round(d["meta"]["n_frames"] / d["meta"]["fps"] / 60, 1)
        rows.append(m)
    if not rows:
        sys.exit("no results in outputs/")
    with open(Path(out) / "summary.csv", "w", newline="") as fh:
        w = csv.writer(fh)
        w.writerow([c for c, _ in COLS])
        for m in rows:
            w.writerow([m.get(c, "") for c, _ in COLS])

    def avg(k):
        v = [m[k] for m in rows if isinstance(m.get(k), (int, float))]
        return round(sum(v) / len(v), 1) if v else ""

    def tot(k):
        return sum(m.get(k) or 0 for m in rows)
    lines = ["| " + " | ".join(h for _, h in COLS) + " |", "|" + "---|" * len(COLS)]
    for m in rows:
        lines.append("| " + " | ".join(str(m.get(c, "")) for c, _ in COLS) + " |")
    lines.append("| **All {} clips** | {} | {} | {} | {} | {} | {} | {} | {} | {} | {} | {} | {} | {} | {} |".format(
        len(rows), round(sum(m["minutes"] for m in rows), 1), avg("left_share_baseline"), avg("left_share_final"),
        avg("right_share_baseline"), avg("right_share_final"), avg("both_hands_baseline"), avg("both_hands_final"),
        avg("jitter_baseline"), avg("jitter_final"), tot("baseline_label_flips"), tot("rejected_coworker_hands"),
        tot("grabs"), tot("fixed_contacts"), avg("fps_video_equiv")))
    (Path(out) / "summary.md").write_text("\n".join(lines) + "\n\nPercentages and jitter are averages per clip; counts are totals.\n")
    print("\n".join(lines))


if __name__ == "__main__":
    main(*sys.argv[1:])
