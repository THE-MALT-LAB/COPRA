import json
import glob
from collections import defaultdict

VALID = {"A", "B", "C", "D"}

def normalize_answer(ans):
    ans = str(ans).strip()

    if ans == "":
        return None

    if len(ans) > 1:
        ans = ans[0]

    ans = ans.upper()

    if ans not in VALID:
        return None

    return ans


def get_dataset(video_name):
    if "DADA_2000" in video_name:
        return "DADA"
    if "CAP_DATA" in video_name:
        return "CAP"
    if "DoTA" in video_name:
        return "DoTA"
    if "MANUAL_DATA" in video_name:
        return "MANU"
    return "UNKNOWN"


def compute_accuracy_all(folder="scores_vru_accident"):
    stats = defaultdict(lambda: {"correct": 0, "total": 0})

    files = sorted(glob.glob(f"{folder}/vru_accident_cuda:*.jsonl"))

    print("Found files:", files)

    for path in files:
        with open(path, "r") as f:
            for line in f:
                if not line.strip():
                    continue

                item = json.loads(line)

                dataset = get_dataset(item["video_name"])
                pred = normalize_answer(item.get("answer", ""))
                gt = str(item["gt_answer"]).strip().upper()

                stats[dataset]["total"] += 1

                if pred == gt:
                    stats[dataset]["correct"] += 1

    results = {}
    overall_correct = 0
    overall_total = 0

    for dataset in ["DADA", "CAP", "DoTA", "MANU"]:
        c = stats[dataset]["correct"]
        t = stats[dataset]["total"]

        acc = 100 * c / t if t > 0 else 0.0
        results[dataset] = acc

        overall_correct += c
        overall_total += t

    results["OVERALL"] = 100 * overall_correct / overall_total

    return results

import sys
results = compute_accuracy_all(folder=sys.argv[1])

for k in ["DADA", "CAP", "DoTA", "MANU", "OVERALL"]:
    print(f"{k:8s}: {results[k]:.2f}")