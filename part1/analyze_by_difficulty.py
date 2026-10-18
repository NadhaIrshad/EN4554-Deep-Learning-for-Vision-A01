"""Accuracy by question difficulty, from the saved test predictions (no GPU, no retraining).

The LoRA+ paper reports that its gains are larger on harder tasks. This splits each method's test accuracy into
easy / hard question types and reports the LoRA+ - LoRA gap per type, averaged over seeds.
  VQAv2:   yes/no | number | other
  TextVQA: answer appears in the image's OCR tokens (copy the text) | answer not in OCR (must read/reason)

Usage:
  python analyze_by_difficulty.py --dataset vqav2
  python analyze_by_difficulty.py --dataset textvqa
"""
import argparse, glob, json, os, random, re
import numpy as np, pandas as pd

parser = argparse.ArgumentParser()
parser.add_argument("--dataset", default="vqav2", choices=["vqav2", "textvqa"])
parser.add_argument("--out-dir", default=None)
cli = parser.parse_args()

OUT_DIR = cli.out_dir or ("./outputs/part1_loraplus" + ("_textvqa" if cli.dataset == "textvqa" else ""))
DATASET = "lmms-lab-encoder/textvqa" if cli.dataset == "textvqa" else "merve/vqav2-small"
DATA_SEED = 42                        # must match part1_loraplus_smolvlm.py
NUM_WORDS = set("zero one two three four five six seven eight nine ten eleven twelve".split())

def normalize(s):
    s = s.lower().strip().split("\n")[0]
    s = re.sub(r"[^\w\s]", "", s)
    s = re.sub(r"\b(a|an|the)\b", " ", s)
    return " ".join(s.split())

# ---- rebuild the exact test split used in training (same RNG sequence as the main script) ----------------------
test_info = None
if cli.dataset == "textvqa":
    from datasets import load_dataset
    raw = load_dataset(DATASET, split="validation").remove_columns(["image"])
    rng = random.Random(DATA_SEED)
    img_ids = raw["image_id"]
    uniq = sorted(set(img_ids)); rng.shuffle(uniq)
    n10 = len(uniq) // 10
    val_imgs, test_imgs = set(uniq[:n10]), set(uniq[n10:2 * n10])
    buckets = {"train": [], "val": [], "test": []}
    for i, x in enumerate(img_ids):
        buckets["val" if x in val_imgs else "test" if x in test_imgs else "train"].append(i)
    for v in buckets.values():
        rng.shuffle(v)
    test_idx = buckets["test"]

def categorize(preds):
    if cli.dataset == "vqav2":
        out = []
        for r in preds:
            g = normalize(r["gt"])
            out.append("yes/no" if g in ("yes", "no") else
                       "number" if g.isdigit() or g in NUM_WORDS else "other")
        return out
    test_info = raw.select(test_idx[:len(preds)])     # N_TEST (500, or 64 for --quick runs)
    assert [r["question"] for r in preds] == test_info["question"], "test split mismatch"
    out = []
    for r, ocr in zip(preds, test_info["ocr_tokens"]):
        ocr_text = " " + normalize(" ".join(ocr)) + " "
        out.append("answer in OCR" if " " + normalize(r["gt"]) + " " in ocr_text else "answer not in OCR")
    return out

# ---- collect runs that have test predictions ---------------------------------------------------------------------
def method_of(name):
    return re.sub(r"_s\d+$", "", name)    # strip the seed suffix -> one method per config

rows = []
base = json.load(open(f"{OUT_DIR}/baseline.json"))
for cat, acc in zip(categorize(base["test_preds"]), [r["acc"] for r in base["test_preds"]]):
    rows.append(dict(method="baseline", seed=0, cat=cat, acc=acc))
for f in sorted(glob.glob(f"{OUT_DIR}/runs/*/results.json")):
    r = json.load(open(f))
    if "test_preds" not in r:
        continue
    for cat, p in zip(categorize(r["test_preds"]), r["test_preds"]):
        rows.append(dict(method=method_of(r["name"]), seed=r["seed"], cat=cat, acc=p["acc"]))
df = pd.DataFrame(rows)

per_seed = df.groupby(["method", "seed", "cat"]).acc.mean().mul(100).reset_index()
tbl = per_seed.groupby(["method", "cat"]).acc.agg(["mean", "std", "count"]).round(2)
counts = df[df.method == "baseline"].cat.value_counts()
print(f"=== {cli.dataset}: test accuracy (%) by question type, mean/std over seeds ===")
print("questions per type:", counts.to_dict())
wide = tbl["mean"].unstack("cat")
print(wide.round(2).to_string())

# LoRA+ - LoRA gap per type (paired by seed where both exist)
lora = [m for m in wide.index if m.startswith("main_LoRA_tuned")]
if lora:
    ref = per_seed[per_seed.method == lora[0]].set_index(["seed", "cat"]).acc
    gaps = []
    for m in wide.index:
        if m in (lora[0], "baseline") or not m.startswith(("main_", "connector_", "delta_")):
            continue
        d = per_seed[per_seed.method == m].set_index(["seed", "cat"]).acc - ref
        d = d.dropna().groupby("cat").agg(["mean", "std"])
        for cat, (mu, sd) in d.iterrows():
            gaps.append(dict(method=m, cat=cat, gap_vs_LoRA=mu, gap_std=sd))
    if gaps:
        g = pd.DataFrame(gaps).pivot(index="method", columns="cat", values="gap_vs_LoRA")
        print(f"\n=== gap vs {lora[0]} (points; > 0 means better than LoRA), paired by seed ===")
        print(g.round(2).to_string())
        pd.DataFrame(gaps).to_csv(f"{OUT_DIR}/difficulty_gaps.csv", index=False)
tbl.to_csv(f"{OUT_DIR}/difficulty_table.csv")
print(f"\nSaved difficulty_table.csv / difficulty_gaps.csv in {os.path.abspath(OUT_DIR)}")
