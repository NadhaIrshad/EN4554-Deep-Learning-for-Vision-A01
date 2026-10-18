"""Part 1 — Does the LoRA+ learning-rate claim transfer to a vision-language model?

Script version of part1_loraplus_smolvlm.ipynb (LoRA+, Hayou, Ghosh & Yu 2024) on SmolVLM-256M / VQAv2.

Hypotheses:
  H1  (eta_A, lambda) grid optimum lies at lambda > 1 and beats the best lambda = 1 (tuned LoRA) cell.
  H2a tuned LoRA+ has higher test VQA accuracy than tuned LoRA at a fixed budget (3 seeds).
  H2b LoRA+ reaches a target val accuracy / loss in fewer steps.
  H3  with lambda = 1, Delta W = B_t A_t comes mostly from B over near-frozen A_0; LoRA+ raises A's share.
  Bonus: is the benefit carried by the vision encoder or the language decoder?

Usage:
  python part1_loraplus_smolvlm.py --gpu 0 --quick     # smoke test (~minutes)
  python part1_loraplus_smolvlm.py --gpu 0             # full experiment
Finished runs are cached as results.json under the output dir, so rerunning the same command resumes.
"""
import argparse
import os

parser = argparse.ArgumentParser()
parser.add_argument("--gpu", default="0", help="GPU index to use (sets CUDA_VISIBLE_DEVICES)")
parser.add_argument("--quick", action="store_true", help="tiny smoke test of the whole pipeline")
parser.add_argument("--dataset", default="vqav2", choices=["vqav2", "textvqa"],
                    help="vqav2 = merve/vqav2-small; textvqa = harder OCR-heavy task (10 answers, split by image)")
parser.add_argument("--out-dir", default=None, help="output directory (default ./outputs/part1_loraplus[_quick])")
parser.add_argument("--experiment", default="main", choices=["main", "delta", "connector", "tower", "single"],
                    help="main = grid + main comparison + bonus; delta = paper's per-step efficiency terms "
                         "(delta_A, delta_B) for the main configs, 1 seed; connector = LoRA also on the "
                         "vision->text connector, global vs width-scaled lambda, 3 seeds; tower = 2x2 ablation of "
                         "lambda in the vision encoder x language decoder, 3 seeds; single = one full-budget run "
                         "from scratch with --lr-a/--ratio/--seed (no grid; for reproducing a main result on Colab)")
parser.add_argument("--lr-a", type=float, default=2e-4, help="single: eta_A")
parser.add_argument("--ratio", type=float, default=1.0, help="single: lambda = eta_B / eta_A (1 = plain LoRA)")
parser.add_argument("--seed", type=int, default=42, help="single: training seed")
parser.add_argument("--seeds", type=int, nargs="+", default=None, help="tower: override the training seeds")
parser.add_argument("--arms", nargs="+", default=None, choices=["neither", "text", "vision", "both"],
                    help="tower: run only these arms (to split the ablation across GPUs)")
parser.add_argument("--stage", default="all", choices=["baseline", "grid", "main", "all"],
                    help="stop after this stage (all = baseline + grid + main + bonus)")
cli = parser.parse_args()
os.environ["CUDA_VISIBLE_DEVICES"] = cli.gpu  # must be set before torch is imported
os.environ.setdefault("NCCL_P2P_DISABLE", "1")  # accelerate refuses RTX 40xx cards without these (no P2P/IB)
os.environ.setdefault("NCCL_IB_DISABLE", "1")

import re, gc, json, math, time, random, itertools, collections
import numpy as np, pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import torch
import transformers, peft
from datasets import load_dataset
from transformers import AutoProcessor, TrainingArguments, Trainer, TrainerCallback, set_seed
try:
    from transformers import AutoModelForImageTextToText as AutoVLM
except ImportError:
    from transformers import AutoModelForVision2Seq as AutoVLM
from peft import LoraConfig, get_peft_model

pd.set_option("display.width", 200); pd.set_option("display.max_columns", 30)
print("transformers", transformers.__version__, "| peft", peft.__version__, "| torch", torch.__version__)
assert torch.cuda.is_available(), "CUDA not available"
print(f"GPU {cli.gpu}:", torch.cuda.get_device_name(0))

# ----------------------------------------------------------------------------------------------------------------
# 1. Config
# ----------------------------------------------------------------------------------------------------------------
DATA_SEED = 42                     # fixes the data split for every run
SEEDS     = [42, 43, 44]           # training seeds for the main comparison
MODEL_ID  = "HuggingFaceTB/SmolVLM-256M-Instruct"
DATASET   = "merve/vqav2-small"    # VQAv2 validation subset (parquet)
N_TRAIN, N_VAL, N_TEST = 3000, 300, 500
BATCH     = 8
RANK, ALPHA, DROPOUT = 8, 16, 0.05
PROMPT    = "{q}\nAnswer with a single word or short phrase."
# LoRA on every attention + MLP linear layer of both towers (vision encoder and language decoder)
TARGET_REGEX = r".*(vision_model|text_model).*\.(q_proj|k_proj|v_proj|o_proj|out_proj|fc1|fc2|gate_proj|up_proj|down_proj)"
# connector experiment: also adapt the pixel-shuffle connector (fan-in 768*16 = 12288 -> 576), unique to the VLM
TARGET_REGEX_CONN = TARGET_REGEX + r"|.*connector\.modality_projection\.proj"
TEXT_WIDTH = 576                   # reference fan-in for width-scaled lambda (text decoder hidden size)
DELTA_EVERY = 25                   # steps between measurements of the paper's efficiency terms delta_A / delta_B

# (eta_A, lambda) grid — short runs, selected on the validation split. lambda = 1 is standard LoRA (an LR sweep).
GRID_LR_A   = [5e-5, 1e-4, 2e-4, 5e-4]
GRID_RATIO  = [1, 4, 16, 32]
GRID_EPOCHS, GRID_N_TRAIN = 1, 1500
MAIN_EPOCHS = 2
N_PROBES    = 8                    # checkpoints at which val accuracy + A/B shares are measured in main runs

if cli.dataset == "textvqa":
    # Harder, OCR-heavy task: tests the paper's caveat that LoRA+ matters more when feature learning is needed.
    # Validation split only (5000 Qs, 10 answers each), split by image like VQAv2. Smaller re-tuning grid.
    DATASET = "lmms-lab-encoder/textvqa"
    GRID_LR_A, GRID_RATIO = [1e-4, 2e-4, 5e-4, 1e-3], [1, 4, 16]   # 1e-3 added: 5e-4 was the edge of the grid

QUICK = cli.quick
if QUICK:
    N_TRAIN, N_VAL, N_TEST, GRID_N_TRAIN = 200, 64, 64, 200
    GRID_LR_A, GRID_RATIO, SEEDS, MAIN_EPOCHS = [2e-4], [1, 16], [42], 1

random.seed(DATA_SEED); np.random.seed(DATA_SEED); set_seed(DATA_SEED)
OUT_DIR = cli.out_dir or ("./outputs/part1_loraplus" + ("_textvqa" if cli.dataset == "textvqa" else "")
                          + ("_quick" if QUICK else ""))
os.makedirs(OUT_DIR, exist_ok=True); print("Outputs ->", os.path.abspath(OUT_DIR))

# ----------------------------------------------------------------------------------------------------------------
# 2. Data — VQAv2 subset, split by image into train / val (model selection) / test (final report)
# ----------------------------------------------------------------------------------------------------------------
raw = load_dataset(DATASET, split="validation")
print(raw)

rng = random.Random(DATA_SEED)
if "image_id" in raw.column_names:
    img_ids = raw["image_id"]
    uniq = sorted(set(img_ids)); rng.shuffle(uniq)
    n10 = len(uniq) // 10
    val_imgs, test_imgs = set(uniq[:n10]), set(uniq[n10:2 * n10])
    split_of = lambda x: "val" if x in val_imgs else "test" if x in test_imgs else "train"
    buckets = {"train": [], "val": [], "test": []}
    for i, x in enumerate(img_ids):
        buckets[split_of(x)].append(i)
else:
    idx = list(range(len(raw))); rng.shuffle(idx); n10 = len(idx) // 10
    buckets = {"val": idx[:n10], "test": idx[n10:2 * n10], "train": idx[2 * n10:]}
for v in buckets.values():
    rng.shuffle(v)
train_ds = raw.select(buckets["train"][:N_TRAIN])
val_ds   = raw.select(buckets["val"][:N_VAL])
test_ds  = raw.select(buckets["test"][:N_TEST])
print(f"train={len(train_ds)} val={len(val_ds)} test={len(test_ds)}  ({DATASET}/validation, data seed {DATA_SEED})")

def gt_answers(e):
    a = e.get("answers")
    if a:
        return [x["answer"] if isinstance(x, dict) else x for x in a]
    return [e["multiple_choice_answer"]]

def target(e):
    # training label / reported gt: VQAv2's consensus answer, else the most common of the annotator answers (TextVQA)
    if e.get("multiple_choice_answer"):
        return e["multiple_choice_answer"]
    return collections.Counter(gt_answers(e)).most_common(1)[0][0]

ex = train_ds[0]; print(ex["question"], "->", target(ex), "| all:", gt_answers(ex)[:10])

# ----------------------------------------------------------------------------------------------------------------
# 3. Processor and assistant-only-loss collator
# ----------------------------------------------------------------------------------------------------------------
processor = AutoProcessor.from_pretrained(MODEL_ID)
processor.image_processor.do_image_splitting = False    # one 512px tile -> 64 image tokens
tok = processor.tokenizer
ASSIST_IDS = tok.encode("Assistant:", add_special_tokens=False)

def build_msgs(q, a=None):
    msgs = [{"role": "user", "content": [{"type": "image"}, {"type": "text", "text": PROMPT.format(q=q)}]}]
    if a is not None:
        msgs.append({"role": "assistant", "content": [{"type": "text", "text": a}]})
    return msgs

def find_last(seq, sub):
    for i in range(len(seq) - len(sub), -1, -1):
        if seq[i:i + len(sub)] == sub:
            return i
    return -1

def train_collate(examples):
    texts  = [processor.apply_chat_template(build_msgs(e["question"], target(e))) for e in examples]
    images = [[e["image"].convert("RGB")] for e in examples]
    tok.padding_side = "right"
    batch = processor(text=texts, images=images, return_tensors="pt", padding=True)
    labels = batch["input_ids"].clone()
    for i, ids in enumerate(batch["input_ids"].tolist()):
        j = find_last(ids, ASSIST_IDS)
        labels[i, : (j + len(ASSIST_IDS)) if j >= 0 else len(ids)] = -100   # loss on the assistant turn only
    labels[batch["attention_mask"] == 0] = -100
    batch["labels"] = labels
    return batch

b = train_collate([train_ds[0], train_ds[1]])
for i in range(2):
    print("supervised:", repr(tok.decode(b["labels"][i][b["labels"][i] != -100])), "| seq len", b["input_ids"].shape[1])

# ----------------------------------------------------------------------------------------------------------------
# 4. VQA evaluation (VQAv2 soft accuracy min(#annotators agreeing / 3, 1) when all 10 answers are available,
#    else exact match against multiple_choice_answer — the case for merve/vqav2-small)
# ----------------------------------------------------------------------------------------------------------------
def normalize(s):
    s = s.lower().strip().split("\n")[0]
    s = re.sub(r"[^\w\s]", "", s)
    s = re.sub(r"\b(a|an|the)\b", " ", s)
    return " ".join(s.split())

@torch.no_grad()
def evaluate(model, ds, bs=32, max_new_tokens=10):
    was_training = model.training
    model.eval(); tok.padding_side = "left"
    rows = []
    for s in range(0, len(ds), bs):
        exs = ds.select(range(s, min(s + bs, len(ds))))
        texts  = [processor.apply_chat_template(build_msgs(e["question"]), add_generation_prompt=True) for e in exs]
        images = [[e["image"].convert("RGB")] for e in exs]
        inputs = processor(text=texts, images=images, return_tensors="pt", padding=True).to(model.device)
        with torch.autocast("cuda", dtype=torch.float16):
            out = model.generate(**inputs, max_new_tokens=max_new_tokens, do_sample=False)
        gens = processor.batch_decode(out[:, inputs["input_ids"].shape[1]:], skip_special_tokens=True)
        for e, g in zip(exs, gens):
            p = normalize(g); gts = [normalize(a) for a in gt_answers(e)]
            # VQAv2 soft accuracy needs the 10 annotator answers; merve/vqav2-small only has the consensus
            # multiple_choice_answer, so fall back to exact match (otherwise a correct answer scores only 1/3).
            acc = min(sum(a == p for a in gts) / 3, 1.0) if len(gts) > 1 else float(p == gts[0])
            rows.append(dict(question=e["question"], pred=p, gt=target(e), acc=acc))
    if was_training:
        model.train()
    return float(np.mean([r["acc"] for r in rows])), rows

def load_model():
    return AutoVLM.from_pretrained(MODEL_ID, torch_dtype=torch.float32).float().to("cuda")  # fp32 master weights + fp16 autocast

# ----------------------------------------------------------------------------------------------------------------
# 5. LoRA+ optimiser and probes
#    LoRA+ = two AdamW groups: lora_B gets lambda * eta_A, everything else eta_A. The linear schedule scales every
#    group by the same factor, so the ratio is kept. ratio_vision / ratio_text override lambda for one tower (bonus).
#    A/B share probe (H3): B_t A_t = B_t A_0 (B learns on random features) + B_t (A_t - A_0) (A learns features).
# ----------------------------------------------------------------------------------------------------------------
def tower(n):
    return "connector" if "connector" in n else "vision" if "vision_model" in n else "text"

def make_optimizer(model, lr_A, ratio, ratio_vision=None, ratio_text=None, width_scale=False):
    # width_scale: lambda_layer = ratio * fan_in / TEXT_WIDTH (LoRA+ theory: optimal eta_B/eta_A = Theta(fan-in))
    fan_in = {n: m.lora_A["default"].weight.shape[1] for n, m in lora_modules(model).items()}
    groups = {}
    for n, p in model.named_parameters():
        if not p.requires_grad:
            continue
        r = 1.0
        if "lora_B" in n:
            r = ratio
            if ratio_vision is not None and "vision_model" in n: r = ratio_vision
            if ratio_text is not None and "text_model" in n: r = ratio_text
            if width_scale:
                r *= fan_in[n.replace("base_model.model.", "").split(".lora_B")[0]] / TEXT_WIDTH
        groups.setdefault(round(float(r), 6), []).append(p)
    print("  param groups:", {f"lr={lr_A*r:.1e}": sum(p.numel() for p in ps) for r, ps in groups.items()})
    return torch.optim.AdamW([{"params": ps, "lr": lr_A * r} for r, ps in groups.items()], lr=lr_A, weight_decay=0.0)

def lora_modules(model):
    return {n.replace("base_model.model.", ""): m for n, m in model.named_modules()
            if hasattr(m, "lora_A") and "default" in getattr(m, "lora_A", {})}

class Probe(TrainerCallback):
    # At chosen steps: VQA accuracy on the val split (H2b) and A/B update shares per tower (H3).
    # Every DELTA_EVERY steps: the paper's efficiency terms on a fixed val batch, per LoRA layer,
    #   delta_A = B_{t-1} (A_t - A_{t-1}) x   (output change from updating A)   and
    #   delta_B = (B_t - B_{t-1}) A_{t-1} x   (output change from updating B);  LoRA+ claims eta_A = eta_B leaves
    #   them unbalanced and eta_B >> eta_A balances them.
    def __init__(self, model, steps, eval_acc):
        self.model, self.steps, self.eval_acc = model, set(steps), eval_acc
        self.mods = lora_modules(model)
        self.A0 = {n: m.lora_A["default"].weight.detach().float().clone() for n, m in self.mods.items()}
        self.acc_curve, self.shares, self.deltas = [], [], []
        b = train_collate([val_ds[i] for i in range(4)]); b.pop("labels")
        self.probe_batch = {k: v.to("cuda") for k, v in b.items()}
        self.old = None

    def _measure_delta(self, step):
        return step in (1, 2, 3, 5, 10) or step % DELTA_EVERY == 0

    def on_pre_optimizer_step(self, args, state, control, **kw):
        if self._measure_delta(state.global_step + 1):    # global_step is incremented after the optimizer step
            self.old = {n: (m.lora_A["default"].weight.detach().float().clone(),
                            m.lora_B["default"].weight.detach().float().clone()) for n, m in self.mods.items()}

    @torch.no_grad()
    def on_optimizer_step(self, args, state, control, **kw):
        if self.old is None:
            return
        per_layer = {}
        def hook(name, m):
            def f(mod, inp):
                x = inp[0].detach().float().reshape(-1, inp[0].shape[-1])
                A0, B0 = self.old[name]
                A1, B1 = m.lora_A["default"].weight.float(), m.lora_B["default"].weight.float()
                s = m.scaling["default"]
                dA = (x @ (A1 - A0).T) @ B0.T * s
                dB = (x @ A0.T) @ (B1 - B0).T * s
                per_layer[name] = (dA.pow(2).mean().sqrt().item(), dB.pow(2).mean().sqrt().item())
            return f
        hs = [m.register_forward_pre_hook(hook(n, m)) for n, m in self.mods.items()]
        was_training = self.model.training; self.model.eval()
        try:
            with torch.autocast("cuda", dtype=torch.float16):
                self.model(**self.probe_batch)
        finally:
            for h in hs: h.remove()
            if was_training: self.model.train()
        step, agg = state.global_step + 1, {}
        for n, (a, b) in per_layer.items():
            agg.setdefault(tower(n), []).append((a, b))
        for tw, v in agg.items():
            v = np.array(v)
            ratio = v[:, 0] / np.maximum(v[:, 1], 1e-20)
            self.deltas.append(dict(step=step, tower=tw, delta_A=float(np.median(v[:, 0])),
                                    delta_B=float(np.median(v[:, 1])),
                                    log10_ratio_A_over_B=float(np.median(np.log10(np.maximum(ratio, 1e-20))))))
        self.old = None

    @torch.no_grad()
    def measure_shares(self, step):
        agg = {}
        for n, m in self.mods.items():
            A, B = m.lora_A["default"].weight.float(), m.lora_B["default"].weight.float()
            dA = A - self.A0[n]
            fromB, fromA = (B @ self.A0[n]).norm().item(), (B @ dA).norm().item()
            tw = tower(n)
            agg.setdefault(tw, []).append((fromA / (fromA + fromB + 1e-12), (dA.norm() / self.A0[n].norm()).item()))
        for tw, v in agg.items():
            v = np.array(v)
            self.shares.append(dict(step=step, tower=tw, share_A=v[:, 0].mean(), rel_move_A=v[:, 1].mean()))

    def on_step_end(self, args, state, control, **kw):
        if state.global_step in self.steps:
            self.measure_shares(state.global_step)
            if self.eval_acc:
                acc, _ = evaluate(self.model, val_ds)
                self.acc_curve.append((state.global_step, acc))
                print(f"  step {state.global_step}: val acc {100*acc:.2f}%")

# ----------------------------------------------------------------------------------------------------------------
# 6. One run = (eta_A, lambda, seed, budget). Results are cached, so reruns resume.
# ----------------------------------------------------------------------------------------------------------------
def run(name, lr_A, ratio, seed, epochs, n_train, probe_acc=False, ratio_vision=None, ratio_text=None, eval_test=False,
        lora_targets=TARGET_REGEX, width_scale=False):
    path = f"{OUT_DIR}/runs/{name}"; res_file = f"{path}/results.json"
    if os.path.exists(res_file):
        print(f"=== {name}: cached")
        return json.load(open(res_file))
    os.makedirs(path, exist_ok=True); set_seed(seed)
    print(f"=== {name}: eta_A={lr_A:g} lambda={ratio} seed={seed} epochs={epochs} n={n_train}")

    model = get_peft_model(load_model(), LoraConfig(r=RANK, lora_alpha=ALPHA, lora_dropout=DROPOUT,
                                                    target_modules=lora_targets, bias="none"))
    n_tr = sum(p.numel() for p in model.parameters() if p.requires_grad)
    n_tot = sum(p.numel() for p in model.parameters())
    tr_ds = train_ds.select(range(n_train))
    total_steps = math.ceil(n_train / BATCH) * epochs
    probe_steps = sorted(set(np.linspace(total_steps / N_PROBES, total_steps, N_PROBES).round().astype(int).tolist()))
    probe = Probe(model, probe_steps, eval_acc=probe_acc)

    args = TrainingArguments(
        output_dir=f"{path}/tmp", num_train_epochs=epochs, per_device_train_batch_size=BATCH,
        per_device_eval_batch_size=16, learning_rate=lr_A, lr_scheduler_type="linear",
        warmup_steps=int(0.05 * total_steps), max_grad_norm=1.0, fp16=True,
        logging_steps=10, eval_strategy="steps", eval_steps=max(10, total_steps // 20), prediction_loss_only=True,
        save_strategy="no", report_to="none", remove_unused_columns=False, label_names=["labels"],
        dataloader_num_workers=2, seed=seed)
    opt = make_optimizer(model, lr_A, ratio, ratio_vision, ratio_text, width_scale=width_scale)
    trainer = Trainer(model=model, args=args, train_dataset=tr_ds, eval_dataset=val_ds,
                      data_collator=train_collate, callbacks=[probe], optimizers=(opt, None))

    t0 = time.time(); trainer.train(); train_min = (time.time() - t0) / 60
    lh = trainer.state.log_history
    train_loss = [(x["step"], x["loss"]) for x in lh if "loss" in x]
    val_loss   = [(x["step"], x["eval_loss"]) for x in lh if "eval_loss" in x]
    grad_norm  = [x["grad_norm"] for x in lh if "grad_norm" in x]
    val_acc = probe.acc_curve[-1][1] if probe.acc_curve else evaluate(model, val_ds)[0]
    res = dict(name=name, lr_A=lr_A, ratio=ratio, ratio_vision=ratio_vision, ratio_text=ratio_text, seed=seed,
               epochs=epochs, n_train=n_train, steps=total_steps, trainable=n_tr, total=n_tot, train_min=train_min,
               train_loss=train_loss, val_loss=val_loss, max_grad_norm=float(np.nanmax(grad_norm)) if grad_norm else None,
               val_acc=val_acc, acc_curve=probe.acc_curve, shares=probe.shares, deltas=probe.deltas,
               connector=lora_targets != TARGET_REGEX, width_scale=width_scale,
               diverged=bool(not train_loss or not np.isfinite(train_loss[-1][1])))
    if eval_test:
        res["test_acc"], res["test_preds"] = evaluate(model, test_ds)
    print(f"  -> val acc {100*val_acc:.2f}%" + (f" | test acc {100*res['test_acc']:.2f}%" if eval_test else "")
          + f" | {train_min:.1f} min")
    json.dump(res, open(res_file, "w"))
    del model, trainer, probe, opt; gc.collect(); torch.cuda.empty_cache()
    return res

# ---- 6a. Baseline: the untouched model (val and test) ----------------------------------------------------------
base_file = f"{OUT_DIR}/baseline.json"
if os.path.exists(base_file):
    baseline = json.load(open(base_file))
else:
    m = load_model()
    baseline = dict(val_acc=evaluate(m, val_ds)[0]); baseline["test_acc"], baseline["test_preds"] = evaluate(m, test_ds)
    json.dump(baseline, open(base_file, "w")); del m; gc.collect(); torch.cuda.empty_cache()
print(f"baseline: val {100*baseline['val_acc']:.2f}% | test {100*baseline['test_acc']:.2f}%")
if cli.stage == "baseline":
    raise SystemExit

# ---- Single run from scratch (reproduces one main-table cell without the grid; e.g. on Colab) -----------------
if cli.experiment == "single":
    r = run(f"single_lr{cli.lr_a:g}_r{cli.ratio:g}_s{cli.seed}", cli.lr_a, cli.ratio, cli.seed, MAIN_EPOCHS, N_TRAIN,
            probe_acc=True, eval_test=True)
    print(f"\n=== single run: eta_A={cli.lr_a:g} lambda={cli.ratio:g} seed={cli.seed} ({cli.dataset}) ===\n"
          f"zero-shot test {100*baseline['test_acc']:.2f}%  ->  adapted test {100*r['test_acc']:.2f}%  "
          f"(val {100*r['val_acc']:.2f}%, final val loss {r['val_loss'][-1][1]:.4f}, {r['train_min']:.1f} min)")
    raise SystemExit

# ---- 6b. (eta_A, lambda) grid — H1 -----------------------------------------------------------------------------
# lambda = 1 column is standard LoRA swept over LRs, so "LoRA+ only wins because B has a larger LR" is ruled out
# if the best cell has lambda > 1 and beats every lambda = 1 cell.
grid = [run(f"grid_lr{lr:g}_r{r}", lr, r, SEEDS[0], GRID_EPOCHS, GRID_N_TRAIN)
        for lr, r in itertools.product(GRID_LR_A, GRID_RATIO)]
gdf = pd.DataFrame([dict(lr_A=g["lr_A"], ratio=g["ratio"], lr_B=g["lr_A"] * g["ratio"], val_acc=100 * g["val_acc"],
                         final_val_loss=g["val_loss"][-1][1] if g["val_loss"] else np.nan, diverged=g["diverged"],
                         max_grad_norm=g["max_grad_norm"]) for g in grid])
gdf.to_csv(f"{OUT_DIR}/grid.csv", index=False)
print("\n=== Grid (H1) ===\n", gdf.round(4).to_string(index=False))

best_lora = gdf[gdf.ratio == 1].sort_values("val_acc", ascending=False).iloc[0]
best_plus = gdf[gdf.ratio > 1].sort_values("val_acc", ascending=False).iloc[0]
print(f"best LoRA  (lambda=1): eta={best_lora.lr_A:g}  val acc {best_lora.val_acc:.2f}%")
print(f"best LoRA+ (lambda>1): eta_A={best_plus.lr_A:g} lambda={best_plus.ratio:g}  val acc {best_plus.val_acc:.2f}%")

fig, axes = plt.subplots(1, 2, figsize=(13, 4.5))
for ax, col, title, cmap in [(axes[0], "val_acc", "val VQA acc (%)", "viridis"),
                             (axes[1], "final_val_loss", "final val loss", "viridis_r")]:
    piv = gdf.pivot(index="lr_A", columns="ratio", values=col)
    im = ax.imshow(piv.values, cmap=cmap, aspect="auto", origin="lower")
    ax.set_xticks(range(len(piv.columns)), [f"λ={c:g}" for c in piv.columns])
    ax.set_yticks(range(len(piv.index)), [f"{i:g}" for i in piv.index]); ax.set_ylabel("η_A")
    for (i, j), v in np.ndenumerate(piv.values):
        ax.text(j, i, f"{v:.2f}" if col == "val_acc" else f"{v:.3f}", ha="center", va="center", color="w", fontsize=9)
    ax.set_title(title); fig.colorbar(im, ax=ax)
plt.suptitle("H1: where is the optimum in the (η_A, λ = η_B/η_A) grid?"); plt.tight_layout()
plt.savefig(f"{OUT_DIR}/grid_heatmap.png", dpi=150, bbox_inches="tight"); plt.close()
if cli.stage == "grid":
    raise SystemExit

# ---- Extra experiments (reuse the grid's tuned learning rates) --------------------------------------------------
if cli.experiment != "main":
    lr_lora, lr_plus, lam = float(best_lora.lr_A), float(best_plus.lr_A), float(best_plus.ratio)
    if cli.experiment == "delta":
        # Step 5, claim-tied metric: the paper's own efficiency terms delta_A / delta_B, per tower, 1 seed.
        exp = {"LoRA (tuned)":       dict(lr_A=lr_lora, ratio=1),
               "LoRA+ (tuned)":      dict(lr_A=lr_plus, ratio=lam),
               "LoRA+ (drop-in 16)": dict(lr_A=lr_lora, ratio=16)}
        seeds, lora_targets, probe_acc = SEEDS[:1], TARGET_REGEX, False
    elif cli.experiment == "tower":
        # Bonus: 2x2 factorial — lambda on the B matrices of the language decoder (yes/no) x vision encoder (yes/no),
        # all at the tuned LoRA+ eta_A, so the lambda effect of each tower is separated from the learning-rate change.
        arms = {"neither": ("lambda=1 (neither)", dict(lr_A=lr_plus, ratio=1)),
                "text":    ("lambda text only",   dict(lr_A=lr_plus, ratio=1, ratio_text=lam, ratio_vision=1)),
                "vision":  ("lambda vision only", dict(lr_A=lr_plus, ratio=1, ratio_text=1, ratio_vision=lam)),
                "both":    ("lambda both",        dict(lr_A=lr_plus, ratio=lam))}
        exp = {name: c for key, (name, c) in arms.items() if not cli.arms or key in cli.arms}
        seeds, lora_targets, probe_acc = cli.seeds or SEEDS, TARGET_REGEX, True
    else:
        # Bonus: the connector (fan-in 12288) is ~16-21x wider than any other layer. LoRA+ theory says the optimal
        # eta_B/eta_A grows with fan-in, so a single global lambda is wrong for a VLM -> width-scaled lambda.
        exp = {"LoRA +conn":               dict(lr_A=lr_lora, ratio=1),
               "LoRA+ global +conn":       dict(lr_A=lr_plus, ratio=lam),
               "LoRA+ width-scaled +conn": dict(lr_A=lr_lora, ratio=1, width_scale=True)}
        seeds, lora_targets, probe_acc = SEEDS, TARGET_REGEX_CONN, True
    slug = lambda k: re.sub(r"[^A-Za-z0-9]+", "_", k.replace("+", "plus")).strip("_")
    xres = {k: [run(f"{cli.experiment}_{slug(k)}_lr{c['lr_A']:g}_r{c['ratio']:g}_s{s}", c["lr_A"], c["ratio"], s,
                    MAIN_EPOCHS, N_TRAIN, probe_acc=probe_acc, eval_test=True, lora_targets=lora_targets,
                    width_scale=c.get("width_scale", False), ratio_text=c.get("ratio_text"),
                    ratio_vision=c.get("ratio_vision")) for s in seeds]
            for k, c in exp.items()}
    if cli.experiment == "tower" and len(xres) == 4:
        # main effects of lambda per tower, paired by seed: average over the other tower's two levels
        acc = {k: {r["seed"]: 100 * r["test_acc"] for r in rs} for k, rs in xres.items()}
        common = sorted(set.intersection(*[set(v) for v in acc.values()]))
        n, t, v, b = (acc[k] for k in ["lambda=1 (neither)", "lambda text only", "lambda vision only", "lambda both"])
        eff = pd.DataFrame([dict(seed=s, text_effect=((t[s] - n[s]) + (b[s] - v[s])) / 2,
                                 vision_effect=((v[s] - n[s]) + (b[s] - t[s])) / 2,
                                 interaction=(b[s] - v[s]) - (t[s] - n[s])) for s in common])
        eff.to_csv(f"{OUT_DIR}/tower_effects.csv", index=False)
        print(f"\n=== tower: main effect of lambda={lam:g} per tower (test acc points, paired by seed) ===\n",
              eff.round(2).to_string(index=False), "\n mean:", eff.drop(columns="seed").mean().round(2).to_dict(),
              "\n std: ", eff.drop(columns="seed").std(ddof=0).round(2).to_dict())

    ref_runs = next(iter(xres.values()))                       # first config is the LoRA reference
    tgt_loss = np.mean([r["val_loss"][-1][1] for r in ref_runs])
    first = lambda curve, ok: next((s for s, v in curve if ok(v)), np.nan)
    xt = pd.DataFrame([dict(method=k, eta_A=rs[0]["lr_A"], ratio=rs[0]["ratio"], width_scale=rs[0]["width_scale"],
                            test_acc=np.mean([100 * r["test_acc"] for r in rs]),
                            test_acc_std=np.std([100 * r["test_acc"] for r in rs]),
                            final_val_loss=np.mean([r["val_loss"][-1][1] for r in rs]),
                            steps_to_ref_val_loss=np.nanmean([first(r["val_loss"], lambda v: v <= tgt_loss) for r in rs])
                            if any(np.isfinite(first(r["val_loss"], lambda v: v <= tgt_loss)) for r in rs) else np.nan,
                            n_seeds=len(rs)) for k, rs in xres.items()])
    xt.to_csv(f"{OUT_DIR}/{cli.experiment}_table.csv", index=False)
    print(f"\n=== {cli.experiment} experiment (reference = {next(iter(xres))}, baseline test "
          f"{100*baseline['test_acc']:.2f}%) ===\n", xt.round(4).to_string(index=False))

    dl = pd.DataFrame([dict(d, method=k, seed=r["seed"]) for k, rs in xres.items() for r in rs for d in r["deltas"]])
    dl.to_csv(f"{OUT_DIR}/{cli.experiment}_deltas.csv", index=False)
    towers = [t for t in ["text", "vision", "connector"] if t in set(dl.tower)]
    fig, axes = plt.subplots(1, len(towers), figsize=(6 * len(towers), 4), squeeze=False)
    for ax, tw in zip(axes[0], towers):
        for k, g in dl[dl.tower == tw].groupby("method"):
            g = g.groupby("step").log10_ratio_A_over_B.mean()
            ax.plot(g.index, g.values, marker=".", label=k)
        ax.axhline(0, c="gray", ls="--", lw=1)
        ax.set_title(f"{tw}: log10(δ_A / δ_B), median over layers"); ax.set_xlabel("step"); ax.grid(alpha=.3)
        ax.legend(fontsize=8)
    plt.suptitle("Paper's efficiency terms: 0 = A and B contribute equally per step")
    plt.tight_layout(); plt.savefig(f"{OUT_DIR}/{cli.experiment}_deltas.png", dpi=150, bbox_inches="tight"); plt.close()
    late = dl[dl.step >= dl.step.max() / 2].groupby(["method", "tower"])[["delta_A", "delta_B", "log10_ratio_A_over_B"]]
    late = late.median()
    late.to_csv(f"{OUT_DIR}/{cli.experiment}_deltas_summary.csv")
    print("\n=== δ_A / δ_B, second half of training (median) ===\n", late.to_string())
    raise SystemExit

# ---- 6c. Main comparison — H2a (final accuracy) and H2b (speed) ------------------------------------------------
configs = {
    "LoRA (tuned)":       (float(best_lora.lr_A), 1),
    "LoRA+ (tuned)":      (float(best_plus.lr_A), float(best_plus.ratio)),
    "LoRA+ (drop-in 16)": (float(best_lora.lr_A), 16),
}
main = {k: [run(f"main_{k.split()[0]}_{k.split()[1].strip('()')}_lr{lr:g}_r{r:g}_s{s}".replace("+", "plus"),
                lr, r, s, MAIN_EPOCHS, N_TRAIN, probe_acc=True, eval_test=True) for s in SEEDS]
        for k, (lr, r) in configs.items()}

# ----------------------------------------------------------------------------------------------------------------
# 7. Results
# ----------------------------------------------------------------------------------------------------------------
lora_runs = main["LoRA (tuned)"]
target_acc  = 0.95 * np.mean([r["acc_curve"][-1][1] for r in lora_runs])   # 95% of LoRA's final val acc
target_loss = np.mean([r["val_loss"][-1][1] for r in lora_runs])           # LoRA's final val loss

def first_step(curve, ok):
    return next((s for s, v in curve if ok(v)), np.nan)

def steps_summary(steps):
    # seeds that reached the target, and the median step counting never-reached seeds as +inf
    # (so the median is defined only when at least half the seeds got there) — avoids nanmean hiding failures
    x = np.array([np.inf if not np.isfinite(s) else s for s in steps], dtype=float)
    med = float(np.median(x))
    return int(np.isfinite(x).sum()), (med if np.isfinite(med) else np.nan)

rows = [dict(method="Baseline (zero-shot)", test_acc=100 * baseline["test_acc"])]
for k, runs in main.items():
    ta = [100 * r["test_acc"] for r in runs]
    n_acc, med_acc = steps_summary([first_step(r["acc_curve"], lambda v: v >= target_acc) for r in runs])
    n_loss, med_loss = steps_summary([first_step(r["val_loss"], lambda v: v <= target_loss) for r in runs])
    rows.append(dict(method=k, eta_A=runs[0]["lr_A"], ratio=runs[0]["ratio"],
                     test_acc=np.mean(ta), test_acc_std=np.std(ta),
                     final_train_loss=np.mean([r["train_loss"][-1][1] for r in runs]),
                     final_val_loss=np.mean([r["val_loss"][-1][1] for r in runs]),
                     acc95_reached=f"{n_acc}/{len(runs)}", acc95_median_step=med_acc,
                     loss_reached=f"{n_loss}/{len(runs)}", loss_median_step=med_loss,
                     max_grad_norm=np.max([r["max_grad_norm"] for r in runs]),
                     trainable_pct=100 * runs[0]["trainable"] / runs[0]["total"], train_min=np.mean([r["train_min"] for r in runs])))
res_tbl = pd.DataFrame(rows)
res_tbl.to_csv(f"{OUT_DIR}/main_table.csv", index=False)
r0 = lora_runs[0]
acc_res = int(np.diff([s for s, _ in r0["acc_curve"]]).min()) if len(r0["acc_curve"]) > 1 else 0
loss_res = int(np.diff([s for s, _ in r0["val_loss"]]).min()) if len(r0["val_loss"]) > 1 else 0
print(f"\n=== Main table (H2a/H2b) ===\ntarget val acc = {100*target_acc:.2f}%   target val loss = {target_loss:.4f}   "
      f"(seeds {SEEDS}; val acc checked every ~{acc_res} steps, val loss every ~{loss_res} steps)")
print(res_tbl.round(4).to_string(index=False))

fig, axes = plt.subplots(1, 3, figsize=(18, 4.5))
for k, runs in main.items():
    # val accuracy curve (H2b)
    steps = [s for s, _ in runs[0]["acc_curve"]]
    A = np.array([[v for _, v in r["acc_curve"]] for r in runs]) * 100
    x = [0] + steps; mu = np.r_[100 * baseline["val_acc"], A.mean(0)]; sd = np.r_[0, A.std(0)]
    axes[0].plot(x, mu, marker="o", label=k); axes[0].fill_between(x, mu - sd, mu + sd, alpha=.2)
    # val loss curve (H2b)
    L = np.array([[v for _, v in r["val_loss"]] for r in runs]); ls = [s for s, _ in runs[0]["val_loss"]]
    axes[1].plot(ls, L.mean(0), label=k); axes[1].fill_between(ls, L.mean(0) - L.std(0), L.mean(0) + L.std(0), alpha=.2)
    # train loss (stability)
    T = np.array([[v for _, v in r["train_loss"]] for r in runs]); ts = [s for s, _ in runs[0]["train_loss"]]
    axes[2].plot(ts, T.mean(0), label=k, lw=1)
axes[0].axhline(100 * target_acc, ls="--", c="gray", lw=1, label="target (95% of LoRA)")
axes[1].axhline(target_loss, ls="--", c="gray", lw=1, label="LoRA final val loss")
for ax, t in zip(axes, ["val VQA acc (%)", "val loss", "train loss"]):
    ax.set_title(t); ax.set_xlabel("step"); ax.grid(alpha=.3); ax.legend(fontsize=8)
plt.tight_layout(); plt.savefig(f"{OUT_DIR}/curves.png", dpi=150, bbox_inches="tight"); plt.close()

# ---- 7a. Mechanism (H3): how much of Delta W comes from A actually learning? -----------------------------------
sh = pd.concat([pd.DataFrame(r["shares"]).assign(method=k, seed=r["seed"]) for k, runs in main.items() for r in runs])
fig, axes = plt.subplots(1, 2, figsize=(13, 4))
for (k, tw), g in sh.groupby(["method", "tower"]):
    g = g.groupby("step")[["share_A", "rel_move_A"]].mean()
    ls = "-" if tw == "vision" else "--"
    axes[0].plot(g.index, g.share_A, ls, marker=".", label=f"{k} / {tw}")
    axes[1].plot(g.index, g.rel_move_A, ls, marker=".", label=f"{k} / {tw}")
axes[0].set_title("share of ΔW from A learning:  ‖B(A−A₀)‖ / (‖BA₀‖+‖B(A−A₀)‖)")
axes[1].set_title("relative movement of A:  ‖A−A₀‖ / ‖A₀‖"); axes[1].set_yscale("log")
for ax in axes: ax.set_xlabel("step"); ax.grid(alpha=.3); ax.legend(fontsize=7)
plt.tight_layout(); plt.savefig(f"{OUT_DIR}/ab_shares.png", dpi=150, bbox_inches="tight"); plt.close()
final_sh = sh[sh.step == sh.step.max()].groupby(["method", "tower"])[["share_A", "rel_move_A"]].mean()
final_sh.to_csv(f"{OUT_DIR}/ab_shares_final.csv")
print("\n=== A/B shares at final step (H3) ===\n", final_sh.round(4).to_string())
if cli.stage == "main":
    raise SystemExit

# ---- 7b. Bonus (vision-specific): which tower carries the LoRA+ benefit? ---------------------------------------
lr, lam = configs["LoRA+ (tuned)"]
bonus = {
    "λ only in vision encoder":   run(f"bonus_vision_lr{lr:g}_r{lam:g}", lr, 1, SEEDS[0], MAIN_EPOCHS, N_TRAIN,
                                     probe_acc=True, eval_test=True, ratio_vision=lam, ratio_text=1),
    "λ only in language decoder": run(f"bonus_text_lr{lr:g}_r{lam:g}", lr, 1, SEEDS[0], MAIN_EPOCHS, N_TRAIN,
                                     probe_acc=True, eval_test=True, ratio_vision=1, ratio_text=lam),
}
ref = {"LoRA (tuned)": main["LoRA (tuned)"][0], "LoRA+ (both towers)": main["LoRA+ (tuned)"][0]}
btbl = pd.DataFrame([dict(setting=k, test_acc=100 * r["test_acc"], final_val_loss=r["val_loss"][-1][1],
                          steps_to_LoRA_val_loss=first_step(r["val_loss"], lambda v: v <= target_loss))
                     for k, r in {**ref, **bonus}.items()])
btbl.to_csv(f"{OUT_DIR}/bonus_table.csv", index=False)
print("\n=== Bonus: which tower carries the benefit? ===\n", btbl.round(4).to_string(index=False))
print(f"\nDone. All tables (.csv) and figures (.png) are in {os.path.abspath(OUT_DIR)}")
