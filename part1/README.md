# Part 1: Does the LoRA+ claim transfer to a vision–language model?

EN4554 Deep Learning for Vision, Assignment 1, Part 1.

We re-implement **LoRA+** (Hayou, Ghosh & Yu, 2024), which sets the learning rate of the LoRA `B` matrix
λ times higher than that of `A`. We test whether its claims still hold when fine-tuning
**SmolVLM-256M-Instruct** for visual question answering on **VQAv2** and **TextVQA**.

**Result in one line:** they do not. A tuned plain-LoRA learning rate matches or beats LoRA+ in accuracy and
speed on both datasets. The paper's own efficiency measure (δ<sub>A</sub>/δ<sub>B</sub>) shows that the
imbalance LoRA+ is designed to fix is not present at this model width. A large `B` learning rate specifically
hurts the vision encoder. The full write-up is in `report/part1_report.tex`.

## Main results

Test accuracy, mean ± std over 3 seeds (42/43/44):

| Method | VQAv2 (exact match) | TextVQA (VQA accuracy) |
|---|---|---|
| Zero-shot SmolVLM-256M | 29.4 | 32.7 |
| LoRA, tuned η | **54.3 ± 0.7** (η = 2e-4) | **43.0 ± 1.5** (η = 5e-4) |
| LoRA+, tuned (η<sub>A</sub>, λ) | 53.5 ± 0.5 (1e-4, 4) | 41.3 ± 0.1 (1e-4, 16) |
| LoRA+ with the paper's λ = 16 at LoRA's η | 37.1 ± 6.2 | 11.2 ± 0.5 |

With η<sub>A</sub> fixed at 1e-4 on TextVQA (3 seeds), λ = 16 gives 40.8 ± 1.6 against 45.0 ± 0.2 for λ = 1.

**Which tower is hurt (TextVQA, 2×2 ablation, 3 seeds, η<sub>A</sub> = 1e-4).** Main effect of λ = 16, in
test-accuracy points, paired by seed:

| λ = 16 applied to | Effect | Per seed |
|---|---|---|
| Vision encoder | **−4.5 ± 0.6** | −4.5, −5.3, −3.8 |
| Language decoder | +0.3 ± 1.0 | +1.5, −1.0, +0.4 |
| Interaction | +0.4 ± 2.5 | – |

Other claim-tied results:

| What | Where it is saved |
|---|---|
| δ<sub>A</sub>/δ<sub>B</sub> | `*_deltas_summary.csv` |
| Accuracy by question type | `difficulty_*.csv` |
| Connector experiment | `connector_table.csv` |
| Tower ablation | `tower_table.csv`, `tower_effects.csv` |

The per-run numbers behind every table are in `outputs/*/runs/*/results.json`.

## Repository layout

| Path | What it is |
|---|---|
| `part1_loraplus_smolvlm.py` | **The experiment.** Data split, assistant-only-loss collator, LoRA+ optimiser (two AdamW groups: `lora_B` at λ·η, everything else at η), VQA evaluation, probes (validation accuracy, A/B weight shares, the paper's δ<sub>A</sub>/δ<sub>B</sub>), tables and plots. |
| `analyze_by_difficulty.py` | Accuracy split by question type, from the saved predictions (no GPU needed). |
| `colab_reproduce_part1.ipynb` | Retrains the main LoRA vs LoRA+ comparison from scratch on a free Colab T4 (see below). |
| `outputs/part1_loraplus/` | VQAv2 results: `grid.csv`, `main_table.csv`, `delta_*`, `connector_*`, `difficulty_*`, plots, and one `runs/<name>/results.json` per training run. |
| `outputs/part1_loraplus_textvqa/` | TextVQA results (same structure, plus `tower_*`). |
| `report/` | LaTeX report (`part1_report.tex`), its figures, and `make_report_figures.py`, which rebuilds the figures from the CSVs. |
| `part1_loraplus_smolvlm.ipynb` | The original notebook. It is superseded by the `.py` and still has the old VQAv2 scoring (see the notes below). |
| `part1_dora_smolvlm.ipynb` | A notebook for another LoRA variant (DoRA). It is not part of the reported results. |

## Setup

Python 3.11 with a CUDA GPU. We used an RTX 4090 with NVIDIA driver 535 (CUDA 12.2).

```bash
conda create -n vlm-lora python=3.11 -y && conda activate vlm-lora
python -m pip install torch==2.5.1 torchvision==0.20.1 --index-url https://download.pytorch.org/whl/cu121
python -m pip install "transformers>=4.46,<5" peft datasets accelerate num2words pandas matplotlib scipy
```

The first run downloads the model (about 0.5 GB) and the datasets (VQAv2 subset about 1 GB, TextVQA validation
0.9 GB) to `~/.cache/huggingface`.

## Reproducing the results

Run all commands from `part1/`. Every run saves `results.json`, so rerunning a command skips finished runs and
only rebuilds the tables and plots.

**Quick check (about 10 min):**
```bash
python -u part1_loraplus_smolvlm.py --gpu 0 --quick
```

**One main result from scratch (about 10 min on an RTX 4090), without the grid:**
```bash
python -u part1_loraplus_smolvlm.py --gpu 0 --experiment single --lr-a 2e-4 --ratio 1 --seed 42   # tuned LoRA
python -u part1_loraplus_smolvlm.py --gpu 0 --experiment single --lr-a 1e-4 --ratio 4 --seed 42   # tuned LoRA+
```

**Full experiments:**

| Command | What it runs | Time (RTX 4090) |
|---|---|---|
| `python -u part1_loraplus_smolvlm.py --gpu 0` | VQAv2: baseline, (η<sub>A</sub>, λ) grid, 3-seed main comparison, tower bonus | ~2.5 h |
| `... --dataset textvqa` | The same on TextVQA | ~2.5 h |
| `... --experiment delta` | The paper's δ<sub>A</sub>/δ<sub>B</sub> for the main configurations (1 seed) | ~20 min |
| `... --experiment connector` | Bonus: LoRA also on the 12,288-wide vision→text connector, global vs width-scaled λ (3 seeds) | ~1.5 h |
| `... --dataset textvqa --experiment tower` | Bonus: 2×2 ablation of λ in the vision encoder × language decoder (3 seeds) | ~2.5 h (or ~1.2 h split with `--arms`) |
| `python analyze_by_difficulty.py --dataset vqav2` (or `textvqa`) | Accuracy by question type | seconds |
| `python report/make_report_figures.py` | Report figures | seconds |

Useful flags: `--gpu N` picks the GPU; `--stage baseline|grid|main` stops early; `--out-dir` changes where
results go; `--seeds` and `--arms` split the tower ablation across GPUs.

## Running on Google Colab

Open `colab_reproduce_part1.ipynb` in Colab (File → Open notebook → GitHub → this repository) and select a
**T4 GPU** runtime. Then Run all. The notebook:

1. clones this repository and installs the packages;
2. runs a 5-minute smoke test;
3. trains tuned LoRA and tuned LoRA+ from scratch (VQAv2, seed 42);
4. prints their test accuracy next to the committed results, which are 53.4% and 53.2% for seed 42.

Expect about ±1 point of difference from GPU non-determinism. The T4 time per run is an estimate of roughly
25–40 min; we have not measured it. The script uses fp16 autocast, not bf16, so it runs on a T4.

## Experimental details

- **Model:** `HuggingFaceTB/SmolVLM-256M-Instruct`. One 512 px image tile (64 visual tokens). LoRA rank 8,
  α 16, dropout 0.05 on all attention and MLP projections of both towers (3.8M trainable parameters, 1.45%).
  The connector is adapted only in the `connector` experiment.
- **Training:** AdamW (weight decay 0), linear schedule with 5% warm-up, gradient clipping 1.0, fp16 autocast with
  fp32 master weights, batch 8, 2 epochs (750 steps). The loss is on the assistant answer only.
- **Data (data seed 42):**
  - VQAv2: `merve/vqav2-small` (VQAv2 *val*), 3000 / 300 / 500 questions for train / val / test. The subset has
    one answer per question and no image IDs, so the split is random by question and we use exact-match accuracy.
  - TextVQA: `lmms-lab-encoder/textvqa` *validation*, 3000 / 300 / 492, split by image, with standard VQA
    accuracy over 10 answers.
- **Model selection:** on the val split only. The test split is used only for the final numbers.
- **Grids:**
  - VQAv2: η<sub>A</sub> ∈ {5e-5, 1e-4, 2e-4, 5e-4} × λ ∈ {1, 4, 16, 32}.
  - TextVQA: η<sub>A</sub> ∈ {1e-4, 2e-4, 5e-4, 1e-3} × λ ∈ {1, 4, 16}.
  - Each grid cell is a short run (1500 examples, 1 epoch). The λ = 1 column is the learning-rate sweep for
    plain LoRA.
  - Both selected optima are inside their grids. The 1e-3 row for TextVQA was added after review, because 5e-4 was
    originally at the edge.
  - At full budget, the val ranking of LoRA's η on TextVQA is unchanged (5e-4: 41.0 against 1e-4: 38.9).
    On test the order flips (43.0 against 45.0), so test gaps of about 2 points between learning rates fall within
    the noise between splits.

## Notes and corrections

- **VQAv2 scoring was fixed after the first runs.** Standard VQA accuracy, `min(#matching answers / 3, 1)`,
  assumes 10 annotator answers. The VQAv2 subset has only one, so a correct answer originally scored 1/3. We
  switched to exact match and recomputed all saved results from the stored predictions; no retraining was needed.
  Every method was scaled by the same factor of 3, so no ranking changed. The original notebook still contains
  the old scoring.
- **Hypothesis H3 was restated during the project.** Our original notebook predicted that LoRA+ would *increase*
  the share of the update that comes from `A`. After the first results we re-read §3 of the paper. It says `B`
  is the under-updated matrix under equal learning rates, so we restated H3 as "LoRA+ rebalances the per-step
  terms δ<sub>A</sub>, δ<sub>B</sub>". The weight-share measurement was not changed. The δ measurement was added
  afterwards to test the paper's mechanism directly. Under the restated hypothesis the result is still negative:
  under plain LoRA, δ<sub>B</sub> is already slightly *larger* than δ<sub>A</sub>, the opposite of the paper's
  prediction.
- **Step counts.** "Steps to target" is reported as the number of seeds that reached the target and the median
  step. Validation accuracy is checked every ~94 steps and validation loss every ~37 steps, so differences smaller
  than that are not meaningful.
- **Non-determinism.** Re-running the same configuration and seed varies by about 1 point because of GPU
  non-determinism.

## Troubleshooting

| Problem | Fix |
|---|---|
| `pip` installs into the wrong conda env | Use `python -m pip ...` |
| Packages in `~/.local` override the env (e.g. a numpy/pyarrow clash) | `conda env config vars set PYTHONNOUSERSITE=1`, then reactivate the env |
| `CUDA not available` after installing | pip pulled a newer torch than the driver supports; reinstall `torch==2.5.1` from the cu121 index |
| `accelerate` error on RTX 40-series cards (`NCCL_P2P_DISABLE`) | Already handled inside the script |
| No live output when piping to `tee` | Use `python -u` |

## Reference

S. Hayou, N. Ghosh, B. Yu. *LoRA+: Efficient Low Rank Adaptation of Large Models.* arXiv:2402.12354, 2024.
