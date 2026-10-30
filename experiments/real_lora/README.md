# SPEAR: real-weight LoRA counterfactual feasibility experiment

**Status: code + CPU protocol tests. No Qwen3.5-4B GPU measurements or AndroidWorld results have been produced by this branch.**

This is a **real-model training runner** (Transformers + PEFT, 4-bit NF4 LoRA), evaluated in a **controlled, resettable transaction-tool environment**. The built-in teacher, documentation and probe are deterministic text **fixtures**, not live teacher API calls, web retrieval, or an AndroidWorld emulator. Option `--external-reports` allows importing independently collected evidence with provenance checks. We deliberately distinguish *genuine model parameter updates* from *real-world diagnostic source provenance*.

## Research question

Does conditioning adapter training on a diagnosis improve **persistent held-out tool-trajectory performance** over a blind training baseline with identical action labels? Do the preferred updates reverse across failures that initially look the same? Does that survive comparison to a strong rule based on the same allowed local inspection, and after counting the cost of testing candidates?

**This module is an early G1/G2 feasibility pilot. It does not implement a learned PI-VOI controller, an online evidence-efficient source-selection algorithm, or recursive self-improvement.** Its purpose is to collect trustworthy intervention-outcome data before building those.

## Task construction

Every case starts from the exact same observable error, `E_REJECTED: operation rejected. Reason unavailable.` The initial prompt omits mechanism names, IDs and private values. A local `inspect_failure` tool is permitted both during training and **at test**. It returns ordinary state metadata, unlike the external diagnosis text used only during LoRA training.

| Latent failure | Local inspection fields | Correct tool trajectory | Relevant fixture evidence |
|---|---|---|---|
| Schema mismatch | `observed_field`, `required_field` | inspect → `rename_field(from,to)` → submit | `docs` |
| Policy violation | `requested_units`, `safe_cap` | inspect → `cap_units(units)` → submit | `teacher` |
| Stale revision | `submitted_revision`, `latest_revision` | inspect → `refresh_revision(revision)` → submit | `probe` |

The environment has a **5-step cap**, strict deterministic state checker, and side-effect accounting for inappropriate actions. A wrong repair is a strict failure, even if a later correction would otherwise succeed. Model outputs must be JSON tool calls.

**Identifiability caution:** if the entire state visible before a query is identical, no source policy can predict which hidden mechanism exists without making an observation. A strong same-input source rule is therefore allowed the *same* local inspection as a learned policy. Using the private mechanism label as a router feature is invalid.

## Train / validation / test isolation

A deterministic `manifest.json` contains three mechanisms per family. Train, validation and test have disjoint **family IDs** and generated parameter values. The source-selection / adapter acceptance decision uses the validation split only. The final test is never selected on. Test prompts have **no injected teacher, documentation, or external probe report**; the ordinary `inspect_failure` action remains available.

Each target mechanism can be adapted using the sources `blind`, `docs`, `teacher`, and `probe`, producing **12 independently trained adapters** for the full matrix. Every arm uses the same gold checker-derived action supervision targets, examples, base model, and optimizer step budget; **only diagnostic conditioning differs**. This prevents a trivial benefit caused by giving one source more correctly labeled data. Input token lengths still differ and are recorded, and the source texts are designer-defined fixtures—do not claim an information-theoretic gain from them.

A validation-only acceptance gate requires an improvement in target-skill strict success and no more than a predeclared regression of previous-skill cases (default 2.5 percentage points). This is an engineering pilot gate, **not** a statistical significance test. Candidate adapters are evaluated on all test cases regardless of acceptance to expose regressions; deployed numbers use accepted-or-rollback only.

## Step 1 — CPU preflight, no model download

Run from repository root with Python 3.10+:

```bash
git clone https://github.com/Lendle-King/SPEAR.git
cd SPEAR
git switch experiment/real-lora-counterfactual

python -m unittest discover -s experiments/real_lora/tests -v

python -m experiments.real_lora.prepare \
  --seed 42 --train-groups 48 --validation-groups 12 --test-groups 24 \
  --out runs/real_lora/manifest.json

python -m experiments.real_lora.run plan \
  --manifest runs/real_lora/manifest.json
```

The GitHub Actions workflow only executes these CPU invariants; it **never trains a 4B model automatically**.

## Step 2 — Set up CUDA host (e.g. RTX 4090)

Use a clean Python virtual environment and install CUDA-matched PyTorch, then:

```bash
pip install -r experiments/real_lora/requirements-real.txt
nvidia-smi

export CUDA_VISIBLE_DEVICES=0
export MODEL=/path/to/actual/Qwen3.5-4B/checkpoint
export BASE_ID="your-immutable-checkpoint-snapshot-or-weight-digest"
export REVISION=main  # pin an exact commit when loading a Hub checkpoint
export RUN_ROOT=runs/real_lora_pilot
```

The loader supports `--model-class causal` (default) or `--model-class image-text` and `--quantization 4bit` (default). **Compatibility with any particular Qwen3.5 checkpoint, especially multimodal architectures, has not yet been established here.** The code fails if the checkpoint's expected LoRA projection modules are absent. Change `--target-modules` as necessary based on the *actual* model architecture. HF revision and the operator-supplied `BASE_ID` are checked across arms; weight-byte hashing of the several-GB base model is not performed automatically. For reproducibility, maintain a pinned snapshot and its external cryptographic digest.

## Step 3 — Minimal paired two-arm real-LoRA pilot

A lower-cost initial gate trains `blind` and `docs` for schema recovery, each in a fresh process:

```bash
python -m experiments.real_lora.prepare \
  --seed 42 --train-groups 12 --validation-groups 6 --test-groups 12 \
  --out "$RUN_ROOT/manifest.json"

for source in blind docs; do
  python -m experiments.real_lora.run train \
    --manifest "$RUN_ROOT/manifest.json" \
    --model "$MODEL" --revision "$REVISION" --base-id "$BASE_ID" \
    --target schema --source "$source" --steps 60 \
    --out "$RUN_ROOT/adapters/schema/$source"

  python -m experiments.real_lora.run evaluate \
    --manifest "$RUN_ROOT/manifest.json" \
    --model "$MODEL" --revision "$REVISION" --base-id "$BASE_ID" \
    --target schema --adapter "$RUN_ROOT/adapters/schema/$source" \
    --out "$RUN_ROOT/eval/schema/$source"
done

python -m experiments.real_lora.run report \
  --root "$RUN_ROOT" \
  --out "$RUN_ROOT/counterfactual_report.json"
```

Training uses real PEFT gradients and saves adapter weights. Evaluation uses **the same loaded base checkpoint with the adapter disabled/enabled**, generating independent full tool trajectories for each validation and test case. The baseline and candidate use the same case and reset state; sampling is deterministic (`do_sample=False`). Models are not sequentially trained from the previous adapter.

## Step 4 — Optional complete 12-arm experiment, serial only

```bash
# SAFE: dry-run generates the split manifest and prints the plan; no GPU work.
MODEL="$MODEL" BASE_ID="$BASE_ID" RUN_ROOT=runs/spear-real-lora-full \
  bash experiments/real_lora/run_matrix.sh

# EXPLICIT consent to GPU training: 3 mechanisms x 4 sources
EXECUTE=1 MODEL="$MODEL" BASE_ID="$BASE_ID" \
  RUN_ROOT=runs/spear-real-lora-full TARGETS="schema policy stale" \
  SOURCES="blind docs teacher probe" STEPS=80 \
  bash experiments/real_lora/run_matrix.sh
```

The guarded script trains **sequentially**, in separate processes, to avoid concurrently loading twelve 4B models. It refuses to overwrite existing adapter directories or evaluation outputs; use a fresh run root when changing the protocol. For a smaller 4-arm test set `TARGETS="schema"`.

## Outputs and validity checks

| Path | What it contains |
|---|---|
| `runs/.../manifest.json` | Private frozen train/validation/test worlds, SHA-256 integrity digest |
| `adapters/<target>/<source>/adapter_model.safetensors` | **Real LoRA weight update**; no checkpoint is committed on failure |
| `adapters/<target>/<source>/spear_training.json` | Source, immutable base identifier, label/dataset hash, steps, train wall time, token counts, peak VRAM, adapter hash |
| `adapters/<target>/<source>/source_provenance.jsonl` | Fixture vs independently supplied evidence provenance |
| `eval/<target>/<source>/paired_trajectories.jsonl` | Every baseline/candidate action, tool observation, strict checker and side effects |
| `eval/<target>/<source>/summary.json` | Validation-only accept/rollback, held-out new/old skill, cross-mechanism matrix, resource ledger |
| `counterfactual_report.json` | All trained arms, same-input source-rule benchmark, validation-best source and full search cost, paired bootstrap vs blind where raw traces are available |

The cost ledger records measured training/evaluation wall seconds and actual source-fixture query counts; **it leaves real API dollar costs null, not zero**. End-to-end budget comparisons still need measured GPU pricing and any external teacher/retrieval invoices. The retrospective validation-best source trains **all** source adapters; it is an expensive exhaustive reference, **not evidence-efficient PI-VOI**. The proposed learned evidence-selection mechanism remains future work.

## Optional: import actual teacher or documentary diagnostics

For `train --external-reports /path/to/reports.jsonl`, provide one text evidence record per **train case**, with exact `case_id`, matching `source`, `content`, and non-empty `provenance`, for example:

```json
{"case_id":"train-family-0000-schema","source":"docs","content":"Versioned API documentation explains the changed argument field","provenance":"docs_sha256:operator-recorded"}
```

The loader rejects missing cases, duplicates, wrong sources and any validation/test IDs. Real external evidence must be collected separately and its costs logged; this runner does not call online services. All arms still use the same checker-derived correct SFT targets to isolate the value of added diagnostic text.

## Scientific go/no-go

1. **G1:** A real LoRA update improves held-out complete tool trajectories beyond the blind **equal-label** baseline. If not, there is no empirical reason to train a sophisticated acquisition policy yet.
2. **G2:** After a valid local inspection, the best durable adapter/evidence source should genuinely depend on the failure context; compare directly to the strong same-input rule, as well as fixed sources and exhaustive candidate validation **with total training/verification costs included**. Do not claim G2 simply because the synthetic environment has three distinct mechanisms.
3. **G3:** Confirm persistent new-skill gains and bounded old-skill regression in held-out episodes with no auxiliary diagnosis injected.
4. **G4 / RSI:** Not yet implemented in this pilot. Would require learned acquisition/update meta-policies outperforming frozen counterparts given the same replay archive and budget.

**Do not promote a passed CPU checker test or a working adapter training loop into a model-quality result.** On this branch only the experiment implementation and CPU checks have been run; actual 4B GPU fitting and the subsequent statistical analysis remain to be executed.
