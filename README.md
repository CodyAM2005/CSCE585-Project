# How Many Latent Layers Are Enough? Exploring the Impact of Latent Depth in VLAs  

## Group Info  
- Cody Miller   
  - Email: codyam@email.sc.edu  
- Ian Attmore  
  - Email: iattmore@email.sc.edu  
- Dave Ringler  
  - Email: dringler@email.sc.edu  

## Project Summary/Abstract  
### A short summary of the project (3–5 sentences maximum).  

Vision-Language-Action (VLA) models for robot control typically bolt a large pretrained LLM backbone onto a vision encoder and action head, but it is unclear how much of that backbone's depth is actually needed for a given manipulation task. This project replicates VLA-Adapter (Wang et al., 2025), a tiny-scale VLA built on a Qwen2.5-0.5B backbone, on the LIBERO manipulation benchmark, then conducts an ablation that removes the final 1, 3, and 12 decoder layers (of 24) and re-trains a matching action head for each configuration. We evaluate the resulting task success rate alongside inference latency, and built a dedicated benchmarking tool to measure the latency impact of genuinely removing layers versus the eval-time hidden-state slicing the naive implementation uses, since the latter executes every layer regardless of how many are "dropped." The goal is to characterize the performance/efficiency tradeoff as latent depth is reduced, and determine how many layers are actually necessary for this class of task.

## Problem Description  
- Problem description: VLA models deploy LLM backbones with many decoder layers inherited from language pretraining, but it is unknown how many of those layers are necessary once the model is fine-tuned for a narrower robotic control task. We quantify this by removing the final *k* decoder layers of VLA-Adapter's Qwen2.5-0.5B backbone for k ∈ {1, 3, 12}, re-training the action head for each *k*, and measuring the resulting task success rate and inference latency relative to the unpruned (k=0) baseline.
- Motivation  
  - Robot control is often latency- and compute-constrained (real-time actuation on embedded/edge hardware), so a shallower backbone that preserves task performance would directly reduce inference cost without a smaller/retrained-from-scratch model.
  - Every task gets the same model depth, whether it’s “pick up the bowl” or a long multi-step task
  - Measuring this directly, rather than assuming layer count scales with performance, gives a principled way to decide how much of a pretrained backbone a given deployment target actually needs to keep.
- Challenges  
  - The straightforward way to "drop" layers — slicing off the last *k* hidden states before the action head — does not actually skip executing those layers in the underlying LLM forward pass, so it produces zero real latency savings; isolating the true effect of removal required building a separate benchmark that patches the model to truly skip layers and verifies the result is numerically equivalent to the naive approach.
  - Each value of *k* changes the number of decoder hidden states available to the action head, so each configuration needs its own re-trained action head and checkpoint rather than reusing one model across all k values — multiplying the training cost of the ablation by the number of configurations tested.
  - Training and evaluating four full model variants (k=0, 1, 3, 12) plus running LIBERO benchmark rollouts for each is computationally expensive, requiring access to shared HPC/cloud GPU resources, which constrained how many configurations and seeds could be tested within the project timeline.

## Contribution  
### [`Replication of existing work`], [`Extension of existing work`]  
We use 'VLA-Adapter: An Effective Paradigm for Tiny-Scale Vision-Language-Action Model' [https://arxiv.org/abs/2509.09372] to reproduce their results. We improve their method as follows:  
- Contribution 1  
- Contribution 2  

## References  
### `BibTeX` of all references used in the project. Can be included as a `references.bib` file in the repo.  

---

# < The following is only applicable for the final project submission >  

## Dependencies

- Python 3.10.16
- Linux with a CUDA-capable GPU (an A100/H100 class card for training; any CUDA GPU works for the latency benchmark)
- conda (Anaconda or Miniconda)
- PyTorch 2.2.0, torchvision 0.17.0, torchaudio 2.2.0
- transformers 4.40.1, flash-attn 2.5.5, plus the rest of the pinned dependencies in each `EfficientVLA*/pyproject.toml` (accelerate, peft, timm, tensorflow 2.15, etc.) — these install automatically via `pip install -e .`
- **Only required for the LIBERO simulated-task accuracy eval** (`experiments/robot/libero/run_libero_eval.py`), not for training or the latency benchmark: the [LIBERO](https://github.com/Lifelong-Robot-Learning/LIBERO) simulator package and `robosuite`

⚠️ Current Deviation from the course template: this project currently uses `conda` + `pip`, matching the upstream [VLA-Adapter](https://github.com/OpenHelix-Team/VLA-Adapter) codebase this project is built on, rather than `uv`. There is no `uv.lock` file. Migrating to `uv` would mean re-resolving `tensorflow`/`flash-attn`/CUDA-specific wheels, which is a non-trivial separate task — flagging this now rather than silently claiming compliance.
TODO: figure out if we need to change this to be compatible with uv instead of pip. 

## Directory Structure

```
CSCE585-Project/
├── EfficientVLA/                 # baseline VLA-Adapter (k=0): all 24 Qwen2.5-0.5B decoder layers run
│   ├── prismatic/                #   model code: vision backbone (DINO+SigLIP) + Qwen2.5-0.5B LLM + action head
│   ├── vla-scripts/               #   finetune.py, train.py, evaluate_calvin.py, deploy.py, merge_lora_weights_and_save.py
│   ├── experiments/robot/         #   eval harness: run_libero_eval.py, openvla_utils.py, robot_utils.py
│   ├── scripts/                   #   preprocess.py, pretrain.py, generate.py
│   ├── outputs/                   #   (populated by the setup steps below) trained checkpoints — not tracked in git
│   ├── data/, pretrained_models/  #   (populated by the setup steps below) downloaded datasets / VLM backbone — not tracked in git
│   └── run*.sh                    #   SLURM job scripts for training/eval on a Slurm cluster
├── EfficientVLALayerDrop/        # drops the final 1 decoder-layer hidden state before the action head (k=1); same layout
├── EfficientVLAManyLayerDrop/    # drops the final 3 decoder-layer hidden states (k=3); same layout
├── EfficientVLATwelveLayerDrop/  # drops the final 12 decoder-layer hidden states (k=12); same layout
├── latency_bench/                # our benchmark: compares "hidden states ignored" vs. "layers truly removed"
│   ├── bench_end_removal.py       #   core benchmark: loads a checkpoint, times inference both ways, verifies equivalence
│   ├── run_end_removal_latency.sh #   Slurm batch script running k=0,1,3 back-to-back on one GPU/node
│   ├── summarize_end_removal.py   #   combines per-k JSON results into a markdown/CSV table
│   └── results/                   #   (created when you run it) per-run JSON + summary tables — not tracked in git
├── Project-Proposal.md
├── CSCE585_Partial_Results.pdf
├── 585 Presentation.pptx
└── README.md
```

⚠️ Deviation from the course template: the mandatory root-level `run.<ext>` / `result.<ext>` / `data/` layout shown in the template doesn't exist at the project root today. The real equivalents are nested under `latency_bench/`: `run_end_removal_latency.sh` is the run script, `summarize_end_removal.py` is the result/plot script, and `latency_bench/results/` holds the result CSV/JSON instead of a root `data/` folder. Flagging this rather than describing files that aren't actually there — if the grading rubric needs literally-named root files, that's a separate follow-up to add thin wrapper scripts.
TODO: add wrapper scripts

## How to Run

These steps assume a Slurm-managed CUDA GPU cluster (what this project has been run on so far); adjust step 7 if running on a non-Slurm machine instead (e.g. drop the `#SBATCH` lines and invoke the `python` commands directly).

### 1. Clone the repo
```bash
git clone https://github.com/CodyAM2005/CSCE585-Project.git
cd CSCE585-Project
```

### 2. Set up the conda environment
```bash
conda create -n vla-adapter python=3.10.16 -y
conda activate vla-adapter
pip install torch==2.2.0 torchvision==0.17.0 torchaudio==2.2.0
cd EfficientVLA
pip install -e .
pip install packaging ninja
pip install "flash-attn==2.5.5" --no-build-isolation
cd ..
```
⚠️ Important: `EfficientVLA`, `EfficientVLALayerDrop`, `EfficientVLAManyLayerDrop`, and `EfficientVLATwelveLayerDrop` all install an editable package literally named `prismatic`. If you `pip install -e .` more than one of them into the *same* conda environment, the second install silently overwrites which repo's code `import prismatic` resolves to — there is no warning. To train or eval against a specific variant, re-run `pip install -e .` from inside that repo's directory first (or keep one conda env per variant if you need more than one installed at once). The `latency_bench` scripts below sidestep this entirely by inserting each repo's path directly rather than relying on the installed package, so they do not require re-running `pip install -e .` when switching between k values.

### 3. Download the VLM backbone — only if you intend to train a new checkpoint
Not needed to run the latency benchmark (step 6): a trained checkpoint is fully self-contained (weights, config, `dataset_statistics.json`, and the action-head/proprio-projector `.pt` files) — the backbone below is only read via `--vlm_path` during fine-tuning.
```bash
# Download once, then symlink into each repo instead of downloading it 4x
git clone https://huggingface.co/Stanford-ILIAD/prism-qwen25-extra-dinosiglip-224px-0_5b pretrained_models/prism-qwen25-extra-dinosiglip-224px-0_5b
for repo in EfficientVLA EfficientVLALayerDrop EfficientVLAManyLayerDrop EfficientVLATwelveLayerDrop; do
  mkdir -p "$repo/pretrained_models"
  ln -s "$(pwd)/pretrained_models/prism-qwen25-extra-dinosiglip-224px-0_5b" "$repo/pretrained_models/"
done
```

### 4. Download the LIBERO training dataset — only if you intend to train a new checkpoint
Also not needed for the latency benchmark, and not needed for training itself unless you're fine-tuning (the benchmark uses a synthetic observation and never touches LIBERO's simulator or raw dataset at all — only `prismatic/vla/datasets/rlds/dataset.py`'s training data loader does).
```bash
git clone git@hf.co:datasets/openvla/modified_libero_rlds data/libero
# Per EfficientVLA/README.md: strip the "modified_" prefix from each dataset folder name, then
# symlink data/libero into each EfficientVLA*/data/libero the same way as step 3.
```

### 5. Get checkpoints
- **k=0 baseline**: can use VLA-Adapter's official published checkpoints directly, e.g. [VLA-Adapter/LIBERO-Spatial-Pro](https://huggingface.co/VLA-Adapter/LIBERO-Spatial-Pro), placed under `EfficientVLA/outputs/`.
- **k=1 / k=3 / k=12 (layer-drop variants)**: cannot use the official checkpoints — their action head has a different block count (24 − k; see `prismatic/models/action_heads.py` in each repo) that only matches a checkpoint trained with that specific repo. Train your own via `vla-scripts/finetune.py` (see each repo's README) from inside the matching `EfficientVLA*` directory.

### 6. Run the latency benchmark
```bash
cd latency_bench
python bench_end_removal.py \
  --repo_dir ../EfficientVLALayerDrop \
  --pretrained_checkpoint <path-to-k1-checkpoint> \
  --num_end_layers_drop 1 \
  --output_json results/k1.json
```
Or run all configured k values at once via the batch script (edit `ROOT` and the `CKPT_K0/K1/K3` paths at the top first to match where your checkpoints actually live):
```bash
sbatch run_end_removal_latency.sh
```

### 7. Summarize the results
```bash
python summarize_end_removal.py --results_dir results/<run-folder>
```
This writes `end_removal_summary.md` and `.csv` with per-k latency, and `e2e_speedup_vs_k0` — the actual number this project is trying to produce.

## Demo  
- All projects must include video(s) demonstrating your project.  
- Please use annotations/explanations to clarify what is happening in the demo.  
---
